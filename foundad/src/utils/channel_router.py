from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ChannelRouter(nn.Module):
    """Estimate and route anomaly-sensitive feature channels.

    Inputs are ViT-style token features with shape [B, N, C], where B is the
    batch size, N is the token count, and C is the channel dimension.
    """

    def __init__(
        self,
        top_k: int = 32,
        momentum: float = 0.9,
        use_ema: bool = True,
        use_stability: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.top_k = int(top_k)
        self.momentum = float(momentum)
        self.use_ema = bool(use_ema)
        self.use_stability = bool(use_stability)
        self.eps = float(eps)

        self.register_buffer("running_score", torch.empty(0), persistent=True)
        self.register_buffer("latest_score", torch.empty(0), persistent=True)
        self.register_buffer("latest_sensitivity", torch.empty(0), persistent=True)
        self.register_buffer("latest_leakage", torch.empty(0), persistent=True)
        self.register_buffer("latest_consistency", torch.empty(0), persistent=True)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def update(
        self,
        F_normal: torch.Tensor,
        F_synthetic_probe: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        F_synthetic_probe_2: Optional[torch.Tensor] = None,
        token_mask_2: Optional[torch.Tensor] = None,
        F_aug1: Optional[torch.Tensor] = None,
        F_aug2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update channel scores from a normal/probe feature pair.

        Args:
            F_normal: Normal features, shape [B, N, C].
            F_synthetic_probe: Weak pixel-synthetic probe features, shape [B, N, C].
            token_mask: Optional synthetic token mask, shape [B, N] or [B, N, 1].
                When provided, scoring rewards masked-region sensitivity and
                penalizes background leakage.
            F_synthetic_probe_2: Optional second synthetic probe for consistency.
            token_mask_2: Optional mask for F_synthetic_probe_2. Defaults to token_mask.
            F_aug1: Optional normal augmented features, shape [B, N, C].
            F_aug2: Optional normal augmented features, shape [B, N, C].
        """

        self._validate_feature_pair(F_normal, F_synthetic_probe, "F_normal", "F_synthetic_probe")
        if (F_aug1 is None) ^ (F_aug2 is None):
            raise ValueError("F_aug1 and F_aug2 must be provided together")

        normal = F_normal.detach().float()
        probe = F_synthetic_probe.detach().float()

        delta = probe - normal
        abs_delta = delta.abs()

        if token_mask is None:
            # Sensitivity: average absolute response to weak synthetic anomaly, [C].
            sensitivity = abs_delta.mean(dim=(0, 1))
            leakage = torch.zeros_like(sensitivity)
            consistency = torch.ones_like(sensitivity)
        else:
            mask = self._format_token_mask(token_mask, F_normal).float()
            sensitivity = self._masked_channel_mean(abs_delta, mask)
            leakage = self._masked_channel_mean(abs_delta, 1.0 - mask)
            consistency = torch.ones_like(sensitivity)

            if F_synthetic_probe_2 is not None:
                self._validate_feature_pair(F_normal, F_synthetic_probe_2, "F_normal", "F_synthetic_probe_2")
                mask_2 = mask if token_mask_2 is None else self._format_token_mask(token_mask_2, F_normal).float()
                delta_2 = F_synthetic_probe_2.detach().float() - normal
                response_1 = self._masked_channel_mean(delta, mask)
                response_2 = self._masked_channel_mean(delta_2, mask_2)
                gap = (response_1 - response_2).abs()
                scale = response_1.abs() + response_2.abs() + self.eps
                consistency = (1.0 - gap / scale).clamp(0.0, 1.0)

        preservation = 1.0 / (leakage + self.eps)
        score = sensitivity * preservation * consistency

        if self.use_stability and F_aug1 is not None and F_aug2 is not None:
            self._validate_feature_pair(F_aug1, F_aug2, "F_aug1", "F_aug2")
            if F_aug1.shape != F_normal.shape:
                raise ValueError(
                    f"Augmented features must match F_normal shape, got {F_aug1.shape} and {F_normal.shape}"
                )
            aug1 = F_aug1.detach().float()
            aug2 = F_aug2.detach().float()
            # Stability: inverse channel variance under normal augmentations, [C].
            stability = 1.0 / ((aug1 - aug2).pow(2).mean(dim=(0, 1)) + self.eps)
            score = score * stability

        current_score = self._normalize_score(score).to(device=F_normal.device)
        self.latest_score = current_score.detach().clone()
        self.latest_sensitivity = self._normalize_score(sensitivity).to(device=F_normal.device).detach().clone()
        self.latest_leakage = self._normalize_score(leakage).to(device=F_normal.device).detach().clone()
        self.latest_consistency = consistency.to(device=F_normal.device).detach().clone()

        if self.use_ema:
            if self.running_score.numel() != current_score.numel() or int(self.num_updates.item()) == 0:
                self.running_score = current_score.detach().clone()
            else:
                running = self.running_score.to(device=current_score.device, dtype=current_score.dtype)
                running = self.momentum * running + (1.0 - self.momentum) * current_score
                self.running_score = self._normalize_score(running).detach().clone()
        else:
            self.running_score = current_score.detach().clone()

        self.num_updates += 1
        return self.get_topk()

    def get_diagnostics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latest normalized sensitivity, leakage, and consistency scores."""

        return (
            self.latest_sensitivity.detach().clone(),
            self.latest_leakage.detach().clone(),
            self.latest_consistency.detach().clone(),
        )

    def get_topk(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top channel indices, normalized top weights, and full score."""

        score = self._score_for_selection()
        k = min(self.top_k, score.numel())
        top_values, top_indices = torch.topk(score, k=k, largest=True)
        top_weights = self._normalize_score(top_values)
        return top_indices.long(), top_weights, score.detach().clone()

    def select_channels(self, F: torch.Tensor) -> torch.Tensor:
        """Select routed channels from features.

        Args:
            F: Feature tensor with shape [B, N, C].

        Returns:
            Tensor with shape [B, N, top_k].
        """

        if F.dim() != 3:
            raise ValueError(f"Expected F with shape [B, N, C], got {tuple(F.shape)}")
        top_indices, _, _ = self.get_topk()
        return F.index_select(dim=2, index=top_indices.to(device=F.device))

    def weighted_reduce(self, F_selected: torch.Tensor) -> torch.Tensor:
        """Apply top-k channel weights and reduce selected channels.

        Args:
            F_selected: Selected feature tensor with shape [B, N, top_k].

        Returns:
            Weighted response map with shape [B, N].
        """

        if F_selected.dim() != 3:
            raise ValueError(f"Expected F_selected with shape [B, N, K], got {tuple(F_selected.shape)}")
        _, top_weights, _ = self.get_topk()
        if F_selected.shape[-1] != top_weights.numel():
            raise ValueError(
                f"F_selected has {F_selected.shape[-1]} channels, but router has {top_weights.numel()} weights"
            )
        weights = top_weights.to(device=F_selected.device, dtype=F_selected.dtype).view(1, 1, -1)
        return (F_selected * weights).sum(dim=-1)

    def _score_for_selection(self) -> torch.Tensor:
        score = self.running_score if self.use_ema else self.latest_score
        if score.numel() == 0:
            raise RuntimeError("ChannelRouter has no scores yet. Call update(...) before get_topk().")
        return score

    def _normalize_score(self, score: torch.Tensor) -> torch.Tensor:
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if score.numel() == 0:
            return score

        total = score.sum()
        if float(total.detach().cpu()) <= self.eps:
            return torch.full_like(score, 1.0 / float(score.numel()))
        return score / (total + self.eps)

    @staticmethod
    def _validate_feature_pair(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_name: str,
        rhs_name: str,
    ) -> None:
        if lhs.dim() != 3 or rhs.dim() != 3:
            raise ValueError(
                f"Expected {lhs_name} and {rhs_name} with shape [B, N, C], got "
                f"{tuple(lhs.shape)} and {tuple(rhs.shape)}"
            )
        if lhs.shape != rhs.shape:
            raise ValueError(f"{lhs_name} and {rhs_name} shapes must match, got {lhs.shape} and {rhs.shape}")

    def _format_token_mask(self, token_mask: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        B, N, _ = F.shape
        mask = token_mask.to(device=F.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] != 1:
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        elif mask.dim() not in (2, 3):
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        if mask.shape[:2] != (B, N):
            raise ValueError(f"Expected token_mask batch/token shape {(B, N)}, got {tuple(mask.shape[:2])}")
        return (mask > 0).to(dtype=F.dtype)

    def _masked_channel_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != values.shape[:2] + (1,):
            raise ValueError(f"Expected mask shape {values.shape[:2] + (1,)}, got {tuple(mask.shape)}")
        denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
        return (values * mask).sum(dim=(0, 1)) / denom


def masked_delta_contrastive_loss(
    F_normal: torch.Tensor,
    F_synthetic_1: torch.Tensor,
    F_synthetic_2: torch.Tensor,
    token_mask_1: torch.Tensor,
    token_mask_2: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Contrast masked synthetic delta directions across two synthetic views.

    Positive pairs are two synthetic deltas from the same sample. Other samples
    in the batch act as negatives. The loss is intentionally small and local:
    it filters synthetic-specific directions instead of replacing the projector
    objective.
    """

    if F_normal.dim() != 3:
        raise ValueError(f"Expected features with shape [B, N, C], got {tuple(F_normal.shape)}")
    if F_normal.shape != F_synthetic_1.shape or F_normal.shape != F_synthetic_2.shape:
        raise ValueError("F_normal, F_synthetic_1, and F_synthetic_2 must have matching shapes")

    B, _, C = F_normal.shape
    top_indices = top_indices.to(device=F_normal.device, dtype=torch.long).flatten()
    if top_indices.numel() == 0 or B <= 1:
        return F_normal.new_zeros(())
    if int(top_indices.min().item()) < 0 or int(top_indices.max().item()) >= C:
        raise ValueError(f"top_indices must be in [0, {C})")

    top_weights = top_weights.to(device=F_normal.device, dtype=F_normal.dtype).flatten()
    if top_weights.numel() != top_indices.numel():
        raise ValueError(f"Expected top_weights shape [{top_indices.numel()}], got {tuple(top_weights.shape)}")
    weights = top_weights / top_weights.sum().clamp_min(eps)

    r1 = _pooled_weighted_delta(F_normal, F_synthetic_1, token_mask_1, top_indices, weights, eps)
    r2 = _pooled_weighted_delta(F_normal, F_synthetic_2, token_mask_2, top_indices, weights, eps)
    r1 = torch.nn.functional.normalize(r1, dim=-1, eps=eps)
    r2 = torch.nn.functional.normalize(r2, dim=-1, eps=eps)

    logits = r1 @ r2.transpose(0, 1) / max(float(temperature), eps)
    labels = torch.arange(B, device=F_normal.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
    )


def _pooled_weighted_delta(
    F_normal: torch.Tensor,
    F_synthetic: torch.Tensor,
    token_mask: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mask = token_mask.to(device=F_normal.device, dtype=F_normal.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    if mask.dim() != 3 or mask.shape[-1] != 1 or mask.shape[:2] != F_normal.shape[:2]:
        raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")

    delta = F_synthetic.index_select(2, top_indices) - F_normal.index_select(2, top_indices)
    delta = delta * top_weights.view(1, 1, -1)
    return (delta * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(eps)


ChannelSelector = ChannelRouter
