from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureSynthMode = str
VALID_FEATURE_SYNTH_MODES = ("dropout", "noise", "replacement", "shuffle", "mixed")
VALID_TOKEN_MASK_MODES = ("rectangle", "random_blob", "random_tokens")


def generate_token_mask(
    B: int,
    N: int,
    grid_size: Union[int, Sequence[int]],
    mask_ratio_range: Tuple[float, float] = (0.05, 0.25),
    mode: str = "rectangle",
    device: Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate token-space anomaly masks for square ViT token grids.

    Args:
        B: Batch size.
        N: Number of tokens, equal to grid_h * grid_w.
        grid_size: Integer grid size or (grid_h, grid_w).
        mask_ratio_range: Min/max token ratio to mask.
        mode: One of rectangle, random_blob, or random_tokens.

    Returns:
        Token mask with shape [B, N].
    """

    if mode not in VALID_TOKEN_MASK_MODES:
        raise ValueError(f"Unsupported token mask mode {mode}. Choices: {VALID_TOKEN_MASK_MODES}")
    if B <= 0 or N <= 0:
        raise ValueError(f"B and N must be positive, got B={B}, N={N}")

    grid_h, grid_w = _resolve_grid_size(grid_size, N)
    min_ratio, max_ratio = mask_ratio_range
    if min_ratio <= 0.0 or max_ratio <= 0.0 or min_ratio > max_ratio:
        raise ValueError(f"Invalid mask_ratio_range: {mask_ratio_range}")

    device = torch.device(device) if device is not None else torch.device("cpu")
    ratios = torch.empty(B, device=device).uniform_(float(min_ratio), float(max_ratio)).clamp(0.0, 1.0)
    score_dtype = torch.float32

    if mode == "rectangle":
        masks = torch.zeros(B, grid_h, grid_w, device=device, dtype=dtype)
        for i in range(B):
            target_area = max(1, int(round(float(ratios[i].item()) * N)))
            aspect = float(torch.empty((), device=device).uniform_(0.3, 1.0 / 0.3).item())
            box_h = int(round(math.sqrt(target_area / aspect)))
            box_w = int(round(math.sqrt(target_area * aspect)))
            box_h = min(max(box_h, 1), grid_h)
            box_w = min(max(box_w, 1), grid_w)
            top = int(torch.randint(0, grid_h - box_h + 1, (1,), device=device).item())
            left = int(torch.randint(0, grid_w - box_w + 1, (1,), device=device).item())
            masks[i, top : top + box_h, left : left + box_w] = 1.0
        return masks.view(B, N)

    if mode == "random_blob":
        low_h = max(2, grid_h // 4)
        low_w = max(2, grid_w // 4)
        noise = torch.rand(B, 1, low_h, low_w, device=device, dtype=score_dtype)
        smooth = F.interpolate(noise, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        return _topk_mask_from_scores(smooth.view(B, N), ratios, dtype=dtype)

    scores = torch.rand(B, N, device=device, dtype=score_dtype)
    return _topk_mask_from_scores(scores, ratios, dtype=dtype)


class ChannelAwareFeatureSynthesizer(nn.Module):
    """Create feature-space synthetic anomalies in routed channels only."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        F_normal: torch.Tensor,
        token_mask: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        mode: FeatureSynthMode = "mixed",
        alpha: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Corrupt selected channels inside masked tokens.

        Args:
            F_normal: Normal features, shape [B, N, C].
            token_mask: Token mask, shape [B, N] or [B, N, 1].
            top_indices: Selected channel indices, shape [K].
            top_weights: Selected channel weights, shape [K].
            mode: Corruption mode.

        Returns:
            F_feat_syn: Synthetic features, shape [B, N, C].
            applied_token_mask: Float mask, shape [B, N].
            applied_channel_indices: Long tensor, shape [K].
        """

        if mode not in VALID_FEATURE_SYNTH_MODES:
            raise ValueError(f"Unsupported feature synthesis mode {mode}. Choices: {VALID_FEATURE_SYNTH_MODES}")
        if F_normal.dim() != 3:
            raise ValueError(f"Expected F_normal with shape [B, N, C], got {tuple(F_normal.shape)}")

        B, N, C = F_normal.shape
        device = F_normal.device
        dtype = F_normal.dtype
        alpha_value = self.alpha if alpha is None else float(alpha)

        top_indices = top_indices.to(device=device, dtype=torch.long).flatten()
        if top_indices.numel() == 0:
            return F_normal.clone(), self._format_token_mask(token_mask, B, N, device, dtype).squeeze(-1), top_indices
        if int(top_indices.min().item()) < 0 or int(top_indices.max().item()) >= C:
            raise ValueError(f"top_indices must be in [0, {C}), got min/max {top_indices.min()} / {top_indices.max()}")

        K = top_indices.numel()
        top_weights = top_weights.to(device=device, dtype=dtype).flatten()
        if top_weights.numel() != K:
            raise ValueError(f"Expected top_weights shape [{K}], got {tuple(top_weights.shape)}")
        top_weights = self._normalize_weights(top_weights)

        mask = self._format_token_mask(token_mask, B, N, device, dtype)  # [B, N, 1]
        weights = top_weights.view(1, 1, K)  # [1, 1, K]

        F_syn = F_normal.clone()
        selected = F_syn.index_select(dim=2, index=top_indices)  # [B, N, K]
        actual_mode = self._resolve_mode(mode, device)
        corrupted_selected = self._corrupt(selected, weights, actual_mode, alpha_value)

        selected_new = selected * (1.0 - mask) + corrupted_selected * mask
        F_syn = F_syn.index_copy(dim=2, index=top_indices, source=selected_new)
        return F_syn, mask.squeeze(-1), top_indices

    def _corrupt(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        mode: str,
        alpha: float,
    ) -> torch.Tensor:
        if mode == "dropout":
            strength = self._blend_strength(weights, alpha)
            return selected * (1.0 - strength)

        if mode == "noise":
            noise = torch.randn_like(selected)
            return selected + alpha * weights * noise

        if mode == "replacement":
            replacement = self._sample_from_other_tokens(selected)
            strength = self._blend_strength(weights, alpha)
            return selected + strength * (replacement - selected)

        if mode == "shuffle":
            shuffled = self._shuffle_within_image(selected)
            strength = self._blend_strength(weights, alpha)
            return selected + strength * (shuffled - selected)

        raise ValueError(f"Unsupported feature synthesis mode {mode}")

    def _blend_strength(self, weights: torch.Tensor, alpha: float) -> torch.Tensor:
        # top_weights sum to one; multiplying by K makes alpha the average blend strength.
        K = weights.shape[-1]
        return (alpha * weights * float(K)).clamp(0.0, 1.0)

    def _resolve_mode(self, mode: str, device: torch.device) -> str:
        if mode != "mixed":
            return mode
        choices = ("dropout", "noise", "replacement", "shuffle")
        idx = int(torch.randint(0, len(choices), (1,), device=device).item())
        return choices[idx]

    def _format_token_mask(
        self,
        token_mask: torch.Tensor,
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = token_mask.to(device=device, dtype=dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        elif mask.dim() == 3:
            if mask.shape[-1] != 1:
                raise ValueError(f"Expected token_mask shape [B, N, 1], got {tuple(mask.shape)}")
        else:
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(mask.shape)}")

        if mask.shape[:2] != (B, N):
            raise ValueError(f"Expected token_mask batch/token shape {(B, N)}, got {tuple(mask.shape[:2])}")
        return (mask > 0).to(dtype=dtype)

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        total = weights.sum()
        if float(total.detach().cpu()) <= self.eps:
            return torch.full_like(weights, 1.0 / float(weights.numel()))
        return weights / (total + self.eps)

    @staticmethod
    def _sample_from_other_tokens(selected: torch.Tensor) -> torch.Tensor:
        B, N, K = selected.shape
        total = B * N
        if total <= 1:
            return selected

        flat = selected.reshape(total, K)
        offsets = torch.randint(1, total, (total,), device=selected.device)
        indices = (torch.arange(total, device=selected.device) + offsets) % total
        return flat.index_select(dim=0, index=indices).view(B, N, K)

    @staticmethod
    def _shuffle_within_image(selected: torch.Tensor) -> torch.Tensor:
        B, N, K = selected.shape
        if N <= 1:
            return selected

        shuffled = torch.empty_like(selected)
        for i in range(B):
            perm = torch.randperm(N, device=selected.device)
            shuffled[i] = selected[i].index_select(dim=0, index=perm)
        return shuffled


def _resolve_grid_size(grid_size: Union[int, Sequence[int]], N: int) -> Tuple[int, int]:
    if isinstance(grid_size, int):
        grid_h = grid_w = int(grid_size)
    else:
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be an int or a 2-item sequence, got {grid_size}")
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])

    if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != N:
        raise ValueError(f"Expected grid_h * grid_w == N, got grid_size=({grid_h}, {grid_w}) and N={N}")
    return grid_h, grid_w


def _topk_mask_from_scores(scores: torch.Tensor, ratios: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    B, N = scores.shape
    mask = torch.zeros(B, N, device=scores.device, dtype=dtype)
    for i in range(B):
        k = max(1, min(N, int(round(float(ratios[i].item()) * N))))
        indices = torch.topk(scores[i], k=k, largest=True).indices
        mask[i, indices] = 1.0
    return mask
