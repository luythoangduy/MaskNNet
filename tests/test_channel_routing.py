from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "foundad"))

from src.utils.channel_router import ChannelRouter, masked_delta_contrastive_loss
from src.utils.feature_synthesis import ChannelAwareFeatureSynthesizer, generate_token_mask


def test_channel_router_topk_shapes() -> None:
    B, N, C, K = 2, 16, 8, 3
    F_normal = torch.zeros(B, N, C)
    F_probe = F_normal.clone()
    F_probe[:, :, 1] = 0.5
    F_probe[:, :, 3] = 1.0
    F_probe[:, :, 5] = 2.0

    router = ChannelRouter(top_k=K, use_ema=True)
    top_indices, top_weights, full_score = router.update(F_normal, F_probe)

    assert top_indices.shape == (K,)
    assert top_weights.shape == (K,)
    assert full_score.shape == (C,)
    assert torch.isclose(top_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert set(top_indices.tolist()) == {1, 3, 5}


def test_channel_router_masked_reliability_penalizes_background_leakage() -> None:
    B, N, C = 1, 6, 4
    F_normal = torch.zeros(B, N, C)
    F_probe = F_normal.clone()
    token_mask = torch.zeros(B, N)
    token_mask[:, :3] = 1

    F_probe[:, :3, 1] = 1.0  # useful: only masked tokens move
    F_probe[:, :, 2] = 1.0   # leaky: masked and background tokens move

    router = ChannelRouter(top_k=1, use_ema=False)
    top_indices, top_weights, full_score = router.update(F_normal, F_probe, token_mask=token_mask)
    sensitivity, leakage, consistency = router.get_diagnostics()

    assert top_indices.tolist() == [1]
    assert top_weights.shape == (1,)
    assert full_score[1] > full_score[2]
    assert sensitivity[1] > 0
    assert leakage[2] > leakage[1]
    assert torch.allclose(consistency, torch.ones_like(consistency))


def test_masked_delta_contrastive_loss_prefers_matching_pairs() -> None:
    B, N, C = 3, 4, 5
    F_normal = torch.zeros(B, N, C)
    token_mask = torch.ones(B, N)
    top_indices = torch.tensor([1, 3])
    top_weights = torch.tensor([0.5, 0.5])

    F_syn_1 = F_normal.clone()
    F_syn_2 = F_normal.clone()
    for i in range(B):
        F_syn_1[i, :, 1] = float(i + 1)
        F_syn_1[i, :, 3] = float(B - i)
        F_syn_2[i, :, 1] = float(i + 1)
        F_syn_2[i, :, 3] = float(B - i)

    matched = masked_delta_contrastive_loss(
        F_normal,
        F_syn_1,
        F_syn_2,
        token_mask,
        token_mask,
        top_indices,
        top_weights,
        temperature=0.1,
    )
    mismatched = masked_delta_contrastive_loss(
        F_normal,
        F_syn_1,
        F_syn_2.flip(0),
        token_mask,
        token_mask,
        top_indices,
        top_weights,
        temperature=0.1,
    )

    assert matched < mismatched


def test_feature_synthesis_modifies_only_masked_selected_channels() -> None:
    B, N, C = 2, 16, 8
    F_normal = torch.ones(B, N, C)
    top_indices = torch.tensor([1, 3, 5])
    top_weights = torch.tensor([0.2, 0.3, 0.5])
    token_mask = torch.zeros(B, N)
    token_mask[0, :4] = 1
    token_mask[1, 8:12] = 1

    synthesizer = ChannelAwareFeatureSynthesizer(alpha=1.0)
    F_syn, applied_mask, applied_indices = synthesizer(
        F_normal,
        token_mask,
        top_indices,
        top_weights,
        mode="dropout",
    )

    assert F_syn.shape == F_normal.shape
    assert applied_mask.shape == (B, N)
    assert applied_indices.shape == top_indices.shape

    changed = (F_syn - F_normal).abs() > 1e-6
    selected_channels = torch.zeros(C, dtype=torch.bool)
    selected_channels[top_indices] = True
    masked_tokens = token_mask.bool()

    assert changed[:, :, top_indices].any()
    assert not changed[0, ~masked_tokens[0], :].any()
    assert not changed[1, ~masked_tokens[1], :].any()
    assert not changed[:, :, ~selected_channels].any()


def test_generate_token_mask_modes() -> None:
    B, N, grid_size = 2, 16, 4
    for mode in ("rectangle", "random_blob", "random_tokens"):
        token_mask = generate_token_mask(
            B=B,
            N=N,
            grid_size=grid_size,
            mask_ratio_range=(0.1, 0.25),
            mode=mode,
        )
        assert token_mask.shape == (B, N)
        assert token_mask.sum() > 0
        assert token_mask.max() <= 1
        assert token_mask.min() >= 0


if __name__ == "__main__":
    test_channel_router_topk_shapes()
    test_feature_synthesis_modifies_only_masked_selected_channels()
    test_generate_token_mask_modes()
