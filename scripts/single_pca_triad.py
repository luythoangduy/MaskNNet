from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDAD_ROOT = REPO_ROOT / "foundad"
if str(FOUNDAD_ROOT) not in sys.path:
    sys.path.insert(0, str(FOUNDAD_ROOT))

from src.utils.synthesis import CutPasteNormal, CutPasteScar, CutPasteUnion, NeuralMaskSynthesizer


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_rgb(path: Path, resize: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((resize, resize), Image.BICUBIC)


def load_mask(path: Path, resize: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((resize, resize), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32)
    return torch.from_numpy(arr > 0)


def to_model_tensor(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(img)


def model_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return float_tensor_to_pil((x * std + mean).cpu())


def pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def float_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def make_synthetic(normal: Image.Image, mask: torch.Tensor, strength: float, seed: int) -> Image.Image:
    rng = torch.Generator().manual_seed(seed)
    x = pil_to_float_tensor(normal)
    y = x.clone()
    noise = torch.rand(x.shape, generator=rng) * 2.0 - 1.0
    y[:, mask] = (x[:, mask] + strength * noise[:, mask]).clamp(0, 1)
    return float_tensor_to_pil(y)


def apply_pixel_synth(
    img: Image.Image,
    subclass: str,
    method: str,
    color_jitter: float,
) -> Image.Image:
    x = to_model_tensor(img).unsqueeze(0)
    if method == "cutpaste_normal":
        synth = CutPasteNormal(colorJitter=color_jitter)
        return model_tensor_to_pil(synth.process_image(x.squeeze(0), subclass))
    elif method == "cutpaste_scar":
        synth = CutPasteScar(colorJitter=color_jitter)
        return model_tensor_to_pil(synth.process_image(x.squeeze(0), subclass))
    elif method == "cutpaste_union":
        synth = CutPasteUnion(colorJitter=color_jitter)
    else:
        raise ValueError(f"Unsupported pixel synthesis method: {method}")

    _, y = synth(x, [subclass])
    return model_tensor_to_pil(y.squeeze(0))


def is_neural_method(method: str) -> bool:
    return method in {"neural_suppress", "neural_violate"}


def is_pixel_method(method: str) -> bool:
    return method in {"noise_mask", "cutpaste_normal", "cutpaste_scar", "cutpaste_union"}


def build_neural_synthesizer(method: str) -> NeuralMaskSynthesizer:
    return NeuralMaskSynthesizer(
        area_ratio=(0.02, 0.25),
        aspect_ratio=0.3,
        method="violate" if method == "neural_violate" else "suppress",
        radius=2,
        pca_dim=5,
        channel_topk_ratio=0.1,
        channel_min=32,
        mask_strength=1.0,
    )


@torch.inference_mode()
def extract_tokens(model: torch.nn.Module, batch: torch.Tensor, n_layer: int) -> torch.Tensor:
    tokens = model.get_intermediate_layers(batch, n=n_layer, return_class_token=False)[0]
    return tokens.squeeze(0).float().cpu()


def load_dinov2_model(device: torch.device) -> torch.nn.Module:
    hub_dir = Path(torch.hub.get_dir())
    local_repo = hub_dir / "facebookresearch_dinov2_main"
    if local_repo.exists():
        model = torch.hub.load(str(local_repo), "dinov2_vitb14", source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    return model.eval().to(device)


def token_mask(pixel_mask: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    m = pixel_mask.float()[None, None]
    pooled = F.interpolate(m, size=(grid_h, grid_w), mode="area").squeeze()
    return (pooled > 0.0).flatten().numpy()


def classify_components(d_syn: np.ndarray, d_real: np.ndarray, align_ratio: np.ndarray) -> list[str]:
    if float(np.nanmax(d_syn)) <= 1e-8:
        return ["case3_invariant" if dr <= np.quantile(d_real, 0.25) else "mixed" for dr in d_real]

    syn_hi = np.quantile(d_syn, 0.75)
    syn_lo = np.quantile(d_syn, 0.25)
    real_hi = np.quantile(d_real, 0.75)
    real_lo = np.quantile(d_real, 0.25)
    align_lo = np.quantile(align_ratio, 0.25)
    align_hi = np.quantile(align_ratio, 0.75)

    cases = []
    for ds, dr, ar in zip(d_syn, d_real, align_ratio):
        if ds >= syn_hi and dr >= real_hi and ar <= align_lo:
            cases.append("case1_synthetic_aligned")
        elif ds >= syn_hi and ar >= align_hi:
            cases.append("case2_wrong_direction")
        elif ds <= syn_lo and dr <= real_lo:
            cases.append("case3_invariant")
        else:
            cases.append("mixed")
    return cases


def spatial_entropy(values: np.ndarray, temperature: float = 1.0) -> float:
    x = np.abs(values.astype(np.float64).reshape(-1))
    if x.size == 0 or float(x.max()) <= 1e-12:
        return 0.0
    x = x / (float(x.mean()) + 1e-12)
    logits = x / max(temperature, 1e-12)
    logits = logits - float(logits.max())
    p = np.exp(logits)
    p = p / (float(p.sum()) + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def normalized_spatial_entropy(values: np.ndarray, temperature: float = 1.0) -> float:
    if values.size <= 1:
        return 0.0
    return spatial_entropy(values, temperature=temperature) / np.log(values.size)


def save_preview(out_dir: Path, normal: Image.Image, synthetic: Image.Image, anomaly: Image.Image, mask: torch.Tensor) -> None:
    mask_img = Image.fromarray((mask.numpy().astype(np.uint8) * 255))
    normal.save(out_dir / "normal.png")
    synthetic.save(out_dir / "synthetic.png")
    anomaly.save(out_dir / "real_anomaly.png")
    mask_img.save(out_dir / "real_mask.png")


def plot_scatter(out_dir: Path, d_syn: np.ndarray, d_real: np.ndarray, cases: list[str]) -> None:
    colors = {
        "case1_synthetic_aligned": "#2ca02c",
        "case2_wrong_direction": "#d62728",
        "case3_invariant": "#1f77b4",
        "mixed": "#7f7f7f",
    }
    plt.figure(figsize=(7, 5))
    for case, color in colors.items():
        idx = np.array([c == case for c in cases])
        if idx.any():
            plt.scatter(d_syn[idx], d_real[idx], s=28, c=color, label=case, alpha=0.85)
    plt.xlabel("d_syn = mean |PCA(f(x+b)) - PCA(f(x))|")
    plt.ylabel("d_real = mean |PCA(f(x^)) - PCA(f(x))|")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_delta_cases.png", dpi=180)
    plt.close()


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cmap = plt.get_cmap("magma")
    hm = normalize_map(heatmap)
    rgba = (cmap(hm) * 255).astype(np.uint8)
    heat = Image.fromarray(rgba[:, :, :3]).resize(base.size, Image.BICUBIC)
    return Image.blend(base.convert("RGB"), heat.convert("RGB"), alpha)


def component_score(row: dict[str, float], case: str) -> float:
    if case == "case1_synthetic_aligned":
        return row["d_syn"] * row["d_real"] / (row["align_ratio"] + 1e-8)
    if case == "case2_wrong_direction":
        return row["d_syn"] * row["align_ratio"]
    if case == "case3_invariant":
        return -1.0 * (row["d_syn"] + row["d_real"])
    return row["d_syn"]


def choose_representatives(rows: list[dict[str, float]], top_n: int) -> dict[str, list[int]]:
    reps: dict[str, list[int]] = {}
    for case in ["case1_synthetic_aligned", "case2_wrong_direction", "case3_invariant"]:
        case_rows = [r for r in rows if r["case"] == case]
        case_rows.sort(key=lambda r: component_score(r, case), reverse=True)
        reps[case] = [int(r["component"]) for r in case_rows[:top_n]]
    return reps


def save_component_visuals(
    out_dir: Path,
    normal: Image.Image,
    synthetic: Image.Image,
    anomaly: Image.Image,
    real_mask: torch.Tensor,
    zp: np.ndarray,
    zsp: np.ndarray,
    zrp: np.ndarray,
    grid: int,
    rows: list[dict[str, float]],
    top_n: int,
) -> None:
    viz_dir = out_dir / "component_visuals"
    viz_dir.mkdir(parents=True, exist_ok=True)
    representatives = choose_representatives(rows, top_n=top_n)

    mask_arr = real_mask.numpy().astype(np.float32)
    mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(normal.size, Image.NEAREST)

    summary_lines = []
    for case, comps in representatives.items():
        summary_lines.append(f"{case}: {comps}")
        for comp in comps:
            row = next(r for r in rows if int(r["component"]) == comp)
            syn_map = np.abs(zsp[:, comp] - zp[:, comp]).reshape(grid, grid)
            real_map = np.abs(zrp[:, comp] - zp[:, comp]).reshape(grid, grid)
            align_map = np.abs(zsp[:, comp] - zrp[:, comp]).reshape(grid, grid)

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for ax in axes.flatten():
                ax.axis("off")

            axes[0, 0].imshow(normal)
            axes[0, 0].set_title("normal x")
            axes[0, 1].imshow(synthetic)
            axes[0, 1].set_title("synthetic x+b")
            axes[0, 2].imshow(anomaly)
            axes[0, 2].imshow(mask_img, alpha=0.35, cmap="Reds")
            axes[0, 2].set_title("real anomaly x^ + mask")

            axes[1, 0].imshow(overlay_heatmap(normal, syn_map))
            axes[1, 0].set_title("|PCA_j(x+b)-PCA_j(x)|")
            axes[1, 1].imshow(overlay_heatmap(anomaly, real_map))
            axes[1, 1].set_title("|PCA_j(x^)-PCA_j(x)|")
            axes[1, 2].imshow(overlay_heatmap(anomaly, align_map))
            axes[1, 2].set_title("|PCA_j(x+b)-PCA_j(x^)|")

            fig.suptitle(
                (
                    f"{case} | component {comp} | "
                    f"d_syn={row['d_syn']:.3f}, d_real={row['d_real']:.3f}, "
                    f"align_ratio={row['align_ratio']:.3f}, sign_match={row['sign_match']:.3f}"
                ),
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(viz_dir / f"{case}_component_{comp:03d}.png", dpi=170)
            plt.close(fig)

    (viz_dir / "representatives.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def save_impact_summary(
    out_dir: Path,
    zp: np.ndarray,
    zsp: np.ndarray,
    zrp: np.ndarray,
    grid: int,
    rows: list[dict[str, float]],
    top_n: int,
) -> None:
    viz_dir = out_dir / "component_visuals"
    representatives = choose_representatives(rows, top_n=top_n)
    selected: list[tuple[str, int]] = []
    for case, comps in representatives.items():
        selected.extend((case, comp) for comp in comps)
    if not selected:
        return

    maps: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    values = []
    for case, comp in selected:
        syn_map = np.abs(zsp[:, comp] - zp[:, comp]).reshape(grid, grid)
        real_map = np.abs(zrp[:, comp] - zp[:, comp]).reshape(grid, grid)
        align_map = np.abs(zsp[:, comp] - zrp[:, comp]).reshape(grid, grid)
        maps.append((case, comp, syn_map, real_map, align_map))
        values.extend([syn_map.ravel(), real_map.ravel(), align_map.ravel()])

    vmax = float(np.quantile(np.concatenate(values), 0.98))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(len(maps), 3, figsize=(9, 3 * len(maps)))
    if len(maps) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (case, comp, syn_map, real_map, align_map) in enumerate(maps):
        for col_idx, (title, hm) in enumerate(
            [
                ("syn-normal", syn_map),
                ("real-normal", real_map),
                ("syn-real align error", align_map),
            ]
        ):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(hm, cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"{case}\ncomp {comp}", fontsize=8)

    fig.subplots_adjust(right=0.90, hspace=0.15, wspace=0.05)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    fig.colorbar(im, cax=cax)
    fig.suptitle("Shared-scale impact maps across selected PCA components", fontsize=11)
    fig.savefig(viz_dir / "impact_shared_scale.png", dpi=180)
    plt.close(fig)

    labels = [f"{case.replace('case', 'c').split('_')[0]}:{comp}" for case, comp in selected]
    row_by_comp = {int(r["component"]): r for r in rows}
    x = np.arange(len(selected))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 1.2), 4))
    ax.bar(x - width, [row_by_comp[c]["d_syn"] for _, c in selected], width, label="d_syn")
    ax.bar(x, [row_by_comp[c]["d_real"] for _, c in selected], width, label="d_real")
    ax.bar(x + width, [row_by_comp[c]["d_align"] for _, c in selected], width, label="d_align")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean absolute PCA delta on affected tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "impact_metrics_bar.png", dpi=180)
    plt.close(fig)


def save_raw_anomaly_synchronized_visuals(
    out_dir: Path,
    anomaly: Image.Image,
    synthetic_anomaly: Image.Image,
    real_mask: torch.Tensor,
    zap: np.ndarray,
    zsap: np.ndarray,
    grid: int,
    rows: list[dict[str, float]],
    top_n: int,
) -> None:
    viz_dir = out_dir / "component_visuals" / "raw_anomaly_synchronized"
    viz_dir.mkdir(parents=True, exist_ok=True)
    representatives = choose_representatives(rows, top_n=top_n)
    selected: list[tuple[str, int]] = []
    for case, comps in representatives.items():
        selected.extend((case, comp) for comp in comps)
    if not selected:
        return

    values = []
    for _, comp in selected:
        values.extend([zap[:, comp], zsap[:, comp], zsap[:, comp] - zap[:, comp]])
    all_values = np.concatenate(values)
    vmax = float(np.quantile(np.abs(all_values), 0.98))
    vmax = max(vmax, 1e-6)

    mask_img = Image.fromarray((real_mask.numpy().astype(np.uint8) * 255)).resize(anomaly.size, Image.NEAREST)

    for case, comp in selected:
        row = next(r for r in rows if int(r["component"]) == comp)
        anomaly_map = zap[:, comp].reshape(grid, grid)
        synthetic_map = zsap[:, comp].reshape(grid, grid)
        diff_map = (zsap[:, comp] - zap[:, comp]).reshape(grid, grid)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].imshow(anomaly)
        axes[0, 0].imshow(mask_img, alpha=0.35, cmap="Reds")
        axes[0, 0].set_title("real anomaly x^ + mask")
        axes[0, 1].imshow(synthetic_anomaly)
        axes[0, 1].set_title("synthetic on same anomaly x^+b")
        axes[0, 2].imshow(synthetic_anomaly)
        axes[0, 2].imshow(mask_img, alpha=0.35, cmap="Reds")
        axes[0, 2].set_title("same coordinate mask")

        panels = [
            ("raw PCA_j(x^)", anomaly_map),
            ("raw PCA_j(x^+b)", synthetic_map),
            ("PCA_j(x^+b)-PCA_j(x^)", diff_map),
        ]
        for ax, (title, hm) in zip(axes[1], panels):
            im = ax.imshow(hm, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(title)

        fig.colorbar(im, ax=axes[1].ravel().tolist(), shrink=0.72)
        fig.suptitle(
            (
                f"same-anomaly raw visualization | {case} | component {comp} | "
                f"original-case d_syn={row['d_syn']:.3f}, d_real={row['d_real']:.3f}, "
                f"align_ratio={row['align_ratio']:.3f}"
            ),
            fontsize=11,
        )
        fig.savefig(viz_dir / f"{case}_component_{comp:03d}_raw_anomaly.png", dpi=170)
        plt.close(fig)

    summary = "\n".join(f"{case}: {comps}" for case, comps in representatives.items())
    (viz_dir / "representatives.txt").write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample PCA triad probe for VisA/MVTec-style anomaly data.")
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--anomaly", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/single_pca_triad"))
    parser.add_argument("--resize", type=int, default=518)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--synthetic-strength", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz-top-n", type=int, default=3)
    parser.add_argument(
        "--synth-method",
        choices=[
            "noise_mask",
            "cutpaste_normal",
            "cutpaste_scar",
            "cutpaste_union",
            "neural_suppress",
            "neural_violate",
        ],
        default="neural_suppress",
    )
    parser.add_argument("--subclass", default="candle")
    parser.add_argument("--color-jitter", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normal = load_rgb(args.normal, args.resize)
    anomaly = load_rgb(args.anomaly, args.resize)
    real_mask = load_mask(args.mask, args.resize)
    if args.synth_method == "noise_mask":
        synthetic = make_synthetic(normal, real_mask, args.synthetic_strength, args.seed)
        synthetic_anomaly = make_synthetic(anomaly, real_mask, args.synthetic_strength, args.seed)
    elif is_pixel_method(args.synth_method):
        synthetic = apply_pixel_synth(normal, args.subclass, args.synth_method, args.color_jitter)
        synthetic_anomaly = apply_pixel_synth(anomaly, args.subclass, args.synth_method, args.color_jitter)
    else:
        synthetic = normal.copy()
        synthetic_anomaly = anomaly.copy()
    save_preview(args.out_dir, normal, synthetic, anomaly, real_mask)

    batch = torch.stack(
        [
            to_model_tensor(normal),
            to_model_tensor(synthetic),
            to_model_tensor(anomaly),
            to_model_tensor(synthetic_anomaly),
        ]
    ).to(device)

    model = load_dinov2_model(device)
    z, z_syn_img, z_real, z_syn_anomaly_img = [
        extract_tokens(model, batch[i : i + 1], args.n_layer) for i in range(4)
    ]

    synth_token_mask = None
    synth_anomaly_token_mask = None
    if is_neural_method(args.synth_method):
        neural = build_neural_synthesizer(args.synth_method)
        normal_batch = batch[0:1].detach().cpu()
        anomaly_batch = batch[2:3].detach().cpu()
        z_syn, synth_token_mask = neural(normal_batch, z.unsqueeze(0), [args.subclass])
        z_syn_anomaly, synth_anomaly_token_mask = neural(anomaly_batch, z_real.unsqueeze(0), [args.subclass])
        z_syn = z_syn.squeeze(0).float().cpu()
        z_syn_anomaly = z_syn_anomaly.squeeze(0).float().cpu()
        synth_token_mask = synth_token_mask.squeeze(0).bool().cpu().numpy()
        synth_anomaly_token_mask = synth_anomaly_token_mask.squeeze(0).bool().cpu().numpy()
    else:
        z_syn = z_syn_img
        z_syn_anomaly = z_syn_anomaly_img

    n_tokens, feat_dim = z.shape
    grid = int(round(n_tokens ** 0.5))
    if grid * grid != n_tokens:
        raise RuntimeError(f"Expected square token grid, got {n_tokens} tokens")

    affected = token_mask(real_mask, grid, grid)
    if not affected.any():
        raise RuntimeError("Mask produced no affected tokens after patch pooling")
    affected_syn = synth_token_mask if synth_token_mask is not None and synth_token_mask.any() else affected

    n_components = min(args.pca_dim, n_tokens, feat_dim)
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(z.numpy())

    zp = pca.transform(z.numpy())
    zsp = pca.transform(z_syn.numpy())
    zrp = pca.transform(z_real.numpy())
    zsap = pca.transform(z_syn_anomaly.numpy())

    syn_delta = zsp - zp
    real_delta = zrp - zp
    d_syn = np.abs(syn_delta[affected_syn]).mean(axis=0)
    d_real = np.abs(real_delta[affected]).mean(axis=0)
    if affected_syn.shape == affected.shape and np.array_equal(affected_syn, affected):
        d_align = np.abs(zsp[affected] - zrp[affected]).mean(axis=0)
        sign_match = (np.sign(syn_delta[affected]) == np.sign(real_delta[affected])).mean(axis=0)
    else:
        mean_syn_delta = syn_delta[affected_syn].mean(axis=0)
        mean_real_delta = real_delta[affected].mean(axis=0)
        d_align = np.abs(mean_syn_delta - mean_real_delta)
        sign_match = (np.sign(mean_syn_delta) == np.sign(mean_real_delta)).astype(np.float32)
    align_ratio = d_align / (d_syn + d_real + 1e-8)
    cases = classify_components(d_syn, d_real, align_ratio)

    rows: list[dict[str, float | str]] = []
    csv_path = args.out_dir / "component_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "component",
                "explained_variance_ratio",
                "d_syn",
                "d_real",
                "d_align",
                "align_ratio",
                "sign_match",
                "entropy_normal",
                "entropy_synthetic",
                "entropy_real",
                "entropy_syn_delta",
                "entropy_real_delta",
                "entropy_align_error",
                "entropy_normal_norm",
                "entropy_synthetic_norm",
                "entropy_real_norm",
                "entropy_syn_delta_norm",
                "entropy_real_delta_norm",
                "entropy_align_error_norm",
                "case",
            ]
        )
        for j in range(n_components):
            normal_map = zp[:, j]
            synthetic_map = zsp[:, j]
            real_map = zrp[:, j]
            syn_delta_map = zsp[:, j] - zp[:, j]
            real_delta_map = zrp[:, j] - zp[:, j]
            align_error_map = zsp[:, j] - zrp[:, j]
            row = {
                "component": j,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[j]),
                "d_syn": float(d_syn[j]),
                "d_real": float(d_real[j]),
                "d_align": float(d_align[j]),
                "align_ratio": float(align_ratio[j]),
                "sign_match": float(sign_match[j]),
                "entropy_normal": spatial_entropy(normal_map),
                "entropy_synthetic": spatial_entropy(synthetic_map),
                "entropy_real": spatial_entropy(real_map),
                "entropy_syn_delta": spatial_entropy(syn_delta_map),
                "entropy_real_delta": spatial_entropy(real_delta_map),
                "entropy_align_error": spatial_entropy(align_error_map),
                "entropy_normal_norm": normalized_spatial_entropy(normal_map),
                "entropy_synthetic_norm": normalized_spatial_entropy(synthetic_map),
                "entropy_real_norm": normalized_spatial_entropy(real_map),
                "entropy_syn_delta_norm": normalized_spatial_entropy(syn_delta_map),
                "entropy_real_delta_norm": normalized_spatial_entropy(real_delta_map),
                "entropy_align_error_norm": normalized_spatial_entropy(align_error_map),
                "case": cases[j],
            }
            rows.append(row)
            writer.writerow(
                [
                    row["component"],
                    row["explained_variance_ratio"],
                    row["d_syn"],
                    row["d_real"],
                    row["d_align"],
                    row["align_ratio"],
                    row["sign_match"],
                    row["entropy_normal"],
                    row["entropy_synthetic"],
                    row["entropy_real"],
                    row["entropy_syn_delta"],
                    row["entropy_real_delta"],
                    row["entropy_align_error"],
                    row["entropy_normal_norm"],
                    row["entropy_synthetic_norm"],
                    row["entropy_real_norm"],
                    row["entropy_syn_delta_norm"],
                    row["entropy_real_delta_norm"],
                    row["entropy_align_error_norm"],
                    row["case"],
                ]
            )

    plot_scatter(args.out_dir, d_syn, d_real, cases)
    save_component_visuals(
        out_dir=args.out_dir,
        normal=normal,
        synthetic=synthetic,
        anomaly=anomaly,
        real_mask=real_mask,
        zp=zp,
        zsp=zsp,
        zrp=zrp,
        grid=grid,
        rows=rows,  # type: ignore[arg-type]
        top_n=args.viz_top_n,
    )
    save_impact_summary(
        out_dir=args.out_dir,
        zp=zp,
        zsp=zsp,
        zrp=zrp,
        grid=grid,
        rows=rows,  # type: ignore[arg-type]
        top_n=args.viz_top_n,
    )
    save_raw_anomaly_synchronized_visuals(
        out_dir=args.out_dir,
        anomaly=anomaly,
        synthetic_anomaly=synthetic_anomaly,
        real_mask=real_mask,
        zap=zrp,
        zsap=zsap,
        grid=grid,
        rows=rows,  # type: ignore[arg-type]
        top_n=args.viz_top_n,
    )

    counts = {case: cases.count(case) for case in sorted(set(cases))}
    print(f"[done] outputs: {args.out_dir}")
    print(f"[info] synth_method: {args.synth_method}")
    print(f"[info] tokens: {n_tokens} ({grid}x{grid}), affected_tokens: {int(affected.sum())}")
    print(f"[info] synthetic_affected_tokens: {int(affected_syn.sum())}")
    print(f"[info] real/synthetic overlap_tokens: {int((affected & affected_syn).sum())}")
    print(f"[info] pca_components: {n_components}")
    print(f"[info] case_counts: {counts}")
    print(f"[info] csv: {csv_path}")
    print(f"[info] plot: {args.out_dir / 'pca_delta_cases.png'}")
    print(f"[info] component visuals: {args.out_dir / 'component_visuals'}")
    print(f"[info] raw same-anomaly visuals: {args.out_dir / 'component_visuals' / 'raw_anomaly_synchronized'}")


if __name__ == "__main__":
    main()
