
from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from src.utils.channel_router import ChannelRouter, masked_delta_contrastive_loss
from src.utils.feature_synthesis import (
    ChannelAwareFeatureSynthesizer,
    VALID_FEATURE_SYNTH_MODES,
    VALID_TOKEN_MASK_MODES,
    generate_token_mask,
)
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.utils.synthesis import CutPasteUnion, NeuralMaskSynthesizer
from src.foundad import VisionModule

_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def build_synthesizer(meta_cfg: Dict[str, Any]):
    synth_mode = meta_cfg.get("synthesis_mode", "neural_mask")
    synth_cfg = meta_cfg.get("synthesis", {})

    if synth_mode == "cutpaste":
        return CutPasteUnion(colorJitter=synth_cfg.get("color_jitter", 0.5))

    if synth_mode == "neural_mask":
        return NeuralMaskSynthesizer(
            area_ratio=tuple(synth_cfg.get("area_ratio", [0.02, 0.25])),
            aspect_ratio=synth_cfg.get("aspect_ratio", 0.3),
            method=synth_cfg.get("method", "suppress"),
            radius=synth_cfg.get("radius", 2),
            pca_dim=synth_cfg.get("pca_dim", 5),
            channel_topk_ratio=synth_cfg.get("channel_topk_ratio", 0.1),
            channel_min=synth_cfg.get("channel_min", 32),
            mask_strength=synth_cfg.get("mask_strength", 1.0),
        )

    raise ValueError(f"Unsupported synthesis mode: {synth_mode}")

class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"],
            mcfg["pred_depth"],
            mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True),
            feat_normed=mcfg.get("feat_normed", False),
            encoder_cfg=mcfg,
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        self.synthesis_mode = args["meta"].get("synthesis_mode", "neural_mask")
        self.synth_probability = args["meta"].get("synth_probability", 0.5)
        self.synthetic_weight = args["meta"].get("synthetic_weight", 2.0)
        self.use_channel_routing = bool(mcfg.get("use_channel_routing", False))
        self.use_channel_weighted_loss = bool(mcfg.get("use_channel_weighted_loss", False))
        self.lambda_selected = float(mcfg.get("lambda_selected", 1.0))
        self.lambda_background_preservation = float(mcfg.get("lambda_background_preservation", 0.0))
        self.lambda_delta_contrastive = float(mcfg.get("lambda_delta_contrastive", 0.0))
        self.delta_contrastive_temperature = float(mcfg.get("delta_contrastive_temperature", 0.2))
        self.use_masked_channel_reliability = bool(mcfg.get("use_masked_channel_reliability", True))
        self.feature_synth_mode = mcfg.get("feature_synth_mode", "mixed")
        self.feature_synth_alpha = float(mcfg.get("feature_synth_alpha", 1.0))
        self.token_mask_mode = mcfg.get("token_mask_mode", "rectangle")
        self.token_mask_ratio_range = (
            float(mcfg.get("token_mask_min_ratio", 0.05)),
            float(mcfg.get("token_mask_max_ratio", 0.25)),
        )
        if self.feature_synth_mode not in VALID_FEATURE_SYNTH_MODES:
            raise ValueError(f"Unsupported feature_synth_mode: {self.feature_synth_mode}")
        if self.token_mask_mode not in VALID_TOKEN_MASK_MODES:
            raise ValueError(f"Unsupported token_mask_mode: {self.token_mask_mode}")
        logger.info(f"Loss mode {self.loss_mode}")

        # ---------- data ----------
        dcfg = args["data"]
        assert dcfg["dataset"] in dcfg["data_name"] # check if the dataset aligns with the few-shot folder
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip",False),
            use_vflip=dcfg.get("use_vflip",False),
            use_rotate90=dcfg.get("use_rotate90",False),
            use_color_jitter=dcfg.get("use_color_jitter",False),
            use_gray=dcfg.get("use_gray",False),
            use_blur=dcfg.get("use_blur",False),
        )
        self.synthesizer = build_synthesizer(mcfg)
        self.channel_router = None
        self.feature_synthesizer = None
        self.channel_probe_synthesizer = None
        if self.use_channel_routing:
            synth_cfg = mcfg.get("synthesis", {})
            self.channel_router = ChannelRouter(
                top_k=int(mcfg.get("channel_topk", 32)),
                momentum=float(mcfg.get("channel_score_momentum", 0.9)),
                use_ema=bool(mcfg.get("use_channel_ema", True)),
                use_stability=bool(mcfg.get("use_channel_stability", False)),
            ).to(self.device)
            self.feature_synthesizer = ChannelAwareFeatureSynthesizer(alpha=self.feature_synth_alpha)
            self.channel_probe_synthesizer = CutPasteUnion(colorJitter=synth_cfg.get("color_jitter", 0.5))
            logger.info(
                "Channel routing enabled: top_k=%d, mode=%s, token_mask=%s, weighted_loss=%s",
                self.channel_router.top_k,
                self.feature_synth_mode,
                self.token_mask_mode,
                self.use_channel_weighted_loss,
            )
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],                         # for cosine_warmup
            min_lr=ocfg.get("min_lr", 1e-6),                  # for cosine_warmup
            warmup_epoch=ocfg.get("warmup_epoch", 5),         # for cosine_warmup
            step_size=ocfg.get("step_size", 300),             # for step
            gamma=ocfg.get("gamma", 0.1),                     # for step
        )
        self.epochs = ocfg["epochs"]
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        # log_dir.mkdir(parents=True, exist_ok=True)     
        self.ckpt_dir = log_dir

        self.tag = lcfg.get("write_tag", "train")      
        
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

    def _loss_fn(
        self,
        h,
        p,
        syn_mask=None,
        top_indices=None,
        top_weights=None,
        p_normal=None,
        preservation_mask=None,
    ) -> torch.Tensor:
        if self.use_channel_weighted_loss and syn_mask is not None and top_indices is not None and top_weights is not None:
            loss = self._channel_weighted_loss(h, p, syn_mask, top_indices, top_weights)
        else:
            if self.loss_mode == "l2":
                patch_loss = F.mse_loss(h, p, reduction="none").mean(dim=2)
            elif self.loss_mode == "smooth_l1":
                patch_loss = F.smooth_l1_loss(h, p, reduction="none").mean(dim=2)
            else:
                raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

            if syn_mask is None:
                loss = patch_loss.mean()
            else:
                weights = torch.ones_like(patch_loss)
                weights = weights + syn_mask * self.synthetic_weight
                loss = (patch_loss * weights).sum() / weights.sum().clamp_min(1.0)

        if (
            self.lambda_background_preservation > 0.0
            and preservation_mask is not None
            and p_normal is not None
        ):
            loss = loss + self.lambda_background_preservation * self._background_preservation_loss(
                p,
                p_normal,
                preservation_mask,
            )

        return loss

    def _channel_weighted_loss(self, h, p, token_mask, top_indices, top_weights) -> torch.Tensor:
        eps = 1e-8
        loss_global = F.mse_loss(p, h)

        top_indices = top_indices.to(device=p.device, dtype=torch.long).flatten()
        top_weights = top_weights.to(device=p.device, dtype=p.dtype).flatten()
        if top_indices.numel() == 0:
            return loss_global

        rec_selected = p.index_select(dim=2, index=top_indices)
        normal_selected = h.index_select(dim=2, index=top_indices)

        mask = token_mask.to(device=p.device, dtype=p.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] != 1:
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        elif mask.dim() not in (2, 3):
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        if mask.shape[:2] != p.shape[:2]:
            raise ValueError(f"Expected token_mask batch/token shape {tuple(p.shape[:2])}, got {tuple(mask.shape[:2])}")

        weights = top_weights / top_weights.sum().clamp_min(eps)
        weights = weights.view(1, 1, -1)
        selected_error = (rec_selected - normal_selected).pow(2)
        denom = (mask.sum() * weights.sum()).clamp_min(eps)
        loss_selected = (selected_error * mask * weights).sum() / denom
        return loss_global + self.lambda_selected * loss_selected

    def _background_preservation_loss(self, p_synthetic, p_normal, token_mask) -> torch.Tensor:
        eps = 1e-8
        mask = token_mask.to(device=p_synthetic.device, dtype=p_synthetic.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] != 1:
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        elif mask.dim() not in (2, 3):
            raise ValueError(f"Expected token_mask shape [B, N] or [B, N, 1], got {tuple(token_mask.shape)}")
        bg_mask = 1.0 - mask
        bg_error = (p_synthetic - p_normal.detach()).pow(2).mean(dim=2, keepdim=True)
        return (bg_error * bg_mask).sum() / bg_mask.sum().clamp_min(eps)

    def _build_channel_routed_context(self, imgs, labels, paths, h):
        _, imgs_probe = self.channel_probe_synthesizer(imgs, labels)
        F_probe = self.model.target_features(imgs_probe, paths, n_layer=self.n_layer).detach()

        self.channel_router.update(h.detach(), F_probe)
        top_indices, top_weights, _ = self.channel_router.get_topk()

        B, N, _ = h.shape
        grid_size = self._infer_square_grid_size(N)
        token_mask = generate_token_mask(
            B=B,
            N=N,
            grid_size=grid_size,
            mask_ratio_range=self.token_mask_ratio_range,
            mode=self.token_mask_mode,
            device=h.device,
            dtype=h.dtype,
        )
        F_feat_syn, applied_token_mask, applied_channel_indices = self.feature_synthesizer(
            h.detach(),
            token_mask,
            top_indices,
            top_weights,
            mode=self.feature_synth_mode,
            alpha=self.feature_synth_alpha,
        )
        contrastive_loss = h.new_zeros(())
        if self.use_masked_channel_reliability:
            top_indices, top_weights, _ = self.channel_router.update(
                h.detach(),
                F_feat_syn.detach(),
                token_mask=applied_token_mask,
            )

        if self.lambda_delta_contrastive > 0.0:
            token_mask_2 = generate_token_mask(
                B=B,
                N=N,
                grid_size=grid_size,
                mask_ratio_range=self.token_mask_ratio_range,
                mode=self.token_mask_mode,
                device=h.device,
                dtype=h.dtype,
            )
            F_feat_syn_2, applied_token_mask_2, _ = self.feature_synthesizer(
                h.detach(),
                token_mask_2,
                top_indices,
                top_weights,
                mode=self.feature_synth_mode,
                alpha=self.feature_synth_alpha,
            )
            if self.use_masked_channel_reliability:
                top_indices, top_weights, _ = self.channel_router.update(
                    h.detach(),
                    F_feat_syn.detach(),
                    token_mask=applied_token_mask,
                    F_synthetic_probe_2=F_feat_syn_2.detach(),
                    token_mask_2=applied_token_mask_2,
                )
            contrastive_loss = masked_delta_contrastive_loss(
                h.detach(),
                F_feat_syn.detach(),
                F_feat_syn_2.detach(),
                applied_token_mask,
                applied_token_mask_2,
                top_indices,
                top_weights,
                temperature=self.delta_contrastive_temperature,
            )

        return F_feat_syn, applied_token_mask, applied_channel_indices, top_weights, contrastive_loss

    @staticmethod
    def _infer_square_grid_size(num_tokens: int) -> int:
        grid_size = int(round(num_tokens ** 0.5))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected a square token grid, got {num_tokens} tokens")
        return grid_size

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save({"predictor": self.model.predictor.state_dict(),
                    "projector": self.model.projector.state_dict() if self.model.projector else None,
                    "epoch": ep, "lr": self.optimizer.param_groups[0]["lr"]}, self.ckpt_dir/name)

    def train(self):
        mp.set_start_method("spawn", force=True); gstep = 0
        for ep in range(self.epochs):
            logger.info("Epoch %d", ep+1); self.sampler.set_epoch(ep); loss_m, time_m = AverageMeter(), AverageMeter()
            for itr, (imgs, labels, paths) in enumerate(self.loader):
                imgs = imgs.to(self.device, non_blocking=True)
                def _step():
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        h = self.model.target_features(imgs, paths, n_layer=self.n_layer).detach()
                        syn_mask = None
                        top_indices = None
                        top_weights = None
                        contrastive_loss = h.new_zeros(())

                        if np.random.rand() < self.synth_probability:
                            if self.use_channel_routing:
                                z_ctx, syn_mask, top_indices, top_weights, contrastive_loss = self._build_channel_routed_context(
                                    imgs, labels, paths, h
                                )
                            elif self.synthesis_mode == "cutpaste":
                                _, imgs_abn = self.synthesizer(imgs, labels)
                                z_ctx = self.model.target_features(imgs_abn, paths, n_layer=self.n_layer).detach()
                                syn_mask = None
                            else:
                                z_ctx, syn_mask = self.synthesizer(imgs, h, labels)
                        else:
                            z_ctx = h

                        p = self.model.predict(self.model.dropout(z_ctx))
                        p_normal = None
                        preservation_mask = syn_mask
                        if self.lambda_background_preservation > 0.0 and syn_mask is not None:
                            with torch.no_grad():
                                p_normal = self.model.predict(h)
                        if self.use_channel_routing and not self.use_channel_weighted_loss:
                            syn_mask = None
                        loss = self._loss_fn(
                            h,
                            p,
                            syn_mask=syn_mask,
                            top_indices=top_indices,
                            top_weights=top_weights,
                            p_normal=p_normal,
                            preservation_mask=preservation_mask,
                        )
                        if self.lambda_delta_contrastive > 0.0:
                            loss = loss + self.lambda_delta_contrastive * contrastive_loss
                        return loss
                (loss,), t = gpu_timer(lambda: [_step()])
                if self.use_bf16: self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                else: loss.backward(); self.optimizer.step()
                grad_stats = grad_logger(self.model.predictor.named_parameters()); self.optimizer.zero_grad()
                loss_m.update(loss.item()); time_m.update(t); gstep += 1
                if gstep % 100 == 0: self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep+1, itr, loss.item(), t)
                if itr % 100 == 0:
                    logger.info("[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB (%.1fms)", ep+1, itr, loss.item(), loss_m.avg, torch.cuda.max_memory_allocated()/1024**2, time_m.avg)
                    if grad_stats:
                        logger.info("    grad: [%.2e %.2e] (%.2e %.2e)", grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max)
            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]['lr']
            )
            if self.scheduler is not None:
                self.scheduler.step()

def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml");
        if not cfg_path.exists(): raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f: args = yaml.safe_load(f)
    Trainer(args).train()

if __name__ == "__main__":
    main()
