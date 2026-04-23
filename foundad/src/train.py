
from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

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
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        self.synthesis_mode = args["meta"].get("synthesis_mode", "neural_mask")
        self.synth_probability = args["meta"].get("synth_probability", 0.5)
        self.synthetic_weight = args["meta"].get("synthetic_weight", 2.0)
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

    def _loss_fn(self, h, p, syn_mask=None) -> torch.Tensor:
        if self.loss_mode == "l2":
            patch_loss = F.mse_loss(h, p, reduction="none").mean(dim=2)
        elif self.loss_mode == "smooth_l1":
            patch_loss = F.smooth_l1_loss(h, p, reduction="none").mean(dim=2)
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

        if syn_mask is None:
            return patch_loss.mean()

        weights = torch.ones_like(patch_loss)
        weights = weights + syn_mask * self.synthetic_weight
        return (patch_loss * weights).sum() / weights.sum().clamp_min(1.0)

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
                        h = self.model.target_features(imgs, paths, n_layer=self.n_layer)
                        syn_mask = None

                        if np.random.rand() < self.synth_probability:
                            if self.synthesis_mode == "cutpaste":
                                _, imgs_abn = self.synthesizer(imgs, labels)
                                z_ctx = self.model.target_features(imgs_abn, paths, n_layer=self.n_layer)
                                syn_mask = None
                            else:
                                z_ctx, syn_mask = self.synthesizer(imgs, h, labels)
                        else:
                            z_ctx = h

                        p = self.model.predict(self.model.dropout(z_ctx))
                        return self._loss_fn(h, p, syn_mask=syn_mask)
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
