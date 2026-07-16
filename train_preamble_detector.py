"""
train_preamble_detector.py — Preamble Detector Training Script

Improvements over the notebook:
- All hyperparameters via argparse (no hardcoded values)
- Unified file collection from multiple directories
- Progress bar per epoch instead of per-sample prints
- Correct AMP usage (GradScaler + autocast)
- Best model checkpoint saves model / cfg / sf
- CSV metrics log (same format as notebook)

Usage:
    python train_preamble_detector.py \\
        --sf 7 \\
        --data_dirs ./data_symbol/preamble_train/sf7/gen_symbol2_spec_ \\
                    ./data_symbol/preamble_train/sf7/gen_symbol2_spec_cfo \\
        --C 128 --epochs 50 --save_dir ckpt_sf7
"""

import argparse
import csv
import glob
import math
import os
import random
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from model import PreambleDetector, SoftIoU, metrics_from_probs

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SF_TO_H = {7: 128, 8: 256, 9: 512}

FNAME_RE = re.compile(
    r"^(?P<symidx>\d+)_(?P<snr>-?\d+)_(?P<sf>\d+)_(?P<bw>\d+)_0_(?P<val>.+)_0_0\.mat$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PreambleDataset(Dataset):
    """Loads (H, W) chirp spectrograms from .mat files.

    Augmentation (use_aug=True):
      - Frequency (CFO) shift: roll along H axis by ±10% of H
      - Time (STO) shift: roll along W axis by ±max_t_shift; label moves with it
    """

    def __init__(
        self,
        file_list: List[str],
        target_W: int,
        target_H: int,
        L_candidates: Tuple[int, ...],
        use_aug: bool = True,
        max_t_shift: int = 20,
    ):
        self.files = list(file_list)
        self.W = target_W
        self.H = target_H
        self.Ls = L_candidates
        self.L = int(np.median(self.Ls))
        self.use_aug = use_aug
        self.max_t_shift = max_t_shift

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        md = loadmat(path)
        spec = md["chirp"]
        if spec.shape != (self.H, self.W):
            raise ValueError(f"Expected ({self.H},{self.W}), got {spec.shape} in {path}")
        spec = spec.astype(np.float32)

        t_start, t_end = 0, self.L

        if self.use_aug:
            # Frequency (CFO) shift
            max_f = int(self.H * 0.1)
            shift_f = np.random.randint(-max_f, max_f + 1)
            spec = np.roll(spec, shift_f, axis=0)

            # Time (STO) shift — label tracks the shift
            shift_t = np.random.randint(-self.max_t_shift, self.max_t_shift + 1)
            spec = np.roll(spec, shift_t, axis=1)
            t_start += shift_t
            t_end += shift_t

        spec_t = torch.from_numpy(spec)
        mu = float(spec_t.mean())
        sigma = float(spec_t.std() + 1e-6)
        spec_t = (spec_t - mu) / sigma

        x = spec_t.unsqueeze(0).float()  # (1, H, W)

        y = torch.zeros(self.W, dtype=torch.float32)
        valid_start = max(0, min(self.W, t_start))
        valid_end = max(0, min(self.W, t_end))
        if valid_start < valid_end:
            y[valid_start:valid_end] = 1.0

        return x, y, path


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.mat")))
        print(f"  {d}: {len(found)} files")
        files.extend(found)
    return files


def append_metrics_csv(path: str, sf: int, epoch: int, tr_loss: float,
                       val_loss: float, m_agg: dict, dt: float):
    p = Path(path)
    is_new = not p.exists()
    fields = ["sf", "epoch", "tr_loss", "val_loss",
              "MAE_start", "MAE_end", "Hit@±1_start", "Hit@±1_end", "seconds"]
    row = {
        "sf": sf, "epoch": epoch,
        "tr_loss": round(float(tr_loss), 6),
        "val_loss": round(float(val_loss), 6),
        "MAE_start": round(float(m_agg.get("MAE_start", math.nan)), 4),
        "MAE_end":   round(float(m_agg.get("MAE_end",   math.nan)), 4),
        "Hit@±1_start": round(float(m_agg.get("Hit@±1_start", math.nan)), 4),
        "Hit@±1_end":   round(float(m_agg.get("Hit@±1_end",   math.nan)), 4),
        "seconds": round(float(dt), 1),
    }
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    sf = args.sf
    H = SF_TO_H[sf]
    Ls = tuple(args.l_candidates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    print(f"\n===== Training SF{sf} | H={H} | C={args.C} | Ls={Ls} | device={device} =====")

    # Data
    all_files = collect_files(args.data_dirs)
    if not all_files:
        raise RuntimeError("No .mat files found. Check --data_dirs.")
    random.seed(args.seed)
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_ratio))
    val_files = all_files[:n_val]
    tr_files = all_files[n_val:]
    print(f"Train: {len(tr_files)}  Val: {len(val_files)}")

    ds_tr = PreambleDataset(tr_files, 660, H, Ls, use_aug=not args.no_aug)
    ds_val = PreambleDataset(val_files, 660, H, Ls, use_aug=False)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=device.type == "cuda")
    ld_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Model
    model = PreambleDetector(C=args.C, Ls=Ls, use_se_tcn=not args.no_se_tcn).to(device)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            ckpt = torch.load(args.pretrained, map_location=device)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state)
            print(f"Loaded pretrained weights from {args.pretrained}")
        else:
            print(f"[WARN] Pretrained path not found: {args.pretrained} — training from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, threshold=1e-4
    )
    bce = nn.BCEWithLogitsLoss()
    soft_iou = SoftIoU()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_dir = args.save_dir or f"ckpt_sf{sf}"
    os.makedirs(save_dir, exist_ok=True)
    best_val = float("inf")
    L_median = int(np.median(Ls))

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        for step, (x, y, _) in enumerate(ld_tr):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, p, _ = model(x)
                loss = (args.bce_w * bce(logits, y)
                        + args.iou_w * soft_iou(p, y))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * x.size(0)

        tr_loss = loss_sum / len(ld_tr.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        m_sum = {"MAE_start": 0.0, "MAE_end": 0.0,
                 "Hit@±1_start": 0.0, "Hit@±1_end": 0.0}
        cnt = 0

        with torch.no_grad():
            for x, y, _ in ld_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, p, _ = model(x)
                loss = (args.bce_w * bce(logits, y)
                        + args.iou_w * soft_iou(p, y))
                val_loss += loss.item() * x.size(0)
                m = metrics_from_probs(p, y, L=L_median)
                for k in m_sum:
                    m_sum[k] += m[k] * x.size(0)
                cnt += x.size(0)

        val_loss /= max(1, len(ld_val.dataset))
        m_agg = {k: v / max(1, cnt) for k, v in m_sum.items()}
        scheduler.step(val_loss)
        dt = time.time() - t0
        curr_lr = opt.param_groups[0]["lr"]

        print(
            f"[SF{sf}][Ep {epoch:03d}] "
            f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
            f"LR={curr_lr:.1e}  "
            f"Hit(s)={m_agg['Hit@±1_start']:.3f}  "
            f"{dt:.1f}s"
        )

        append_metrics_csv(
            f"{save_dir}/train_metrics.csv",
            sf, epoch, tr_loss, val_loss, m_agg, dt,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": {
                        "C": args.C,
                        "Ls": Ls,
                        "use_se_tcn": not args.no_se_tcn,
                        "L_map": {sf: Ls},
                    },
                    "sf": sf,
                    "optimizer": opt.state_dict(),
                },
                f"{save_dir}/best_finetuned.pth",
            )
            print(f"  -> Saved best (val={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f}  Saved to {save_dir}/")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PreambleDetector")

    # Required
    p.add_argument("--sf", type=int, required=True, choices=[7, 8, 9],
                   help="Spreading Factor (determines spectrogram height H)")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="One or more directories containing .mat training files")

    # Model
    p.add_argument("--C", type=int, default=128,
                   help="Base channel count for ColumnNetV2 (default: 128)")
    p.add_argument("--l_candidates", nargs="+", type=int, default=[262, 264, 266],
                   help="Preamble length candidates in columns (default: 262 264 266)")
    p.add_argument("--no_se_tcn", action="store_true",
                   help="Use original TCN without SEBlock (matches SF7 paper weights)")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--bce_w", type=float, default=0.3,
                   help="BCEWithLogitsLoss weight (default: 0.3)")
    p.add_argument("--iou_w", type=float, default=0.7,
                   help="SoftIoU loss weight (default: 0.7)")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of data for validation (default: 0.1)")
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--no_aug", action="store_true",
                   help="Disable data augmentation (CFO + STO shift)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision")

    # I/O
    p.add_argument("--save_dir", type=str, default=None,
                   help="Checkpoint directory (default: ckpt_sf{sf})")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to pretrained .pth for fine-tuning")

    args = p.parse_args()
    args.amp = not args.no_amp
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
