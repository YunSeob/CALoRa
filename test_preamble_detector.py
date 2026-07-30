"""
test_preamble_detector.py — Preamble Detector Evaluation Script

Improvements over the notebook:
- Robust SNR parsing from filename (regex, not fragile path split)
- No per-sample print — periodic progress updates only
- Per-SNR detection rate table printed to stdout
- Optional CSV export of per-sample and per-SNR results
- Batch-safe inference (single sample per call, GPU-ready)

Usage:
    python test_preamble_detector.py \\
        --sf 7 \\
        --model_path ckpt_sf7/best_finetuned.pth \\
        --data_dirs ./data_symbol/preamble_train/sf7/test2_spec_ \\
                    ./data_symbol/preamble_train/sf7/test2_spec_cfo \\
        --shift_size 66 --thresh 0.7

Notes on TP/TN/FP/FN:
    The notebook defines these jointly on confidence AND location:
      TP: prob > thresh  AND  |t_start - shift_size| <= location_tol
      FN: prob > thresh  BUT  |t_start - shift_size| >  location_tol  (wrong location)
      FP: prob <= thresh AND  |t_start - shift_size| <= location_tol  (missed)
      TN: prob <= thresh AND  |t_start - shift_size| >  location_tol  (double miss)
    This differs from standard binary classification; it penalises both missed
    detections and false alarms at wrong positions simultaneously.
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat

from models.model_components import PreambleDetector, best_interval_from_p, metrics_from_counts

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SF_TO_H = {7: 128, 8: 256, 9: 512}

FNAME_RE = re.compile(
    r"^(?P<symidx>\d+)_(?P<snr>-?\d+)_(?P<sf>\d+)_(?P<bw>\d+)_0_(?P<val>.+)_0_0\.mat$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_snr(path: str) -> Optional[int]:
    """Extract SNR from filename using regex (robust vs. path-split approach)."""
    m = FNAME_RE.match(os.path.basename(path))
    if m:
        return int(m.group("snr"))
    return None


def collect_files(dirs: List[str], snr_min: Optional[int],
                  snr_max: Optional[int]) -> List[str]:
    files: List[str] = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.mat")))
        print(f"  {d}: {len(found)} files")
        files.extend(found)

    if snr_min is not None or snr_max is not None:
        lo = snr_min if snr_min is not None else -50
        hi = snr_max if snr_max is not None else 35
        before = len(files)
        files = [f for f in files if lo <= (parse_snr(f) or lo) <= hi]
        print(f"SNR filter [{lo}, {hi}]: {before} -> {len(files)} files")

    return files


def preprocess(spec: np.ndarray, shift_size: int) -> torch.Tensor:
    """Shift spectrogram right by shift_size, zero-pad left, then normalise."""
    spec_b = np.roll(spec, shift=shift_size, axis=1).astype(np.float32)
    spec_b[:, :shift_size] = 0.0
    t = torch.from_numpy(spec_b)
    mu = float(t.mean())
    sigma = float(t.std() + 1e-6)
    return (t - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: PreambleDetector,
    spec: np.ndarray,
    shift_size: int,
    Ls: Tuple[int, ...],
    device: torch.device,
) -> Tuple[np.ndarray, int, int, float]:
    """Return (p_np, t_start, t_end, mean_prob_in_interval)."""
    x = preprocess(spec, shift_size)
    x = x.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    logits, p, _ = model(x)
    p_np = p.squeeze(0).cpu().numpy()            # (W,)
    L_hat, t_start, t_end = best_interval_from_p(p_np, Ls)
    mean_prob = float(np.mean(p_np[t_start : t_end + 1]))
    return p_np, t_start, t_end, mean_prob


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sf = args.sf
    H = SF_TO_H[sf]

    # --- Load model ---
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    cfg_ckpt = ckpt.get("cfg", {})
    C = cfg_ckpt.get("C", args.C)
    use_se = cfg_ckpt.get("use_se_tcn", not args.no_se_tcn)

    L_map = cfg_ckpt.get("L_map", {})
    ckpt_sf = ckpt.get("sf", sf)
    Ls = tuple(int(x) for x in L_map.get(ckpt_sf, ()))
    if not Ls:
        Ls = tuple(args.l_candidates)

    model = PreambleDetector(C=C, Ls=Ls, use_se_tcn=use_se).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model: C={C}, Ls={Ls}, use_se_tcn={use_se}")
    print(f"Loaded from {args.model_path}")

    # --- Collect files ---
    mat_files = collect_files(args.data_dirs, args.snr_min, args.snr_max)
    if not mat_files:
        raise RuntimeError("No test files found. Check --data_dirs and SNR range.")
    print(f"Total test files: {len(mat_files)}\n")

    # --- Inference loop ---
    tp = tn = fp = fn = 0
    per_snr: dict = defaultdict(list)   # snr -> list of (prob, bool_pred, bool_exist)
    sample_rows = []                    # for CSV output

    for i, path in enumerate(mat_files):
        snr = parse_snr(path)
        md = loadmat(path)
        spec = md["chirp"].astype(np.float32)
        assert spec.shape[0] == H and spec.shape[1] == 660, (
            f"Unexpected shape {spec.shape} in {path}"
        )

        p_np, t_start, t_end, mean_prob = run_inference(
            model, spec, args.shift_size, Ls, device
        )

        bool_pred  = mean_prob > args.thresh
        bool_exist = abs(t_start - args.shift_size) <= args.location_tol

        if   bool_pred and     bool_exist: tp += 1
        elif bool_pred and not bool_exist: fn += 1
        elif not bool_pred and bool_exist: fp += 1
        else:                              tn += 1

        if snr is not None:
            per_snr[snr].append((mean_prob, bool_pred, bool_exist))

        sample_rows.append({
            "path": path, "snr": snr,
            "t_start": t_start, "t_end": t_end,
            "mean_prob": round(mean_prob, 4),
            "bool_pred": int(bool_pred),
            "bool_exist": int(bool_exist),
        })

        if (i + 1) % 200 == 0 or (i + 1) == len(mat_files):
            print(f"  [{i+1}/{len(mat_files)}] TP={tp} FN={fn} FP={fp} TN={tn}")

    # --- Overall metrics ---
    print(f"\n{'='*55}")
    print(f" Overall  (TP={tp}  FN={fn}  FP={fp}  TN={tn})")
    print(f"{'='*55}")
    m = metrics_from_counts(tp, tn, fp, fn)
    for k, v in m.items():
        print(f"  {k:<22}: {v:.4f}" if not np.isnan(v) else f"  {k:<22}: nan")

    # --- Per-SNR table ---
    if per_snr:
        print(f"\n{'SNR':>6} | {'N':>5} | {'AvgProb':>8} | {'DetRate(TP/(TP+FP+FN+TN))':>10}")
        print("-" * 45)
        for snr in sorted(per_snr.keys()):
            recs = per_snr[snr]
            avg_prob = np.mean([r[0] for r in recs])
            det_rate = np.mean([r[1] and r[2] for r in recs])   # TP / total
            print(f"{snr:>6} | {len(recs):>5} | {avg_prob:>8.4f} | {det_rate:>10.4f}")

    # --- Save results ---
    if args.output_csv:
        _save_csv(args.output_csv, sample_rows, per_snr)
        print(f"\nResults saved to {args.output_csv}")

    return tp, tn, fp, fn, per_snr


def _save_csv(path: str, sample_rows: list, per_snr: dict):
    base, ext = os.path.splitext(path)
    ext = ext or ".csv"

    # Per-sample CSV
    sample_path = f"{base}_samples{ext}"
    fields = ["path", "snr", "t_start", "t_end", "mean_prob", "bool_pred", "bool_exist"]
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample_rows)

    # Per-SNR CSV
    snr_path = f"{base}_per_snr{ext}"
    with open(snr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr", "n", "avg_prob", "det_rate"])
        for snr in sorted(per_snr.keys()):
            recs = per_snr[snr]
            avg_prob = float(np.mean([r[0] for r in recs]))
            det_rate = float(np.mean([r[1] and r[2] for r in recs]))
            w.writerow([snr, len(recs), round(avg_prob, 4), round(det_rate, 4)])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PreambleDetector")

    # Required
    p.add_argument("--sf", type=int, required=True, choices=[7, 8, 9])
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to checkpoint .pth file")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="One or more directories containing .mat test files")

    # Model (fallback when not in checkpoint)
    p.add_argument("--C", type=int, default=128,
                   help="Base channel count (used if not stored in checkpoint)")
    p.add_argument("--l_candidates", nargs="+", type=int, default=[264],
                   help="Preamble length candidates (fallback if not in checkpoint)")
    p.add_argument("--no_se_tcn", action="store_true",
                   help="Use original TCN without SEBlock (fallback if not in checkpoint)")

    # Test parameters
    p.add_argument("--shift_size", type=int, default=66,
                   help="Time-axis shift applied to test spectrograms (default: 66)")
    p.add_argument("--thresh", type=float, default=0.7,
                   help="Probability threshold for preamble detection (default: 0.7)")
    p.add_argument("--location_tol", type=int, default=37,
                   help="Column tolerance for correct location (default: 37 ≈ 1 LoRa symbol)")

    # SNR filter
    p.add_argument("--snr_min", type=int, default=None)
    p.add_argument("--snr_max", type=int, default=None)

    # Output
    p.add_argument("--output_csv", type=str, default=None,
                   help="Base path for CSV output (generates *_samples.csv and *_per_snr.csv)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
