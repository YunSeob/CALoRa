"""
generate_preamble_embedded_spectrogram.py

Pipeline:
    /datasets/sfX/preamble_train/gen_symbol/*.mat  (raw IQ, shape (1, 2^sf*8*20))
        -> per-symbol STFT (20 symbols × 33 time-bins = 660 columns)
        -> chirp restorer  (CALoRa)
        -> magnitude spectrogram  (n_classes, 660)
    /datasets/sfX/preamble_train/spectrogram/*.mat

Usage:
    python generate_preamble_embedded_spectrogram.py \\
        --sf 7,8,9 \\
        --root_path /datasets \\
        --weights_dir /phd/ys/edlora/CALoRa/weights \\
        --calora_dir /phd/ys/calora

    # CFO 변형 추가 (ppm 랜덤 ±20)
    python generate_preamble_embedded_spectrogram.py --sf 7 --cfo
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SF_TO_N_CLASSES = {7: 128, 8: 256, 9: 512, 10: 1024}
N_SYMBOLS_PER_PACKET = 20   # 8 preamble + 2 sync + 2 downchirp + 8 data
STFT_FRAMES_PER_SYMBOL = 33
SPEC_WIDTH = N_SYMBOLS_PER_PACKET * STFT_FRAMES_PER_SYMBOL  # 660


# ─────────────────────────────────────────────────────────────────────────────
# SF-specific STFT parameters
# ─────────────────────────────────────────────────────────────────────────────

def get_stft_params(sf: int) -> dict:
    n_classes = SF_TO_N_CLASSES[sf]
    return {
        "n_classes":  n_classes,
        "nsamp":      n_classes * 8,
        "n_fft":      n_classes * 8,
        "win_length": n_classes // 2,
        "hop_length": n_classes // 4,
        "freq_size":  n_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_impairments(iq: np.ndarray, fs: float = 1e6,
                      ppm: float = 0.0, fc: float = 915e6) -> np.ndarray:
    """Apply CFO + SFO to a single-symbol IQ signal. ppm=0 → no distortion."""
    if ppm == 0.0:
        return iq.copy()

    n = np.arange(len(iq))
    t = n / fs

    cfo_hz = fc * (ppm * 1e-6)
    signal_cfo = iq * np.exp(1j * 2 * np.pi * cfo_hz * t)

    sfo = 1 + (ppm * 1e-6)
    interpolator = interp1d(t, signal_cfo, kind="linear", fill_value="extrapolate")
    return interpolator(t / sfo)


def spec_to_network_input(stft_raw: torch.Tensor, freq_size: int,
                          device: torch.device) -> torch.Tensor:
    """STFT complex → (1, 2, freq_size, W) real/imag network input.

    Matches spec_to_network_input2 from the notebook:
      cyclic frequency trim → per-sample max normalisation → real/imag split
    """
    trim = freq_size // 2
    y = torch.cat([stft_raw[:, -trim:, :], stft_raw[:, :trim, :]], dim=1)
    y = y.to(device)

    y_abs = torch.abs(y)
    y_max = torch.tensor(
        [torch.max(y_abs[b]) for b in range(y_abs.shape[0])],
        device=device,
    ).unsqueeze(1).unsqueeze(2)
    y = y / y_max

    y = torch.view_as_real(y)        # (B, freq_size, W, 2)
    y = y.permute(0, 3, 1, 2)        # (B, 2, freq_size, W)
    return y.float()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(sf: int, weights_dir: str, calora_dir: str,
               device: torch.device) -> torch.nn.Module:
    weight_path = os.path.join(weights_dir, f"chirp_restorer_sf{sf}.pth")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    if calora_dir not in sys.path:
        sys.path.insert(0, calora_dir)
    from models.model_components import CALoRa

    model = CALoRa().to(device)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded chirp restorer SF{sf} from {weight_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def process_file(mat_path: str, model: torch.nn.Module,
                 params: dict, device: torch.device,
                 ppm: float) -> np.ndarray:
    """Convert one gen_symbol .mat → (n_classes, 660) magnitude spectrogram."""
    md = loadmat(mat_path)
    iq_all = md["chirp"][0]   # (nsamp * N_SYMBOLS,)

    n_classes  = params["n_classes"]
    nsamp      = params["nsamp"]
    freq_size  = params["freq_size"]
    n_fft      = params["n_fft"]
    win_length = params["win_length"]
    hop_length = params["hop_length"]

    result = torch.zeros(n_classes, SPEC_WIDTH)

    for i in range(N_SYMBOLS_PER_PACKET):
        sym_iq = apply_impairments(iq_all[i * nsamp : (i + 1) * nsamp], ppm=ppm)

        an_spec = torch.tensor(sym_iq).unsqueeze(0)  # (1, nsamp)
        stft_raw = torch.stft(
            input=an_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            pad_mode="constant",
            return_complex=True,
        )  # (1, n_fft//2+1, 33)

        net_in = spec_to_network_input(stft_raw, freq_size, device)  # (1, 2, freq_size, 33)
        pred   = model(net_in)                                        # (1, 2, freq_size, 33)

        mag = torch.abs(pred[0, 0].cpu()) + torch.abs(pred[0, 1].cpu())
        result[:, i * STFT_FRAMES_PER_SYMBOL : (i + 1) * STFT_FRAMES_PER_SYMBOL] = mag

    return result.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-SF processing
# ─────────────────────────────────────────────────────────────────────────────

def process_sf(sf: int, root_path: str, weights_dir: str, calora_dir: str,
               device: torch.device, ppm_mode: str, max_ppm: float):
    params  = get_stft_params(sf)
    src_dir = os.path.join(root_path, f"sf{sf}", "preamble_train", "gen_symbol")
    out_dir = os.path.join(root_path, f"sf{sf}", "preamble_train", "spectrogram")
    os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(src_dir, "*.mat")))
    if not mat_files:
        print(f"  [WARN] No .mat files in {src_dir} — skipping SF{sf}")
        return

    print(f"\n[SF{sf}] {len(mat_files)} files")
    print(f"  src : {src_dir}")
    print(f"  out : {out_dir}")

    model = load_model(sf, weights_dir, calora_dir, device)

    for k, path in enumerate(mat_files):
        if ppm_mode == "random":
            ppm = float(np.random.uniform(-max_ppm, max_ppm))
        elif ppm_mode == "fixed":
            ppm = max_ppm
        else:
            ppm = 0.0

        spec = process_file(path, model, params, device, ppm)

        out_path = os.path.join(out_dir, os.path.basename(path))
        savemat(out_path, {
            "__header__":  b"Preamble-embedded chirp spectrogram",
            "__version__": "1.0",
            "__globals__": [],
            "__ppm__":     ppm,
            "chirp":       spec,
        })

        if (k + 1) % 200 == 0 or (k + 1) == len(mat_files):
            print(f"  [{k+1}/{len(mat_files)}] {os.path.basename(path)}")

    print(f"[SF{sf}] 완료 ✅  ({len(mat_files)} files -> {out_dir})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Preamble-embedded spectrogram generator")

    p.add_argument("--sf", type=str, default="7,8,9",
                   help="생성할 SF 목록, 쉼표로 구분 (기본값: '7,8,9'). 예: --sf 7  또는  --sf 7,8,9")
    p.add_argument("--root_path", type=str, default="/datasets",
                   help="데이터 베이스 경로 (기본값: /datasets). "
                        "입력: {root_path}/sfX/preamble_train/gen_symbol/  "
                        "출력: {root_path}/sfX/preamble_train/spectrogram/")
    p.add_argument("--weights_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
                   help="chirp_restorer_sfX.pth 가중치 폴더 (기본값: ./weights)")
    p.add_argument("--calora_dir", type=str,
                   default=os.path.dirname(os.path.abspath(__file__)),
                   help="models/model_components.py 가 있는 디렉토리 (기본값: 이 스크립트와 같은 폴더)")

    cfo_group = p.add_mutually_exclusive_group()
    cfo_group.add_argument("--cfo", action="store_true",
                           help="각 파일마다 CFO/SFO를 랜덤 적용 (±max_ppm)")
    cfo_group.add_argument("--fixed_ppm", type=float, default=None,
                           help="모든 파일에 고정 ppm 값으로 CFO/SFO 적용")
    p.add_argument("--max_ppm", type=float, default=20.0,
                   help="--cfo 사용 시 최대 ppm 범위 (기본값: 20.0)")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sf_list = [int(x.strip()) for x in args.sf.split(",")]
    for sf in sf_list:
        if sf not in SF_TO_N_CLASSES:
            raise ValueError(f"지원하지 않는 SF: {sf}. 가능한 값: {list(SF_TO_N_CLASSES.keys())}")

    if args.cfo:
        ppm_mode = "random"
    elif args.fixed_ppm is not None:
        ppm_mode = "fixed"
    else:
        ppm_mode = "none"

    print("--- Preamble-embedded 스펙트로그램 생성 시작 ---")
    print(f"  SF       : {sf_list}")
    print(f"  root_path: {args.root_path}")
    print(f"  CFO 모드  : {ppm_mode}"
          + (f" (max ±{args.max_ppm} ppm)" if ppm_mode == "random" else
             f" ({args.fixed_ppm} ppm)"    if ppm_mode == "fixed"  else ""))
    print(f"  device   : {device}")
    print("------------------------------------------------\n")

    for sf in sf_list:
        process_sf(
            sf=sf,
            root_path=args.root_path,
            weights_dir=args.weights_dir,
            calora_dir=args.calora_dir,
            device=device,
            ppm_mode=ppm_mode,
            max_ppm=args.max_ppm if ppm_mode == "random" else (args.fixed_ppm or 0.0),
        )

    print("\n--- 모든 작업 완료 ---")


if __name__ == "__main__":
    main()
