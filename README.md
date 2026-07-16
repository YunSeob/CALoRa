# CALoRa (Chirp-Aware LoRa)
Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultralow SNR

<img src="https://img.shields.io/badge/-python3.10-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/-pytorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/-IQ Signal-8CAAE6?style=flat&logo=scipy&logoColor=white"/> <img src ="https://img.shields.io/badge/-Spectrogram-7B16FF?style=flat&logo=spectrum&logoColor=white"/>

This repository contains the official implementation of the paper: **"[Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultralow SNR](https://ieeexplore.ieee.org/abstract/document/11386915)"**, accepted in _IEEE Internet of Things Journal (2026)_.

## Overview

**Problem:** In LoRa networks, preamble detection fails in ultra-low SNR environments (below −20 dB) because the signal is buried in noise.

**Solution:** CALoRa introduces a two-stage pipeline:
1. **Chirp Restorer** — A CNN-Transformer network reconstructs the chirp structure from a noisy spectrogram using self-attention.
2. **Preamble Detector** — A temporal classifier localizes the preamble position in the restored spectrogram.

**Result:** Achieves **>90% detection accuracy** at SNRs between **−21.7 dB and −24.3 dB**, significantly outperforming conventional methods.

![Model Architecture](./images/Figure_architecture.png)

**Example of Symbol Restoration**

![Symbol Restoration](./images/Figure_symbol_restoration_example.png)

## 🛠️ Environment & Prerequisites

### Tested System
* **OS**: Ubuntu 22.04 LTS
* **GPU**: NVIDIA GeForce RTX 4090
* **CUDA**: 12.1
* **Python**: 3.10

### Key Dependencies
* **PyTorch**: 2.1.0 (docker image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)
* **NumPy**: 1.26.4 — numerical operations
* **SciPy**: 1.15.3 — signal processing & LoRa channel simulation
* **Matplotlib**: 3.10.3 — visualization

## ⚙️ Installation

```bash
git clone https://github.com/YunSeob/CALoRa.git
cd CALoRa
pip install -r requirements.txt
```

## 🚀 Usage

### 📡 Generating Datasets (Symbol Restoration Training)

Key parameters used for LoRa signal generation:
* **Bandwidth**: 125 kHz
* **Sampling Rate**: 1 MHz
* **Modulation**: LoRa Symbol (IQ Data)

**Arguments**
* **`--symbol`**: Signal type — `noisy` (SNR −40 to 0 dB) or `clean` (fixed 35 dB)
* **`--generate_size`**: Number of samples per SNR level (default: 16,384)

**Output**: `.mat` files in `./data_symbol/sfX/gen_symbol/`  
**Filename convention**: `{sym_index}_{snr}_{sf}_{bw}_0_{val}_0_0.mat`

```bash
# Generate noisy symbols
python generate_symbols.py --symbol noisy --generate_size 1000

# Generate clean symbols
python generate_symbols.py --symbol clean --generate_size 1000
```

---

### 📡 Preamble Detection Data Pipeline

Generating training/test data for the preamble detector requires **two steps**:

```
Step 1: generate_preamble_embedded.py            →  raw IQ packets        (.mat)
Step 2: generate_preamble_embedded_spectrogram.py →  restored spectrograms (.mat)
```

#### Step 1 — Generate Preamble-Embedded IQ Signals

Each generated frame consists of 20 LoRa symbols:

| Position | Symbols | Description |
|---|---|---|
| 0 – 7 | 8 | Preamble (value = 0) |
| 8 – 9 | 2 | Sync word (random, range 16–96) |
| 10 – 11 | 2 | Down-chirp |
| 12 – 19 | 8 | Payload (random) |

- **SNR range**: −40 dB to 0 dB (41 levels)
- **Output format**: `.mat` file, `chirp` field of shape `(1, 2^sf × 8 × 20)`
- **Output path**: `{root_path}/sf{sf}/preamble_train/gen_symbol/`
- **Filename convention**: `{index}_{snr}_{sf}_{bw}_0_{payload_list}_0_0.mat`

| Argument | Default | Description |
|---|---|---|
| `--sf` | `7,8,9,10` | Spreading Factor(s), comma-separated |
| `--generate_size` | `100` | Number of samples per SNR level per SF |
| `--root_path` | `/datasets` | Base output directory (e.g. a Docker volume mount — not the `datasets/` folder inside this repo) |

```bash
# Generate SF7 and SF8 data, 200 samples per SNR level
python generate_preamble_embedded.py \
    --sf 7,8 \
    --generate_size 200 \
    --root_path /datasets

# Generate all SFs (7, 8, 9, 10)
python generate_preamble_embedded.py --root_path /datasets
```

> **Output example** (SF7, 200 samples × 41 SNR levels = 8,200 files):
> ```
> /datasets/sf7/preamble_train/gen_symbol/
>     0_0_7_125000_0_[0,0,...,42]_0_0.mat
>     1_-1_7_125000_0_[0,0,...,87]_0_0.mat
>     ...
> ```

#### Step 2 — Convert IQ Signals to Restored Spectrograms

This step runs each raw IQ packet through the **pre-trained chirp restorer** (symbol-by-symbol STFT → `CNNTransformerHybrid` → magnitude) and saves the result as a `(n_classes, 660)` spectrogram.

- **Input**: `{root_path}/sf{sf}/preamble_train/gen_symbol/*.mat`
- **Output**: `{root_path}/sf{sf}/preamble_train/spectrogram/*.mat`, `chirp` field of shape `(n_classes, 660)`
- **Requires**: pre-trained chirp restorer weights (`weights/chirp_restorer_sfX.pth`)

| Argument | Default | Description |
|---|---|---|
| `--sf` | `7,8,9` | SF(s) to process, comma-separated |
| `--root_path` | `/datasets` | Base output directory (same as Step 1 — not the `datasets/` folder inside this repo) |
| `--weights_dir` | `./weights` | Folder containing `chirp_restorer_sfX.pth` |
| `--calora_dir` | `/phd/ys/calora` | Path to the directory where the model class is defined |
| `--cfo` | off | Apply random CFO/SFO impairment per file (±`max_ppm`) |
| `--fixed_ppm` | — | Apply a fixed ppm CFO/SFO to all files |
| `--max_ppm` | `20.0` | Max ppm range when `--cfo` is used |

```bash
# Basic (no CFO)
python generate_preamble_embedded_spectrogram.py \
    --sf 7,8,9 \
    --root_path /datasets \
    --weights_dir <weight_path>

# With random CFO (±20 ppm) — simulates carrier frequency offset
python generate_preamble_embedded_spectrogram.py \
    --sf 7 \
    --root_path /datasets \
    --weights_dir <weight_path> \
    --cfo --max_ppm 20

# With fixed CFO
python generate_preamble_embedded_spectrogram.py \
    --sf 7 \
    --root_path /datasets \
    --weights_dir <weight_path> \
    --fixed_ppm 15.0
```

> **Output example** (SF7):
> ```
> /datasets/sf7/preamble_train/spectrogram/
>     0_0_7_125000_0_[0,0,...,42]_0_0.mat     # chirp: (128, 660)
>     1_-1_7_125000_0_[0,0,...,87]_0_0.mat
>     ...
> ```

#### Complete Data Generation Example

```bash
# Step 1 — Generate raw IQ packets (SF7, 200 samples/SNR)
python generate_preamble_embedded.py \
    --sf 7 \
    --generate_size 200 \
    --root_path /datasets

# Step 2 — Convert to spectrograms (no CFO)
python generate_preamble_embedded_spectrogram.py \
    --sf 7 \
    --root_path /datasets \
    --weights_dir <weight_path>

# Step 2-b — Convert to spectrograms with random CFO (augmented test set)
python generate_preamble_embedded_spectrogram.py \
    --sf 7 \
    --root_path /datasets \
    --weights_dir <weight_path> \
    --cfo
```

---

### 📡 Train

#### 1. Symbol Restoration Model

To train the proposed symbol restoration model, run:

```bash
# Train with Spreading Factor 7
python train.py --sf 7 --train_iters 100000
```

#### 2. Preamble Detection Model

Use `train_preamble_detector.py` to train the preamble detection model. The spectrogram dataset from the pipeline above is required as input.

```bash
# SF7 (original architecture: C=64, no SEBlock)
python train_preamble_detector.py \
    --sf 7 \
    --data_dirs <path/to/sf7/preamble_train/spectrogram> \
    --C 64 \
    --no_se_tcn \
    --epochs 100 \
    --save_dir ckpt_sf7

# SF8 / SF9 (updated architecture: C=128, SEBlock enabled)
python train_preamble_detector.py \
    --sf 8 \
    --data_dirs <path/to/sf8/preamble_train/spectrogram> \
    --epochs 100 \
    --save_dir ckpt_sf8
```

---

### 📊 Preamble Detection Evaluation

After training, evaluate the preamble detector using `test_preamble_detector.py`.

> **Note:** Paths to the model checkpoint and data directory will differ per environment. Adjust them accordingly.

| Argument | Description |
|---|---|
| `--sf` | Spreading Factor of the model to evaluate |
| `--model_path` | Path to the trained checkpoint (`.pth`) |
| `--data_dirs` | Directory containing test spectrogram `.mat` files |
| `--shift_size` | Time-axis offset applied to test spectrograms (default: `66`) |
| `--thresh` | Probability threshold for preamble detection |
| `--location_tol` | Column tolerance for correct location (default: `37` ≈ 1 LoRa symbol) |
| `--C` | Base channel count used during training (default: `128`) |
| `--no_se_tcn` | Use original TCN without SEBlock (required for SF7 weights) |

**SF7 Example** — trained with original TCN (no SEBlock, C=64):

```bash
python test_preamble_detector.py \
    --sf 7 \
    --model_path <path/to/ckpt_sf7/best_finetuned.pth> \
    --data_dirs <path/to/sf7/preamble_train/spectrogram> \
    --shift_size 66 \
    --thresh 0.5 \
    --location_tol 37 \
    --C 64 \
    --no_se_tcn
```

**SF8 Example** — trained with updated architecture (SEBlock, C=128), default settings apply:

```bash
python test_preamble_detector.py \
    --sf 8 \
    --model_path <path/to/ckpt_sf8/best_finetuned.pth> \
    --data_dirs <path/to/sf8/preamble_train/spectrogram> \
    --shift_size 66 \
    --thresh 0.6 \
    --location_tol 37
```

The script outputs a per-SNR detection rate table and the following overall metrics:

| Metric | Description |
|---|---|
| **TPR (Recall)** | Fraction of correctly detected preambles |
| **Precision** | Fraction of detections that are correct |
| **F0.5 / F1 / F2** | F-score variants weighting precision vs. recall |
| **Accuracy** | Overall correct classifications |
| **AvgProb** | Mean model confidence per SNR bin |
| **DetRate** | Fraction of samples with correct detection AND location |

To save results as CSV, add `--output_csv <output_path>`.

---

### 📊 Prediction & Analysis

#### Symbol Restoration

Use `predict.py` for symbol restoration inference. The model takes a noisy LoRa spectrogram as input and outputs a restored version that suppresses noise while preserving the chirp structure.

## Acknowledgement
This code is built upon the official implementation of **NELoRa**. We appreciate their contributions to the open-source community.
- NELoRa Repository: [Github Link Here](https://github.com/hanqingguo/NELoRa-Sensys)

## Citation
If you find this work useful in your research, please consider citing our paper:
```bibtex
@article{kim2026chirp,
  title={Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultra-Low SNR},
  author={Kim, Yun-Seob and Byeon, Seunggyu and Kim, Dong-Hyun and Hasegawa, Mikio and Kim, Jong-Deok},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```
