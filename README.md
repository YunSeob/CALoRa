# CALoRa (Chirp-Aware LoRa)
Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultralow SNR

<img src="https://img.shields.io/badge/-python3.10-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/-pytorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/-IQ Signal-8CAAE6?style=flat&logo=scipy&logoColor=white"/> <img src ="https://img.shields.io/badge/-Spectrogram-7B16FF?style=flat&logo=spectrum&logoColor=white"/>

This repository contains the official implementation of the paper: **"Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultralow SNR"**, accepted in _IEEE Internet of Things Journal (2026)_.

## 💡 Key Features
- **Enhanced Preamble Detection**: Achieves high detection probability even in ultra-low SNR environments (e.g., under -20dB).
	-  **Symbol Restoration**
    
- **Chirp-Aware Mechanism:** Utilizes a novel self-attention module to effectively capture LoRa chirp characteristics.
	- **Convolutional Neural Network (CNN)**
	- **Transformer Encoder (Self-attention)**
    
- **End-to-End Pipeline:** Includes full support for signal generation, channel simulation, and model training.
	- **Restore-then-Detect**

<h2>Abstract</h2>
In Low-Power Wide-Area Networks (LPWANs) such as LoRa, the preamble is essential for detecting highly attenuated signals. Its repetitive pattern allows a receiver to identify the presence of the signal and its precise starting point. However, in ultra-low Signal-to-Noise Ratio (SNR) environments, the preamble becomes undetectable as it is buried in strong noise, causing the entire detection process to fail. Although existing methods, such as those based on preamble symbol energy accumulation or deep learning-based spectrogram restoration, have been proposed, their performance remains limited under these extreme conditions. To address this limitation, this paper proposes a novel two-stage preamble detection scheme. The first stage employs a Convolutional-Transformer Encoder-Deconvolutional network that leverages self-attention to capture the distinct linear patterns of chirp signals even in the presence of severe noise. In the second stage, a classifier determines the presence of the preamble. Experimental results demonstrate that our proposed method significantly outperforms conventional approaches, lowering the minimum required SNR for preamble detection. To validate its performance, we utilized metrics including True Positive Rate (TPR) and F-scores. Under these evaluations, our scheme achieves a detection accuracy of over 90% in the ultra-low SNR range of -21.7 dB to -24.3 dB, confirming its robustness and practical viability.


![모델 구조](./images/Figure_architecture.png)

%% ## Symbol Restoration
설명 %%

### Example of Symbol Restoration
![프리앰블 탐지기](./images/Figure_symbol_restoration_example.png)

## 🛠️ Environment & Prerequisites
### Tested System
* ***OS**: Ubuntu 22.04 LTS
* **GPU**: NVIDIA GeForce RTX 3090 (24GB) or higher 
* **CUDA**: 12.1 (Compatible with PyTorch 2.1.0)
* **Python**: 3.10 
### Key Dependencies 
* **PyTorch**: 2.1.0 (docker image : pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime)
* **NumPy**: 1.26.4; For numerical operations 
* **SciPy**: 1.15.3; For signal processing & LoRa channel simulation 
* **Matplotlib**: 3.10.3; For visualization of detection results

## ⚙️ Installation

```bash
git clone https://github.com/YunSeob/CALoRa.git
cd CALoRa 
pip install -r requirements.txt
```

## 🚀사용 방법 (Usage)

### 📡Generating Datasets 
**1. Signal Specification** 
Key parameters used for LoRa signal generation:
* **Bandwidth** : 125 kHz 
* **Sampling Rate** : 1 MHz 
* **Modulation:** LoRa Symbol (IQ Data) 
  
**2. Usage**
You can configure the dataset generation using the following arguments:
* **`--symbol`**: Specifies the signal type (Noise level).
	* *Noisy* : SNR range from **-40 dB** to **0 dB** 
	* *Clean* : Fixed SNR of **35 dB** 
* **`--generate_size`**: Number of data samples to generate per SNR level (Default: 32,768).


**3. Output Structure** 
Generated data is saved in '.mat' format within the `./data_symbol/sfX/gen_symbol/` directory.
* **Filename Convention:** `{sym_index}_{snr}_{sf}_{bw}_0_{val}_0_0.mat`

```bash
# Generate Noisy Symbol (generate_size default : 32768)
python generate_symbols.py --symbol noisy --generate_size 1000

# Generate Clean Symbol (generate_size default : 32768)
python generate_symbols.py --symbol clean --generate_size 1000
```

### 📡Preamble-Embedded Data Generation
1. **Frame Structure**
The generated LoRa frames are structured as follows:
	- **Sequence** : Preamble (8 Symbols) + Down-chirp (2 Symbols) + Payload (10 Symbols)
	- **Payload** : Composed of random symbol values.
	- **SNR Range** : -40 dB ~ - 0 dB

2.  **Usage Arguments**
	- **--generate_size** : Defines the number of data samples generated for each SNR level (Default: 100).

3. **출력 (Output)**
	- The output files are stored in `.mat` format within the `./data_symbol/preamble_train/sfX/gen_symbol/` 
	- **File Naming Rule** : `{sym_index}_{snr}_{sf}_{bw}_0_{payload list}_0_0.mat`

```bash
python generate_preamble_embedded.py --generate_size 100
```

### 📡 Train
Symbol Restoration 모델을 학습하기 위해 다음 명령어를 통해 학습을 진행

```bash
python train.py --sf 7 --train_iters 1000000
```

<h4>Prediction</h4>
- `predict.ipynb` 참고

## 📂 폴더 구조 (Directory Structure)
```
. 
├── datasets/ # 데이터셋 폴더 
├── images/
├── models/ # 모델 코드
├── LoRa.py # LoRa util 코드 
├── generate_preamble_embedded.py # 프리앰블이 포함된 IQ 신호 생성 코드
├── generate_symbols.py # LoRa 심볼 생성 코드
├── train.py # 학습 실행 파일 
├── predict.ipynb # Symbol Restoration 예측 및 성능 평가
└── README.md # 프로젝트 설명서
```


## Notes
```
@article{kim2026chirp,
  title={Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultra-Low SNR},
  author={Kim, Yun-Seob and Byeon, Seunggyu and Kim, Dong-Hyun and Hasegawa, Mikio and Kim, Jong-Deok},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

Acknowledgement 
