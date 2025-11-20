# CALoRa (Chirp-Aware LoRa)
Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultra-Low SNR

<img src="https://img.shields.io/badge/-python3.10-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/-pytorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/-IQ Signal-8CAAE6?style=flat&logo=scipy&logoColor=white"/> <img src ="https://img.shields.io/badge/-Spectrogram-7B16FF?style=flat&logo=spectrum&logoColor=white"/>

- **Symbol Restoration**
- **Enhanced Preamble Detection**
- **Restore-then-Detect**
- Convolutional Neural Network (CNN)
- Transformer Encoder (Self-attention)

<h2>Abstract</h2>
In Low-Power Wide-Area Networks (LPWANs) such as LoRa, the preamble is essential for detecting highly attenuated signals. Its repetitive pattern allows a receiver to identify the presence of the signal and its precise starting point. However, in ultra-low Signal-to-Noise Ratio (SNR) environments, the preamble becomes undetectable as it is buried in strong noise, causing the entire detection process to fail. Although existing methods, such as those based on preamble symbol energy accumulation or deep learning-based spectrogram restoration, have been proposed, their performance remains limited under these extreme conditions. To address this limitation, this paper proposes a novel two-stage preamble detection scheme. The first stage employs a Convolutional-Transformer Encoder-Deconvolutional network that leverages self-attention to capture the distinct linear patterns of chirp signals even in the presence of severe noise. In the second stage, a classifier determines the presence of the preamble. Experimental results demonstrate that our proposed method significantly outperforms conventional approaches, lowering the minimum required SNR for preamble detection. To validate its performance, we utilized metrics including True Positive Rate (TPR) and F-scores. Under these evaluations, our scheme achieves a detection accuracy of over 90% in the ultra-low SNR range of -21 dB to -24 dB, confirming its robustness and practical viability.


![모델 구조](./images/Figure_architecture.png)

## 진행 상황
* LoRa 심볼 데이터 생성 코드 구현
* LoRa 프리앰블을 포함한 20개의 심볼을 생성하는 코드 구현
* Symbol Restoration
	* 심볼 학습 코드 구현
	* 성능 평가를 위한 코드를 편의성을 위해 쥬피터 노트북으로 구현

## 추가적으로 필요한 코드
* Preamble Detector
	* 학습 코드 구현
	* 성능 평가 코드 구현

## Symbol Restoration
설명

### 심볼 복원 예시
![프리앰블 탐지기](./images/Figure_symbol_restoration_example.png)

## 개발 환경 (Prerequsites)
* Python 3.10
* Pytorch 2.1.0

## ⚙️ 설치 방법 (Installation)
이 프로젝트를 실행하기 위해 먼저 저장소를 클론하고 필수 라이브러리를 설치해주세요.

```bash
git clone https://github.com/YunSeob/CALoRa.git
cd CALoRa 
pip install -r requirements.txt
```

## 🚀사용 방법 (Usage)

<h4>Generating Datasets</h4>

### 📡 Generating Datasets 
**1. 신호 사양 (Signal Specs)** 
* **Bandwidth** : 125 kHz 
* **Sampling Rate** : 1 MHz 
* **Modulation:** LoRa Symbol IQ Signal 
  
**2. 실행 옵션 (Usage)** 
* **`--symbol`**: 신호의 노이즈 여부를 결정합니다. 
	* *Noisy* : SNR [-40, 0] dB 
	* *Clean* : SNR 35 dB 
* **`--generate_size`**: SNR 별 생성할 데이터 수를 정의합니다. (Default: 32,768) 

**3. 출력 (Output)** 
데이터는 `./data_symbol/sfX/gen_symbol/` 폴더에 `.mat` 형식으로 저장됩니다. 
* **파일명 규칙:** `{sym_index}_{snr}_{sf}_{bw}_0_{val}_0_0.mat`

```bash
# Noisy 심볼 생성 (generate_size default : 32768)
python generate_symbols.py --symbol noisy --generate_size 1000

# Clean 심볼 생성 (generate_size default : 32768)
python generate_symbols.py --symbol clean --generate_size 1000
```
* 프리앰블이 포함된 데이터 생성

```bash

```

<h4>Train</h4>

<h4>Prediction</h4>
- `predict.ipynb` 참고

## 📂 폴더 구조 (Directory Structure)
```bash
. 
├── data/ # 데이터셋 폴더 
├── models/ # 모델 코드 
├── train.py # 학습 실행 파일 
└── README.md # 프로젝트 설명서
```