# CALoRa (Chirp-Aware LoRa)
Chirp-Aware Self-Attention for Robust LoRa Preamble Detection under Ultra-Low SNR

<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python3.11"/>
<img src="https://img.shields.io/badge/-pytorch-EE4C2C?style=flat&logo=PyTorch"/>


- Symbol Restoration
- Enhanced Preamble Detection
- Restore-then-Detect
- Convolutional Neural Network (CNN)
- Transformer Encoder (Self-attention)

<h2>Abstract</h2>
In Low-Power Wide-Area Networks (LPWANs) such as LoRa, the preamble is essential for detecting highly attenuated signals. Its repetitive pattern allows a receiver to identify the presence of the signal and its precise starting point. However, in ultra-low Signal-to-Noise Ratio (SNR) environments, the preamble becomes undetectable as it is buried in strong noise, causing the entire detection process to fail. Although existing methods, such as those based on preamble symbol energy accumulation or deep learning-based spectrogram restoration, have been proposed, their performance remains limited under these extreme conditions. To address this limitation, this paper proposes a novel two-stage preamble detection scheme. The first stage employs a Convolutional-Transformer Encoder-Deconvolutional network that leverages self-attention to capture the distinct linear patterns of chirp signals even in the presence of severe noise. In the second stage, a classifier determines the presence of the preamble. Experimental results demonstrate that our proposed method significantly outperforms conventional approaches, lowering the minimum required SNR for preamble detection. To validate its performance, we utilized metrics including True Positive Rate (TPR) and F-scores. Under these evaluations, our scheme achieves a detection accuracy of over 90% in the ultra-low SNR range of -21 dB to -24 dB, confirming its robustness and practical viability.