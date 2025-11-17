
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import datasets.data_loader as data_loader
from utils import generate_dataset

import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from collections import Counter
import csv

from models.model_components import maskCNNModel, classificationHybridModel, CALoRa, CALoRa_extract
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Training")

parser.add_argument('--sf',
                        type=int, 
                        default=7, 
                        help="Spreading Factor (기본값 : 7)")

parser.add_argument('--train_iters', 
                        type=int, 
                        default=100000, 
                        help="Epochs (기본값 : 100000)")

args = parser.parse_args()

# Define dummy options class for maskCNNModel
class Opts:
    # LoRa Configuration
    sf=args.sf
    fs = 1e6
    bw = 125e3
    n_classes = int(2**sf)      # 128
    
    # Model Configuration
    x_image_channel = 2
    y_image_channel = 2
    conv_dim_lstm = int(n_classes*8)
    ratio_bt_train_and_test = 0.8
    batch_size = 16
    num_workers = 1
    normalization = True
    sorting_type = 4
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    train_iters = args.train_iters
    scaling_for_imaging_loss = n_classes
    log_step = 100
    val_steps = 500
    val_ratio = 0.1

    # For NELoRa
    lstm_dim = 400
    fc1_dim = 600
    conv_dim_out_ = n_classes
    conv_dim_lstm_ = n_classes * 8

    # Spectrogram Configuration
    stft_nfft = int(n_classes * fs // bw)
    stft_window = int(n_classes // 2)
    stft_overlap = stft_window // 2
    freq_size = n_classes
    
    # Path Configuration
    root_path='./'
    data_dir = f'./data_symbol/sf{sf}/gen_symbol'
    # data_dir = f'./data_symbol/sf{sf}/gen_symbol_new'
    save_dir = './'

    # Chirp Configuration
    code_list = [round(i, 1) for i in list(np.arange(0, n_classes, 0.1))]
    snr_list = list(range(-40, 1))
    bw_list = [125000]
    sf_list = [sf]
    instance_list = list(range(0, 7))
    feature_name = 'chirp'
    groundtruth_code = 35
opts = Opts()

def to_var(x):
    """GPU 사용 가능 시 텐서를 CUDA로 이동시키고 Variable로 감싼다."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def spec_to_network_input(x, opts):
    """스펙트로그램 텐서를 네트워크 입력 형식으로 변환"""
    freq_size = opts.freq_size
    # 주파수 축 절반 길이를 기준으로 잘라 사용할 크기를 계산
    trim_size = freq_size // 2
    # 뒤 절반과 앞 절반을 이어 붙여 주파수 축을 순환 시프트한다(fftshift 유사 동작을 수행한다).
    # 스펙트로그램의 주파수 관점에서는 위와 아래
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if opts.normalization:
        # 배치별 절대값 최댓값으로 정규화
        y_abs = torch.abs(y)
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y = torch.div(y, y_abs_max)

    if opts.x_image_channel == 2:
        # 복소 입력을 실수 채널 2개로 분해하여 [B,2,H,W]로 변환
        y = torch.view_as_real(y)  # 형상: [B, H, W, 2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        # 위상 정보를 사용하여 단일 채널로 구성
        y = torch.angle(y)         # 형상: [B, H, W]
        y = torch.unsqueeze(y, 1)  # 형상: [B, 1, H, W]
    return y  # 반환 형상: 채널=2일 때 [B, 2, H, W], 채널=1일 때 [B, 1, H, W]을 반환한다.

#  Data Loader
[files_train, files_test] = generate_dataset(opts.root_path, opts.data_dir, opts.ratio_bt_train_and_test,
                          opts.code_list, opts.snr_list, opts.bw_list, opts.sf_list,
                          opts.instance_list, opts.sorting_type)

training_dataloader_X, testing_dataloader_X = data_loader.lora_loader(opts, files_train, files_test, False)
training_dataloader_Y, testing_dataloader_Y = data_loader.lora_loader(opts, files_train, files_test, True)

files_train_split, files_val_split = train_test_split(
    files_train, test_size=opts.val_ratio, random_state=32, shuffle=True
)

# validation dataloader (새로 생성)
validation_dataloader_X, _ = data_loader.lora_loader(opts, files_val_split, files_test, False)
validation_dataloader_Y, _ = data_loader.lora_loader(opts, files_val_split, files_test, True)

# Loss 정의
loss_spec = torch.nn.MSELoss(reduction='mean')
loss_class = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
calora = CALoRa().to(device)

C_XtoY = classificationHybridModel(conv_dim_in=opts.y_image_channel,
                                       conv_dim_out=opts.conv_dim_out_,
                                       conv_dim_lstm=opts.conv_dim_lstm_).to(device)

# Optimizer
g_params = list(calora.parameters())
# g_params = list(calora.parameters()) + list(C_XtoY.parameters())
g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])

iter_X = iter(training_dataloader_X)
iter_Y = iter(training_dataloader_Y)
test_iter_X = iter(testing_dataloader_X)
test_iter_Y = iter(testing_dataloader_Y)

fixed_X, name_X_fixed = next(test_iter_X)
fixed_X = to_var(fixed_X)

fixed_Y, name_Y_fixed = next(test_iter_Y)
fixed_Y = to_var(fixed_Y)
fixed_X_spectrum_raw = torch.stft(input=fixed_X, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                      win_length=opts.stft_window, pad_mode='constant')
fixed_X_spectrum = spec_to_network_input(fixed_X_spectrum_raw, opts)
fixed_Y_spectrum_raw = torch.stft(input=fixed_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                      win_length=opts.stft_window, pad_mode='constant')
fixed_Y_spectrum = spec_to_network_input(fixed_Y_spectrum_raw, opts)
iter_per_epoch = min(len(iter_X), len(iter_Y))


# 초기 설정
best_loss = float('inf')
best_val_loss = float('inf')
csv_log_path = os.path.join(opts.save_dir, f'training_log_enhanced_sf{opts.sf}_nelora_ver6.csv')
os.makedirs(opts.save_dir, exist_ok=True)

# CSV 파일 헤더 작성
with open(csv_log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'G_Y_loss', 'G_Image_loss', 'G_Class_loss', 'Val_Loss'])

# epoch 수 계산 (총 iteration 수와 iteration-per-epoch 활용)
num_epochs = opts.train_iters // iter_per_epoch
iteration = 0

for epoch in range(num_epochs):
    for (images_X, name_X), (images_Y, name_Y) in zip(training_dataloader_X, training_dataloader_Y):

        iteration += 1

        # NELoRa
        # snr_X_mapping = list(map(lambda x: int(x.split('_')[1]), name_X))
        labels_X_mapping = list(map(lambda x: int(x.split('_')[5]), name_X))
        labels_Y_mapping = list(map(lambda x: int(x.split('_')[5]), name_Y))

        images_X, labels_X = to_var(images_X), to_var(torch.tensor(labels_X_mapping))
        images_Y, labels_Y = to_var(images_Y), to_var(torch.tensor(labels_Y_mapping))
        # snr_X = to_var(torch.tensor(snr_X_mapping).float().unsqueeze(1))

        # STFT + Network input 변환
        images_X_spectrum_raw = torch.stft(images_X, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                           win_length=opts.stft_window, pad_mode='constant')
        images_X_spectrum = spec_to_network_input(images_X_spectrum_raw, opts)

        images_Y_spectrum_raw = torch.stft(images_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                           win_length=opts.stft_window, pad_mode='constant')
        images_Y_spectrum = spec_to_network_input(images_Y_spectrum_raw, opts)

        # Forward
        fake_Y_spectrum = calora(images_X_spectrum)
        g_y_pix_loss = loss_spec(fake_Y_spectrum, images_Y_spectrum)
        labels_X_estimated = C_XtoY(fake_Y_spectrum)
        g_y_class_loss = loss_class(labels_X_estimated, labels_X)

        # Backward & optimize
        g_optimizer.zero_grad()
        G_Image_loss = opts.scaling_for_imaging_loss * g_y_pix_loss
        G_Class_loss = g_y_class_loss
        G_Y_loss = G_Image_loss + G_Class_loss
        # G_Y_loss = G_Image_loss
        G_Y_loss.backward()
        g_optimizer.step()

        # Validation
        if iteration % opts.val_steps == 0:
            val_loss_total = 0.0
            val_batches = 0

            calora.eval()
            C_XtoY.eval()

            val_iter_Y = iter(validation_dataloader_Y)
            for val_images_X, val_name_X in validation_dataloader_X:
                try:
                    val_images_Y, val_name_Y = next(val_iter_Y)
                except StopIteration:
                    val_iter_Y = iter(validation_dataloader_Y)
                    val_images_Y, val_name_Y = next(val_iter_Y)

                # val_snr_X = torch.tensor(list(map(lambda x: int(x.split('_')[1]), val_name_X)))
                val_labels_X = torch.tensor(list(map(lambda x: int(x.split('_')[5]), val_name_X)))
                val_labels_Y = torch.tensor(list(map(lambda x: int(x.split('_')[5]), val_name_Y)))

                val_images_X, val_labels_X = to_var(val_images_X), to_var(val_labels_X)
                val_images_Y, val_labels_Y = to_var(val_images_Y), to_var(val_labels_Y)
                # val_snr_X = to_var(torch.tensor(val_snr_X).float().unsqueeze(1))

                val_images_X_spectrum = spec_to_network_input(
                    torch.stft(val_images_X, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                               win_length=opts.stft_window, pad_mode='constant'), opts)

                val_images_Y_spectrum = spec_to_network_input(
                    torch.stft(val_images_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                               win_length=opts.stft_window, pad_mode='constant'), opts)

                val_fake_Y = calora(val_images_X_spectrum)
                val_loss = loss_spec(val_fake_Y, val_images_Y_spectrum)

                val_loss_total += val_loss.item()
                val_batches += 1

            if val_batches > 0:
                val_loss_avg = val_loss_total / val_batches
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    print(f"📅 [Iteration {iteration}] Best val model saved with val_loss: {val_loss_avg:.6f}")
            # torch.cuda.empty_cache() 

            calora.train()
            C_XtoY.train()

        # 모델 저장 (training 기준)
        if G_Y_loss.item() < best_loss:
            best_loss = G_Y_loss.item()
            torch.save(calora.state_dict(), os.path.join(opts.save_dir, f'training_log_enhanced_sf{opts.sf}_denoising_calora.pth'))
            torch.save(C_XtoY.state_dict(), os.path.join(opts.save_dir, f'training_log_enhanced_sf{opts.sf}_classification_calora.pth'))
            if (iteration % 200)==0 and (iteration <= 50000):
                torch.save(calora.state_dict(), os.path.join(opts.save_dir, f'./weights_history/training_log_enhanced_sf{opts.sf}_denoising_iter{iteration}_calora.pth'))
                torch.save(C_XtoY.state_dict(), os.path.join(opts.save_dir, f'./weights_history/training_log_enhanced_sf{opts.sf}_classification_iter{iteration}_calora.pth'))
            print(f"✅ [Iteration {iteration}] Best model saved with loss: {best_loss:.6f}  |  G_Image_loss : {G_Image_loss.item():.4f}  |  G_Class_loss : {G_Class_loss.item():.4f}")
            # print(f"✅ [Iteration {iteration}] Best model saved with loss: {best_loss:.6f}  |  G_Image_loss : {G_Image_loss.item():.4f}  |  G_Class_loss : {0:.4f}")

        # 로그 기록
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, G_Y_loss.item(), G_Image_loss.item(), G_Class_loss.item(), best_val_loss])
            # writer.writerow([iteration, G_Y_loss.item(), G_Image_loss.item(), 0, best_val_loss])

        # 콘솔 출력
        if iteration % opts.log_step == 0:
            print(f'Iteration [{iteration:5d}/{opts.train_iters:5d}] | G_Y_loss: {G_Y_loss.item():.4f} '
                  f'| G_Image_loss: {G_Image_loss.item():.4f} | G_Class_loss: {G_Class_loss:.4f} | Val_loss: {best_val_loss:.4f}')
