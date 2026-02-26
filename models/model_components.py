# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Classification Model
# NELoRa (For Decoding Spectrogram's Symbols)
class classificationHybridModel(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim_in=2, conv_dim_out=128, conv_dim_lstm=1024):
        super(classificationHybridModel, self).__init__()

        self.out_size = conv_dim_out
        self.conv1 = nn.Conv2d(conv_dim_in, 16, (3, 3), stride=(2, 2), padding=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dense = nn.Linear(conv_dim_lstm * 4, conv_dim_out * 4)
        self.fcn1 = nn.Linear(conv_dim_out * 4, conv_dim_out * 2)
        self.fcn2 = nn.Linear(2 * conv_dim_out, conv_dim_out)
        self.softmax = nn.Softmax(dim=1)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.pool1(out)
        out = out.view(out.size(0), -1)

        out = self.act(self.dense(out))
        out = self.drop2(out)

        out = self.act(self.fcn1(out))
        out = self.drop1(out)
        out = self.fcn2(out)

        # out = self.softmax(out)
        return out


# Denoising Model
# NeLoRa (For compare)
class maskCNNModel(nn.Module):
    def __init__(self, opts):
        super(maskCNNModel, self).__init__()
        self.opts = opts

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(opts.x_image_channel, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),

        )

        self.lstm = nn.LSTM(
            opts.conv_dim_lstm,
            opts.lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2 * opts.lstm_dim, opts.fc1_dim)
        self.fc2 = nn.Linear(opts.fc1_dim, opts.freq_size * opts.y_image_channel)

    def forward(self, x):
        out = x.transpose(2, 3).contiguous()
        out = self.conv(out)
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        out = out.view(out.size(0), out.size(1), self.opts.y_image_channel, -1)
        out = torch.sigmoid(out)
        out = out.transpose(1, 2).contiguous()
        out = out.transpose(2, 3).contiguous()
        masked = out * x  # out is mask, masked is denoised
        return masked
    
# For extracting mask from maskCNNModel
class maskCNNModel_return_mask(nn.Module):
    def __init__(self, opts):
        super(maskCNNModel_return_mask, self).__init__()
        self.opts = opts

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(opts.x_image_channel, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),

        )

        self.lstm = nn.LSTM(
            opts.conv_dim_lstm,
            opts.lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2 * opts.lstm_dim, opts.fc1_dim)
        self.fc2 = nn.Linear(opts.fc1_dim, opts.freq_size * opts.y_image_channel)

    def forward(self, x):
        out = x.transpose(2, 3).contiguous()
        out = self.conv(out)
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        out = out.view(out.size(0), out.size(1), self.opts.y_image_channel, -1)
        out = torch.sigmoid(out)
        out = out.transpose(1, 2).contiguous()
        out = out.transpose(2, 3).contiguous()
        mask = out
        masked = out * x  # out is mask, masked is denoised
        return mask

# Ours
class CALoRa(nn.Module):
    def __init__(self, img_channels=2, hidden_dim=64, trans_dim=128, nhead=4, num_layers=4, max_h_enc=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans_dim = trans_dim

        # CNN Encoder: Height만 다운샘플링 (stride=(4,1))
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=(4,3), stride=(4,1), padding=(0,1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Linear Projection for Transformer
        self.linear_proj = nn.Linear(hidden_dim, trans_dim)
        self.linear_proj_back = nn.Linear(trans_dim, hidden_dim)

        # 학습 가능한 Positional Encoding
        self.max_h_enc = max_h_enc  # 최대 SF에 따라 변하는 H_enc 최대값 (예: SF=10일 때는 1024//4=256)
        self.pos_embedding = nn.Parameter(torch.randn(self.max_h_enc * 33, trans_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dim_feedforward=256, batch_first=False),
            num_layers=num_layers
        )

        # CNN Decoder: Height만 업샘플링 (stride=(4,1))
        self.decoder_convt = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=(4,3), stride=(4,1), padding=(0,1))
        self.decoder_bn = nn.BatchNorm2d(img_channels)
        self.decoder_final_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()

    def _generate_pos_embed(self, H, W, B):
        pos_embed = self.pos_embedding[:H*W, :].unsqueeze(1).repeat(1, B, 1)
        return pos_embed

    def forward(self, x):
        B, C, input_height, input_width = x.size()

        # Encode
        feat = self.encoder(x)  # [B, hidden_dim, H_enc, W_enc]
        B_enc, C_enc, H_enc, W_enc = feat.shape

        # Flatten
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()  # [H_enc*W_enc, B, C_enc]

        # Positional Encoding
        pos_embed = self._generate_pos_embed(H_enc, W_enc, B)
        feat_embed = self.linear_proj(feat_flat) + pos_embed

        # Transformer
        trans_out = self.transformer(feat_embed)

        # Restore shape
        trans_out = self.linear_proj_back(trans_out)
        trans_out = trans_out.permute(1, 2, 0).contiguous()
        trans_out = trans_out.view(B, C_enc, H_enc, W_enc)

        # Decode
        out = self.decoder_convt(trans_out)
        out = self.decoder_bn(out)

        # Final Conv & Sigmoid
        out = self.decoder_final_conv(out)
        out = self.decoder_sigmoid(out)

        # 최종적으로 입력과 정확히 맞춤
        if out.shape[2] != input_height or out.shape[3] != input_width:
            out = F.interpolate(out, size=(input_height, input_width), mode='bilinear', align_corners=False)

        return out

# Ours (For extracting global features)  
class CALoRa_extract(nn.Module):
    def __init__(self, img_channels=2, hidden_dim=64, trans_dim=128, nhead=4, num_layers=4, max_h_enc=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans_dim = trans_dim

        # CNN Encoder: Height만 다운샘플링 (stride=(4,1))
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=(4,3), stride=(4,1), padding=(0,1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Linear Projection for Transformer
        self.linear_proj = nn.Linear(hidden_dim, trans_dim)
        self.linear_proj_back = nn.Linear(trans_dim, hidden_dim)

        # 학습 가능한 Positional Encoding
        self.max_h_enc = max_h_enc
        self.pos_embedding = nn.Parameter(torch.randn(self.max_h_enc * 33, trans_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dim_feedforward=256, batch_first=False),
            num_layers=num_layers
        )

        # CNN Decoder
        self.decoder_convt = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=(4,3), stride=(4,1), padding=(0,1))
        self.decoder_bn = nn.BatchNorm2d(img_channels)
        self.decoder_final_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()

    def _generate_pos_embed(self, H, W, B):
        pos_embed = self.pos_embedding[:H*W, :].unsqueeze(1).repeat(1, B, 1)
        return pos_embed

    # forward 함수를 수정하여 중간 출력을 반환하도록 변경
    def forward(self, x, return_intermediate=False):
        B, C, input_height, input_width = x.size()

        # 1. CNN-Encoder 출력
        feat = self.encoder(x)
        B_enc, C_enc, H_enc, W_enc = feat.shape
        
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()
        pos_embed = self._generate_pos_embed(H_enc, W_enc, B)
        feat_embed = self.linear_proj(feat_flat) + pos_embed

        # 2. Transformer-Encoder 출력
        trans_out_seq = self.transformer(feat_embed) # 시퀀스 형태의 출력

        # Transformer 출력을 다시 이미지 형태로 복원
        trans_out_flat = self.linear_proj_back(trans_out_seq)
        trans_out_flat = trans_out_flat.permute(1, 2, 0).contiguous()
        trans_out_img = trans_out_flat.view(B, C_enc, H_enc, W_enc)

        # Decode
        out = self.decoder_convt(trans_out_img)
        out = self.decoder_bn(out)
        out = self.decoder_final_conv(out)
        out = self.decoder_sigmoid(out)

        if out.shape[2] != input_height or out.shape[3] != input_width:
            out = F.interpolate(out, size=(input_height, input_width), mode='bilinear', align_corners=False)

        # `return_intermediate` 플래그가 True일 때만 중간 출력을 반환
        if return_intermediate:
            return out, feat, trans_out_img
        else:
            return out