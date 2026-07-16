import torch
import numpy as np
from collections import Counter
from scipy.io import savemat
import os
import argparse
from LoRa import LoRa
import random

def create_lora_packet(sf: int) -> list:
    packet = [0]*8
    # packet.extend([random.randint(0,3) for _ in range(2)])
    packet.extend([random.randint(16,96) for _ in range(2)])

    packet.extend([-1]*2)

    upper_bound = int(2**sf)
    packet.extend([random.randint(0, upper_bound) for _ in range(8)])

    return packet

def generate_packet_with_noise(sf, bw, generate_size, target_snr=-5, root_path=None):
    lora_init = LoRa(sf,bw)
    sym_index = 0
    if root_path is None:
        root_path = '/datasets'
    root_path = os.path.join(root_path, f'sf{sf}', 'preamble_train', 'gen_symbol', '')
    os.makedirs(root_path, exist_ok=True)
    for i in range(generate_size):
        packet_chirp = np.zeros((int(2**sf)*8*20), dtype=np.complex128)
        packet_list = create_lora_packet(sf)
        for j in range(len(packet_list)):
            val = packet_list[j]
            if val >= 0:
                chirp = lora_init.gen_symbol_fs(val,sf,bw,down=False, Fs=int(8*bw))
            else:
                chirp = lora_init.gen_symbol_fs(0,sf,bw,down=True, Fs=int(8*bw))
            gen_snr = target_snr
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            packet_chirp[j*int(2**sf*8):(j+1)*int(2**sf*8)] = chirp_awgn
        chirp_signal = np.array(packet_chirp).reshape(1,-1)
        mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
        save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{packet_list}_0_0.mat'
        savemat(root_path + save_name, mat_data)
        sym_index += 1

if __name__ == "__main__":
    # 1. ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="LoRa 심볼 생성 스크립트")

    # 2. 인자 추가하기

    parser.add_argument('--generate_size',
                        type=int,
                        default=100,
                        help="각 SNR/SF 당 생성할 심볼의 개수 (기본값: 100)")
    parser.add_argument('--root_path',
                        type=str,
                        default='/datasets',
                        help="저장 베이스 경로 (기본값: /datasets). "
                             "실제 저장 경로: {root_path}/sf{sf}/preamble_train/gen_symbol/")
    parser.add_argument('--sf',
                        type=str,
                        default='7,8,9,10',
                        help="생성할 SF 목록, 쉼표로 구분 (기본값: '7,8,9,10'). "
                             "예: --sf 7  또는  --sf 7,8")

    # 3. 인자 파싱
    args = parser.parse_args()

    sf_list = [int(x.strip()) for x in args.sf.split(',')]
    for sf in sf_list:
        if sf not in (7, 8, 9, 10):
            parser.error(f"지원하지 않는 SF 값: {sf}. 7, 8, 9, 10 중에서 선택하세요.")

    # 4. 파싱된 인자에 따라 적절한 함수 실행
    print(f"--- Preamble-embedded 신호 생성 시작 ---")
    print(f"  생성할 SF: {sf_list}")
    print(f"  생성할 개수: {args.generate_size}")
    print(f"  저장 베이스 경로: {args.root_path}")
    print(f"----------------------\n")

    snr_list = list(range(-40,1))
    bw = 125000
    for sf in sf_list:
        lora_init = LoRa(sf, bw)
        for snr in snr_list:
            print(f"[진행상황] SF={sf}, SNR={snr} dB에서 패킷 {args.generate_size}개 생성 중...")
            generate_packet_with_noise(sf, bw, generate_size=args.generate_size, target_snr=snr, root_path=args.root_path)
        print(f'[완료] SF={sf} 완료 ✅\n')

    print("--- 모든 작업 완료 ---")