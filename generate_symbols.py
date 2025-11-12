import torch
from collections import Counter
import os
import argparse
from LoRa import LoRa


bw = 125000

# SNR 범위가 -40~0인 LoRa 심볼 데이터를 mat 파일 형식으로 저장
# 각 SF마다 생성되는 심볼의 수는 generate_size에 따라 결정됨
# 사용되는 SF는 7,8,9
# 심볼 저장 경로 예시 (SF=7) : 심볼은 ./data_symbol/sf7/gen_symbol/[filename]
# filename : {sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat
def generate_symbol(generate_size=32768):
    # generate_size = generate_size
    snr_list = list(range(-40, 1))

    for i in range(3):
        sf = i +7
        root_path = f'./data_symbol/sf{str(sf)}/gen_symbol/'
        lora_init = LoRa(sf, bw)
        for j in range(len(snr_list)):
            snr = snr_list[j]
            print(f"[진행상황] SF={sf}, SNR={snr} dB에서 심볼 {generate_size}개 생성 중...")
            lora_init.generate_symbol_with_noise(sf, bw, generate_size, root_path, target_snr=snr_list[j])
        print(f"[완료] SF={sf} 완료 ✅\n")
    
# Clean Symbol의 SNR은 +35 dB로 설정
def generate_clean_symbol(generate_size=32768):
    snr_list = [35]
    for i in range(3):
        sf = i + 7
        root_path = f'./data_symbol/sf{str(sf)}/gen_symbol/'
        lora_init = LoRa(sf, bw)
        for j in range(len(snr_list)):
            snr = snr_list[j]
            print(f"[진행상황] SF={sf}, SNR={snr} dB에서 심볼 {generate_size}개 생성 중...")
            lora_init.generate_symbol_with_noise(sf, bw, generate_size, root_path, target_snr=snr_list[0])
        print(f"[완료] SF={sf} 완료 ✅\n")

if __name__ == "__main__":
    # 1. ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="LoRa 심볼 생성 스크립트")

    # 2. 인자 추가하기
    
    # --symbol 인자: 'noisy' 또는 'clean'만 받도록 설정 (required=True로 필수 인자 지정)
    parser.add_argument('--symbol', 
                        type=str, 
                        required=True, 
                        choices=['noisy', 'clean'], 
                        help="생성할 심볼의 종류. 'noisy' 또는 'clean' 중 선택")

    # --generate_size 인자: 정수형(int)으로 받고, 기본값을 32768로 설정
    parser.add_argument('--generate_size', 
                        type=int, 
                        default=32768, 
                        help="각 SNR/SF 당 생성할 심볼의 개수 (기본값: 32768)")

    # 3. 인자 파싱
    args = parser.parse_args()

    # 4. 파싱된 인자에 따라 적절한 함수 실행
    print(f"--- 심볼 생성 시작 ---")
    print(f"  선택된 모드: {args.symbol}")
    print(f"  생성할 개수: {args.generate_size}")
    print(f"----------------------\n")

    if args.symbol == 'noisy':
        generate_symbol(generate_size=args.generate_size)
    elif args.symbol == 'clean':
        generate_clean_symbol(generate_size=args.generate_size)

    print("--- 모든 작업 완료 ---")