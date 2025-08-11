#!/usr/bin/env python3
"""
ToW 데이터셋의 실제 토큰 길이 분석
"""

import json
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset_lengths():
    # 데이터 로드
    with open('ToW_koconovel_complete.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"전체 데이터 개수: {len(data)}")
    
    # 토크나이저 로드 (DeepSeek 모델 기준)
    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        trust_remote_code=True
    )
    
    # ToW 토큰이 있는 데이터만 필터링
    tow_data = [entry for entry in data if entry['tow_count'] > 0]
    print(f"ToW 토큰이 있는 데이터: {len(tow_data)}")
    
    # 각 텍스트의 토큰 길이 계산
    lengths = []
    tow_positions = []
    
    for i, entry in enumerate(tow_data[:100]):  # 처음 100개 샘플링
        augmented_text = entry['augmented_text']
        
        # 토큰화
        tokens = tokenizer.tokenize(augmented_text)
        length = len(tokens)
        lengths.append(length)
        
        # ToW 토큰 위치 찾기
        text = augmented_text
        tow_start = text.find('<ToW>')
        if tow_start != -1:
            # ToW 시작점까지의 토큰 수
            before_tow = tokenizer.tokenize(text[:tow_start])
            tow_positions.append(len(before_tow))
        
        if i < 5:  # 처음 5개 예시 출력
            print(f"\n--- 샘플 {i+1} ---")
            print(f"총 토큰 길이: {length}")
            print(f"ToW 시작 위치: {tow_positions[-1] if tow_positions else 'N/A'}")
            print(f"텍스트 일부: {augmented_text[:100]}...")
    
    # 통계 분석
    lengths = np.array(lengths)
    tow_positions = np.array(tow_positions)
    
    print(f"\n=== 길이 통계 ===")
    print(f"평균 길이: {lengths.mean():.1f} 토큰")
    print(f"중간값: {np.median(lengths):.1f} 토큰")
    print(f"최소 길이: {lengths.min()} 토큰")
    print(f"최대 길이: {lengths.max()} 토큰")
    print(f"표준편차: {lengths.std():.1f} 토큰")
    
    print(f"\n=== 길이별 분포 ===")
    ranges = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, float('inf'))]
    for start, end in ranges:
        if end == float('inf'):
            count = np.sum(lengths >= start)
            print(f"{start}+ 토큰: {count}개 ({count/len(lengths)*100:.1f}%)")
        else:
            count = np.sum((lengths >= start) & (lengths < end))
            print(f"{start}-{end} 토큰: {count}개 ({count/len(lengths)*100:.1f}%)")
    
    print(f"\n=== ToW 위치 통계 ===")
    if len(tow_positions) > 0:
        print(f"ToW 평균 시작 위치: {tow_positions.mean():.1f} 토큰")
        print(f"ToW 위치 중간값: {np.median(tow_positions):.1f} 토큰")
        print(f"ToW 최빠름/최늦음: {tow_positions.min()}-{tow_positions.max()} 토큰")
    
    # 512 토큰으로 자를 때 손실 분석
    print(f"\n=== 512 토큰 제한시 손실 분석 ===")
    truncated_count = np.sum(lengths > 512)
    print(f"512 토큰 초과하는 데이터: {truncated_count}개 ({truncated_count/len(lengths)*100:.1f}%)")
    
    if len(tow_positions) > 0:
        tow_lost = np.sum(tow_positions > 512)
        print(f"ToW 토큰이 512 이후에 있는 경우: {tow_lost}개 ({tow_lost/len(tow_positions)*100:.1f}%)")
    
    # 권장사항
    percentiles = [90, 95, 99]
    print(f"\n=== 권장 최대 길이 ===")
    for p in percentiles:
        length = np.percentile(lengths, p)
        print(f"{p}% 데이터를 포함하려면: {int(length)} 토큰")

if __name__ == "__main__":
    analyze_dataset_lengths()