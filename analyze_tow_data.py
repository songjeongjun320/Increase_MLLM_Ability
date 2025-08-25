#!/usr/bin/env python3
"""
TOW 데이터셋 Context 토큰 길이 분석 및 필터링 스크립트

이 스크립트는:
1. tow_data 폴더의 모든 JSON 파일을 분석
2. context 필드의 토큰 길이를 계산
3. 토큰 길이별 데이터 분포 시각화
4. 지정된 토큰 길이 이하의 데이터만 필터링하여 새로운 JSON 파일로 저장
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# 한글 토큰 길이 계산을 위한 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def count_korean_tokens(text: str) -> int:
    """
    한국어 텍스트의 단어 수를 띄어쓰기 기준으로 계산합니다.
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        int: 띄어쓰기 기준 단어 수
    """
    if not text:
        return 0
    
    # 텍스트를 공백으로 분리하여 단어 수 계산
    words = text.strip().split()
    
    # 빈 문자열 제거
    words = [word for word in words if word.strip()]
    
    return len(words)

def load_json_files(data_dir: str) -> List[Dict]:
    """
    데이터 디렉토리에서 모든 JSON 파일을 로드합니다.
    
    Args:
        data_dir (str): JSON 파일들이 있는 디렉토리 경로
        
    Returns:
        List[Dict]: 모든 데이터 항목들의 리스트
    """
    data_dir = Path(data_dir)
    all_data = []
    
    json_files = list(data_dir.glob("*.json"))
    print(f"발견된 JSON 파일 수: {len(json_files)}")
    
    for json_file in tqdm(json_files, desc="JSON 파일 로딩"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 파일명 정보 추가
                for item in data:
                    item['source_file'] = json_file.name
                
                all_data.extend(data)
                print(f"  {json_file.name}: {len(data)}개 항목")
                
        except Exception as e:
            print(f"  ❌ {json_file.name}: 로딩 실패 - {e}")
    
    print(f"총 로딩된 데이터 항목 수: {len(all_data)}")
    return all_data

def analyze_token_lengths(data: List[Dict]) -> Tuple[List[int], Dict]:
    """
    데이터의 context 단어 길이를 분석합니다.
    
    Args:
        data (List[Dict]): 분석할 데이터
        
    Returns:
        Tuple[List[int], Dict]: 단어 길이 리스트와 통계 정보
    """
    token_lengths = []
    file_stats = defaultdict(list)
    
    print("단어 수 분석 중...")
    for item in tqdm(data, desc="단어 수 계산"):
        context = item.get('context', '')
        word_count = count_korean_tokens(context)
        token_lengths.append(word_count)
        
        # 파일별 통계
        source_file = item.get('source_file', 'unknown')
        file_stats[source_file].append(word_count)
    
    # 전체 통계
    stats = {
        'total_items': len(token_lengths),
        'min_words': min(token_lengths) if token_lengths else 0,
        'max_words': max(token_lengths) if token_lengths else 0,
        'avg_words': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        'median_words': sorted(token_lengths)[len(token_lengths)//2] if token_lengths else 0,
    }
    
    # 파일별 통계
    stats['file_stats'] = {}
    for filename, lengths in file_stats.items():
        stats['file_stats'][filename] = {
            'count': len(lengths),
            'avg_words': sum(lengths) / len(lengths),
            'min_words': min(lengths),
            'max_words': max(lengths)
        }
    
    return token_lengths, stats

def plot_token_distribution(token_lengths: List[int], stats: Dict, output_dir: str = "."):
    """
    단어 길이 분포를 시각화합니다.
    
    Args:
        token_lengths (List[int]): 단어 길이 리스트
        stats (Dict): 통계 정보
        output_dir (str): 출력 디렉토리
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TOW 데이터셋 Context 단어 길이 분석', fontsize=16, fontweight='bold')
    
    # 1. 히스토그램
    axes[0, 0].hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('단어 길이 분포 (히스토그램)')
    axes[0, 0].set_xlabel('단어 수')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].axvline(stats['avg_words'], color='red', linestyle='--', label=f'평균: {stats["avg_words"]:.1f}')
    axes[0, 0].axvline(stats['median_words'], color='green', linestyle='--', label=f'중앙값: {stats["median_words"]}')
    axes[0, 0].legend()
    
    # 2. 박스플롯
    axes[0, 1].boxplot(token_lengths)
    axes[0, 1].set_title('단어 길이 분포 (박스플롯)')
    axes[0, 1].set_ylabel('단어 수')
    
    # 3. 누적 분포
    sorted_lengths = sorted(token_lengths)
    cumulative_pct = [i/len(sorted_lengths)*100 for i in range(1, len(sorted_lengths)+1)]
    axes[1, 0].plot(sorted_lengths, cumulative_pct)
    axes[1, 0].set_title('누적 분포 함수')
    axes[1, 0].set_xlabel('단어 수')
    axes[1, 0].set_ylabel('누적 비율 (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 주요 percentile 표시
    percentiles = [50, 80, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100) - 1
        if 0 <= idx < len(sorted_lengths):
            axes[1, 0].axvline(sorted_lengths[idx], color='red', alpha=0.5, linestyle=':', 
                             label=f'{p}%ile: {sorted_lengths[idx]}')
    axes[1, 0].legend(fontsize=8)
    
    # 4. 파일별 통계
    if 'file_stats' in stats:
        file_names = list(stats['file_stats'].keys())
        avg_words = [stats['file_stats'][f]['avg_words'] for f in file_names]
        
        axes[1, 1].bar(range(len(file_names)), avg_words)
        axes[1, 1].set_title('파일별 평균 단어 수')
        axes[1, 1].set_xlabel('파일')
        axes[1, 1].set_ylabel('평균 단어 수')
        axes[1, 1].set_xticks(range(len(file_names)))
        axes[1, 1].set_xticklabels([f.replace('.json', '') for f in file_names], rotation=45)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'word_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"분포 차트 저장됨: {output_path}")
    plt.show()

def print_statistics(token_lengths: List[int], stats: Dict):
    """
    통계 정보를 출력합니다.
    
    Args:
        token_lengths (List[int]): 단어 길이 리스트
        stats (Dict): 통계 정보
    """
    print("\n" + "="*60)
    print("📊 TOW 데이터셋 단어 길이 분석 결과")
    print("="*60)
    
    print(f"총 데이터 항목 수: {stats['total_items']:,}")
    print(f"최소 단어 수: {stats['min_words']}")
    print(f"최대 단어 수: {stats['max_words']}")
    print(f"평균 단어 수: {stats['avg_words']:.2f}")
    print(f"중앙값 단어 수: {stats['median_words']}")
    
    # Percentile 분석
    sorted_lengths = sorted(token_lengths)
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    
    print(f"\n📈 단어 길이 Percentile 분석:")
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100) - 1
        if 0 <= idx < len(sorted_lengths):
            value = sorted_lengths[idx]
            count_below = len([x for x in token_lengths if x <= value])
            percentage = count_below / len(token_lengths) * 100
            print(f"  {p:2d}%ile: {value:4d} 단어 (전체의 {percentage:5.1f}%가 이 값 이하)")
    
    # 단어 길이 구간별 분포
    print(f"\n📊 단어 길이 구간별 데이터 분포:")
    ranges = [
        (0, 10, "매우 짧음"),
        (11, 20, "짧음"),
        (21, 40, "보통"),
        (41, 60, "긺"),
        (61, 100, "매우 긺"),
        (101, float('inf'), "극도로 긺")
    ]
    
    for min_len, max_len, desc in ranges:
        if max_len == float('inf'):
            count = len([x for x in token_lengths if x >= min_len])
            range_desc = f"{min_len}+ 단어"
        else:
            count = len([x for x in token_lengths if min_len <= x <= max_len])
            range_desc = f"{min_len}-{max_len} 단어"
            
        percentage = count / len(token_lengths) * 100
        print(f"  {range_desc:12s} ({desc:8s}): {count:5,}개 ({percentage:5.1f}%)")
    
    # 파일별 통계
    if 'file_stats' in stats:
        print(f"\n📁 파일별 통계:")
        for filename, file_stat in stats['file_stats'].items():
            print(f"  {filename:40s}: "
                  f"{file_stat['count']:5,}개, "
                  f"평균 {file_stat['avg_words']:6.1f} 단어, "
                  f"범위 {file_stat['min_words']}-{file_stat['max_words']} 단어")

def filter_and_save_data(data: List[Dict], min_word_length: int, output_dir: str = "."):
    """
    지정된 단어 길이 초과의 데이터만 필터링하여 저장합니다.
    
    Args:
        data (List[Dict]): 원본 데이터
        min_word_length (int): 최소 단어 길이 (이 값을 초과하는 데이터만 저장)
        output_dir (str): 출력 디렉토리
    """
    filtered_data = []
    
    print(f"\n🔍 최소 단어 길이 {min_word_length} 초과 데이터 필터링 중...")
    
    for item in tqdm(data, desc="데이터 필터링"):
        context = item.get('context', '')
        word_count = count_korean_tokens(context)
        
        if word_count > min_word_length:
            # source_file 필드 제거 (불필요한 메타데이터)
            filtered_item = {k: v for k, v in item.items() if k != 'source_file'}
            filtered_data.append(filtered_item)
    
    # 결과 저장
    output_path = Path(output_dir) / f'training_dataset_over_{min_word_length}_words.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # 필터링 결과 출력
    original_count = len(data)
    filtered_count = len(filtered_data)
    filtered_ratio = filtered_count / original_count * 100
    removed_count = original_count - filtered_count
    removed_ratio = removed_count / original_count * 100
    
    print(f"\n✅ 필터링 완료!")
    print(f"  원본 데이터: {original_count:,}개")
    print(f"  필터링된 데이터: {filtered_count:,}개 ({filtered_ratio:.1f}%) - {min_word_length} 단어 초과")
    print(f"  제거된 데이터: {removed_count:,}개 ({removed_ratio:.1f}%) - {min_word_length} 단어 이하")
    print(f"  저장 경로: {output_path}")
    
    return filtered_data, output_path

def get_recommended_word_lengths(word_lengths: List[int]) -> List[int]:
    """
    데이터 분포를 기반으로 권장 단어 길이 임계값들을 제안합니다.
    (긴 데이터 필터링을 위한 최소값 기준)
    
    Args:
        word_lengths (List[int]): 단어 길이 리스트
        
    Returns:
        List[int]: 권장 최소 단어 길이 임계값들
    """
    sorted_lengths = sorted(word_lengths)
    
    # 상위 데이터를 위한 percentile 지점들 (낮은 percentile = 더 많은 긴 데이터)
    percentiles = [20, 25, 30, 50, 75]
    recommendations = []
    
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100) - 1
        if 0 <= idx < len(sorted_lengths):
            recommendations.append(sorted_lengths[idx])
    
    # 중복 제거 및 정렬
    recommendations = sorted(list(set(recommendations)))
    
    return recommendations

def main():
    """메인 함수"""
    # 설정
    DATA_DIR = "4_tow_generation/old/tow_data"
    OUTPUT_DIR = "4_tow_generation/processed"
    
    # 출력 디렉토리 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("🚀 TOW 데이터셋 분석 시작")
    print(f"  데이터 디렉토리: {DATA_DIR}")
    print(f"  출력 디렉토리: {OUTPUT_DIR}")
    
    # 1. 데이터 로딩
    data = load_json_files(DATA_DIR)
    if not data:
        print("❌ 데이터를 찾을 수 없습니다.")
        return
    
    # 2. 단어 길이 분석
    word_lengths, stats = analyze_token_lengths(data)
    
    # 3. 통계 출력
    print_statistics(word_lengths, stats)
    
    # 4. 분포 시각화
    plot_token_distribution(word_lengths, stats, OUTPUT_DIR)
    
    # 5. 권장 임계값 제안
    recommended_lengths = get_recommended_word_lengths(word_lengths)
    print(f"\n💡 권장 최소 단어 길이 임계값들 (이 값을 초과하는 긴 데이터만 저장):")
    for length in recommended_lengths:
        count_above = len([x for x in word_lengths if x > length])
        percentage = count_above / len(word_lengths) * 100
        print(f"  {length} 단어 초과: 전체의 {percentage:.1f}% 데이터 유지 ({count_above:,}개)")
    
    # 6. 사용자 입력으로 필터링할 단어 길이 설정
    print(f"\n🎯 필터링할 최소 단어 길이를 선택하세요 (이 값을 초과하는 데이터만 저장됩니다):")
    print(f"  (권장: {recommended_lengths})")
    
    while True:
        try:
            user_input = input("최소 단어 길이 입력 (예: 20): ").strip()
            if not user_input:
                print("값을 입력해주세요.")
                continue
                
            min_word_length = int(user_input)
            
            if min_word_length < 0:
                print("0 이상의 값을 입력해주세요.")
                continue
            
            # 예상 결과 미리보기
            count_above = len([x for x in word_lengths if x > min_word_length])
            percentage = count_above / len(word_lengths) * 100
            print(f"\n📋 예상 결과: {min_word_length} 단어 초과 데이터 {count_above:,}개 ({percentage:.1f}%) 저장 예정")
            
            confirm = input("계속하시겠습니까? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', '네', 'ㅇ']:
                break
            elif confirm in ['n', 'no', '아니요', 'ㄴ']:
                continue
            else:
                print("y(예) 또는 n(아니요)을 입력해주세요.")
            
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            return
    
    # 7. 데이터 필터링 및 저장
    filtered_data, output_path = filter_and_save_data(data, min_word_length, OUTPUT_DIR)
    
    print(f"\n🎉 분석 완료! 결과 파일들:")
    print(f"  📈 분포 차트: {OUTPUT_DIR}/word_distribution_analysis.png")
    print(f"  📄 필터링된 데이터셋: {output_path}")

if __name__ == "__main__":
    # 필요한 라이브러리 import 확인
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from tqdm import tqdm
    except ImportError as e:
        print(f"❌ 필요한 라이브러리가 설치되어 있지 않습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install matplotlib seaborn pandas tqdm")
        exit(1)
    
    main()