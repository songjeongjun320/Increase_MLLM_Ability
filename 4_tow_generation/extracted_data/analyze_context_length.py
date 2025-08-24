import json
import os
from collections import defaultdict
import argparse

def count_tokens_simple(text):
    """간단한 토큰 카운팅 (공백 기준)"""
    return len(text.split()) if text else 0

def count_characters(text):
    """문자 수 카운팅"""
    return len(text) if text else 0

def analyze_context_lengths(json_file, min_char_threshold=10, min_token_threshold=3):
    """JSON 파일에서 context 길이를 분석하고 통계를 제공"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n=== {os.path.basename(json_file)} 분석 결과 ===")
    print(f"총 데이터 개수: {len(data)}")
    
    # 통계 수집
    char_counts = []
    token_counts = []
    context_lengths = defaultdict(int)
    
    for item in data:
        context = item.get('context', '')
        char_count = count_characters(context)
        token_count = count_tokens_simple(context)
        
        char_counts.append(char_count)
        token_counts.append(token_count)
        context_lengths[char_count] += 1
    
    # 기본 통계
    print(f"\n--- 문자 수 기준 통계 ---")
    print(f"최소: {min(char_counts)} 문자")
    print(f"최대: {max(char_counts)} 문자")
    print(f"평균: {sum(char_counts)/len(char_counts):.1f} 문자")
    print(f"중간값: {sorted(char_counts)[len(char_counts)//2]} 문자")
    
    print(f"\n--- 토큰 수 기준 통계 ---")
    print(f"최소: {min(token_counts)} 토큰")
    print(f"최대: {max(token_counts)} 토큰")
    print(f"평균: {sum(token_counts)/len(token_counts):.1f} 토큰")
    print(f"중간값: {sorted(token_counts)[len(token_counts)//2]} 토큰")
    
    # 필터링 기준별 분석
    print(f"\n--- 필터링 기준별 분석 ---")
    
    # 문자 수 기준
    removed_by_char = sum(1 for count in char_counts if count < min_char_threshold)
    remaining_by_char = len(data) - removed_by_char
    
    print(f"문자 수 {min_char_threshold}자 미만:")
    print(f"  - 제거될 데이터: {removed_by_char}개 ({removed_by_char/len(data)*100:.1f}%)")
    print(f"  - 남을 데이터: {remaining_by_char}개 ({remaining_by_char/len(data)*100:.1f}%)")
    
    # 토큰 수 기준
    removed_by_token = sum(1 for count in token_counts if count < min_token_threshold)
    remaining_by_token = len(data) - removed_by_token
    
    print(f"토큰 수 {min_token_threshold}개 미만:")
    print(f"  - 제거될 데이터: {removed_by_token}개 ({removed_by_token/len(data)*100:.1f}%)")
    print(f"  - 남을 데이터: {remaining_by_token}개 ({remaining_by_token/len(data)*100:.1f}%)")
    
    # 구간별 분포
    print(f"\n--- 문자 수 구간별 분포 ---")
    ranges = [(0, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, float('inf'))]
    for start, end in ranges:
        count = sum(1 for c in char_counts if start <= c < end or (end == float('inf') and c >= start))
        percentage = count / len(data) * 100
        range_str = f"{start}-{end if end != float('inf') else '+'}"
        print(f"  {range_str:>8}자: {count:>4}개 ({percentage:>5.1f}%)")
    
    print(f"\n--- 토큰 수 구간별 분포 ---")
    token_ranges = [(0, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, float('inf'))]
    for start, end in token_ranges:
        count = sum(1 for c in token_counts if start <= c < end or (end == float('inf') and c >= start))
        percentage = count / len(data) * 100
        range_str = f"{start}-{end if end != float('inf') else '+'}"
        print(f"  {range_str:>8}토큰: {count:>4}개 ({percentage:>5.1f}%)")
    
    return {
        'total': len(data),
        'char_stats': {
            'min': min(char_counts),
            'max': max(char_counts),
            'avg': sum(char_counts)/len(char_counts),
            'median': sorted(char_counts)[len(char_counts)//2]
        },
        'token_stats': {
            'min': min(token_counts),
            'max': max(token_counts),
            'avg': sum(token_counts)/len(token_counts),
            'median': sorted(token_counts)[len(token_counts)//2]
        },
        'filtering': {
            'char_threshold': min_char_threshold,
            'token_threshold': min_token_threshold,
            'removed_by_char': removed_by_char,
            'remaining_by_char': remaining_by_char,
            'removed_by_token': removed_by_token,
            'remaining_by_token': remaining_by_token
        }
    }

def main():
    parser = argparse.ArgumentParser(description='JSON 파일의 context 길이 분석')
    parser.add_argument('--dir', default='tow_data', help='JSON 파일들이 있는 디렉토리 (기본값: tow_data)')
    parser.add_argument('--char-threshold', type=int, default=10, help='최소 문자 수 임계값 (기본값: 10)')
    parser.add_argument('--token-threshold', type=int, default=3, help='최소 토큰 수 임계값 (기본값: 3)')
    parser.add_argument('--file', help='특정 파일만 분석 (선택사항)')
    
    args = parser.parse_args()
    
    if args.file:
        # 특정 파일만 분석
        json_files = [args.file]
    else:
        # 디렉토리의 모든 JSON 파일 분석
        json_files = [f for f in os.listdir(args.dir) if f.endswith('.json')]
        json_files = [os.path.join(args.dir, f) for f in json_files]
    
    if not json_files:
        print("JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"문자 수 임계값: {args.char_threshold}자")
    print(f"토큰 수 임계값: {args.token_threshold}개")
    print("="*60)
    
    all_results = {}
    
    for json_file in json_files:
        try:
            result = analyze_context_lengths(json_file, args.char_threshold, args.token_threshold)
            all_results[json_file] = result
        except Exception as e:
            print(f"오류 발생 ({json_file}): {e}")
    
    # 전체 요약
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("=== 전체 요약 ===")
        total_data = sum(r['total'] for r in all_results.values())
        total_removed_char = sum(r['filtering']['removed_by_char'] for r in all_results.values())
        total_removed_token = sum(r['filtering']['removed_by_token'] for r in all_results.values())
        
        print(f"총 파일 수: {len(all_results)}개")
        print(f"총 데이터 개수: {total_data:,}개")
        print(f"\n문자 수 {args.char_threshold}자 미만 제거시:")
        print(f"  - 제거될 데이터: {total_removed_char:,}개 ({total_removed_char/total_data*100:.1f}%)")
        print(f"  - 남을 데이터: {total_data-total_removed_char:,}개 ({(total_data-total_removed_char)/total_data*100:.1f}%)")
        print(f"\n토큰 수 {args.token_threshold}개 미만 제거시:")
        print(f"  - 제거될 데이터: {total_removed_token:,}개 ({total_removed_token/total_data*100:.1f}%)")
        print(f"  - 남을 데이터: {total_data-total_removed_token:,}개 ({(total_data-total_removed_token)/total_data*100:.1f}%)")

if __name__ == "__main__":
    main()