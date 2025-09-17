#!/usr/bin/env python3
"""
ARC vs KO-ARC 결과 비교 스크립트

이 스크립트는 ARC와 KO-ARC 결과를 비교하여:
1. ARC에서는 맞췄지만 KO-ARC에서는 틀린 문제들을 찾아 모델_gap.json에 저장
2. KO-ARC에서는 맞았지만 ARC에서는 틀린 문제들을 찾아 모델_Rgap.json에 저장
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_results(file_path: str) -> Dict[str, Any]:
    """결과 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_first_bracket_answer(raw_output: str) -> str:
    """raw output에서 첫 번째 {A}, {B}, {C}, {D} 형태의 답을 추출합니다."""
    if not raw_output:
        return ""

    pattern = r'\{([A-D])\}'
    matches = re.findall(pattern, raw_output)
    if matches:
        return matches[0]
    return ""


def extract_dataset_results(datasets: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    datasets에서 ARC와 KO-ARC 결과를 분리합니다.

    Returns:
        arc_results: {id: result_dict}
        ko_arc_results: {id: result_dict}
    """
    arc_results = {}
    ko_arc_results = {}

    # ARC 데이터 처리
    if 'ARC' in datasets and 'details' in datasets['ARC']:
        for result in datasets['ARC']['details']:
            item_id = result['id']
            # 필드명을 통일하기 위해 변환
            converted_result = {
                'id': item_id,
                'dataset': 'ARC',
                'ground_truth': result['ground_truth'],
                'raw_output': result['model_raw_output'],
                'new_extracted_answer': result['predicted_answer'],
                'new_correct': result['is_correct']
            }
            arc_results[item_id] = converted_result

    # Ko-ARC 데이터 처리
    if 'Ko-ARC' in datasets and 'details' in datasets['Ko-ARC']:
        for result in datasets['Ko-ARC']['details']:
            item_id = result['id']
            # 필드명을 통일하기 위해 변환
            converted_result = {
                'id': item_id,
                'dataset': 'Ko-ARC',
                'ground_truth': result['ground_truth'],
                'raw_output': result['model_raw_output'],
                'new_extracted_answer': result['predicted_answer'],
                'new_correct': result['is_correct']
            }
            ko_arc_results[item_id] = converted_result

    return arc_results, ko_arc_results


def extract_dataset_results_legacy(detailed_results: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    이전 구조의 detailed_results에서 ARC와 KO-ARC 결과를 분리합니다. (하위 호환성)
    """
    arc_results = {}
    ko_arc_results = {}

    for result in detailed_results:
        dataset = result['dataset']
        item_id = result['id']

        if dataset == 'ARC':
            arc_results[item_id] = result
        elif dataset == 'Ko-ARC':
            ko_arc_results[item_id] = result

    return arc_results, ko_arc_results


def find_performance_gaps(arc_results: Dict[str, Dict], ko_arc_results: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    성능 차이를 분석합니다.

    Returns:
        gap_items: ARC에서는 맞췄지만 KO-ARC에서는 틀린 문제들
        reverse_gap_items: KO-ARC에서는 맞았지만 ARC에서는 틀린 문제들
        correct_items: 둘 다 맞춘 문제들
    """
    gap_items = []
    reverse_gap_items = []
    correct_items = []

    # 공통된 ID들만 비교
    common_ids = set(arc_results.keys()) & set(ko_arc_results.keys())

    for item_id in common_ids:
        arc_result = arc_results[item_id]
        ko_arc_result = ko_arc_results[item_id]

        # raw output에서 답 추출
        arc_raw_answer = extract_first_bracket_answer(arc_result.get('raw_output', ''))
        ko_arc_raw_answer = extract_first_bracket_answer(ko_arc_result.get('raw_output', ''))

        # 추출된 답과 ground truth 비교
        ground_truth = arc_result['ground_truth']
        arc_raw_correct = arc_raw_answer == ground_truth
        ko_arc_raw_correct = ko_arc_raw_answer == ground_truth

        # 기존 추출된 답도 비교
        arc_extracted_correct = arc_result['new_correct']
        ko_arc_extracted_correct = ko_arc_result['new_correct']

        # ARC는 맞췄지만 KO-ARC는 틀린 경우 (raw output 기준)
        if arc_raw_correct and not ko_arc_raw_correct:
            gap_items.append({
                'id': item_id,
                'ground_truth': ground_truth,
                'arc_raw_answer': arc_raw_answer,
                'ko_arc_raw_answer': ko_arc_raw_answer,
                'arc_raw_output': arc_result.get('raw_output', ''),
                'ko_arc_raw_output': ko_arc_result.get('raw_output', ''),
            })

        # KO-ARC는 맞췄지만 ARC는 틀린 경우 (raw output 기준)
        elif ko_arc_raw_correct and not arc_raw_correct:
            reverse_gap_items.append({
                'id': item_id,
                'ground_truth': ground_truth,
                'arc_raw_answer': arc_raw_answer,
                'ko_arc_raw_answer': ko_arc_raw_answer,
                'arc_raw_output': arc_result.get('raw_output', ''),
                'ko_arc_raw_output': ko_arc_result.get('raw_output', ''),
            })

        # 둘 다 맞춘 경우 (raw output 기준)
        elif arc_raw_correct and ko_arc_raw_correct:
            correct_items.append({
                'id': item_id,
                'ground_truth': ground_truth,
                'arc_raw_answer': arc_raw_answer,
                'ko_arc_raw_answer': ko_arc_raw_answer,
                'arc_raw_output': arc_result.get('raw_output', ''),
                'ko_arc_raw_output': ko_arc_result.get('raw_output', ''),
            })

    return gap_items, reverse_gap_items, correct_items


def save_gap_analysis(model_name: str, gap_items: List[Dict], reverse_gap_items: List[Dict], correct_items: List[Dict], output_dir: str):
    """gap 분석 결과를 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 통계 계산
    common_count = len(gap_items) + len(reverse_gap_items) + len(correct_items)

    # 통계 계산
    both_raw_correct = len(correct_items)

    # ARC에서는 맞췄지만 KO-ARC에서는 틀린 경우
    gap_file = output_path / f"{model_name}_gap.json"
    gap_data = {
        'model': model_name,
        'description': 'ARC에서는 맞췄지만 KO-ARC에서는 틀린 문제들 (raw output 기준)',
        'count': len(gap_items),
        'items': gap_items
    }

    with open(gap_file, 'w', encoding='utf-8') as f:
        json.dump(gap_data, f, ensure_ascii=False, indent=2)

    # KO-ARC에서는 맞았지만 ARC에서는 틀린 경우
    reverse_gap_file = output_path / f"{model_name}_Rgap.json"
    reverse_gap_data = {
        'model': model_name,
        'description': 'KO-ARC에서는 맞았지만 ARC에서는 틀린 문제들 (raw output 기준)',
        'count': len(reverse_gap_items),
        'items': reverse_gap_items
    }

    with open(reverse_gap_file, 'w', encoding='utf-8') as f:
        json.dump(reverse_gap_data, f, ensure_ascii=False, indent=2)

    # 둘 다 맞춘 경우
    correct_file = output_path / f"{model_name}_correct.json"
    correct_data = {
        'model': model_name,
        'description': '둘 다 맞춘 문제들 (raw output 기준)',
        'count': len(correct_items),
        'statistics': {
            'both_raw_correct': both_raw_correct,
            'common_problems_total': common_count
        },
        'items': correct_items
    }

    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=2)

    print(f"Model: {model_name}")
    print(f"   Gap (ARC correct, KO-ARC wrong): {len(gap_items)} items")
    print(f"   Reverse Gap (KO-ARC correct, ARC wrong): {len(reverse_gap_items)} items")
    print(f"   Both Correct (raw): {both_raw_correct} items")
    print(f"   Saved to: {gap_file}, {reverse_gap_file}, {correct_file}")


def extract_model_name(filename: str) -> str:
    """파일명에서 모델명을 추출합니다."""
    # results_모델명_3shot_re_extracted.json 패턴에서 모델명 추출
    if filename.startswith('results_'):
        parts = filename.replace('results_', '').replace('_3shot_re_extracted.json', '').replace('_3shot.json', '')
        return parts
    return filename


def process_single_file(json_file: Path, output_dir: Path):
    """단일 JSON 파일을 처리합니다."""
    try:
        # 결과 로드
        results = load_results(json_file)

        # 새로운 구조와 이전 구조 모두 지원
        if 'datasets' in results:
            # 새로운 구조: datasets.ARC.details, datasets.Ko-ARC.details
            arc_results, ko_arc_results = extract_dataset_results(results['datasets'])
        elif 'detailed_results' in results:
            # 이전 구조: detailed_results (하위 호환성)
            arc_results, ko_arc_results = extract_dataset_results_legacy(results['detailed_results'])
        else:
            print(f"Error: Unknown JSON structure in {json_file.name}")
            return

        # ARC와 Ko-ARC 데이터가 모두 있는지 확인
        if not arc_results:
            print(f"Warning: No ARC data found in {json_file.name}")
        if not ko_arc_results:
            print(f"Warning: No Ko-ARC data found in {json_file.name}")

        if not arc_results or not ko_arc_results:
            print(f"Skipping {json_file.name} - incomplete dataset")
            return

        # 성능 차이 분석
        gap_items, reverse_gap_items, correct_items = find_performance_gaps(arc_results, ko_arc_results)

        # 모델명 추출
        model_name = extract_model_name(json_file.name)

        # 결과 저장
        save_gap_analysis(model_name, gap_items, reverse_gap_items, correct_items, output_dir)

    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 실행 함수"""
    import sys

    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")

    # 명령행 인수로 JSON 파일을 받거나, 현재 디렉토리에서 찾기
    if len(sys.argv) > 1:
        # 명령행에서 특정 파일들이 지정된 경우
        json_files = []
        for arg in sys.argv[1:]:
            file_path = Path(arg)
            if file_path.exists() and file_path.suffix == '.json':
                json_files.append(file_path)
            else:
                print(f"Warning: File not found or not a JSON file: {arg}")

        if json_files:
            print(f"Processing {len(json_files)} specified JSON files...")
            for json_file in json_files:
                print(f"\nProcessing file: {json_file}")
                process_single_file(json_file, json_file.parent / "gap_analysis")
        else:
            print("No valid JSON files provided.")
    else:
        # 현재 디렉토리에서 re_extracted.json 파일들 찾기
        json_files = list(current_dir.glob("*re_extracted.json"))

        if json_files:
            print(f"Found {len(json_files)} re_extracted.json files in current directory")
            output_dir = current_dir / "gap_analysis"
            output_dir.mkdir(exist_ok=True)

            for json_file in json_files:
                print(f"\nProcessing file: {json_file.name}")
                process_single_file(json_file, output_dir)
        else:
            print("No re_extracted.json files found in current directory.")
            print("Usage: python compare_arc_ko_arc.py [file1.json] [file2.json] ...")
            print("Or place re_extracted.json files in the current directory.")


if __name__ == "__main__":
    main()