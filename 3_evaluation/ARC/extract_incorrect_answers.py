"""
Ko-ARC 데이터셋에서 틀린 답변(is_correct=False) 추출 스크립트

사용법:
1. 아래 설정 변수들을 원하는 값으로 수정
2. python extract_incorrect_answers.py 실행

설정 변수:
- INPUT_PATH: 입력 파일 또는 디렉토리 경로
- OUTPUT_PATH: 출력 디렉토리 경로
- PROCESS_ALL_FILES: True면 디렉토리의 모든 파일 처리, False면 단일 파일 처리
"""

import json
import argparse
import os
from pathlib import Path


def extract_incorrect_answers(input_file, output_dir):
    """
    Extract entries where is_correct is False from evaluation results.
    Save results by model name.

    Args:
        input_file (str): Path to input JSON file with evaluation results
        output_dir (str): Directory to save output files
    """
    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract model name from model_config or filename
    model_name = "unknown_model"
    if 'model_config' in data and 'name' in data['model_config']:
        model_name = data['model_config']['name']
    else:
        # Extract from filename if model_config not available
        filename = Path(input_file).stem
        if 'results_' in filename:
            model_name = filename.replace('results_', '').replace('_3shot', '')

    incorrect_entries = []

    # Check if datasets exist and process each dataset
    if 'datasets' in data:
        for dataset_name, dataset_info in data['datasets'].items():
            # Only process Ko-ARC dataset
            if dataset_name == 'Ko-ARC':
                print(f"Processing {dataset_name} dataset...")

                if 'details' in dataset_info:
                    for entry in dataset_info['details']:
                        # Check if is_correct is False
                        if entry.get('is_correct') == False:
                            incorrect_entry = {
                                'dataset': dataset_name,
                                'index': entry.get('index'),
                                'id': entry.get('id'),
                                'ground_truth': entry.get('ground_truth'),
                                'predicted_answer': entry.get('predicted_answer'),
                                'model_raw_output': entry.get('model_raw_output', ''),
                                'is_correct': entry.get('is_correct')
                            }
                            incorrect_entries.append(incorrect_entry)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save incorrect entries to output file
    output_file = os.path.join(output_dir, f"{model_name}_incorrect_answers.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_entries, f, indent=2, ensure_ascii=False)

    print(f"Model: {model_name}")
    print(f"Found {len(incorrect_entries)} incorrect answers")
    print(f"Results saved to: {output_file}")

    return len(incorrect_entries)


def process_all_files(input_dir, output_dir):
    """
    Process all result files in a directory.

    Args:
        input_dir (str): Directory containing result files
        output_dir (str): Directory to save output files
    """
    input_path = Path(input_dir)
    total_incorrect = 0
    processed_files = 0

    # Find all JSON files that look like result files
    result_files = []
    for json_file in input_path.glob("**/*.json"):
        if "result" in json_file.name.lower():
            result_files.append(json_file)

    if not result_files:
        print("No result files found!")
        return

    print(f"Found {len(result_files)} result files to process:")
    for file in result_files:
        print(f"  - {file}")

    print("\nProcessing files...")
    for json_file in result_files:
        try:
            print(f"\n--- Processing: {json_file.name} ---")
            incorrect_count = extract_incorrect_answers(str(json_file), output_dir)
            total_incorrect += incorrect_count
            processed_files += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\n=== Summary ===")
    print(f"Processed {processed_files} files")
    print(f"Total incorrect answers found: {total_incorrect}")


# ===== 설정 변수 =====
# 여기서 input과 output 경로를 직접 설정할 수 있습니다
INPUT_PATH = "./basemodels"  # 입력 파일 또는 디렉토리 경로
OUTPUT_PATH = "./ko_arc_incorrect_answers"  # 출력 디렉토리 경로
PROCESS_ALL_FILES = True  # True: 디렉토리의 모든 파일 처리, False: 단일 파일 처리

def main():
    parser = argparse.ArgumentParser(
        description="Extract incorrect answers from evaluation results, separated by model"
    )
    parser.add_argument(
        '--input',
        default=INPUT_PATH,
        help=f'Path to input JSON file or directory with result files (default: {INPUT_PATH})'
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_PATH,
        help=f'Directory to save output files (default: {OUTPUT_PATH})'
    )
    parser.add_argument(
        '--process-all',
        action='store_true',
        default=PROCESS_ALL_FILES,
        help='Process all result files in the input directory'
    )

    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Process all files: {args.process_all}")
    print("-" * 50)

    input_path = Path(args.input)

    if args.process_all or input_path.is_dir():
        # Process all files in directory
        process_all_files(args.input, args.output)
    else:
        # Process single file
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' does not exist")
            return

        extract_incorrect_answers(args.input, args.output)


if __name__ == "__main__":
    main()