#!/usr/bin/env python3
"""
KLUE 데이터셋 다운로드 스크립트
HuggingFace에서 KLUE 벤치마크의 모든 태스크를 다운로드하여 로컬에 저장
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def download_klue_datasets():
    """KLUE 데이터셋의 모든 태스크를 다운로드하여 저장"""

    # 저장할 디렉토리 설정
    base_dir = Path(__file__).parent / "KLUE"
    base_dir.mkdir(exist_ok=True)

    # KLUE의 8개 태스크
    klue_tasks = [
        "ynat",  # Topic Classification (TC)
        "sts",   # Semantic Textual Similarity
        "nli",   # Natural Language Inference
        "ner",   # Named Entity Recognition
        "re",    # Relation Extraction
        "dp",    # Dependency Parsing
        "mrc",   # Machine Reading Comprehension
        "wos"    # Dialogue State Tracking (DST)
    ]

    # 각 태스크별로 데이터 다운로드
    for task in klue_tasks:
        print(f"\n>> Downloading: {task}")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("klue/klue", task)

            # Create task directory
            task_dir = base_dir / task
            task_dir.mkdir(exist_ok=True)

            # Save each split (train, validation, test)
            for split_name, split_data in dataset.items():
                print(f"  >> Saving: {task}/{split_name} ({len(split_data)} samples)")

                # JSON 형태로 저장
                json_path = task_dir / f"{split_name}.json"
                split_data.to_json(json_path, orient="records", force_ascii=False, indent=2)

                # 추가로 CSV도 저장 (분석용)
                csv_path = task_dir / f"{split_name}.csv"
                split_data.to_pandas().to_csv(csv_path, index=False, encoding='utf-8')

                # 데이터 구조 정보 저장
                info_path = task_dir / f"{split_name}_info.json"
                info = {
                    "task": task,
                    "split": split_name,
                    "num_samples": len(split_data),
                    "features": list(split_data.features.keys()),
                    "feature_types": {k: str(v) for k, v in split_data.features.items()}
                }

                # 첫 번째 샘플 예시도 포함
                if len(split_data) > 0:
                    info["sample_example"] = dict(split_data[0])

                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

            print(f"  >> {task} completed!")

        except Exception as e:
            print(f"  >> {task} error: {str(e)}")
            continue

    # 전체 요약 정보 생성
    summary_path = base_dir / "KLUE_summary.json"
    summary = {
        "dataset_name": "KLUE",
        "source": "https://huggingface.co/datasets/klue/klue",
        "description": "Korean Language Understanding Evaluation Benchmark",
        "tasks": {},
        "download_timestamp": pd.Timestamp.now().isoformat()
    }

    # 각 태스크 정보 수집
    for task in klue_tasks:
        task_dir = base_dir / task
        if task_dir.exists():
            task_info = {"splits": {}}

            for info_file in task_dir.glob("*_info.json"):
                with open(info_file, 'r', encoding='utf-8') as f:
                    split_info = json.load(f)
                    split_name = split_info["split"]
                    task_info["splits"][split_name] = {
                        "num_samples": split_info["num_samples"],
                        "features": split_info["features"]
                    }

            summary["tasks"][task] = task_info

    # 요약 저장
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n>> Summary saved: {summary_path}")
    print("\n>> KLUE dataset download completed!")

    # Print downloaded files
    print("\n>> Downloaded files:")
    for task in klue_tasks:
        task_dir = base_dir / task
        if task_dir.exists():
            print(f"  >> {task}/")
            for file in sorted(task_dir.iterdir()):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"    >> {file.name} ({size_mb:.1f}MB)")

def inspect_klue_data():
    """다운로드된 KLUE 데이터의 구조를 분석"""

    base_dir = Path(__file__).parent / "KLUE"
    summary_path = base_dir / "KLUE_summary.json"

    if not summary_path.exists():
        print(">> KLUE data not downloaded. Please run download_klue_datasets() first.")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    print(">> KLUE Dataset Analysis")
    print("=" * 50)

    for task_name, task_info in summary["tasks"].items():
        print(f"\n>> {task_name.upper()}")
        for split_name, split_info in task_info["splits"].items():
            print(f"  >> {split_name}: {split_info['num_samples']:,} samples")
            print(f"     Fields: {', '.join(split_info['features'])}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KLUE dataset download and analysis")
    parser.add_argument("--download", action="store_true", help="Download KLUE datasets")
    parser.add_argument("--inspect", action="store_true", help="Analyze downloaded data")

    args = parser.parse_args()

    if args.download:
        download_klue_datasets()
    elif args.inspect:
        inspect_klue_data()
    else:
        print("Usage:")
        print("  python download_klue_datasets.py --download  # Download datasets")
        print("  python download_klue_datasets.py --inspect   # Analyze data")

        # Default action: download
        print("\nDefault action: Download KLUE datasets")
        download_klue_datasets()