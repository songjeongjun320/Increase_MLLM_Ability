#!/usr/bin/env python3
"""
KLUE 데이터셋 구조 파악 및 다운로드 테스트
"""

from datasets import load_dataset, get_dataset_config_names
import json

def test_klue_access():
    """KLUE 데이터셋 접근 방법 테스트"""

    print("Testing KLUE dataset access methods...")

    # 방법 1: klue/klue 형태로 접근
    try:
        print("\n1. Trying 'klue/klue'...")
        configs = get_dataset_config_names("klue/klue")
        print(f"Available configs: {configs}")

        # 각 config별로 테스트
        for config in configs[:3]:  # 처음 3개만 테스트
            try:
                print(f"  Loading config: {config}")
                dataset = load_dataset("klue/klue", config)
                print(f"    Splits: {list(dataset.keys())}")
                if dataset:
                    first_split = list(dataset.keys())[0]
                    sample = dataset[first_split][0]
                    print(f"    Sample fields: {list(sample.keys())}")
                    break
            except Exception as e:
                print(f"    Error with {config}: {str(e)}")

    except Exception as e:
        print(f"Error with klue/klue: {str(e)}")

    # 방법 2: klue만 사용
    try:
        print("\n2. Trying 'klue'...")
        configs = get_dataset_config_names("klue")
        print(f"Available configs: {configs}")

    except Exception as e:
        print(f"Error with klue: {str(e)}")

    # 방법 3: 개별 태스크 직접 접근 시도
    print("\n3. Trying individual task access...")
    task_names = ["ynat", "sts", "nli", "ner", "re", "dp", "mrc", "wos"]

    for task in task_names:
        try:
            print(f"  Trying klue/{task}...")
            dataset = load_dataset("klue", task)
            print(f"    {task} - Splits: {list(dataset.keys())}")
            break
        except Exception as e:
            print(f"    {task} failed: {str(e)}")

def download_klue_samples():
    """KLUE 샘플 데이터 다운로드"""

    try:
        # klue/klue 방식으로 시도
        print("Downloading KLUE samples...")
        configs = get_dataset_config_names("klue/klue")

        for config in configs:
            try:
                print(f"\nDownloading {config}...")
                dataset = load_dataset("klue/klue", config)

                # 첫 번째 split의 첫 번째 샘플 저장
                if dataset:
                    first_split = list(dataset.keys())[0]
                    sample = dataset[first_split][0]

                    # 샘플 저장
                    with open(f"sample_{config}.json", "w", encoding="utf-8") as f:
                        json.dump(dict(sample), f, ensure_ascii=False, indent=2)

                    print(f"  Sample saved: sample_{config}.json")
                    print(f"  Fields: {list(sample.keys())}")

            except Exception as e:
                print(f"  Error with {config}: {str(e)}")

    except Exception as e:
        print(f"Overall error: {str(e)}")

if __name__ == "__main__":
    test_klue_access()
    print("\n" + "="*50)
    download_klue_samples()