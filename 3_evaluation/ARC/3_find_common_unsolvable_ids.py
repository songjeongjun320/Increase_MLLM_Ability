import json
import os
from pathlib import Path

def find_common_unsolvable_ids():
    """
    eng_correct-kr_incorrect와 kr_input_eng_reasoning_incorrect에서
    공통되는 ID들을 찾아서 저장하는 함수
    """
    current_dir = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/3_evaluation/ARC")

    # 모델 리스트
    models = ["gemma-3-4b-pt", "llama-3.2-3b-pt", "qwem-2.5-3b-pt"]

    for model in models:
        print(f"Processing {model}...")

        # 파일 경로 설정
        eng_correct_kr_incorrect_file = current_dir / f"step1_results/eng_correct-kr_incorrect_{model}.json"
        kr_input_eng_reasoning_incorrect_file = current_dir / f"step2_results/kr_input_eng_reasoning_incorrect_{model}.json"

        # 파일 존재 여부 확인
        if not eng_correct_kr_incorrect_file.exists():
            print(f"  Warning: {eng_correct_kr_incorrect_file.name} not found")
            continue
        if not kr_input_eng_reasoning_incorrect_file.exists():
            print(f"  Warning: {kr_input_eng_reasoning_incorrect_file.name} not found")
            continue

        # JSON 파일들 읽기
        with open(eng_correct_kr_incorrect_file, 'r', encoding='utf-8') as f:
            eng_correct_kr_incorrect_data = json.load(f)

        with open(kr_input_eng_reasoning_incorrect_file, 'r', encoding='utf-8') as f:
            kr_input_eng_reasoning_incorrect_data = json.load(f)

        # ID 리스트 추출
        eng_correct_kr_incorrect_ids = set(eng_correct_kr_incorrect_data.get('ids', []))
        kr_input_eng_reasoning_incorrect_ids = set(kr_input_eng_reasoning_incorrect_data.get('ids', []))

        # 공통 ID 찾기
        common_ids = list(eng_correct_kr_incorrect_ids.intersection(kr_input_eng_reasoning_incorrect_ids))
        common_ids.sort()  # 정렬하여 일관성 유지

        # 결과 JSON 구조 생성
        result_data = {
            "model_name": model,
            "analysis_type": "korean_input_unsolvable_regardless_of_reasoning_language",
            "total_count": len(common_ids),
            "description": "영어로 문제를 받아서 영어 reasoning 할때는 해결 가능했지만, 한글로 질문을 받고, 영어로 reasoning 하려고 해도 여전히 틀린, 그러니까 한국 input일경우에는 어떠한 reasoning언어로도 해결이 안된 문제 모음",
            "source_analysis": {
                "eng_correct_kr_incorrect_count": len(eng_correct_kr_incorrect_ids),
                "kr_input_eng_reasoning_incorrect_count": len(kr_input_eng_reasoning_incorrect_ids),
                "common_count": len(common_ids)
            },
            "ids": common_ids
        }

        # 결과 파일로 저장
        output_file = current_dir / f"step3_results/kr_input_unsolvable_{model}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"  {model}:")
        print(f"    English correct but Korean incorrect: {len(eng_correct_kr_incorrect_ids)} items")
        print(f"    Korean input with English reasoning incorrect: {len(kr_input_eng_reasoning_incorrect_ids)} items")
        print(f"    Common (unsolvable with Korean input): {len(common_ids)} items")
        print(f"    Saved to: {output_file.name}")
        print()

    print("All results saved successfully!")

if __name__ == "__main__":
    find_common_unsolvable_ids()