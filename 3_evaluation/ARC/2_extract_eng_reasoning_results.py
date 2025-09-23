import json
import os
from pathlib import Path

def extract_eng_reasoning_results():
    """
    basemodel_eng_reasoning 폴더에서 Ko-ARC 결과를 분석하여
    is_correct가 true인 것과 false인 것의 ID들을 추출하는 함수
    """
    basemodel_eng_reasoning_dir = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/3_evaluation/ARC/basemodel_eng_reasoning")

    # 모델별로 결과 저장
    for json_file in basemodel_eng_reasoning_dir.glob("*.json"):
        print(f"Processing {json_file.name}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ko-ARC 데이터 추출
        ko_arc_data = data.get('datasets', {}).get('Ko-ARC', {}).get('details', [])

        # is_correct가 true인 것과 false인 것 분리
        correct_ids = []
        incorrect_ids = []

        for item in ko_arc_data:
            item_id = item['id']
            is_correct = item['is_correct']

            if is_correct:
                correct_ids.append(item_id)
            else:
                incorrect_ids.append(item_id)

        # 모델 이름 추출 (gemma, llama, qwen 등)
        model_name = json_file.stem.replace('results_', '').replace('_3shot', '')

        # is_correct가 true인 것들 저장
        correct_result_data = {
            "model_name": model_name,
            "analysis_type": "korean_input_english_reasoning_correct",
            "total_count": len(correct_ids),
            "description": "Korean input IDs that were solved correctly with English reasoning",
            "ids": correct_ids
        }

        # is_correct가 false인 것들 저장
        incorrect_result_data = {
            "model_name": model_name,
            "analysis_type": "korean_input_english_reasoning_incorrect",
            "total_count": len(incorrect_ids),
            "description": "Korean input IDs that could not be solved even with English reasoning",
            "ids": incorrect_ids
        }

        # 모델별 JSON 파일로 저장
        correct_output_file = f"step2_results/kr_input_eng_reasoning_correct_{model_name}.json"
        incorrect_output_file = f"step2_results/kr_input_eng_reasoning_incorrect_{model_name}.json"

        with open(correct_output_file, 'w', encoding='utf-8') as f:
            json.dump(correct_result_data, f, ensure_ascii=False, indent=2)

        with open(incorrect_output_file, 'w', encoding='utf-8') as f:
            json.dump(incorrect_result_data, f, ensure_ascii=False, indent=2)

        print(f"  {model_name}:")
        print(f"    Correct (solved with English reasoning): {len(correct_ids)} items saved to {correct_output_file}")
        print(f"    Incorrect (still unsolved with English reasoning): {len(incorrect_ids)} items saved to {incorrect_output_file}")

    print("\nAll results saved successfully!")

if __name__ == "__main__":
    extract_eng_reasoning_results()