import json
import os
from pathlib import Path

def find_eng_correct_kr_incorrect():
    """
    ARC에서는 정답이지만 Ko-ARC에서는 틀린 ID들을 찾는 함수
    """
    basemodel_results_dir = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/3_evaluation/ARC/basemodel_results")

    # 모델별로 결과 저장
    for json_file in basemodel_results_dir.glob("*.json"):
        print(f"Processing {json_file.name}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ARC와 Ko-ARC 데이터 추출
        arc_data = data.get('datasets', {}).get('ARC', {}).get('details', [])
        ko_arc_data = data.get('datasets', {}).get('Ko-ARC', {}).get('details', [])

        # ID를 키로 하는 딕셔너리 생성
        arc_results = {item['id']: item['is_correct'] for item in arc_data}
        ko_arc_results = {item['id']: item['is_correct'] for item in ko_arc_data}

        # ARC에서는 정답이지만 Ko-ARC에서는 틀린 ID 찾기
        eng_correct_kr_incorrect_ids = []

        for item_id in arc_results:
            if item_id in ko_arc_results:
                arc_correct = arc_results[item_id]
                ko_arc_correct = ko_arc_results[item_id]

                if arc_correct and not ko_arc_correct:
                    eng_correct_kr_incorrect_ids.append(item_id)

        # 모델 이름 추출 (gemma, llama, qwen 등)
        model_name = json_file.stem.replace('results_', '').replace('_3shot', '')

        # 결과 JSON 구조 생성
        result_data = {
            "model_name": model_name,
            "analysis_type": "english_correct_korean_incorrect",
            "total_count": len(eng_correct_kr_incorrect_ids),
            "description": "IDs where ARC is correct but Ko-ARC is incorrect",
            "ids": eng_correct_kr_incorrect_ids
        }

        # 모델별 JSON 파일로 저장
        output_file = f"step1_results/eng_correct-kr_incorrect_{model_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"  {model_name}: {len(eng_correct_kr_incorrect_ids)} items saved to {output_file}")

    print("\nAll results saved successfully!")

if __name__ == "__main__":
    find_eng_correct_kr_incorrect()