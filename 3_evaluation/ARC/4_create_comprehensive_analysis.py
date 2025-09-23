import json
import os
from pathlib import Path

def create_comprehensive_analysis():
    """
    각 unsolvable ID에 대해 3가지 reasoning 방식의 raw output을 모두 수집하는 함수
    """
    current_dir = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/3_evaluation/ARC")
    basemodel_results_dir = current_dir / "basemodel_results"
    basemodel_eng_reasoning_dir = current_dir / "basemodel_eng_reasoning"

    # 모델 리스트
    models = ["gemma-3-4b-pt", "llama-3.2-3b-pt", "qwem-2.5-3b-pt"]

    for model in models:
        print(f"Processing {model}...")

        # unsolvable IDs 파일 읽기
        unsolvable_file = current_dir / f"step3_results/kr_input_unsolvable_{model}.json"
        if not unsolvable_file.exists():
            print(f"  Warning: {unsolvable_file.name} not found")
            continue

        with open(unsolvable_file, 'r', encoding='utf-8') as f:
            unsolvable_data = json.load(f)

        unsolvable_ids = set(unsolvable_data.get('ids', []))

        # 원본 basemodel_results 파일 읽기 (영어 입력 + 영어 추론, 한국어 입력 + 한국어 추론)
        basemodel_file = basemodel_results_dir / f"results_{model}_3shot.json"
        if not basemodel_file.exists():
            print(f"  Warning: {basemodel_file.name} not found")
            continue

        with open(basemodel_file, 'r', encoding='utf-8') as f:
            basemodel_data = json.load(f)

        # basemodel_eng_reasoning 파일 읽기 (한국어 입력 + 영어 추론)
        eng_reasoning_file = basemodel_eng_reasoning_dir / f"results_{model}_3shot.json"
        if not eng_reasoning_file.exists():
            print(f"  Warning: {eng_reasoning_file.name} not found")
            continue

        with open(eng_reasoning_file, 'r', encoding='utf-8') as f:
            eng_reasoning_data = json.load(f)

        # 각 데이터셋에서 details 추출
        arc_details = {item['id']: item for item in basemodel_data.get('datasets', {}).get('ARC', {}).get('details', [])}
        ko_arc_details = {item['id']: item for item in basemodel_data.get('datasets', {}).get('Ko-ARC', {}).get('details', [])}
        kr_eng_details = {item['id']: item for item in eng_reasoning_data.get('datasets', {}).get('Ko-ARC', {}).get('details', [])}

        # 종합 분석 결과 생성
        comprehensive_results = []
        kr_answer_consistency_count = 0  # kr_kr와 kr_eng의 답이 같은 경우 카운트

        for item_id in sorted(unsolvable_ids):
            # 각 방식에서 해당 ID의 정보 수집
            eng_eng_item = arc_details.get(item_id, {})
            kr_kr_item = ko_arc_details.get(item_id, {})
            kr_eng_item = kr_eng_details.get(item_id, {})

            # 종합 정보 구성
            comprehensive_item = {
                "id": item_id,
                "ground_truth": eng_eng_item.get('ground_truth', kr_kr_item.get('ground_truth', kr_eng_item.get('ground_truth', 'Unknown'))),
                "reasoning_comparisons": {
                    "eng_eng_reasoning": {
                        "description": "English input + English reasoning",
                        "predicted_answer": eng_eng_item.get('predicted_answer', 'Not found'),
                        "is_correct": eng_eng_item.get('is_correct', False),
                        "model_raw_output": eng_eng_item.get('model_raw_output', 'Not found')
                    },
                    "kr_kr_reasoning": {
                        "description": "Korean input + Korean reasoning",
                        "predicted_answer": kr_kr_item.get('predicted_answer', 'Not found'),
                        "is_correct": kr_kr_item.get('is_correct', False),
                        "model_raw_output": kr_kr_item.get('model_raw_output', 'Not found')
                    },
                    "kr_eng_reasoning": {
                        "description": "Korean input + English reasoning",
                        "predicted_answer": kr_eng_item.get('predicted_answer', 'Not found'),
                        "is_correct": kr_eng_item.get('is_correct', False),
                        "model_raw_output": kr_eng_item.get('model_raw_output', 'Not found')
                    }
                }
            }

            comprehensive_results.append(comprehensive_item)

            # kr_kr와 kr_eng의 답이 같은지 확인
            kr_kr_answer = kr_kr_item.get('predicted_answer', 'Not found')
            kr_eng_answer = kr_eng_item.get('predicted_answer', 'Not found')

            if kr_kr_answer != 'Not found' and kr_eng_answer != 'Not found' and kr_kr_answer == kr_eng_answer:
                kr_answer_consistency_count += 1

        # 결과 JSON 구조 생성
        result_data = {
            "model_name": model,
            "analysis_type": "comprehensive_reasoning_comparison",
            "total_count": len(comprehensive_results),
            "kr_answer_consistency_count": kr_answer_consistency_count,
            "kr_answer_consistency_rate": round(kr_answer_consistency_count / len(comprehensive_results) * 100, 2) if len(comprehensive_results) > 0 else 0,
            "description": "Detailed analysis of problems that can be solved when given in English but remain unsolvable in Korean regardless of reasoning language. Compares results from 3 different reasoning approaches for each problem.",
            "reasoning_methods": {
                "eng_eng_reasoning": "English input + English reasoning (from basemodel_results)",
                "kr_kr_reasoning": "Korean input + Korean reasoning (from basemodel_results)",
                "kr_eng_reasoning": "Korean input + English reasoning (from basemodel_eng_reasoning)"
            },
            "answer_consistency_analysis": {
                "kr_kr_vs_kr_eng_same_answer_count": kr_answer_consistency_count,
                "kr_kr_vs_kr_eng_same_answer_rate": f"{round(kr_answer_consistency_count / len(comprehensive_results) * 100, 2)}%" if len(comprehensive_results) > 0 else "0%",
                "description": "Count and percentage of problems where Korean+Korean reasoning and Korean+English reasoning produced the same answer"
            },
            "items": comprehensive_results
        }

        # 결과 파일로 저장
        output_file = current_dir / f"step4_results/comprehensive_reasoning_analysis_{model}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"  {model}:")
        print(f"    Total unsolvable items analyzed: {len(comprehensive_results)}")
        print(f"    Korean answer consistency (kr_kr vs kr_eng): {kr_answer_consistency_count}/{len(comprehensive_results)} ({round(kr_answer_consistency_count / len(comprehensive_results) * 100, 2)}%)")
        print(f"    Saved to: {output_file.name}")
        print()

    print("All comprehensive analyses saved successfully!")

if __name__ == "__main__":
    create_comprehensive_analysis()