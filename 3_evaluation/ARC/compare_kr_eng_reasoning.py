"""
한국어 추론 vs 영어 추론 결과 비교 분석 스크립트

한국어 추론으로 틀린 문제들 중에서:
1. 영어 추론으로는 맞춘 문제들 (개선된 문제들)
2. 영어 추론으로도 여전히 틀린 문제들 (지속적으로 어려운 문제들)

결과:
- {model}_kr_wrong_eng_correct.json: 한국어로 틀렸지만 영어로는 맞춘 문제들
- {model}_kr_wrong_eng_wrong.json: 한국어로도 영어로도 틀린 문제들
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# ===== 설정 변수 =====
KR_REASONING_DIR = "./ko_arc_incorrect_answers_kr_input_kr_reasoning"  # 한국어 추론 틀린 답변 폴더
ENG_REASONING_DIR = "./ko_arc_incorrect_answers_kr_input_eng_reasoning"  # 영어 추론 틀린 답변 폴더
BASEMODELS_KR_DIR = "./basemodels"  # 한국어 추론 원본 결과 폴더 (basemodels에 모든 결과가 있음)
BASEMODELS_ENG_DIR = "./basemodels"  # 영어 추론 원본 결과 폴더 (basemodels에 모든 결과가 있음)
OUTPUT_DIR = "./kr_eng_reasoning_comparison"  # 비교 결과 저장 폴더


def load_incorrect_answers(directory):
    """
    틀린 답변 폴더에서 모델별 틀린 문제 ID들을 로드

    Returns:
        dict: {model_name: set of incorrect question IDs}
    """
    incorrect_by_model = {}

    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist")
        return incorrect_by_model

    for json_file in Path(directory).glob("*.json"):
        # 파일명에서 모델명 추출 (예: qwem-2.5-3b-pt_incorrect_answers.json -> qwem-2.5-3b-pt)
        model_name = json_file.stem.replace('_incorrect_answers', '')

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 틀린 문제 ID들을 set으로 저장
            incorrect_ids = {entry.get('id') for entry in data if entry.get('id')}
            incorrect_by_model[model_name] = incorrect_ids

            print(f"Loaded {len(incorrect_ids)} incorrect answers for {model_name} from {directory}")

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return incorrect_by_model


def load_full_incorrect_data(directory):
    """
    틀린 답변 폴더에서 모델별 전체 데이터를 로드 (ID를 키로 하는 딕셔너리)

    Returns:
        dict: {model_name: {question_id: full_entry_data}}
    """
    data_by_model = {}

    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist")
        return data_by_model

    for json_file in Path(directory).glob("*.json"):
        model_name = json_file.stem.replace('_incorrect_answers', '')

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ID를 키로 하는 딕셔너리로 변환
            data_dict = {entry.get('id'): entry for entry in data if entry.get('id')}
            data_by_model[model_name] = data_dict

        except Exception as e:
            print(f"Error loading full data from {json_file}: {e}")

    return data_by_model


def load_full_original_data(directory):
    """
    원본 결과 폴더에서 모델별 전체 데이터를 로드 (Ko-ARC 데이터만, ID를 키로 하는 딕셔너리)

    Returns:
        dict: {model_name: {question_id: full_entry_data}}
    """
    data_by_model = {}

    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist")
        return data_by_model

    for json_file in Path(directory).glob("*.json"):
        # 파일명에서 모델명 추출 (예: results_qwem-2.5-3b-pt_3shot.json -> qwem-2.5-3b-pt)
        filename = json_file.stem
        if 'results_' in filename:
            model_name = filename.replace('results_', '').replace('_3shot', '')
        else:
            model_name = filename

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Ko-ARC 데이터만 추출
            if 'datasets' in data and 'Ko-ARC' in data['datasets']:
                ko_arc_data = data['datasets']['Ko-ARC']
                if 'details' in ko_arc_data:
                    # ID를 키로 하는 딕셔너리로 변환
                    data_dict = {entry.get('id'): entry for entry in ko_arc_data['details'] if entry.get('id')}
                    data_by_model[model_name] = data_dict

        except Exception as e:
            print(f"Error loading original data from {json_file}: {e}")

    return data_by_model


def compare_reasoning_results():
    """
    한국어 추론과 영어 추론 결과를 비교하여 분석
    """
    print("Loading Korean reasoning incorrect answers...")
    kr_incorrect = load_incorrect_answers(KR_REASONING_DIR)

    print("Loading English reasoning incorrect answers...")
    eng_incorrect = load_incorrect_answers(ENG_REASONING_DIR)

    print("Loading full Korean reasoning data...")
    kr_full_data = load_full_incorrect_data(KR_REASONING_DIR)

    print("Loading full English reasoning data...")
    eng_full_data = load_full_incorrect_data(ENG_REASONING_DIR)

    print("Loading original Korean reasoning results...")
    kr_original_data = load_full_original_data(BASEMODELS_KR_DIR)

    print("Loading original English reasoning results...")
    eng_original_data = load_full_original_data(BASEMODELS_ENG_DIR)

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 공통 모델들 찾기
    common_models = set(kr_incorrect.keys()) & set(eng_incorrect.keys())
    print(f"\nCommon models found: {list(common_models)}")

    results_summary = {}

    for model in common_models:
        print(f"\n--- Analyzing {model} ---")

        kr_wrong_ids = kr_incorrect[model]
        eng_wrong_ids = eng_incorrect[model]

        # 한국어로 틀렸지만 영어로는 맞춘 문제들
        kr_wrong_eng_correct_ids = kr_wrong_ids - eng_wrong_ids

        # 한국어로도 영어로도 틀린 문제들
        kr_wrong_eng_wrong_ids = kr_wrong_ids & eng_wrong_ids

        print(f"Korean wrong, English correct: {len(kr_wrong_eng_correct_ids)} problems")
        print(f"Korean wrong, English wrong: {len(kr_wrong_eng_wrong_ids)} problems")
        print(f"Total Korean wrong: {len(kr_wrong_ids)} problems")

        # 개선율 계산 (한국어로 틀린 문제들 중에서 영어로 맞춘 비율)
        improvement_rate_among_wrong = len(kr_wrong_eng_correct_ids) / len(kr_wrong_ids) * 100 if kr_wrong_ids else 0
        print(f"Improvement rate (among Korean wrong): {improvement_rate_among_wrong:.2f}%")

        # 전체 Ko-ARC 문제 수 추정 (1167개 정도)
        total_koarc_problems = 1167  # Ko-ARC 전체 문제 수
        overall_improvement_rate = len(kr_wrong_eng_correct_ids) / total_koarc_problems * 100
        print(f"Overall improvement rate (total problems): {overall_improvement_rate:.2f}%")

        # 결과 저장을 위한 데이터 준비
        kr_wrong_eng_correct_data = []
        kr_wrong_eng_wrong_data = []

        # 한국어로 틀렸지만 영어로는 맞춘 문제들의 데이터 수집
        for question_id in kr_wrong_eng_correct_ids:
            if question_id in kr_full_data[model]:
                kr_entry = kr_full_data[model][question_id].copy()

                # 영어 추론 결과를 원본 데이터에서 가져오기
                eng_entry = {}
                if model in eng_original_data and question_id in eng_original_data[model]:
                    eng_entry = eng_original_data[model][question_id]

                # 기본 정보 설정
                entry = {
                    'dataset': kr_entry.get('dataset'),
                    'index': kr_entry.get('index'),
                    'id': kr_entry.get('id'),
                    'ground_truth': kr_entry.get('ground_truth'),
                    'kr_predicted_answer': kr_entry.get('predicted_answer'),
                    'eng_predicted_answer': eng_entry.get('predicted_answer', 'CORRECT'),  # 영어로는 맞췄으므로
                    'model_raw_output_kr': kr_entry.get('model_raw_output', ''),
                    'model_raw_output_eng': eng_entry.get('model_raw_output', 'Problem solved with English reasoning'),
                    'reasoning_comparison': 'kr_wrong_eng_correct',
                    'improvement_status': 'English reasoning helped solve this problem'
                }
                kr_wrong_eng_correct_data.append(entry)

        # 한국어로도 영어로도 틀린 문제들의 데이터 수집
        for question_id in kr_wrong_eng_wrong_ids:
            if question_id in kr_full_data[model]:
                kr_entry = kr_full_data[model][question_id].copy()
                eng_entry = eng_full_data[model].get(question_id, {})

                entry = {
                    'dataset': kr_entry.get('dataset'),
                    'index': kr_entry.get('index'),
                    'id': kr_entry.get('id'),
                    'ground_truth': kr_entry.get('ground_truth'),
                    'kr_predicted_answer': kr_entry.get('predicted_answer'),
                    'eng_predicted_answer': eng_entry.get('predicted_answer', 'UNKNOWN'),
                    'model_raw_output_kr': kr_entry.get('model_raw_output', ''),
                    'model_raw_output_eng': eng_entry.get('model_raw_output', ''),
                    'reasoning_comparison': 'kr_wrong_eng_wrong',
                    'improvement_status': 'Both Korean and English reasoning failed'
                }
                kr_wrong_eng_wrong_data.append(entry)

        # 파일 저장 (상단에 통계 정보 포함)
        kr_wrong_eng_correct_file = os.path.join(OUTPUT_DIR, f"{model}_kr_wrong_eng_correct.json")
        kr_wrong_eng_wrong_file = os.path.join(OUTPUT_DIR, f"{model}_kr_wrong_eng_wrong.json")

        # kr_wrong_eng_correct 파일에 통계 추가
        kr_wrong_eng_correct_with_stats = {
            "summary": {
                "model": model,
                "total_ko_arc_problems": total_koarc_problems,
                "total_kr_wrong": len(kr_wrong_ids),
                "kr_wrong_eng_correct_count": len(kr_wrong_eng_correct_ids),
                "description": "Problems that were wrong with Korean reasoning but correct with English reasoning"
            },
            "problems": kr_wrong_eng_correct_data
        }

        # kr_wrong_eng_wrong 파일에 통계 추가
        kr_wrong_eng_wrong_with_stats = {
            "summary": {
                "model": model,
                "total_ko_arc_problems": total_koarc_problems,
                "total_kr_wrong": len(kr_wrong_ids),
                "kr_wrong_eng_wrong_count": len(kr_wrong_eng_wrong_ids),
                "description": "Problems that were wrong with both Korean and English reasoning"
            },
            "problems": kr_wrong_eng_wrong_data
        }

        with open(kr_wrong_eng_correct_file, 'w', encoding='utf-8') as f:
            json.dump(kr_wrong_eng_correct_with_stats, f, indent=2, ensure_ascii=False)

        with open(kr_wrong_eng_wrong_file, 'w', encoding='utf-8') as f:
            json.dump(kr_wrong_eng_wrong_with_stats, f, indent=2, ensure_ascii=False)

        print(f"Saved: {kr_wrong_eng_correct_file}")
        print(f"Saved: {kr_wrong_eng_wrong_file}")

        # 요약 정보 저장
        results_summary[model] = {
            'total_koarc_problems': total_koarc_problems,
            'total_kr_wrong': len(kr_wrong_ids),
            'kr_wrong_eng_correct': len(kr_wrong_eng_correct_ids),
            'kr_wrong_eng_wrong': len(kr_wrong_eng_wrong_ids),
            'improvement_rate_among_wrong': improvement_rate_among_wrong,
            'overall_improvement_rate': overall_improvement_rate,
            'kr_accuracy': ((total_koarc_problems - len(kr_wrong_ids)) / total_koarc_problems * 100),
            'eng_accuracy_estimated': ((total_koarc_problems - len(kr_wrong_ids) + len(kr_wrong_eng_correct_ids)) / total_koarc_problems * 100),
            'kr_wrong_eng_correct_ids': list(kr_wrong_eng_correct_ids),
            'kr_wrong_eng_wrong_ids': list(kr_wrong_eng_wrong_ids)
        }

    # 전체 요약 저장
    summary_file = os.path.join(OUTPUT_DIR, "comparison_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Overall Summary ===")
    print(f"Summary saved to: {summary_file}")

    # 테이블 형태로 요약 출력
    print("\nModel-wise Comparison Results:")
    print("=" * 120)
    print(f"{'Model':<25} {'KR Wrong':<10} {'Improved':<10} {'Still Wrong':<12} {'Improve%(Wrong)':<15} {'Overall Improve%':<16} {'KR Acc%':<10} {'Eng Acc%':<10}")
    print("=" * 120)

    for model, stats in results_summary.items():
        print(f"{model:<25} {stats['total_kr_wrong']:<10} {stats['kr_wrong_eng_correct']:<10} "
              f"{stats['kr_wrong_eng_wrong']:<12} {stats['improvement_rate_among_wrong']:<15.2f} "
              f"{stats['overall_improvement_rate']:<16.2f} {stats['kr_accuracy']:<10.2f} {stats['eng_accuracy_estimated']:<10.2f}")

    print("\nExplanation:")
    print("- Improve%(Wrong): Percentage improved by English among Korean wrong problems")
    print("- Overall Improve%: Percentage improved by English among total Ko-ARC problems")
    print("- KR Acc%: Korean reasoning accuracy")
    print("- Eng Acc%: Estimated English reasoning accuracy (Korean correct + English improvement)")

    return results_summary


def main():
    print("Korean vs English Reasoning Comparison Analysis")
    print("=" * 60)
    print(f"Korean reasoning directory: {KR_REASONING_DIR}")
    print(f"English reasoning directory: {ENG_REASONING_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    results = compare_reasoning_results()

    print(f"\nAnalysis completed! Check the '{OUTPUT_DIR}' directory for detailed results.")


if __name__ == "__main__":
    main()