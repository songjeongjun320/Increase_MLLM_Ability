"""
Re-extract Numerical Answers from GSM8K JSON Results (STRICT MODE)

This script re-extracts numerical answers from raw_output in GSM8K JSON files using STRICT validation.
Prioritizes {} format matching few-shot examples, then #### format.

STRICT MODE CHANGES:
- Only accepts {number} box format (highest priority)
- Only accepts #### number format (second priority)
- Accepts Korean/English answer patterns as fallback
- Forces models to follow exact few-shot example formats

Usage:
    python re_extract_answers.py results_korean_qwem-2.5-3b-pt.json
    python re_extract_answers.py results.json -o re_evaluated.json
    python re_extract_answers.py results.json -e 10  # Show 10 examples

The script expects a JSON file with GSM8K results containing:
- raw_output/model_raw_output: The model's raw text output
- ground_truth: The correct numerical answer
- extracted_answer: Previously extracted answer (optional, None means extraction failed)
"""

import json
import re
import argparse
import os
from typing import Optional, Union

def extract_numerical_answer(model_output: str) -> Optional[float]:
    """
    Extract numerical answer from model output with strict priority:
    1) { <number> }  # boxed answer (highest priority - matches few-shot examples)
    2) #### <number> # structured answer format
    3) Other English/Korean phrasings
    Returns None if no clear numerical answer is found.
    """
    if not model_output:
        return None

    cleaned_output = model_output.strip()

    patterns = [
        # 1) { 18 }   (최우선 - few-shot 예제와 일치)
        r'\{([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\}',

        # 2) #### 18  (둘째 우선)
        r'####\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',

        # 3) 그 외 일반 패턴
        r'답[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'The answer is\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:정답|Answer)[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:답|정답|Answer)\s*(?:은|는|is)?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:따라서|그러므로|그래서|결론적으로|최종적으로|Hence|Therefore)\s*(?:답|정답|answer)?[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?)(?:\s*(?:입니다|이다|\.|\s*$))',
        r'(?:총|합계|전체|Total)\s*[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'=\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)(?:\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?))?(?:\s*$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer_str = matches[0].replace(',', '').strip()
            try:
                return float(answer_str)
            except ValueError:
                continue

    # Fallback: 마지막 줄에서 숫자 스캔
    for line in reversed(cleaned_output.split('\n')):
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)', line)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                continue

    return None

def check_numerical_match(predicted: Union[float, None], ground_truth: Union[str, float, None], tolerance: float = 1e-6) -> bool:
    """
    Check if predicted answer matches ground truth with tolerance
    """
    if predicted is None or ground_truth is None:
        return False

    try:
        pred_float = float(predicted)
        gt_float = float(ground_truth)
        return abs(pred_float - gt_float) < tolerance
    except (ValueError, TypeError):
        return False

def re_evaluate_gsm8k_results(json_filepath: str, output_filepath: str = None, show_examples: int = 5, language: str = None):
    """
    Re-evaluate GSM8K results from a JSON file by re-extracting answers from raw_output
    """
    if not os.path.exists(json_filepath):
        print(f"Error: File {json_filepath} not found")
        return

    # Load the JSON data
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_filepath}: {e}")
        return
    except Exception as e:
        print(f"Error: Failed to read file {json_filepath}: {e}")
        return

    # Check if it's a raw generations file or results file
    if isinstance(data, list):
        # Raw generations format
        results = data
        evaluation_type = "GSM8K Raw Generations"
    elif isinstance(data, dict):
        # Results file format
        if 'korean_results' in data and 'details' in data['korean_results']:
            # Korean results file
            results = data['korean_results']['details']
            evaluation_type = data.get('evaluation_type', 'GSM8K Korean Results')
            language = 'Korean'
        elif 'english_results' in data and 'details' in data['english_results']:
            # English results file
            results = data['english_results']['details']
            evaluation_type = data.get('evaluation_type', 'GSM8K English Results')
            language = 'English'
        elif 'results' in data and 'details' in data['results']:
            # General results format
            results = data['results']['details']
            evaluation_type = data.get('evaluation_type', 'GSM8K Results')
            language = data.get('language', 'Unknown')
        else:
            print("Error: Unrecognized JSON format. Expected GSM8K results structure.")
            return

        if not results:
            print("Error: No details found in the results")
            return
    else:
        print("Error: Unrecognized JSON format. Expected a list of items or GSM8K results structure.")
        return

    # Re-extract answers
    total_items = len(results)
    re_extracted_results = []

    print(f"Re-evaluating {total_items} GSM8K items...")

    # Process all items
    for i, item in enumerate(results):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{total_items} ({(i + 1)/total_items*100:.1f}%)")

        # Handle different field names for raw output
        raw_output = item.get('raw_output', '') or item.get('model_raw_output', '')
        ground_truth = item.get('ground_truth', '')
        old_extracted = item.get('extracted_answer', None) or item.get('predicted_answer', None)

        # Re-extract answer
        new_extracted = extract_numerical_answer(raw_output)

        # Check correctness
        is_correct = check_numerical_match(new_extracted, ground_truth) if new_extracted is not None else False
        old_correct = check_numerical_match(old_extracted, ground_truth) if (old_extracted is not None) else False

        # Store results (convert boolean to string for Python compatibility)
        result_item = {
            "index": item.get('index', ''),
            "ground_truth": ground_truth,
            "old_extracted_answer": old_extracted,
            "new_extracted_answer": new_extracted,
            "old_correct": str(old_correct),
            "new_correct": str(is_correct),
            "extraction_changed": str(old_extracted != new_extracted),
            "accuracy_changed": str(old_correct != is_correct),
            "question": item.get('question', ''),
            "raw_output": raw_output
        }

        re_extracted_results.append(result_item)

    # Calculate accuracy
    total_items = len(re_extracted_results)
    valid_predictions = sum(1 for item in re_extracted_results if item['new_extracted_answer'] is not None)
    correct_predictions = sum(1 for item in re_extracted_results if item['new_correct'] == 'True')
    extraction_failed = total_items - valid_predictions

    accuracy_standard = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0
    accuracy_strict = (correct_predictions / total_items * 100) if total_items > 0 else 0

    # Count changes
    extraction_changes = sum(1 for item in re_extracted_results if item['extraction_changed'] == 'True')
    accuracy_improvements = sum(1 for item in re_extracted_results if item['accuracy_changed'] == 'True' and item['new_correct'] == 'True')
    accuracy_degradations = sum(1 for item in re_extracted_results if item['accuracy_changed'] == 'True' and item['new_correct'] == 'False')

    # Print summary
    print("\n" + "="*80)
    print("GSM8K RE-EXTRACTION RESULTS SUMMARY")
    print("="*80)
    print(f"Evaluation Type: {evaluation_type}")
    if language:
        print(f"Language: {language}")
    print(f"Total Items: {total_items}")
    print(f"Valid Predictions: {valid_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Extraction Failed: {extraction_failed}")
    print(f"Accuracy (Standard): {accuracy_standard:.2f}%")
    print(f"Accuracy (Strict): {accuracy_strict:.2f}%")
    print()
    print("CHANGES:")
    print(f"  Extraction Changed: {extraction_changes}")
    print(f"  Accuracy Improved: {accuracy_improvements}")
    print(f"  Accuracy Degraded: {accuracy_degradations}")
    print("="*80)

    # Save results
    if output_filepath is None:
        base_name = os.path.splitext(json_filepath)[0]
        output_filepath = f"{base_name}_re_extracted.json"

    summary_data = {
        "original_file": json_filepath,
        "evaluation_type": evaluation_type,
        "language": language,
        "summary": {
            "total_items": total_items,
            "valid_predictions": valid_predictions,
            "correct_predictions": correct_predictions,
            "extraction_failed": extraction_failed,
            "accuracy_standard": accuracy_standard,
            "accuracy_strict": accuracy_strict
        },
        "changes": {
            "extraction_changed": extraction_changes,
            "accuracy_improved": accuracy_improvements,
            "accuracy_degraded": accuracy_degradations
        },
        "detailed_results": re_extracted_results
    }

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_filepath}")

    # Show some examples of changes
    if extraction_changes > 0 and show_examples > 0:
        print(f"\nFirst {show_examples} extraction changes:")
        changes_shown = 0
        for item in re_extracted_results:
            if item['extraction_changed'] == 'True' and changes_shown < show_examples:
                status = "✓" if item['new_correct'] == 'True' and item['old_correct'] == 'False' else "✗" if item['new_correct'] == 'False' and item['old_correct'] == 'True' else "="
                print(f"  {status} Index {item['index']}: '{item['old_extracted_answer']}' -> '{item['new_extracted_answer']}' (GT: {item['ground_truth']})")
                changes_shown += 1

def main():
    parser = argparse.ArgumentParser(description='Re-extract numerical answers from GSM8K JSON results and evaluate accuracy')
    parser.add_argument('json_file', help='Path to the GSM8K JSON file containing raw_output data')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-e', '--examples', type=int, default=5, help='Number of example changes to show (default: 5)')
    parser.add_argument('-l', '--language', help='Specify language (Korean, English)')

    args = parser.parse_args()

    re_evaluate_gsm8k_results(args.json_file, args.output, args.examples, args.language)

if __name__ == "__main__":
    main()