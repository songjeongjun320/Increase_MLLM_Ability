"""
Re-extract Answers from HellaSwag JSON Results (STRICT MODE)

This script re-extracts answers from raw_output in HellaSwag JSON files using STRICT validation.
Only accepts {} format matching few-shot examples.

STRICT MODE CHANGES:
- Only accepts {A}, {B}, {C}, {D} box format
- Converts A,B,C,D to 0,1,2,3 for HellaSwag evaluation
- Rejects all other formats (e.g., "The answer is A", "A.", "(A)", etc.)
- Forces models to follow exact few-shot example formats

Usage:
    python re_extract_answers.py results_qwem-2.5-3b-tow-09_11_2epoch_fix_tow-merged_5shot.json
    python re_extract_answers.py results.json -o re_evaluated.json
    python re_extract_answers.py results.json -e 10  # Show 10 examples

The script expects a JSON file with HellaSwag results containing:
- raw_output/model_raw_output: The model's raw text output
- ground_truth: The correct answer (0, 1, 2, or 3)
- predicted_answer: Previously extracted answer (optional, None means extraction failed)
"""

import json
import re
import argparse
import os
from typing import Optional, Union

def extract_answer_robust(model_output: str) -> Optional[int]:
    """
    Extract the final answer (A, B, C, D) from model output and convert to 0,1,2,3
    STRICT MODE: Only accepts {} format - unified across all evaluation scripts.
    Returns None if no clear structured answer is found.
    """
    if not model_output:
        return None

    cleaned_output = model_output.strip().upper()

    # STRICT: Only accept {} format for consistency across all evaluation scripts
    box_pattern = r'\{([A-D])\}'
    box_matches = re.findall(box_pattern, cleaned_output)
    if box_matches:
        # Convert A,B,C,D to 0,1,2,3 for HellaSwag
        return ord(box_matches[0]) - ord('A')  # Use last match (final answer)

    # No fallback patterns - forces models to use {} format only
    return None

def check_answer_match(predicted: Union[int, None], ground_truth: Union[str, int, None]) -> bool:
    """
    Check if predicted answer matches ground truth
    """
    if predicted is None or ground_truth is None:
        return False

    try:
        pred_int = int(predicted)
        gt_int = int(ground_truth)
        return pred_int == gt_int
    except (ValueError, TypeError):
        return False

def re_evaluate_hellaswag_results(json_filepath: str, output_filepath: str = None, show_examples: int = 5, dataset_name: str = None):
    """
    Re-evaluate HellaSwag results from a JSON file by re-extracting answers from raw_output
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
        evaluation_type = "HellaSwag Raw Generations"
    elif isinstance(data, dict) and 'datasets' in data:
        # Results file with datasets structure
        print("Detected results file with 'datasets' structure")
        datasets = data.get('datasets', {})

        # Check for HellaSwag or Ko-HellaSwag dataset
        if dataset_name:
            # Use specified dataset
            if dataset_name in datasets:
                print(f"Using specified dataset: {dataset_name}")
                results = datasets[dataset_name].get('details', [])
                evaluation_type = f"HellaSwag {dataset_name} Results"
            else:
                available_datasets = list(datasets.keys())
                print(f"Error: Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
                return
        else:
            # Process both HellaSwag and Ko-HellaSwag if available
            all_results = []
            processed_datasets = []

            # Check for HellaSwag dataset
            if 'HellaSwag' in datasets:
                hellaswag_results = datasets['HellaSwag'].get('details', [])
                if hellaswag_results:
                    # Add dataset field to each item
                    for item in hellaswag_results:
                        item['dataset'] = 'HellaSwag'
                    all_results.extend(hellaswag_results)
                    processed_datasets.append('HellaSwag')
                    print(f"Found HellaSwag dataset with {len(hellaswag_results)} items")

            # Check for Ko-HellaSwag dataset
            if 'Ko-HellaSwag' in datasets:
                ko_hellaswag_results = datasets['Ko-HellaSwag'].get('details', [])
                if ko_hellaswag_results:
                    # Add dataset field to each item
                    for item in ko_hellaswag_results:
                        item['dataset'] = 'Ko-HellaSwag'
                    all_results.extend(ko_hellaswag_results)
                    processed_datasets.append('Ko-HellaSwag')
                    print(f"Found Ko-HellaSwag dataset with {len(ko_hellaswag_results)} items")

            if not all_results:
                available_datasets = list(datasets.keys())
                print(f"Error: No HellaSwag or Ko-HellaSwag dataset found. Available datasets: {available_datasets}")
                return

            results = all_results
            evaluation_type = f"HellaSwag {', '.join(processed_datasets)} Results"
            print(f"Processing both datasets: {', '.join(processed_datasets)} (Total: {len(results)} items)")

        if not results:
            print("Error: No details found in the dataset")
            return
    else:
        print("Error: Unrecognized JSON format. Expected a list of items or HellaSwag results structure.")
        return

    # Re-extract answers
    total_items = len(results)
    re_extracted_results = []

    # Separate results by dataset
    hellaswag_results = [item for item in results if item.get('dataset') == 'HellaSwag']
    ko_hellaswag_results = [item for item in results if item.get('dataset') == 'Ko-HellaSwag']

    hellaswag_count = len(hellaswag_results)
    ko_hellaswag_count = len(ko_hellaswag_results)

    print(f"Re-evaluating {total_items} items...")
    if hellaswag_count > 0 and ko_hellaswag_count > 0:
        print(f"  - HellaSwag: {hellaswag_count} items")
        print(f"  - Ko-HellaSwag: {ko_hellaswag_count} items")

    # Process all items
    for i, item in enumerate(results):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{total_items} ({(i + 1)/total_items*100:.1f}%)")

        # Handle different field names for raw output
        raw_output = item.get('raw_output', '') or item.get('model_raw_output', '')
        ground_truth = item.get('ground_truth', None)
        old_extracted = item.get('predicted_answer', None) or item.get('extracted_answer', None)

        # Re-extract answer
        new_extracted = extract_answer_robust(raw_output)

        # Check correctness (only if ground truth is available and valid)
        is_correct = False
        old_correct = False

        if ground_truth is not None and ground_truth != -1:
            is_correct = check_answer_match(new_extracted, ground_truth) if new_extracted is not None else False
            old_correct = check_answer_match(old_extracted, ground_truth) if old_extracted is not None else False

        # Store results (convert boolean to string for Python compatibility)
        result_item = {
            "dataset": item.get('dataset', ''),
            "index": item.get('index', ''),
            "id": item.get('id', '') or item.get('ind', ''),
            "ground_truth": ground_truth,
            "old_extracted_answer": old_extracted,
            "new_extracted_answer": new_extracted,
            "old_correct": str(old_correct),
            "new_correct": str(is_correct),
            "extraction_changed": str(old_extracted != new_extracted),
            "accuracy_changed": str(old_correct != is_correct),
            "source_dataset": item.get('dataset', 'Unknown')
        }

        re_extracted_results.append(result_item)

    # Calculate accuracy separately for each dataset
    def calculate_dataset_accuracy(dataset_name):
        # Filter re_extracted_results by dataset
        dataset_items = [item for item in re_extracted_results if item['dataset'] == dataset_name]
        total_items = len(dataset_items)
        if total_items == 0:
            return {
                "total_items": 0,
                "valid_predictions": 0,
                "correct_predictions": 0,
                "extraction_failed": 0,
                "accuracy_standard": 0.0,
                "accuracy_strict": 0.0
            }

        valid_predictions = sum(1 for item in dataset_items if item['new_extracted_answer'] is not None)
        correct_predictions = sum(1 for item in dataset_items if item['new_correct'] == 'True')
        extraction_failed = total_items - valid_predictions

        accuracy_standard = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0
        accuracy_strict = (correct_predictions / total_items * 100) if total_items > 0 else 0

        return {
            "total_items": total_items,
            "valid_predictions": valid_predictions,
            "correct_predictions": correct_predictions,
            "extraction_failed": extraction_failed,
            "accuracy_standard": accuracy_standard,
            "accuracy_strict": accuracy_strict
        }

    # Calculate accuracy for each dataset
    hellaswag_accuracy = calculate_dataset_accuracy("HellaSwag")
    ko_hellaswag_accuracy = calculate_dataset_accuracy("Ko-HellaSwag")

    # Overall accuracy
    total_valid_predictions = hellaswag_accuracy["valid_predictions"] + ko_hellaswag_accuracy["valid_predictions"]
    total_correct_predictions = hellaswag_accuracy["correct_predictions"] + ko_hellaswag_accuracy["correct_predictions"]
    total_extraction_failed = hellaswag_accuracy["extraction_failed"] + ko_hellaswag_accuracy["extraction_failed"]

    overall_accuracy_standard = (total_correct_predictions / total_valid_predictions * 100) if total_valid_predictions > 0 else 0
    overall_accuracy_strict = (total_correct_predictions / total_items * 100) if total_items > 0 else 0

    # Count changes
    extraction_changes = sum(1 for item in re_extracted_results if item['extraction_changed'] == 'True')
    accuracy_improvements = sum(1 for item in re_extracted_results if item['accuracy_changed'] == 'True' and item['new_correct'] == 'True')
    accuracy_degradations = sum(1 for item in re_extracted_results if item['accuracy_changed'] == 'True' and item['new_correct'] == 'False')

    # Print summary
    print("\n" + "="*80)
    print("HELLASWAG RE-EXTRACTION RESULTS SUMMARY")
    print("="*80)
    print(f"OVERALL:")
    print(f"  Total Items: {total_items}")
    print(f"  Valid Predictions: {total_valid_predictions}")
    print(f"  Correct Predictions: {total_correct_predictions}")
    print(f"  Extraction Failed: {total_extraction_failed}")
    print(f"  Accuracy (Standard): {overall_accuracy_standard:.2f}%")
    print(f"  Accuracy (Strict): {overall_accuracy_strict:.2f}%")
    print()

    if hellaswag_count > 0:
        print(f"HELLASWAG DATASET:")
        print(f"  Items: {hellaswag_accuracy['total_items']}")
        print(f"  Valid Predictions: {hellaswag_accuracy['valid_predictions']}")
        print(f"  Correct Predictions: {hellaswag_accuracy['correct_predictions']}")
        print(f"  Extraction Failed: {hellaswag_accuracy['extraction_failed']}")
        print(f"  Accuracy (Standard): {hellaswag_accuracy['accuracy_standard']:.2f}%")
        print(f"  Accuracy (Strict): {hellaswag_accuracy['accuracy_strict']:.2f}%")
        print()

    if ko_hellaswag_count > 0:
        print(f"KO-HELLASWAG DATASET:")
        print(f"  Items: {ko_hellaswag_accuracy['total_items']}")
        print(f"  Valid Predictions: {ko_hellaswag_accuracy['valid_predictions']}")
        print(f"  Correct Predictions: {ko_hellaswag_accuracy['correct_predictions']}")
        print(f"  Extraction Failed: {ko_hellaswag_accuracy['extraction_failed']}")
        print(f"  Accuracy (Standard): {ko_hellaswag_accuracy['accuracy_standard']:.2f}%")
        print(f"  Accuracy (Strict): {ko_hellaswag_accuracy['accuracy_strict']:.2f}%")
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
        "overall": {
            "total_items": total_items,
            "valid_predictions": total_valid_predictions,
            "correct_predictions": total_correct_predictions,
            "extraction_failed": total_extraction_failed,
            "accuracy_standard": overall_accuracy_standard,
            "accuracy_strict": overall_accuracy_strict
        },
        "hellaswag_dataset": hellaswag_accuracy,
        "ko_hellaswag_dataset": ko_hellaswag_accuracy,
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
    parser = argparse.ArgumentParser(description='Re-extract answers from HellaSwag JSON results and evaluate accuracy')
    parser.add_argument('json_file', help='Path to the HellaSwag JSON file containing raw_output data')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-e', '--examples', type=int, default=5, help='Number of example changes to show (default: 5)')
    parser.add_argument('-d', '--dataset', help='Specify dataset name (e.g., HellaSwag, Ko-HellaSwag)')

    args = parser.parse_args()

    re_evaluate_hellaswag_results(args.json_file, args.output, args.examples, args.dataset)

if __name__ == "__main__":
    main()