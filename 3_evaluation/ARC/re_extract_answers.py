"""
Re-extract Answers from JSON Results

This script re-extracts answers from raw_output in JSON files and evaluates accuracy.
It's useful for re-evaluating model results with improved extraction logic.

Usage:
    python re_extract_answers.py results.json
    python re_extract_answers.py results.json -o re_evaluated.json
    python re_extract_answers.py results.json -e 10  # Show 10 examples

The script expects a JSON file with a list of items, each containing:
- raw_output: The model's raw text output
- ground_truth: The correct answer (A, B, C, or D)
- extracted_answer: Previously extracted answer (optional)
"""

import json
import re
import argparse
import os
from typing import Optional

def extract_answer_robust(model_output: str) -> Optional[str]:
    """
    Extract the final answer (A, B, C, D) from model output using structured patterns first.
    Supports A-D for 4 options (ARC format).
    """
    if not model_output:
        return None
        
    cleaned_output = model_output.strip().upper()
    valid_answers = ['A', 'B', 'C', 'D']
    
    # Priority 1: Structured answer patterns (most reliable)
    structured_patterns = [
        r'####\s*(?:정답|답|ANSWER|THEREFORE\s+ANSWER)\s*:?\s*\{?([A-D])\}?',  # #### Answer: A or #### 정답: A or {A}
        r'\{([A-D])\}',  # {A} box format matching prompt style
        r'(?:정답|답|ANSWER)\s*:?\s*\{?([A-D])\}?',        # Answer: A or 정답: A or {A}
        r'(?:따라서|그러므로|SO|THEREFORE)\s+(?:정답은|답은|정답|답|THE\s+ANSWER\s+IS|ANSWER\s+IS)\s*:?\s*\{?([A-D])\}?',  # So the answer is A or {A}
    ]
    
    for pattern in structured_patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[0]  # Return the first match (avoid repetitions/hallucinations)
    
    # Priority 2: Start of text patterns
    start_patterns = [
        r'^\s*([A-D])[\.\)\]\s]',  # A. or A) or A] at start
        r'^\s*\(?([A-D])\)?\s*[\.:;]',  # (A): or A. or A:
        r'^\s*([A-D])\s*$',          # Just A at start of line
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, cleaned_output, re.MULTILINE)
        if match:
            return match.group(1)
    
    # Priority 3: Last resort - find A-D near end of text (avoid random letters in middle)
    # Only look in last 100 characters to avoid picking up random letters
    last_part = cleaned_output[-100:] if len(cleaned_output) > 100 else cleaned_output
    
    # Look for isolated A-D characters near the end
    end_patterns = [
        r'([A-D])(?:\s*[\.:;]?\s*$)',  # A at end with optional punctuation
        r'(?:\s|^)([A-D])(?:\s|$)',    # A surrounded by whitespace
    ]
    
    for pattern in end_patterns:
        matches = re.findall(pattern, last_part)
        if matches:
            return matches[0]  # Return the first match (avoid repetitions)
    
    # Priority 4: Absolute fallback - scan from end backwards
    # This avoids picking random letters from the beginning/middle of text
    for i in range(len(cleaned_output) - 1, -1, -1):
        if cleaned_output[i] in valid_answers:
            # Check if this letter appears to be part of an answer pattern
            context_start = max(0, i - 20)
            context_end = min(len(cleaned_output), i + 20)
            context = cleaned_output[context_start:context_end]
            
            # Avoid letters that are clearly part of words
            if i > 0 and cleaned_output[i-1].isalnum():
                continue
            if i < len(cleaned_output) - 1 and cleaned_output[i+1].isalnum():
                continue
                
            return cleaned_output[i]
    
    return None

def re_evaluate_json_results(json_filepath: str, output_filepath: str = None, show_examples: int = 5, dataset_name: str = None):
    """
    Re-evaluate results from a JSON file by re-extracting answers from raw_output
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
    elif isinstance(data, dict) and 'datasets' in data:
        # Results file with datasets structure
        print("Detected results file with 'datasets' structure")
        datasets = data.get('datasets', {})
        
        # Check for ARC or Ko-ARC dataset
        if dataset_name:
            # Use specified dataset
            if dataset_name in datasets:
                print(f"Using specified dataset: {dataset_name}")
                results = datasets[dataset_name].get('details', [])
            else:
                available_datasets = list(datasets.keys())
                print(f"Error: Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
                return
        else:
            # Process both ARC and Ko-ARC if available
            all_results = []
            processed_datasets = []
            
            # Check for ARC dataset
            if 'ARC' in datasets:
                arc_results = datasets['ARC'].get('details', [])
                if arc_results:
                    # Add dataset field to each item
                    for item in arc_results:
                        item['dataset'] = 'ARC'
                    all_results.extend(arc_results)
                    processed_datasets.append('ARC')
                    print(f"Found ARC dataset with {len(arc_results)} items")
            
            # Check for Ko-ARC dataset
            if 'Ko-ARC' in datasets:
                ko_arc_results = datasets['Ko-ARC'].get('details', [])
                if ko_arc_results:
                    # Add dataset field to each item
                    for item in ko_arc_results:
                        item['dataset'] = 'Ko-ARC'
                    all_results.extend(ko_arc_results)
                    processed_datasets.append('Ko-ARC')
                    print(f"Found Ko-ARC dataset with {len(ko_arc_results)} items")
            
            if not all_results:
                available_datasets = list(datasets.keys())
                print(f"Error: No ARC or Ko-ARC dataset found. Available datasets: {available_datasets}")
                return
            
            results = all_results
            print(f"Processing both datasets: {', '.join(processed_datasets)} (Total: {len(results)} items)")
        
        if not results:
            print("Error: No details found in the dataset")
            return
    else:
        print("Error: Unrecognized JSON format. Expected a list of items or datasets structure.")
        return
    
    # Re-extract answers
    total_items = len(results)
    re_extracted_results = []
    
    # Separate results by dataset
    arc_results = [item for item in results if item.get('dataset') == 'ARC']
    ko_arc_results = [item for item in results if item.get('dataset') == 'Ko-ARC']
    
    arc_count = len(arc_results)
    ko_arc_count = len(ko_arc_results)
    
    print(f"Re-evaluating {total_items} items...")
    if arc_count > 0 and ko_arc_count > 0:
        print(f"  - ARC: {arc_count} items")
        print(f"  - Ko-ARC: {ko_arc_count} items")
    
    # Process all items
    for i, item in enumerate(results):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{total_items} ({(i + 1)/total_items*100:.1f}%)")
        # Handle different field names for raw output
        raw_output = item.get('raw_output', '') or item.get('model_raw_output', '')
        ground_truth = item.get('ground_truth', '')
        old_extracted = item.get('extracted_answer', '') or item.get('predicted_answer', '')
        
        # Re-extract answer
        new_extracted = extract_answer_robust(raw_output)
        
        # Check correctness
        is_correct = new_extracted == ground_truth if new_extracted else False
        old_correct = old_extracted == ground_truth if old_extracted else False
        
        # Store results
        result_item = {
            "dataset": item.get('dataset', ''),
            "index": item.get('index', ''),
            "id": item.get('id', ''),
            "ground_truth": ground_truth,
            "old_extracted_answer": old_extracted,
            "new_extracted_answer": new_extracted,
            "old_correct": old_correct,
            "new_correct": is_correct,
            "extraction_changed": old_extracted != new_extracted,
            "accuracy_changed": old_correct != is_correct,
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
        correct_predictions = sum(1 for item in dataset_items if item['new_correct'])
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
    arc_accuracy = calculate_dataset_accuracy("ARC")
    ko_arc_accuracy = calculate_dataset_accuracy("Ko-ARC")
    
    # Overall accuracy
    total_valid_predictions = arc_accuracy["valid_predictions"] + ko_arc_accuracy["valid_predictions"]
    total_correct_predictions = arc_accuracy["correct_predictions"] + ko_arc_accuracy["correct_predictions"]
    total_extraction_failed = arc_accuracy["extraction_failed"] + ko_arc_accuracy["extraction_failed"]
    
    overall_accuracy_standard = (total_correct_predictions / total_valid_predictions * 100) if total_valid_predictions > 0 else 0
    overall_accuracy_strict = (total_correct_predictions / total_items * 100) if total_items > 0 else 0
    
    # Count changes
    extraction_changes = sum(1 for item in re_extracted_results if item['extraction_changed'])
    accuracy_improvements = sum(1 for item in re_extracted_results if item['accuracy_changed'] and item['new_correct'])
    accuracy_degradations = sum(1 for item in re_extracted_results if item['accuracy_changed'] and not item['new_correct'])
    
    # Print summary
    print("\n" + "="*80)
    print("RE-EXTRACTION RESULTS SUMMARY")
    print("="*80)
    print(f"OVERALL:")
    print(f"  Total Items: {total_items}")
    print(f"  Valid Predictions: {total_valid_predictions}")
    print(f"  Correct Predictions: {total_correct_predictions}")
    print(f"  Extraction Failed: {total_extraction_failed}")
    print(f"  Accuracy (Standard): {overall_accuracy_standard:.2f}%")
    print(f"  Accuracy (Strict): {overall_accuracy_strict:.2f}%")
    print()
    
    if arc_count > 0:
        print(f"ARC DATASET:")
        print(f"  Items: {arc_accuracy['total_items']}")
        print(f"  Valid Predictions: {arc_accuracy['valid_predictions']}")
        print(f"  Correct Predictions: {arc_accuracy['correct_predictions']}")
        print(f"  Extraction Failed: {arc_accuracy['extraction_failed']}")
        print(f"  Accuracy (Standard): {arc_accuracy['accuracy_standard']:.2f}%")
        print(f"  Accuracy (Strict): {arc_accuracy['accuracy_strict']:.2f}%")
        print()
    
    if ko_arc_count > 0:
        print(f"KO-ARC DATASET:")
        print(f"  Items: {ko_arc_accuracy['total_items']}")
        print(f"  Valid Predictions: {ko_arc_accuracy['valid_predictions']}")
        print(f"  Correct Predictions: {ko_arc_accuracy['correct_predictions']}")
        print(f"  Extraction Failed: {ko_arc_accuracy['extraction_failed']}")
        print(f"  Accuracy (Standard): {ko_arc_accuracy['accuracy_standard']:.2f}%")
        print(f"  Accuracy (Strict): {ko_arc_accuracy['accuracy_strict']:.2f}%")
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
        "overall": {
            "total_items": total_items,
            "valid_predictions": total_valid_predictions,
            "correct_predictions": total_correct_predictions,
            "extraction_failed": total_extraction_failed,
            "accuracy_standard": overall_accuracy_standard,
            "accuracy_strict": overall_accuracy_strict
        },
        "arc_dataset": arc_accuracy,
        "ko_arc_dataset": ko_arc_accuracy,
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
            if item['extraction_changed'] and changes_shown < show_examples:
                status = "✓" if item['new_correct'] and not item['old_correct'] else "✗" if not item['new_correct'] and item['old_correct'] else "="
                print(f"  {status} Index {item['index']}: '{item['old_extracted_answer']}' -> '{item['new_extracted_answer']}' (GT: {item['ground_truth']})")
                changes_shown += 1

def main():
    parser = argparse.ArgumentParser(description='Re-extract answers from JSON results and evaluate accuracy')
    parser.add_argument('json_file', help='Path to the JSON file containing raw_output data')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-e', '--examples', type=int, default=5, help='Number of example changes to show (default: 5)')
    parser.add_argument('-d', '--dataset', help='Specify dataset name (e.g., ARC, Ko-ARC)')
    
    args = parser.parse_args()
    
    re_evaluate_json_results(args.json_file, args.output, args.examples, args.dataset)

if __name__ == "__main__":
    main()