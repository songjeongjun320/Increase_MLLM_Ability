#!/usr/bin/env python3
"""
Filter out items with unbalanced ToW tags.
This script removes items where the number of <ToW> and </ToW> tags don't match.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple

def check_tow_balance(text: str) -> Tuple[bool, int, int]:
    """
    Check if ToW tags are balanced in a text.
    
    Args:
        text: The completion text to check
        
    Returns:
        Tuple of (is_balanced, open_count, close_count)
    """
    open_tags = re.findall(r'<ToW>', text)
    close_tags = re.findall(r'</ToW>', text)
    
    open_count = len(open_tags)
    close_count = len(close_tags)
    is_balanced = open_count == close_count
    
    return is_balanced, open_count, close_count

def process_item(item: Dict) -> Tuple[Dict, bool, int, int]:
    """
    Process a single item from the dataset.
    
    Args:
        item: Dictionary containing prompt and completion
        
    Returns:
        Tuple of (item, is_balanced, open_count, close_count)
    """
    completion = item.get('completion', '')
    is_balanced, open_count, close_count = check_tow_balance(completion)
    
    return item, is_balanced, open_count, close_count

def main():
    parser = argparse.ArgumentParser(description='Filter out items with unbalanced ToW tags')
    parser.add_argument('--input', '-i', 
                       default='final_tow_dataset_context_large_iterative.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o',
                       default='final_tow_dataset_context_large_iterative_filtered.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return 1
    
    print(f"Processing file: {input_path}")
    print(f"Output file: {output_path}")
    print("-" * 50)
    
    processed_count = 0
    balanced_count = 0
    unbalanced_count = 0
    total_opens = 0
    total_closes = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                processed_item, is_balanced, open_count, close_count = process_item(item)
                
                total_opens += open_count
                total_closes += close_count
                
                if is_balanced:
                    balanced_count += 1
                    # Write balanced item to output file
                    outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                else:
                    unbalanced_count += 1
                    if args.verbose:
                        print(f"Line {line_num}: Unbalanced (Open: {open_count}, Close: {close_count})")
                
                processed_count += 1
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} items...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print("-" * 50)
    print("PROCESSING COMPLETE")
    print(f"Total items processed: {processed_count}")
    print(f"Balanced items kept: {balanced_count}")
    print(f"Unbalanced items removed: {unbalanced_count}")
    print(f"Total <ToW> tags: {total_opens}")
    print(f"Total </ToW> tags: {total_closes}")
    print(f"Difference: {total_opens - total_closes}")
    print(f"Output saved to: {output_path}")
    
    # Calculate statistics
    if processed_count > 0:
        balance_rate = (balanced_count / processed_count) * 100
        print(f"Balance rate: {balance_rate:.2f}%")
    
    return 0

if __name__ == "__main__":
    exit(main())














