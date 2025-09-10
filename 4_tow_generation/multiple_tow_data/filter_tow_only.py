#!/usr/bin/env python3
"""
Filter out items that don't contain any ToW tags.
This script keeps only items that have at least one <ToW> tag.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple

def has_tow_tags(text: str) -> bool:
    """
    Check if text contains any ToW tags.
    
    Args:
        text: The completion text to check
        
    Returns:
        True if text contains at least one <ToW> tag, False otherwise
    """
    return bool(re.search(r'<ToW>', text))

def process_item(item: Dict) -> Tuple[Dict, bool]:
    """
    Process a single item from the dataset.
    
    Args:
        item: Dictionary containing prompt and completion
        
    Returns:
        Tuple of (item, has_tow)
    """
    completion = item.get('completion', '')
    has_tow = has_tow_tags(completion)
    
    return item, has_tow

def main():
    parser = argparse.ArgumentParser(description='Filter out items without ToW tags')
    parser.add_argument('--input', '-i', 
                       default='final_tow_dataset_context_large_iterative_filtered.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o',
                       default='final_tow_dataset_09_05.jsonl',
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
    with_tow_count = 0
    without_tow_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                processed_item, has_tow = process_item(item)
                
                if has_tow:
                    with_tow_count += 1
                    # Write item with ToW tags to output file
                    outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                else:
                    without_tow_count += 1
                    if args.verbose:
                        print(f"Line {line_num}: No ToW tags")
                
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
    print(f"Items with ToW tags kept: {with_tow_count}")
    print(f"Items without ToW tags removed: {without_tow_count}")
    print(f"Output saved to: {output_path}")
    
    # Calculate statistics
    if processed_count > 0:
        tow_rate = (with_tow_count / processed_count) * 100
        print(f"ToW retention rate: {tow_rate:.2f}%")
    
    return 0

if __name__ == "__main__":
    exit(main())











