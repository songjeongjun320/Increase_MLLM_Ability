#!/usr/bin/env python3
"""
Filter ToW tokens that appear too early in the completion text.
This script removes ToW content that starts within the first 10 words of the completion.
"""

import json
import re
import argparse
from typing import Dict, List, Tuple
from pathlib import Path


def count_words_before_first_tow(text: str) -> int:
    """
    Count the number of words before the first <ToW> tag in the text.
    
    Args:
        text: The completion text to analyze
        
    Returns:
        Number of words before the first <ToW> tag
    """
    # Find the position of the first <ToW> tag
    tow_match = re.search(r'<ToW>', text)
    if not tow_match:
        return len(text.split())  # No ToW found, return total word count
    
    # Get text before the first <ToW> tag
    text_before_tow = text[:tow_match.start()].strip()
    
    # Count words in the text before ToW
    words = text_before_tow.split()
    return len(words)


def remove_tow_content(text: str) -> str:
    """
    Remove all <ToW>...</ToW> content from the text.
    
    Args:
        text: The text containing ToW tags
        
    Returns:
        Text with all ToW content removed
    """
    # Remove all <ToW>...</ToW> blocks
    cleaned_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
    
    # Clean up any extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def remove_early_tow_blocks(text: str, threshold: int = 10) -> Tuple[str, int]:
    """
    Remove ToW blocks that start within the first 'threshold' words.
    Continue removing until we find a ToW that starts after 'threshold' words.
    
    Args:
        text: The completion text to process
        threshold: Word threshold for early ToW removal
        
    Returns:
        Tuple of (cleaned_text, number_of_blocks_removed)
    """
    import re
    
    current_text = text
    removed_count = 0
    
    while True:
        # Find the first <ToW> tag
        tow_start_match = re.search(r'<ToW>', current_text)
        
        if not tow_start_match:
            # No more ToW blocks found
            break
        
        tow_start = tow_start_match.start()
        
        # Count words before this ToW block
        text_before_tow = current_text[:tow_start].strip()
        words_before_tow = len(text_before_tow.split())
        
        # If this ToW starts within threshold, remove it
        if words_before_tow <= threshold:
            # Find the corresponding </ToW> tag
            tow_end_match = re.search(r'</ToW>', current_text[tow_start:])
            
            if tow_end_match:
                # Found matching closing tag
                tow_end = tow_start + tow_end_match.end()
                current_text = current_text[:tow_start] + current_text[tow_end:]
                removed_count += 1
            else:
                # No closing tag found, remove just the opening tag
                tow_end = tow_start + len('<ToW>')
                current_text = current_text[:tow_start] + current_text[tow_end:]
                removed_count += 1
        else:
            # Found a ToW that starts after threshold, stop removing
            break
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', current_text).strip()
    return cleaned_text, removed_count


def process_item(item: Dict, threshold: int = 10) -> Tuple[Dict, bool, int]:
    """
    Process a single item from the dataset.
    
    Args:
        item: Dictionary containing prompt and completion
        threshold: Word threshold for early ToW removal
        
    Returns:
        Tuple of (processed_item, was_modified, tow_blocks_removed)
    """
    completion = item.get('completion', '')
    
    # Remove early ToW blocks if any
    cleaned_completion, tow_blocks_removed = remove_early_tow_blocks(completion, threshold)
    
    was_modified = tow_blocks_removed > 0
    
    return {
        'prompt': item.get('prompt', ''),
        'completion': cleaned_completion
    }, was_modified, tow_blocks_removed


def main():
    parser = argparse.ArgumentParser(description='Filter ToW tokens that appear too early')
    parser.add_argument('--input', '-i', 
                       default='final_tow_dataset_refined.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o',
                       default='final_tow_dataset_context_large.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--threshold', '-t', type=int, default=10,
                       help='Word threshold for early ToW removal (default: 10)')
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
    print(f"Threshold: {args.threshold} words")
    print("-" * 50)
    
    processed_count = 0
    modified_count = 0
    removed_tow_blocks = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                processed_item, was_modified, tow_blocks_removed = process_item(item, args.threshold)
                
                if was_modified:
                    modified_count += 1
                    removed_tow_blocks += tow_blocks_removed
                    
                    if args.verbose:
                        words_before = count_words_before_first_tow(item.get('completion', ''))
                        print(f"Line {line_num}: Modified (words before ToW: {words_before}, "
                              f"ToW blocks removed: {tow_blocks_removed})")
                
                # Write processed item to output file
                outfile.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
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
    print(f"Items modified: {modified_count}")
    print(f"Total ToW blocks removed: {removed_tow_blocks}")
    print(f"Output saved to: {output_path}")
    
    # Calculate statistics
    if processed_count > 0:
        modification_rate = (modified_count / processed_count) * 100
        print(f"Modification rate: {modification_rate:.2f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())
