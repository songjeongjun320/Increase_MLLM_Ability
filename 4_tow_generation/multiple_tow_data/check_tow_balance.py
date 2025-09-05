#!/usr/bin/env python3
"""
Check for ToW tag balance issues in the dataset.
This script will identify items with mismatched ToW tags.
"""

import json
import re
from pathlib import Path
from collections import Counter

def check_tow_balance(text):
    """
    Check if ToW tags are properly balanced in a text.
    
    Returns:
        Tuple of (is_balanced, open_count, close_count, issues)
    """
    open_tags = re.findall(r'<ToW>', text)
    close_tags = re.findall(r'</ToW>', text)
    
    open_count = len(open_tags)
    close_count = len(close_tags)
    is_balanced = open_count == close_count
    
    issues = []
    if not is_balanced:
        if open_count > close_count:
            issues.append(f"Missing {open_count - close_count} closing tags")
        else:
            issues.append(f"Missing {close_count - open_count} opening tags")
    
    # Check for unclosed ToW blocks
    tow_pattern = r'<ToW>.*?</ToW>'
    complete_blocks = len(re.findall(tow_pattern, text, flags=re.DOTALL))
    
    if complete_blocks != min(open_count, close_count):
        issues.append(f"Found {open_count} opening tags, {close_count} closing tags, but only {complete_blocks} complete blocks")
    
    return is_balanced, open_count, close_count, issues

def analyze_tow_balance(file_path):
    """
    Analyze ToW balance issues in the dataset.
    """
    print(f"Analyzing ToW balance in {file_path}...")
    
    total_items = 0
    balanced_items = 0
    unbalanced_items = 0
    
    open_tag_counts = []
    close_tag_counts = []
    balance_issues = Counter()
    
    problematic_items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                completion = item['completion']
                
                total_items += 1
                is_balanced, open_count, close_count, issues = check_tow_balance(completion)
                
                open_tag_counts.append(open_count)
                close_tag_counts.append(close_count)
                
                if is_balanced:
                    balanced_items += 1
                else:
                    unbalanced_items += 1
                    for issue in issues:
                        balance_issues[issue] += 1
                    
                    # Store first few problematic items for inspection
                    if len(problematic_items) < 10:
                        problematic_items.append({
                            'line': line_num,
                            'open_count': open_count,
                            'close_count': close_count,
                            'issues': issues,
                            'text_preview': completion[:200] + "..." if len(completion) > 200 else completion
                        })
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} items...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Calculate statistics
    print("\n" + "="*60)
    print("ToW BALANCE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total items processed: {total_items:,}")
    print(f"Balanced items: {balanced_items:,} ({balanced_items/total_items*100:.1f}%)")
    print(f"Unbalanced items: {unbalanced_items:,} ({unbalanced_items/total_items*100:.1f}%)")
    
    print(f"\nTotal <ToW> tags: {sum(open_tag_counts):,}")
    print(f"Total </ToW> tags: {sum(close_tag_counts):,}")
    print(f"Difference: {sum(open_tag_counts) - sum(close_tag_counts):,}")
    
    if balance_issues:
        print(f"\nBalance Issues:")
        for issue, count in balance_issues.most_common():
            print(f"  {issue}: {count:,} items")
    
    if problematic_items:
        print(f"\nFirst {len(problematic_items)} problematic items:")
        for i, item in enumerate(problematic_items, 1):
            print(f"\n{i}. Line {item['line']}:")
            print(f"   Open: {item['open_count']}, Close: {item['close_count']}")
            print(f"   Issues: {', '.join(item['issues'])}")
            print(f"   Preview: {item['text_preview']}")
    
    # Check for patterns in unbalanced items
    if unbalanced_items > 0:
        print(f"\nAnalyzing patterns in unbalanced items...")
        
        # Check if items end with unclosed ToW
        unclosed_at_end = 0
        unclosed_in_middle = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 1000:  # Sample first 1000 items
                    break
                    
                try:
                    item = json.loads(line.strip())
                    completion = item['completion']
                    
                    is_balanced, open_count, close_count, _ = check_tow_balance(completion)
                    if not is_balanced and open_count > close_count:
                        # Check if text ends with unclosed ToW
                        if completion.rstrip().endswith('<ToW>') or '<ToW>' in completion.split('</ToW>')[-1]:
                            unclosed_at_end += 1
                        else:
                            unclosed_in_middle += 1
                            
                except:
                    continue
        
        print(f"  Items with unclosed ToW at end: {unclosed_at_end}")
        print(f"  Items with unclosed ToW in middle: {unclosed_in_middle}")

def main():
    """
    Main function to check ToW balance.
    """
    # Check the original file
    input_file = Path("final_tow_dataset_context_large_iterative_filtered.jsonl")
    
    if not input_file.exists():
        print(f"Error: File not found - {input_file}")
        return
    
    analyze_tow_balance(input_file)

if __name__ == "__main__":
    main()
