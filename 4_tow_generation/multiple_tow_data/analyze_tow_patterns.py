#!/usr/bin/env python3
"""
Analyze the specific patterns of ToW imbalance issues.
This script will examine the exact structure of problematic ToW blocks.
"""

import json
import re
from pathlib import Path

def analyze_tow_patterns(text, line_num):
    """
    Analyze the specific pattern of ToW issues in a text.
    
    Returns:
        Dictionary with pattern analysis
    """
    # Find all ToW opening and closing positions
    tow_opens = [(m.start(), m.end()) for m in re.finditer(r'<ToW>', text)]
    tow_closes = [(m.start(), m.end()) for m in re.finditer(r'</ToW>', text)]
    
    open_count = len(tow_opens)
    close_count = len(tow_closes)
    
    pattern_info = {
        'line': line_num,
        'open_count': open_count,
        'close_count': close_count,
        'opens': tow_opens,
        'closes': tow_closes,
        'text_length': len(text),
        'ends_with_tow': text.rstrip().endswith('<ToW>'),
        'ends_with_closed_tow': text.rstrip().endswith('</ToW>'),
        'last_50_chars': text[-50:] if len(text) > 50 else text,
        'patterns': []
    }
    
    # Check for specific patterns
    if open_count > close_count:
        # Case 1: Text ends with unclosed ToW
        if text.rstrip().endswith('<ToW>'):
            pattern_info['patterns'].append("ENDS_WITH_OPEN_TOW")
        
        # Case 2: ToW opens but never closes (check if last ToW is unclosed)
        if tow_opens and tow_closes:
            last_open_pos = tow_opens[-1][0]
            last_close_pos = tow_closes[-1][0]
            if last_open_pos > last_close_pos:
                pattern_info['patterns'].append("LAST_TOW_UNCLOSED")
        
        # Case 3: Multiple consecutive ToW opens without closes
        consecutive_opens = 0
        max_consecutive = 0
        for i, (open_pos, _) in enumerate(tow_opens):
            # Check if this open is followed by another open before any close
            if i < len(tow_opens) - 1:
                next_open_pos = tow_opens[i + 1][0]
                # Find closes between this open and next open
                closes_between = [pos for pos, _ in tow_closes if open_pos < pos < next_open_pos]
                if not closes_between:
                    consecutive_opens += 1
                    max_consecutive = max(max_consecutive, consecutive_opens)
                else:
                    consecutive_opens = 0
            else:
                # Last open - check if it has a close after it
                closes_after = [pos for pos, _ in tow_closes if open_pos < pos]
                if not closes_after:
                    consecutive_opens += 1
                    max_consecutive = max(max_consecutive, consecutive_opens)
        
        if max_consecutive > 1:
            pattern_info['patterns'].append(f"CONSECUTIVE_OPENS_{max_consecutive}")
    
    return pattern_info

def find_tow_sequences(text):
    """
    Find all ToW sequences and their structure.
    """
    # Find all ToW tags with their positions
    tow_tags = []
    for match in re.finditer(r'<ToW>|</ToW>', text):
        tow_tags.append((match.start(), match.group(), match.end()))
    
    sequences = []
    current_sequence = []
    
    for pos, tag, end_pos in tow_tags:
        if tag == '<ToW>':
            current_sequence.append(('OPEN', pos, end_pos))
        elif tag == '</ToW>':
            current_sequence.append(('CLOSE', pos, end_pos))
    
    return tow_tags, current_sequence

def analyze_problematic_items(file_path, max_items=20):
    """
    Analyze the first few problematic items in detail.
    """
    print(f"Analyzing ToW patterns in {file_path}...")
    
    problematic_items = []
    pattern_counts = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                completion = item['completion']
                
                # Check if this item has ToW imbalance
                tow_opens = len(re.findall(r'<ToW>', completion))
                tow_closes = len(re.findall(r'</ToW>', completion))
                
                if tow_opens != tow_closes:
                    pattern_info = analyze_tow_patterns(completion, line_num)
                    problematic_items.append(pattern_info)
                    
                    # Count patterns
                    for pattern in pattern_info['patterns']:
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    
                    if len(problematic_items) >= max_items:
                        break
                        
            except Exception as e:
                continue
    
    return problematic_items, pattern_counts

def main():
    """
    Main function to analyze ToW patterns.
    """
    # Analyze the original file
    input_file = Path("final_tow_dataset_context_large_iterative_filtered.jsonl")
    
    if not input_file.exists():
        print(f"Error: File not found - {input_file}")
        return
    
    problematic_items, pattern_counts = analyze_problematic_items(input_file, 20)
    
    print("\n" + "="*80)
    print("ToW PATTERN ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Analyzed {len(problematic_items)} problematic items")
    
    print(f"\nPattern Counts:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} items")
    
    print(f"\nDetailed Analysis of First {min(10, len(problematic_items))} Items:")
    print("="*80)
    
    for i, item in enumerate(problematic_items[:10], 1):
        print(f"\n{i}. Line {item['line']}:")
        print(f"   Open: {item['open_count']}, Close: {item['close_count']}")
        print(f"   Text length: {item['text_length']} chars")
        print(f"   Ends with ToW: {item['ends_with_tow']}")
        print(f"   Ends with closed ToW: {item['ends_with_closed_tow']}")
        print(f"   Patterns: {', '.join(item['patterns']) if item['patterns'] else 'None'}")
        print(f"   Last 50 chars: '{item['last_50_chars']}'")
        
        # Show ToW tag positions
        print(f"   ToW positions:")
        for j, (pos, tag_type) in enumerate(zip(item['opens'], ['OPEN'] * len(item['opens']))):
            print(f"     {tag_type} at {pos}")
        for j, (pos, tag_type) in enumerate(zip(item['closes'], ['CLOSE'] * len(item['closes']))):
            print(f"     {tag_type} at {pos}")

if __name__ == "__main__":
    main()
