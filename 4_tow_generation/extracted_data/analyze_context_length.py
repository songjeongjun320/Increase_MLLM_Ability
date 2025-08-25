import json
import os
from collections import defaultdict
import argparse

def count_tokens_simple(text):
    """Simple token counting (space-based)"""
    return len(text.split()) if text else 0

def count_characters(text):
    """Character count"""
    return len(text) if text else 0

def analyze_context_lengths(json_file, min_char_threshold=10, min_token_threshold=3):
    """Analyze context lengths from JSON file and provide statistics"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n=== Analysis Results for {os.path.basename(json_file)} ===")
    print(f"Total data count: {len(data)}")
    
    # Collect statistics
    char_counts = []
    token_counts = []
    context_lengths = defaultdict(int)
    
    for item in data:
        context = item.get('context', '')
        char_count = count_characters(context)
        token_count = count_tokens_simple(context)
        
        char_counts.append(char_count)
        token_counts.append(token_count)
        context_lengths[char_count] += 1
    
    # Basic statistics
    print(f"\n--- Character Count Statistics ---")
    print(f"Minimum: {min(char_counts)} characters")
    print(f"Maximum: {max(char_counts)} characters")
    print(f"Average: {sum(char_counts)/len(char_counts):.1f} characters")
    print(f"Median: {sorted(char_counts)[len(char_counts)//2]} characters")
    
    print(f"\n--- Token Count Statistics ---")
    print(f"Minimum: {min(token_counts)} tokens")
    print(f"Maximum: {max(token_counts)} tokens")
    print(f"Average: {sum(token_counts)/len(token_counts):.1f} tokens")
    print(f"Median: {sorted(token_counts)[len(token_counts)//2]} tokens")
    
    # Filtering criteria analysis
    print(f"\n--- Filtering Criteria Analysis ---")
    
    # Character count criteria
    removed_by_char = sum(1 for count in char_counts if count < min_char_threshold)
    remaining_by_char = len(data) - removed_by_char
    
    print(f"Character count less than {min_char_threshold}:")
    print(f"  - Data to be removed: {removed_by_char} items ({removed_by_char/len(data)*100:.1f}%)")
    print(f"  - Data to remain: {remaining_by_char} items ({remaining_by_char/len(data)*100:.1f}%)")
    
    # Token count criteria
    removed_by_token = sum(1 for count in token_counts if count < min_token_threshold)
    remaining_by_token = len(data) - removed_by_token
    
    print(f"Token count less than {min_token_threshold}:")
    print(f"  - Data to be removed: {removed_by_token} items ({removed_by_token/len(data)*100:.1f}%)")
    print(f"  - Data to remain: {remaining_by_token} items ({remaining_by_token/len(data)*100:.1f}%)")
    
    # Distribution by range
    print(f"\n--- Character Count Distribution by Range ---")
    ranges = [(0, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, float('inf'))]
    for start, end in ranges:
        count = sum(1 for c in char_counts if start <= c < end or (end == float('inf') and c >= start))
        percentage = count / len(data) * 100
        range_str = f"{start}-{end if end != float('inf') else '+'}"
        print(f"  {range_str:>8} chars: {count:>4} items ({percentage:>5.1f}%)")
    
    print(f"\n--- Token Count Distribution by Range ---")
    token_ranges = [(0, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, float('inf'))]
    for start, end in token_ranges:
        count = sum(1 for c in token_counts if start <= c < end or (end == float('inf') and c >= start))
        percentage = count / len(data) * 100
        range_str = f"{start}-{end if end != float('inf') else '+'}"
        print(f"  {range_str:>8} tokens: {count:>4} items ({percentage:>5.1f}%)")
    
    return {
        'total': len(data),
        'char_stats': {
            'min': min(char_counts),
            'max': max(char_counts),
            'avg': sum(char_counts)/len(char_counts),
            'median': sorted(char_counts)[len(char_counts)//2]
        },
        'token_stats': {
            'min': min(token_counts),
            'max': max(token_counts),
            'avg': sum(token_counts)/len(token_counts),
            'median': sorted(token_counts)[len(token_counts)//2]
        },
        'filtering': {
            'char_threshold': min_char_threshold,
            'token_threshold': min_token_threshold,
            'removed_by_char': removed_by_char,
            'remaining_by_char': remaining_by_char,
            'removed_by_token': removed_by_token,
            'remaining_by_token': remaining_by_token
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze context lengths in JSON files')
    parser.add_argument('--dir', default='tow_data', help='Directory containing JSON files (default: tow_data)')
    parser.add_argument('--char-threshold', type=int, default=10, help='Minimum character count threshold (default: 10)')
    parser.add_argument('--token-threshold', type=int, default=3, help='Minimum token count threshold (default: 3)')
    parser.add_argument('--file', help='Analyze specific file only (optional)')
    
    args = parser.parse_args()
    
    if args.file:
        # Analyze specific file only
        json_files = [args.file]
    else:
        # Analyze all JSON files in directory
        json_files = [f for f in os.listdir(args.dir) if f.endswith('.json')]
        json_files = [os.path.join(args.dir, f) for f in json_files]
    
    if not json_files:
        print("No JSON files found.")
        return
    
    print(f"Character count threshold: {args.char_threshold} characters")
    print(f"Token count threshold: {args.token_threshold} tokens")
    print("="*60)
    
    all_results = {}
    
    for json_file in json_files:
        try:
            result = analyze_context_lengths(json_file, args.char_threshold, args.token_threshold)
            all_results[json_file] = result
        except Exception as e:
            print(f"Error occurred ({json_file}): {e}")
    
    # Overall summary
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("=== Overall Summary ===")
        total_data = sum(r['total'] for r in all_results.values())
        total_removed_char = sum(r['filtering']['removed_by_char'] for r in all_results.values())
        total_removed_token = sum(r['filtering']['removed_by_token'] for r in all_results.values())
        
        print(f"Total files: {len(all_results)}")
        print(f"Total data count: {total_data:,}")
        print(f"\nWhen removing data with character count less than {args.char_threshold}:")
        print(f"  - Data to be removed: {total_removed_char:,} items ({total_removed_char/total_data*100:.1f}%)")
        print(f"  - Data to remain: {total_data-total_removed_char:,} items ({(total_data-total_removed_char)/total_data*100:.1f}%)")
        print(f"\nWhen removing data with token count less than {args.token_threshold}:")
        print(f"  - Data to be removed: {total_removed_token:,} items ({total_removed_token/total_data*100:.1f}%)")
        print(f"  - Data to remain: {total_data-total_removed_token:,} items ({(total_data-total_removed_token)/total_data*100:.1f}%)")

if __name__ == "__main__":
    main()