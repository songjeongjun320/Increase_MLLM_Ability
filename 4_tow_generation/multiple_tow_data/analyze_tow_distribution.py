import json
import re
from pathlib import Path

def count_words_before_first_tow(completion):
    """
    Count the number of words before the first <ToW> token in completion.
    """
    # Find the first <ToW> occurrence
    first_tow_match = re.search(r'<ToW>', completion)
    
    if not first_tow_match:
        # No <ToW> found, return total word count
        words = completion.split()
        return len(words), 0
    
    # Extract text before first <ToW>
    text_before_first_tow = completion[:first_tow_match.start()]
    
    # Count words (split by whitespace)
    words_before = text_before_first_tow.split()
    
    return len(words_before), first_tow_match.start()

def count_tow_tokens(completion):
    """
    Count total number of <ToW> and </ToW> tokens in completion.
    """
    tow_open_count = len(re.findall(r'<ToW>', completion))
    tow_close_count = len(re.findall(r'</ToW>', completion))
    
    return tow_open_count, tow_close_count

def analyze_jsonl_file(file_path):
    """
    Analyze the JSONL file to get statistics about ToW distribution.
    """
    print(f"Analyzing {file_path}...")
    
    total_items = 0
    total_words_before_first_tow = 0
    total_words_in_all_completions = 0
    total_tow_open = 0
    total_tow_close = 0
    
    items_with_tow = 0
    items_without_tow = 0
    
    word_counts_before_first_tow = []
    completion_lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                completion = item['completion']
                
                total_items += 1
                
                # Count total words in completion
                total_words = len(completion.split())
                total_words_in_all_completions += total_words
                completion_lengths.append(total_words)
                
                # Count words before first ToW
                words_before, first_tow_pos = count_words_before_first_tow(completion)
                
                # Count ToW tokens
                tow_open, tow_close = count_tow_tokens(completion)
                total_tow_open += tow_open
                total_tow_close += tow_close
                
                if tow_open > 0:
                    items_with_tow += 1
                    word_counts_before_first_tow.append(words_before)
                    total_words_before_first_tow += words_before
                else:
                    items_without_tow += 1
                
                # Print progress every 1000 items
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
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total items processed: {total_items:,}")
    print(f"Items with ToW tokens: {items_with_tow:,}")
    print(f"Items without ToW tokens: {items_without_tow:,}")
    
    print(f"\nToW Token Counts:")
    print(f"Total <tow> tokens: {total_tow_open:,}")
    print(f"Total </tow> tokens: {total_tow_close:,}")
    print(f"Average ToW pairs per item: {total_tow_open / total_items:.2f}")
    
    if items_with_tow > 0:
        avg_words_before_first_tow = total_words_before_first_tow / items_with_tow
        avg_completion_length = total_words_in_all_completions / total_items
        
        print(f"\nWord Distribution Analysis:")
        print(f"Average words before first ToW: {avg_words_before_first_tow:.2f}")
        print(f"Average total words per completion: {avg_completion_length:.2f}")
        print(f"Ratio (words before first ToW / total words): {avg_words_before_first_tow / avg_completion_length:.3f}")
        
        # Calculate percentiles for words before first ToW
        word_counts_before_first_tow.sort()
        n = len(word_counts_before_first_tow)
        
        percentiles = [10, 25, 50, 75, 90]
        print(f"\nPercentiles for words before first ToW:")
        for p in percentiles:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            print(f"  {p}th percentile: {word_counts_before_first_tow[idx]} words")
        
        print(f"  Min words before first ToW: {min(word_counts_before_first_tow)}")
        print(f"  Max words before first ToW: {max(word_counts_before_first_tow)}")
        
        # Distribution analysis
        print(f"\nDistribution Analysis:")
        ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for x in word_counts_before_first_tow if x >= start)
                print(f"  {start}+ words: {count:,} items ({count/items_with_tow*100:.1f}%)")
            else:
                count = sum(1 for x in word_counts_before_first_tow if start <= x <= end)
                print(f"  {start}-{end} words: {count:,} items ({count/items_with_tow*100:.1f}%)")

def main():
    """
    Main function to analyze the JSONL file.
    """
    # Define the input file
    input_file = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/4_tow_generation/multiple_tow_data/final_multiple_tow.jsonl")
    
    if not input_file.exists():
        print(f"Error: File not found - {input_file}")
        return
    
    analyze_jsonl_file(input_file)

if __name__ == "__main__":
    main()