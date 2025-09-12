import json
import re
from pathlib import Path
from collections import Counter

# Configuration flags - set to True/False based on your dataset
ANALYZE_TOW = True  # Set to True if your dataset contains <ToW> tokens
ANALYZE_HCOT = False  # Set to True if your dataset contains <hCoT> tokens

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

def count_hcot_tokens(completion):
    """
    Count total number of <hCoT> and </hCoT> tokens in completion.
    """
    hcot_open_count = len(re.findall(r'<hCoT>', completion))
    hcot_close_count = len(re.findall(r'</hCoT>', completion))
    
    return hcot_open_count, hcot_close_count

def remove_tow_content(completion):
    """
    Remove all content between <ToW> and </ToW> tokens, including the tokens themselves.
    Returns the cleaned text with only the original content.
    """
    # Remove everything between <ToW> and </ToW> (including the tags)
    cleaned = re.sub(r'<ToW>.*?</ToW>', '', completion, flags=re.DOTALL)
    return cleaned

def remove_hcot_content(completion):
    """
    Remove all content between <hCoT> and </hCoT> tokens, including the tokens themselves.
    Returns the cleaned text with only the original content.
    """
    # Remove everything between <hCoT> and </hCoT> (including the tags)
    cleaned = re.sub(r'<hCoT>.*?</hCoT>', '', completion, flags=re.DOTALL)
    return cleaned

def count_words_before_first_hcot(completion):
    """
    Count the number of words before the first <hCoT> token in completion.
    """
    # Find the first <hCoT> occurrence
    first_hcot_match = re.search(r'<hCoT>', completion)
    
    if not first_hcot_match:
        # No <hCoT> found, return total word count
        words = completion.split()
        return len(words), 0
    
    # Extract text before first <hCoT>
    text_before_first_hcot = completion[:first_hcot_match.start()]
    
    # Count words (split by whitespace)
    words_before = text_before_first_hcot.split()
    
    return len(words_before), first_hcot_match.start()

def estimate_token_length(text):
    """
    Estimate token length using a simple approximation.
    For more accurate tokenization, you might want to use transformers library.
    """
    # Simple approximation: 1 token ≈ 0.75 words for English
    # This is a rough estimate and may vary by tokenizer
    words = len(text.split())
    return int(words * 1.33)  # More conservative estimate

def create_dynamic_token_categories(max_tokens):
    """
    Create token length categories based on the maximum token length in the dataset.
    Provides detailed breakdown for 0-100 token range: 0-30, 31-60, 61-100.
    """
    if max_tokens <= 1000:
        return ["0-30", "31-60", "61-100", "101-200", "201-500", "501-1000", "1000+"]
    elif max_tokens <= 2000:
        return ["0-30", "31-60", "61-100", "101-200", "201-500", "501-1000", "1001-1500", "1501-2000", "2000+"]
    elif max_tokens <= 4000:
        return ["0-30", "31-60", "61-100", "101-200", "201-500", "501-1000", "1001-1500", "1501-2000", "2001-2500", "2501-3000", "3001-3500", "3501-4000", "4000+"]
    else:
        # For very large datasets, create more granular categories
        categories = ["0-30", "31-60", "61-100", "101-200", "201-500", "501-1000"]
        # Add 500-token increments up to max_tokens
        current = 1000
        while current < max_tokens:
            next_val = min(current + 500, max_tokens)
            categories.append(f"{current+1}-{next_val}")
            current = next_val
        categories.append(f"{max_tokens+1}+")
        return categories

def categorize_token_length(estimated_tokens, categories):
    """
    Categorize token length into the appropriate category.
    """
    for category in categories:
        if category.endswith("+"):
            # Handle the last category (e.g., "4000+")
            min_val = int(category[:-1])
            if estimated_tokens >= min_val:
                return category
        else:
            # Handle range categories (e.g., "1001-1500")
            parts = category.split("-")
            if len(parts) == 2:
                min_val, max_val = int(parts[0]), int(parts[1])
                if min_val <= estimated_tokens <= max_val:
                    return category
    return "unknown"

def analyze_jsonl_file(file_path):
    """
    Analyze the JSONL file to get statistics about ToW and hCoT distribution.
    """
    print(f"Analyzing {file_path}...")
    
    total_items = 0
    total_words_before_first_tow = 0
    total_words_before_first_hcot = 0
    total_words_in_all_completions = 0
    total_words_excluding_tow = 0
    total_words_excluding_hcot = 0
    total_tow_open = 0
    total_tow_close = 0
    total_hcot_open = 0
    total_hcot_close = 0
    
    # Analysis for samples with 100+ original tokens
    samples_100plus_original_tokens = 0
    total_tow_tokens_in_100plus_samples = 0
    tow_counts_in_100plus_samples = []
    
    items_with_tow = 0
    items_without_tow = 0
    items_with_hcot = 0
    items_without_hcot = 0
    
    word_counts_before_first_tow = []
    word_counts_before_first_hcot = []
    completion_lengths = []
    completion_lengths_excluding_tow = []
    completion_lengths_excluding_hcot = []
    token_lengths = []
    token_lengths_excluding_tow = []
    token_lengths_excluding_hcot = []
    token_length_distribution = Counter()
    token_length_distribution_excluding_tow = Counter()
    token_length_distribution_excluding_hcot = Counter()
    
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
                
                # Estimate token length
                estimated_tokens = estimate_token_length(completion)
                token_lengths.append(estimated_tokens)
                
                # Calculate lengths excluding ToW content
                if ANALYZE_TOW:
                    completion_without_tow = remove_tow_content(completion)
                    words_without_tow = len(completion_without_tow.split())
                    total_words_excluding_tow += words_without_tow
                    completion_lengths_excluding_tow.append(words_without_tow)
                    
                    estimated_tokens_without_tow = estimate_token_length(completion_without_tow)
                    token_lengths_excluding_tow.append(estimated_tokens_without_tow)
                    
                    # Check if original text has 100+ tokens
                    if estimated_tokens_without_tow >= 100:
                        samples_100plus_original_tokens += 1
                        tow_open, tow_close = count_tow_tokens(completion)
                        total_tow_tokens_in_100plus_samples += tow_open
                        tow_counts_in_100plus_samples.append(tow_open)
                
                # Calculate lengths excluding hCoT content
                if ANALYZE_HCOT:
                    completion_without_hcot = remove_hcot_content(completion)
                    words_without_hcot = len(completion_without_hcot.split())
                    total_words_excluding_hcot += words_without_hcot
                    completion_lengths_excluding_hcot.append(words_without_hcot)
                    
                    estimated_tokens_without_hcot = estimate_token_length(completion_without_hcot)
                    token_lengths_excluding_hcot.append(estimated_tokens_without_hcot)
                
                # Count words before first ToW (only if ToW analysis is enabled)
                if ANALYZE_TOW:
                    words_before_tow, first_tow_pos = count_words_before_first_tow(completion)
                    tow_open, tow_close = count_tow_tokens(completion)
                    total_tow_open += tow_open
                    total_tow_close += tow_close
                    
                    if tow_open > 0:
                        items_with_tow += 1
                        word_counts_before_first_tow.append(words_before_tow)
                        total_words_before_first_tow += words_before_tow
                    else:
                        items_without_tow += 1
                
                # Count words before first hCoT (only if hCoT analysis is enabled)
                if ANALYZE_HCOT:
                    words_before_hcot, first_hcot_pos = count_words_before_first_hcot(completion)
                    hcot_open, hcot_close = count_hcot_tokens(completion)
                    total_hcot_open += hcot_open
                    total_hcot_close += hcot_close
                    
                    if hcot_open > 0:
                        items_with_hcot += 1
                        word_counts_before_first_hcot.append(words_before_hcot)
                        total_words_before_first_hcot += words_before_hcot
                    else:
                        items_without_hcot += 1
                
                # Print progress every 1000 items
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} items...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Create dynamic token categories based on max token length
    max_tokens = max(token_lengths) if token_lengths else 0
    token_categories = create_dynamic_token_categories(max_tokens)
    
    # Categorize all token lengths
    for estimated_tokens in token_lengths:
        category = categorize_token_length(estimated_tokens, token_categories)
        token_length_distribution[category] += 1
    
    # Categorize token lengths excluding ToW content
    if ANALYZE_TOW and token_lengths_excluding_tow:
        max_tokens_excluding_tow = max(token_lengths_excluding_tow)
        token_categories_excluding_tow = create_dynamic_token_categories(max_tokens_excluding_tow)
        
        for estimated_tokens in token_lengths_excluding_tow:
            category = categorize_token_length(estimated_tokens, token_categories_excluding_tow)
            token_length_distribution_excluding_tow[category] += 1
    
    # Categorize token lengths excluding hCoT content
    if ANALYZE_HCOT and token_lengths_excluding_hcot:
        max_tokens_excluding_hcot = max(token_lengths_excluding_hcot)
        token_categories_excluding_hcot = create_dynamic_token_categories(max_tokens_excluding_hcot)
        
        for estimated_tokens in token_lengths_excluding_hcot:
            category = categorize_token_length(estimated_tokens, token_categories_excluding_hcot)
            token_length_distribution_excluding_hcot[category] += 1
    
    # Calculate statistics
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total items processed: {total_items:,}")
    
    if ANALYZE_TOW:
        print(f"Items with ToW tokens: {items_with_tow:,}")
        print(f"Items without ToW tokens: {items_without_tow:,}")
    
    if ANALYZE_HCOT:
        print(f"Items with hCoT tokens: {items_with_hcot:,}")
        print(f"Items without hCoT tokens: {items_without_hcot:,}")
    
    if ANALYZE_TOW:
        print(f"\nToW Token Counts:")
        print(f"Total <ToW> tokens: {total_tow_open:,}")
        print(f"Total </ToW> tokens: {total_tow_close:,}")
        print(f"Average ToW pairs per item: {total_tow_open / total_items:.2f}")
    
    if ANALYZE_HCOT:
        print(f"\nhCoT Token Counts:")
        print(f"Total <hCoT> tokens: {total_hcot_open:,}")
        print(f"Total </hCoT> tokens: {total_hcot_close:,}")
        print(f"Average hCoT pairs per item: {total_hcot_open / total_items:.2f}")
    
    # Define percentiles for use in both word and token analysis
    percentiles = [10, 25, 50, 75, 90]
    
    # Calculate average completion length (used by both ToW and hCoT analysis)
    avg_completion_length = total_words_in_all_completions / total_items
    
    if ANALYZE_TOW and items_with_tow > 0:
        avg_words_before_first_tow = total_words_before_first_tow / items_with_tow
        
        print(f"\nContext Length Before First ToW Analysis:")
        print(f"Average words before first ToW: {avg_words_before_first_tow:.2f}")
        print(f"Average total words per completion: {avg_completion_length:.2f}")
        print(f"Ratio (words before first ToW / total words): {avg_words_before_first_tow / avg_completion_length:.3f}")
        
        # Calculate percentiles for words before first ToW
        word_counts_before_first_tow.sort()
        n = len(word_counts_before_first_tow)
        print(f"\nPercentiles for Context Length Before First ToW:")
        for p in percentiles:
            idx = int(n * p / 100)
            if idx >= n:
                idx = n - 1
            print(f"  {p}th percentile: {word_counts_before_first_tow[idx]} words")
        
        print(f"  Min context length before first ToW: {min(word_counts_before_first_tow)} words")
        print(f"  Max context length before first ToW: {max(word_counts_before_first_tow)} words")
        
        # Distribution analysis
        print(f"\nContext Length Distribution Before First ToW:")
        ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for x in word_counts_before_first_tow if x >= start)
                print(f"  {start}+ words context: {count:,} items ({count/items_with_tow*100:.1f}%)")
            else:
                count = sum(1 for x in word_counts_before_first_tow if start <= x <= end)
                print(f"  {start}-{end} words context: {count:,} items ({count/items_with_tow*100:.1f}%)")
    
    # hCoT analysis
    if ANALYZE_HCOT and items_with_hcot > 0:
        avg_words_before_first_hcot = total_words_before_first_hcot / items_with_hcot
        
        print(f"\nContext Length Before First hCoT Analysis:")
        print(f"Average words before first hCoT: {avg_words_before_first_hcot:.2f}")
        print(f"Ratio (words before first hCoT / total words): {avg_words_before_first_hcot / avg_completion_length:.3f}")
        
        # Calculate percentiles for words before first hCoT
        word_counts_before_first_hcot.sort()
        n_hcot = len(word_counts_before_first_hcot)
        
        print(f"\nPercentiles for Context Length Before First hCoT:")
        for p in percentiles:
            idx = int(n_hcot * p / 100)
            if idx >= n_hcot:
                idx = n_hcot - 1
            print(f"  {p}th percentile: {word_counts_before_first_hcot[idx]} words")
        
        print(f"  Min context length before first hCoT: {min(word_counts_before_first_hcot)} words")
        print(f"  Max context length before first hCoT: {max(word_counts_before_first_hcot)} words")
        
        # Context length distribution analysis
        print(f"\nContext Length Distribution Before First hCoT:")
        ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for x in word_counts_before_first_hcot if x >= start)
                print(f"  {start}+ words context: {count:,} items ({count/items_with_hcot*100:.1f}%)")
            else:
                count = sum(1 for x in word_counts_before_first_hcot if start <= x <= end)
                print(f"  {start}-{end} words context: {count:,} items ({count/items_with_hcot*100:.1f}%)")
    
    # Token length analysis
    print(f"\nToken Length Analysis:")
    print(f"Average estimated tokens per completion: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"Min estimated tokens: {min(token_lengths)}")
    print(f"Max estimated tokens: {max(token_lengths)}")
    
    # Calculate percentiles for token lengths
    token_lengths.sort()
    n_tokens = len(token_lengths)
    
    print(f"\nToken Length Percentiles:")
    for p in percentiles:
        idx = int(n_tokens * p / 100)
        if idx >= n_tokens:
            idx = n_tokens - 1
        print(f"  {p}th percentile: {token_lengths[idx]} tokens")
    
    # Token length distribution (dynamic categories)
    print(f"\nToken Length Distribution (Dynamic Categories):")
    total_items_for_dist = sum(token_length_distribution.values())
    for category in token_categories:
        count = token_length_distribution[category]
        percentage = count / total_items_for_dist * 100
        print(f"  {category} tokens: {count:,} items ({percentage:.1f}%)")
    
    # Original text ratio analysis (excluding ToW content)
    if ANALYZE_TOW and token_lengths_excluding_tow:
        print(f"\nOriginal Text Analysis (Excluding ToW Content):")
        avg_tokens_excluding_tow = sum(token_lengths_excluding_tow) / len(token_lengths_excluding_tow)
        avg_tokens_total = sum(token_lengths) / len(token_lengths)
        
        print(f"Average estimated tokens (original text only): {avg_tokens_excluding_tow:.2f}")
        print(f"Average estimated tokens (total with ToW): {avg_tokens_total:.2f}")
        print(f"Original text ratio: {avg_tokens_excluding_tow / avg_tokens_total:.3f} ({avg_tokens_excluding_tow / avg_tokens_total * 100:.1f}%)")
        print(f"ToW content ratio: {(avg_tokens_total - avg_tokens_excluding_tow) / avg_tokens_total:.3f} ({(avg_tokens_total - avg_tokens_excluding_tow) / avg_tokens_total * 100:.1f}%)")
        
        # Token length percentiles for original text
        token_lengths_excluding_tow_sorted = sorted(token_lengths_excluding_tow)
        n_tokens_original = len(token_lengths_excluding_tow_sorted)
        
        print(f"\nOriginal Text Token Length Percentiles:")
        for p in percentiles:
            idx = int(n_tokens_original * p / 100)
            if idx >= n_tokens_original:
                idx = n_tokens_original - 1
            print(f"  {p}th percentile: {token_lengths_excluding_tow_sorted[idx]} tokens")
        
        print(f"  Min tokens (original): {min(token_lengths_excluding_tow)} tokens")
        print(f"  Max tokens (original): {max(token_lengths_excluding_tow)} tokens")
        
        # Original text token distribution
        if token_length_distribution_excluding_tow:
            print(f"\nOriginal Text Token Length Distribution:")
            total_items_original = sum(token_length_distribution_excluding_tow.values())
            for category in sorted(token_length_distribution_excluding_tow.keys()):
                count = token_length_distribution_excluding_tow[category]
                percentage = count / total_items_original * 100
                print(f"  {category} tokens: {count:,} items ({percentage:.1f}%)")
        
        # Analysis for samples with 100+ original tokens
        if samples_100plus_original_tokens > 0:
            print(f"\nAnalysis for Samples with 100+ Original Tokens:")
            print(f"Total samples with 100+ original tokens: {samples_100plus_original_tokens:,}")
            print(f"Percentage of total dataset: {samples_100plus_original_tokens / total_items * 100:.2f}%")
            print(f"Total ToW tokens in these samples: {total_tow_tokens_in_100plus_samples:,}")
            print(f"Average ToW tokens per sample (100+ original): {total_tow_tokens_in_100plus_samples / samples_100plus_original_tokens:.2f}")
            
            # ToW count distribution for 100+ samples
            tow_count_dist = {}
            for tow_count in tow_counts_in_100plus_samples:
                tow_count_dist[tow_count] = tow_count_dist.get(tow_count, 0) + 1
            
            print(f"\nToW Count Distribution (for 100+ original token samples):")
            for tow_count in sorted(tow_count_dist.keys()):
                count = tow_count_dist[tow_count]
                percentage = count / samples_100plus_original_tokens * 100
                print(f"  {tow_count} ToW tokens: {count:,} samples ({percentage:.1f}%)")
        else:
            print(f"\nNo samples found with 100+ original tokens.")
    
    # Original text ratio analysis (excluding hCoT content)  
    if ANALYZE_HCOT and token_lengths_excluding_hcot:
        print(f"\nOriginal Text Analysis (Excluding hCoT Content):")
        avg_tokens_excluding_hcot = sum(token_lengths_excluding_hcot) / len(token_lengths_excluding_hcot)
        avg_tokens_total = sum(token_lengths) / len(token_lengths)
        
        print(f"Average estimated tokens (original text only): {avg_tokens_excluding_hcot:.2f}")
        print(f"Average estimated tokens (total with hCoT): {avg_tokens_total:.2f}")
        print(f"Original text ratio: {avg_tokens_excluding_hcot / avg_tokens_total:.3f} ({avg_tokens_excluding_hcot / avg_tokens_total * 100:.1f}%)")
        print(f"hCoT content ratio: {(avg_tokens_total - avg_tokens_excluding_hcot) / avg_tokens_total:.3f} ({(avg_tokens_total - avg_tokens_excluding_hcot) / avg_tokens_total * 100:.1f}%)")
        
        # Token length percentiles for original text
        token_lengths_excluding_hcot_sorted = sorted(token_lengths_excluding_hcot)
        n_tokens_original_hcot = len(token_lengths_excluding_hcot_sorted)
        
        print(f"\nOriginal Text Token Length Percentiles (excluding hCoT):")
        for p in percentiles:
            idx = int(n_tokens_original_hcot * p / 100)
            if idx >= n_tokens_original_hcot:
                idx = n_tokens_original_hcot - 1
            print(f"  {p}th percentile: {token_lengths_excluding_hcot_sorted[idx]} tokens")
        
        print(f"  Min tokens (original): {min(token_lengths_excluding_hcot)} tokens")
        print(f"  Max tokens (original): {max(token_lengths_excluding_hcot)} tokens")
        
        # Original text token distribution
        if token_length_distribution_excluding_hcot:
            print(f"\nOriginal Text Token Length Distribution (excluding hCoT):")
            total_items_original_hcot = sum(token_length_distribution_excluding_hcot.values())
            for category in sorted(token_length_distribution_excluding_hcot.keys()):
                count = token_length_distribution_excluding_hcot[category]
                percentage = count / total_items_original_hcot * 100
                print(f"  {category} tokens: {count:,} items ({percentage:.1f}%)")
    
    # Training recommendations
    print(f"\nTraining Recommendations:")
    max_tokens = max(token_lengths)
    avg_tokens = sum(token_lengths) / len(token_lengths)
    
    print(f"  Maximum token length in dataset: {max_tokens}")
    print(f"  Average token length: {avg_tokens:.0f}")
    
    if max_tokens > 4000:
        print(f"  WARNING: Some samples exceed 4000 tokens. Consider:")
        print(f"     - Using models with larger context windows (8K+)")
        print(f"     - Truncating very long samples")
        print(f"     - Implementing dynamic batching")
    elif max_tokens > 2000:
        print(f"  INFO: Most samples fit within 4K context. Consider:")
        print(f"     - Standard 4K context models should work well")
        print(f"     - Monitor for any truncation issues")
    else:
        print(f"  OK: All samples fit within 2K context. Standard models should work well.")
    
    # Memory estimation for training
    print(f"\nMemory Estimation (rough):")
    print(f"  For batch size 1: ~{max_tokens * 2} tokens per sample")
    print(f"  For batch size 4: ~{max_tokens * 8} tokens per batch")
    print(f"  For batch size 8: ~{max_tokens * 16} tokens per batch")

def main():
    """
    Main function to analyze the JSONL file.
    """
    # Define the input file
    input_file = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/4_tow_generation/multiple_tow_data/tow_09_11.jsonl")
    
    if not input_file.exists():
        print(f"Error: File not found - {input_file}")
        return
    
    analyze_jsonl_file(input_file)

if __name__ == "__main__":
    main()