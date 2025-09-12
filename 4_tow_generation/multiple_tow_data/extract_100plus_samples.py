import json
import re
from pathlib import Path

def remove_tow_content(completion):
    """
    Remove all content between <ToW> and </ToW> tokens, including the tokens themselves.
    Returns the cleaned text with only the original content.
    """
    cleaned = re.sub(r'<ToW>.*?</ToW>', '', completion, flags=re.DOTALL)
    return cleaned

def estimate_token_length(text):
    """
    Estimate token length using a simple approximation.
    """
    words = len(text.split())
    return int(words * 1.33)  # More conservative estimate

def extract_100plus_samples(input_file, output_file):
    """
    Extract samples where original text (excluding ToW content) has 100+ tokens.
    """
    print(f"Extracting 100+ original token samples from {input_file}...")
    
    extracted_count = 0
    total_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                item = json.loads(line.strip())
                completion = item['completion']
                total_processed += 1
                
                # Remove ToW content to get original text
                completion_without_tow = remove_tow_content(completion)
                
                # Estimate token length of original text
                estimated_tokens_without_tow = estimate_token_length(completion_without_tow)
                
                # If original text has 100+ tokens, keep this sample
                if estimated_tokens_without_tow >= 100:
                    outfile.write(line)
                    extracted_count += 1
                
                # Print progress every 1000 items
                if line_num % 1000 == 0:
                    print(f"Processed {line_num:,} items, extracted {extracted_count:,} samples so far...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total items processed: {total_processed:,}")
    print(f"Samples with 100+ original tokens: {extracted_count:,}")
    print(f"Extraction ratio: {extracted_count / total_processed * 100:.2f}%")
    print(f"Output saved to: {output_file}")

def main():
    """
    Main function to extract 100+ original token samples.
    """
    input_file = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/4_tow_generation/multiple_tow_data/tow_09_05.jsonl")
    output_file = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/4_tow_generation/multiple_tow_data/09_11.jsonl")
    
    if not input_file.exists():
        print(f"Error: Input file not found - {input_file}")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    extract_100plus_samples(input_file, output_file)

if __name__ == "__main__":
    main()