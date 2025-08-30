import json
import os
import re
from pathlib import Path
import tiktoken

def process_single_item(item):
    """
    Process a single JSON item to merge ToW with gold labels in the original sentence.
    """
    original_sentence = item['original_sentence']
    gold_labels = item['gold_label']
    tows = item['tows']
    
    # Filter out ToWs containing [ERROR] and check word count before each ToW
    filtered_data = []
    for i, (gold_label, tow) in enumerate(zip(gold_labels, tows)):
        if '[ERROR]' in tow:
            print(f"Warning: Skipping ToW with [ERROR]: {tow[:50]}...")
            continue
        
        # Find position of this gold_label in the sentence
        gold_pos = original_sentence.find(gold_label)
        if gold_pos == -1:
            print(f"Warning: Could not find '{gold_label}' in the sentence")
            continue
        
        # Count words before this position
        text_before = original_sentence[:gold_pos]
        words_before = text_before.split()
        
        if len(words_before) < 5:
            print(f"Warning: Skipping ToW with only {len(words_before)} words before it: '{gold_label}'")
            continue
        
        if len(words_before) > 15:
            print(f"Warning: Skipping ToW with too many ({len(words_before)}) words before it: '{gold_label}'")
            continue
        
        filtered_data.append((gold_label, tow))
    
    # If no valid ToWs were found after filtering, skip this item by returning None.
    if not filtered_data:
        return None, []
    
    # Build the completion by inserting ToWs at gold_label positions
    completion_parts = []
    current_pos = 0
    
    for gold_label, tow in filtered_data:
        # Find the gold label in the remaining part of the sentence
        gold_pos = original_sentence.find(gold_label, current_pos)
        
        if gold_pos == -1:
            print(f"Warning: Could not find '{gold_label}' in the sentence starting from position {current_pos}")
            continue
        
        # Add text from current position to the gold label
        completion_parts.append(original_sentence[current_pos:gold_pos])
        
        # Add the ToW
        completion_parts.append(tow)
        
        # Add the gold label
        completion_parts.append(gold_label)
        
        # Update current position
        current_pos = gold_pos + len(gold_label)
    
    # Add remaining text after the last gold label
    completion_parts.append(original_sentence[current_pos:])
    
    # Create the result with empty prompt and full completion
    completion = ''.join(completion_parts)
    
    processed_item = {
        "prompt": "",
        "completion": completion
    }

    tows_list = [tow for _, tow in filtered_data]
    return processed_item, tows_list

def process_json_file(file_path):
    """
    Process a single JSON file and return the processed items.
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_items = []
    tows_in_file = []
    
    for item in data:
        try:
            processed_item, item_tows = process_single_item(item)
            if processed_item is not None:
                processed_items.append(processed_item)
                tows_in_file.extend(item_tows)
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Processed {len(processed_items)} items from {file_path}")
    return processed_items, tows_in_file

def main():
    """
    Main function to process all JSON files and create the final JSONL file.
    """
    # Define the input directory and output file
    input_dir = Path("C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/4_tow_generation/multiple_tow_data")
    output_file = input_dir / "final_multiple_tow.jsonl"
    
    # Find all JSON files
    json_files = list(input_dir.glob("extract_over_19token_multiple_tow_gemini_2.5-flash_part*.json"))
    json_files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(json_files)} JSON files to process")
    
    all_processed_items = []
    all_tows = []
    
    # Process each file
    for json_file in json_files:
        processed_items, tows_from_file = process_json_file(json_file)
        all_processed_items.extend(processed_items)
        all_tows.extend(tows_from_file)

    # Analysis
    if all_processed_items:
        # Using cl100k_base tokenizer, commonly used for GPT models.
        enc = tiktoken.get_encoding("cl100k_base")
        
        total_tow_pairs = len(all_tows)
        total_tow_tokens = sum(len(enc.encode(tow)) for tow in all_tows)
        
        total_completion_chars = 0
        max_completion_tokens = 0
        
        for item in all_processed_items:
            completion = item['completion']
            total_completion_chars += len(completion)
            token_count = len(enc.encode(completion))
            if token_count > max_completion_tokens:
                max_completion_tokens = token_count
        
        print("\n--- Analysis Results ---")
        print(f"Total ToW pairs: {total_tow_pairs}")
        print(f"Total characters in completions: {total_completion_chars}")
        print(f"Maximum completion tokens: {max_completion_tokens}")
        print(f"Total tokens in ToWs: {total_tow_tokens}")
        print("------------------------\n")
    
    # Write to JSONL file
    print(f"Writing {len(all_processed_items)} items to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_processed_items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully created {output_file} with {len(all_processed_items)} items")

if __name__ == "__main__":
    main()