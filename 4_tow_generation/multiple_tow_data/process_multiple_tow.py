import json
import os
import re
from pathlib import Path

def process_single_item(item):
    """
    Process a single JSON item to merge ToW with gold labels in the original sentence.
    """
    original_sentence = item['original_sentence']
    gold_labels = item['gold_label']
    tows = item['tows']
    
    # Build the completion by inserting ToWs at gold_label positions
    completion_parts = []
    current_pos = 0
    
    for gold_label, tow in zip(gold_labels, tows):
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
    
    return {
        "prompt": "",
        "completion": completion
    }

def process_json_file(file_path):
    """
    Process a single JSON file and return the processed items.
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_items = []
    
    for item in data:
        try:
            processed_item = process_single_item(item)
            processed_items.append(processed_item)
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Processed {len(processed_items)} items from {file_path}")
    return processed_items

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
    
    # Process each file
    for json_file in json_files:
        processed_items = process_json_file(json_file)
        all_processed_items.extend(processed_items)
    
    # Write to JSONL file
    print(f"Writing {len(all_processed_items)} items to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_processed_items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully created {output_file} with {len(all_processed_items)} items")

if __name__ == "__main__":
    main()