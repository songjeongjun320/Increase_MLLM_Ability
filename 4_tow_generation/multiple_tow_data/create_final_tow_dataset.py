import json
import os
import glob
from typing import List, Dict, Any

def find_word_positions(text: str, words: List[str]) -> List[tuple]:
    """
    Find the positions of words in the text and return them sorted by position.
    Returns list of tuples: (position, word_index, word)
    """
    positions = []
    
    for word_idx, word in enumerate(words):
        # Find all occurrences of the word
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            positions.append((pos, word_idx, word))
            start = pos + 1
    
    # Sort by position (first occurrence of each word)
    positions.sort()
    
    # For each word, keep only the first occurrence
    seen_words = set()
    filtered_positions = []
    
    for pos, word_idx, word in positions:
        if word_idx not in seen_words:
            filtered_positions.append((pos, word_idx, word))
            seen_words.add(word_idx)
    
    return filtered_positions

def insert_tows_in_sentence(original_sentence: str, gold_label: List[str], tows: List[str]) -> str:
    """
    Insert ToW elements into the original sentence at the positions of gold_label words.
    """
    if not gold_label or not tows or len(gold_label) != len(tows):
        return original_sentence
    
    # Find positions of gold_label words
    positions = find_word_positions(original_sentence, gold_label)
    
    if not positions:
        return original_sentence
    
    # Build the new sentence by inserting ToWs
    result = ""
    last_pos = 0
    
    for pos, word_idx, word in positions:
        # Add text before the word
        result += original_sentence[last_pos:pos]
        
        # Add the ToW
        result += tows[word_idx]
        
        # Add the word itself
        result += word
        
        last_pos = pos + len(word)
    
    # Add remaining text
    result += original_sentence[last_pos:]
    
    return result

def process_json_files(input_folder: str, output_file: str):
    """
    Process all JSON files in the input folder and create the final dataset.
    """
    # Find all JSON files matching the pattern
    pattern = os.path.join(input_folder, "extract_over_19token_multiple_tow_gemini_2.5-flash_*.json")
    json_files = glob.glob(pattern)
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    all_entries = []
    
    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing {os.path.basename(json_file)}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  Loaded {len(data)} entries")
            
            for entry in data:
                # Extract required fields
                original_sentence = entry.get('original_sentence', '')
                gold_label = entry.get('gold_label', [])
                tows = entry.get('tows', [])
                
                # Skip if required fields are missing
                if not original_sentence or not gold_label or not tows:
                    continue
                
                # Skip if lengths don't match
                if len(gold_label) != len(tows):
                    print(f"    Warning: Mismatch in entry {entry.get('id', 'unknown')} - gold_label length: {len(gold_label)}, tows length: {len(tows)}")
                    print(f"      gold_label: {gold_label}")
                    print(f"      tows count: {len(tows)}")
                    continue
                
                # Create the modified sentence with ToWs inserted
                completion = insert_tows_in_sentence(original_sentence, gold_label, tows)
                
                # Create the final entry
                final_entry = {
                    "prompt": "",
                    "completion": completion
                }
                
                all_entries.append(final_entry)
        
        except Exception as e:
            print(f"  Error processing {json_file}: {str(e)}")
            continue
    
    print(f"\nTotal entries processed: {len(all_entries)}")
    
    # Write to JSONL file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully created {output_file} with {len(all_entries)} entries")

def main():
    # Set paths
    input_folder = "."  # Current directory (multiple_tow_data)
    output_file = "final_tow_dataset.jsonl"
    
    print("Starting ToW dataset creation...")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print(f"Output file: {os.path.abspath(output_file)}")
    
    # Process the files
    process_json_files(input_folder, output_file)
    
    print("\nDataset creation completed!")

if __name__ == "__main__":
    main()
