import json
import argparse
import os

def clean_dataset(input_file, output_file):
    """
    Reads a JSON or JSONL file, removes the trailing string "대화 상태:" 
    from the 'completion' field, and saves the cleaned data to a new file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    
    # Try to read as a single JSON object (likely a list of dicts)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        is_jsonl = False
    except json.JSONDecodeError:
        # If it fails, assume it's JSONL
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line.strip()}")
        is_jsonl = True

    if isinstance(data, dict):
        data = [data] # Handle case where JSON file is a single dictionary

    processed_count = 0
    for item in data:
        if 'completion' in item and isinstance(item['completion'], str):
            original_completion = item['completion']
            # rstrip to handle potential trailing whitespace
            if original_completion.rstrip().endswith("대화 상태:"):
                # Remove "대화 상태:" from the end
                end_index = original_completion.rfind("대화 상태:")
                item['completion'] = original_completion[:end_index].rstrip()
                processed_count += 1
    
    # Save the cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        if is_jsonl:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    print(f"Processing complete.")
    print(f"Total items processed: {len(data)}")
    print(f"Items modified: {processed_count}")
    print(f"Cleaned data saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean a JSON dataset by removing '대화 상태:' from the 'completion' field."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSON or JSONL file."
    )
    parser.add_argument(
        "--output_file",
        help="Path for the output file. If not provided, '_cleaned' will be appended to the input filename."
    )
    
    args = parser.parse_args()
    
    output_file = args.output_file
    if not output_file:
        file_dir, file_name = os.path.split(args.input_file)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join(file_dir, f"{name}_cleaned{ext}")

    clean_dataset(args.input_file, output_file)
