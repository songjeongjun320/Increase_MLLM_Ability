import json
import argparse
from tqdm import tqdm

def convert_to_prompt_completion(input_file_path, output_file_path):
    """
    Converts a JSON dataset with 'original_sentence', 'gold_label', and 'tow'
    into a JSONL file with 'prompt' and 'completion' format.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output JSONL file.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            original_data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Make sure it's a valid JSON file.")
        return

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        print(f"Converting {len(original_data)} records...")
        for record in tqdm(original_data, desc="Converting records"):
            original_sentence = record.get('original_sentence')
            gold_label = record.get('gold_label')
            tow = record.get('tow')

            if not all([original_sentence, gold_label, tow]):
                print(f"Skipping record due to missing data: {record.get('id', 'N/A')}")
                continue

            # Construct the new 'completion' string by inserting the 'tow'
            # content before the 'gold_label' in the 'original_sentence'.
            if gold_label in original_sentence:
                parts = original_sentence.split(gold_label, 1)
                # The structure is: text_before_gold_label + tow_content + gold_label + text_after_gold_label
                completion_text = f"{parts[0]}{tow}{gold_label}{parts[1]}"
            else:
                print(f"Warning: gold_label '{gold_label}' not found in original_sentence for ID {record.get('id', 'N/A')}. Skipping.")
                continue
            
            # Create the new data structure
            new_record = {
                "prompt": "",
                "completion": completion_text
            }

            # Write the new record as a line in the JSONL file
            outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')

    print(f"\nConversion complete. Output saved to '{output_file_path}'")


if __name__ == '__main__':
    # Set up argument parser to get file paths from the command line
    parser = argparse.ArgumentParser(
        description="Convert ToW JSON dataset to prompt/completion JSONL format."
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to the source JSON file."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path for the destination JSONL file."
    )

    args = parser.parse_args()

    # Run the conversion function
    convert_to_prompt_completion(args.input_file, args.output_file)