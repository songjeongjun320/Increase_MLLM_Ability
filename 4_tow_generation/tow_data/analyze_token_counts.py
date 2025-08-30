import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def analyze_token_counts(file_path, model_names):
    """
    Analyzes the average token count for a given JSONL file across different models.

    Args:
        file_path (str): The path to the JSONL file.
        model_names (dict): A dictionary mapping model aliases to Hugging Face model identifiers.
    """
    tokenizers = {}
    for alias, model_name in model_names.items():
        print(f"Loading tokenizer for {alias} ({model_name})...")
        try:
            # trust_remote_code=True is needed for some models like Qwen and Gemma
            tokenizers[alias] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer for {model_name}: {e}")
            return

    total_tokens = {alias: 0 for alias in model_names}
    line_count = 0

    print(f"\nProcessing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'. Please check the path.")
        return

    # First, count the total lines for tqdm
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except Exception as e:
        print(f"Error reading file to get line count: {e}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Analyzing token counts"):
                try:
                    data = json.loads(line)
                    text = data.get("prompt", "") + data.get("completion", "")

                    if not text.strip():
                        continue

                    for alias, tokenizer in tokenizers.items():
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        total_tokens[alias] += len(tokens)
                    
                    line_count += 1
                except json.JSONDecodeError:
                    # Silently skip lines that are not valid JSON (e.g., empty lines)
                    pass
                except Exception as e:
                    print(f"An error occurred on line {line_count + 1}: {e}")
    except Exception as e:
        print(f"Error processing file: {e}")
        return


    if line_count > 0:
        print("\n--- Average Token Counts ---")
        for alias, total in total_tokens.items():
            average = total / line_count
            print(f"{alias.capitalize()}: {average:.2f} tokens")
        print("--------------------------")
    else:
        print("No valid lines were processed from the file.")

if __name__ == "__main__":
    # Define the models and their Hugging Face identifiers
    # Using smaller, but representative models for faster tokenizer downloads.
    models = {
        'qwen': 'Qwen/Qwen2-1.5B-Instruct',
        'llama': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'gemma': 'google/gemma-2b',
        'deepseek': 'deepseek-ai/deepseek-coder-1.3b-instruct'
    }

    # Path to the dataset file, relative to the project root.
    file_to_analyze = os.path.join('4_tow_generation', 'multiple_tow_data', 'final_multiple_tow.jsonl')

    analyze_token_counts(file_to_analyze, models)
