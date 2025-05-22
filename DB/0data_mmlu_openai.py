from datasets import load_dataset
import json
import os
import math

# Create directory
output_dir = "MMLU"
os.makedirs(output_dir, exist_ok=True)

# Load your token
with open(os.path.expanduser("~/.huggingface/token"), "r") as f:
    token = f.read().strip()

# Load the dataset
print("Loading dataset...")
ds = load_dataset("openai/MMMLU", "KO_KR", use_auth_token=token)

# Access the test split
test_data = ds['test']
print(f"Test split contains {len(test_data)} examples")
print(f"Example data: {test_data[0]}")

# Save the dataset in chunks of 1000 examples
batch_size = 1000
total_batches = math.ceil(len(test_data) / batch_size)

print(f"Saving dataset in {total_batches} files (1000 examples per file)...")

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(test_data))
    
    print(f"Processing file {batch_idx+1}/{total_batches} (examples {start_idx+1}-{end_idx})...")
    batch_data = [dict(test_data[i]) for i in range(start_idx, end_idx)]
    
    # Save to file named 1.json, 2.json, etc. (1-based index)
    file_name = f"{batch_idx+1}.json"
    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=4)
    
    print(f"Saved file {file_name} with {len(batch_data)} examples")

print("All files saved successfully!")