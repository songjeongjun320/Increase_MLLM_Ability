#!/usr/bin/env python3
"""
Create cached tokenized dataset for ToW training
This script pre-processes and saves the tokenized dataset to avoid repeated tokenization
"""

import os
import json
import torch
from pathlib import Path
import logging
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_cached_dataset(
    data_path="../4_tow_generation/tow_data/training_dataset_over_6_words.json",
    model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
    output_dir="cached_datasets",
    max_sequence_length=256
):
    """Create and save cached tokenized dataset"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add special tokens
    special_tokens = ["<ToW>", "</ToW>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} new ToW tokens to tokenizer")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Analyze optimal length
    logger.info("Analyzing optimal sequence length...")
    lengths = []
    processed_data = []
    
    for entry in tqdm(data, desc="Processing entries"):
        original_sentence = entry.get('original_sentence', '')
        context = entry.get('context', '')
        tow = entry.get('tow', '')

        if not original_sentence or not context or not tow:
            continue

        # Ensure context is a prefix of the original sentence
        if original_sentence.startswith(context):
            remaining_sentence = original_sentence[len(context):]
        else:
            continue
        
        input_text = context
        output_text = f"{tow}{remaining_sentence}"
        full_text = f"{input_text}{output_text}{tokenizer.eos_token}"
        
        # Analyze length
        tokens = tokenizer.tokenize(full_text)
        lengths.append(len(tokens))
        
        processed_data.append({
            "input": input_text,
            "full_text": full_text
        })
    
    # Determine optimal length
    lengths = np.array(lengths)
    logger.info(f"Length statistics:")
    logger.info(f"  Mean: {lengths.mean():.1f} tokens")
    logger.info(f"  Median: {np.median(lengths):.1f} tokens")
    logger.info(f"  95th percentile: {np.percentile(lengths, 95):.1f} tokens")
    logger.info(f"  Max: {lengths.max()} tokens")
    
    optimal_length = int(np.percentile(lengths, 98))
    optimal_length = max(256, min(optimal_length, tokenizer.model_max_length or 4096))
    logger.info(f"Using optimal length: {optimal_length} tokens")
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    
    def tokenize_function(examples):
        # Tokenize input
        input_tokens = tokenizer(
            examples['input'],
            add_special_tokens=False
        )
        
        # Tokenize full text
        full_tokens = tokenizer(
            examples['full_text'],
            truncation=True,
            max_length=optimal_length,
            padding='max_length',
            return_attention_mask=True
        )
        
        # Create labels
        all_labels = []
        for i in range(len(full_tokens['input_ids'])):
            input_ids = full_tokens['input_ids'][i]
            attention_mask = full_tokens['attention_mask'][i]
            input_len = len(input_tokens['input_ids'][i])
            
            labels = input_ids.copy()
            mask_len = min(input_len, len(labels))
            labels[:mask_len] = [-100] * mask_len
            
            # Mask padding tokens
            for j in range(len(labels)):
                if attention_mask[j] == 0:
                    labels[j] = -100
            
            all_labels.append(labels)
        
        full_tokens['labels'] = all_labels
        return full_tokens
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'full_text'],
        desc="Tokenizing and creating labels"
    )
    
    # Save tokenized dataset
    cache_file = output_dir / "tokenized_dataset.arrow"
    logger.info(f"Saving tokenized dataset to {cache_file}")
    tokenized_dataset.save_to_disk(str(cache_file))
    
    # Save metadata
    metadata = {
        "model_id": model_id,
        "max_sequence_length": optimal_length,
        "num_examples": len(tokenized_dataset),
        "special_tokens": special_tokens
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset cached successfully!")
    logger.info(f"  - Dataset: {cache_file}")
    logger.info(f"  - Metadata: {metadata_file}")
    logger.info(f"  - Examples: {len(tokenized_dataset)}")
    
    return tokenized_dataset, metadata

if __name__ == "__main__":
    create_cached_dataset()