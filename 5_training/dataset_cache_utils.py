#!/usr/bin/env python3
"""
Common dataset caching utilities for ToW training
Provides model-specific caching to avoid re-tokenization
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datasets import Dataset
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class DatasetCacheManager:
    """Manages caching of tokenized datasets for different models"""
    
    def __init__(self, base_cache_dir: str = "cached_datasets"):
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, model_id: str, data_paths: List[str], max_length: int) -> str:
        """Generate unique cache key based on model and data configuration"""
        # Create a hash of the model ID and data paths for unique identification
        content = f"{model_id}:{':'.join(sorted(data_paths))}:{max_length}"
        cache_key = hashlib.md5(content.encode()).hexdigest()[:12]
        return cache_key
    
    def get_cache_dir(self, model_id: str, data_paths: List[str], max_length: int) -> Path:
        """Get cache directory for specific model and data configuration"""
        cache_key = self.get_cache_key(model_id, data_paths, max_length)
        model_name = Path(model_id).name  # Extract just the model name
        cache_dir = self.base_cache_dir / f"{model_name}_{cache_key}"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
    
    def load_cached_dataset(self, model_id: str, data_paths: List[str], max_length: int) -> Optional[Dataset]:
        """Load cached tokenized dataset if available"""
        cache_dir = self.get_cache_dir(model_id, data_paths, max_length)
        cache_file = cache_dir / "tokenized_dataset.arrow"
        metadata_file = cache_dir / "dataset_metadata.json"
        
        if cache_file.exists() and metadata_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            try:
                # Load and validate metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Verify cache is still valid
                if (metadata.get('model_id') == model_id and 
                    metadata.get('max_sequence_length') == max_length and
                    set(metadata.get('data_paths', [])) == set(data_paths)):
                    
                    # Load dataset
                    cached_dataset = Dataset.load_from_disk(str(cache_file))
                    logger.info(f"Successfully loaded cached dataset with {len(cached_dataset)} examples")
                    logger.info(f"Cache key: {self.get_cache_key(model_id, data_paths, max_length)}")
                    return cached_dataset
                else:
                    logger.warning("Cache metadata doesn't match current configuration. Will recreate cache.")
                    return None
                    
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Will create new dataset.")
                return None
        else:
            logger.info(f"No cached dataset found for {Path(model_id).name}. Will create and cache new dataset.")
            return None
    
    def save_cached_dataset(self, dataset: Dataset, model_id: str, data_paths: List[str], 
                          max_length: int, metadata: Dict[str, Any]) -> bool:
        """Save tokenized dataset to cache"""
        try:
            cache_dir = self.get_cache_dir(model_id, data_paths, max_length)
            cache_file = cache_dir / "tokenized_dataset.arrow"
            metadata_file = cache_dir / "dataset_metadata.json"
            
            # Save dataset
            dataset.save_to_disk(str(cache_file))
            
            # Add cache info to metadata
            metadata.update({
                "model_id": model_id,
                "data_paths": data_paths,
                "max_sequence_length": max_length,
                "cache_key": self.get_cache_key(model_id, data_paths, max_length),
                "num_examples": len(dataset),
            })
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully cached dataset to {cache_dir}")
            logger.info(f"Cache key: {metadata['cache_key']}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
            return False
    
    def clear_cache(self, model_id: str = None, data_paths: List[str] = None, max_length: int = None):
        """Clear cache for specific model or all caches"""
        if model_id and data_paths and max_length:
            # Clear specific cache
            cache_dir = self.get_cache_dir(model_id, data_paths, max_length)
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache for {Path(model_id).name}")
        else:
            # Clear all caches
            if self.base_cache_dir.exists():
                import shutil
                shutil.rmtree(self.base_cache_dir)
                self.base_cache_dir.mkdir(exist_ok=True)
                logger.info("Cleared all dataset caches")
    
    def list_caches(self) -> List[Dict[str, Any]]:
        """List all available caches"""
        caches = []
        
        for cache_dir in self.base_cache_dir.iterdir():
            if cache_dir.is_dir():
                metadata_file = cache_dir / "dataset_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        cache_info = {
                            "cache_dir": str(cache_dir),
                            "model_name": Path(metadata.get('model_id', '')).name,
                            "num_examples": metadata.get('num_examples', 0),
                            "max_length": metadata.get('max_sequence_length', 0),
                            "cache_key": metadata.get('cache_key', ''),
                        }
                        caches.append(cache_info)
                    except:
                        continue
        
        return caches


class SmartToWDataProcessor:
    """Smart data processor with model-specific caching"""
    
    def __init__(self, tokenizer, config, model_id: str):
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
        self.cache_manager = DatasetCacheManager()
    
    def analyze_data_lengths(self, data: List[Dict]) -> int:
        """Analyze data to determine optimal max length"""
        logger.info("Analyzing dataset lengths...")
        
        lengths = []
        
        for entry in tqdm(data, desc="Analyzing lengths"):
            input_text = entry.get('input', '')
            output_text = entry.get('output', '')
            text = f"{input_text}{output_text}{self.tokenizer.eos_token}"
            tokens = self.tokenizer.tokenize(text)
            lengths.append(len(tokens))
        
        lengths = np.array(lengths)
        
        if len(lengths) == 0:
            logger.warning("No data to analyze for lengths. Using default max_sequence_length.")
            return self.config.max_sequence_length

        logger.info(f"Length statistics:")
        logger.info(f"  Mean: {lengths.mean():.1f} tokens")
        logger.info(f"  Median: {np.median(lengths):.1f} tokens")
        logger.info(f"  95th percentile: {np.percentile(lengths, 95):.1f} tokens")
        logger.info(f"  Max: {lengths.max()} tokens")
        
        optimal_length = int(np.percentile(lengths, 98))
        optimal_length = max(256, min(optimal_length, self.tokenizer.model_max_length or 4096))
        
        logger.info(f"Setting adaptive max length to: {optimal_length} tokens")
        return optimal_length
    
    def _load_data(self, data_paths: List[str]) -> List[Dict]:
        """Load dataset from json or jsonl files."""
        logger.info(f"Loading data from {len(data_paths)} files...")
        
        all_data = []
        for data_path in data_paths:
            path = Path(data_path)
            if not path.exists():
                logger.warning(f"Data file not found, skipping: {data_path}")
                continue
            
            logger.info(f"  - Loading from {data_path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.suffix == ".jsonl":
                        for line in f:
                            all_data.append(json.loads(line))
                    elif path.suffix == ".json":
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            logger.warning(f"Data in {data_path} is not a list, skipping.")
                    else:
                        logger.warning(f"Unsupported file type {path.suffix} for {data_path}. Skipping.")
            except json.JSONDecodeError as e:
                logger.warning(f"Could not decode JSON from {data_path}: {e}. Skipping line or file.")
        
        logger.info(f"Loaded a total of {len(all_data)} entries")
        return all_data

    def create_training_dataset(self, data_paths: List[str]) -> Dataset:
        """Create dataset with intelligent caching"""
        logger.info("Creating training dataset for converted ToW training...")
        
        # Load data first to determine optimal length if needed
        data = self._load_data(data_paths)
        
        if not data:
            logger.error("No data loaded. Aborting dataset creation.")
            return None

        if self.config.adaptive_max_length:
            optimal_length = self.analyze_data_lengths(data)
            self.config.max_sequence_length = optimal_length
        
        # Try to load cached dataset
        cached_dataset = self.cache_manager.load_cached_dataset(
            self.model_id, data_paths, self.config.max_sequence_length
        )
        if cached_dataset is not None:
            return cached_dataset
        
        # Process data assuming 'prompt'/'completion' format
        processed_data = []
        logger.info("Processing data in 'prompt'/'completion' format.")
        for entry in tqdm(data, desc="Processing prompt/completion data"):
            prompt = entry.get('prompt', '')
            completion = entry.get('completion', '')
            if not completion:
                logger.warning(f"Skipping entry with empty completion: {entry}")
                continue
            
            # If prompt is None, treat it as an empty string
            prompt = prompt if prompt is not None else ''

            full_text = f"{prompt}{completion}{self.tokenizer.eos_token}"
            processed_data.append({"input": prompt, "full_text": full_text})
        
        logger.info(f"Created {len(processed_data)} training examples.")
        
        # Create and tokenize dataset
        dataset = Dataset.from_list(processed_data)
        tokenized_dataset = self._tokenize_dataset(dataset)
        
        # Cache the tokenized dataset
        metadata = {
            "special_tokens": [self.tow_start_token, self.tow_end_token],
            "tokenizer_name": getattr(self.tokenizer, 'name_or_path', 'unknown'),
        }
        
        self.cache_manager.save_cached_dataset(
            tokenized_dataset, self.model_id, data_paths, 
            self.config.max_sequence_length, metadata
        )
        
        return tokenized_dataset
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset with proper labeling"""
        def tokenize_function(examples):
            # Temporarily set truncation side to 'left' if supported
            original_truncation_side = getattr(self.tokenizer, 'truncation_side', None)
            if hasattr(self.tokenizer, 'truncation_side'):
                self.tokenizer.truncation_side = 'left'
            
            input_tokens = self.tokenizer(
                examples['input'],
                add_special_tokens=False
            )
            full_tokens = self.tokenizer(
                examples['full_text'],
                truncation=True,
                max_length=self.config.max_sequence_length,
                padding='max_length',
                return_attention_mask=True
            )
            
            # Restore original truncation side
            if original_truncation_side is not None:
                self.tokenizer.truncation_side = original_truncation_side

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
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['input', 'full_text'],
            desc="Tokenizing and creating labels"
        )
        
        logger.info("Checking tokenized sample lengths...")
        for i in range(min(3, len(tokenized_dataset))):
            input_ids_len = len(tokenized_dataset[i]['input_ids'])
            labels_len = len(tokenized_dataset[i]['labels'])
            actual_labels = sum(1 for label in tokenized_dataset[i]['labels'] if label != -100)
            logger.info(f"Sample {i} - input_ids: {input_ids_len}, labels: {labels_len}, actual_labels: {actual_labels}")

        return tokenized_dataset