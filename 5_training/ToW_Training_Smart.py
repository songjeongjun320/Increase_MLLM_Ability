#!/usr/bin/env python3
"""
ToW Training with Smart Text Handling
- Adaptive max length based on data analysis
- ToW token preservation
- Smart chunking for long texts
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
from tqdm import tqdm

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ModelConfig:
    name: str
    model_id: str
    use_quantization: bool = False
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

@dataclass
class ToWTrainingConfig:
    """ToW training config with smart text handling"""
    tow_data_path: str = "ToW_koconovel_complete.json"
    output_base_dir: str = "ToW_Models"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Smart text handling
    adaptive_max_length: bool = True  # 데이터에 맞춰 최대 길이 조정
    preserve_tow_tokens: bool = True  # ToW 토큰 보존 우선
    enable_chunking: bool = True      # 긴 텍스트 청킹 활성화
    min_chunk_overlap: int = 50       # 청크간 겹치는 토큰 수
    
    # Default settings
    max_sequence_length: int = 1024   # 기본값, 적응형으로 조정됨
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Other settings
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True

MODEL_CONFIGS = [
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),
]

class SmartToWDataProcessor:
    """Smart data processor with adaptive length and ToW preservation"""
    
    def __init__(self, tokenizer, config: ToWTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
        
        # Add ToW tokens
        special_tokens = [self.tow_start_token, self.tow_end_token]
        new_tokens = [token for token in special_tokens if token not in self.tokenizer.get_vocab()]
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new ToW tokens to tokenizer")
    
    def analyze_data_lengths(self, data: List[Dict]) -> int:
        """Analyze data to determine optimal max length"""
        logger.info("Analyzing dataset lengths...")
        
        # Sample data for analysis (use first 200 entries)
        sample_data = [entry for entry in data[:200] if entry['tow_count'] > 0]
        lengths = []
        
        for entry in tqdm(sample_data, desc="Analyzing lengths"):
            text = entry['augmented_text']
            tokens = self.tokenizer.tokenize(text)
            lengths.append(len(tokens))
        
        lengths = np.array(lengths)
        
        # Statistics
        logger.info(f"Length statistics:")
        logger.info(f"  Mean: {lengths.mean():.1f} tokens")
        logger.info(f"  Median: {np.median(lengths):.1f} tokens")
        logger.info(f"  95th percentile: {np.percentile(lengths, 95):.1f} tokens")
        logger.info(f"  Max: {lengths.max()} tokens")
        
        # Choose length that covers 95% of data
        optimal_length = int(np.percentile(lengths, 95))
        
        # Ensure it's reasonable (between 256 and 2048)
        optimal_length = max(256, min(optimal_length, 2048))
        
        logger.info(f"Setting adaptive max length to: {optimal_length} tokens")
        return optimal_length
    
    def find_tow_positions(self, text: str) -> List[Tuple[int, int]]:
        """Find all ToW token positions in text"""
        positions = []
        start_idx = 0
        
        while True:
            tow_start = text.find(self.tow_start_token, start_idx)
            if tow_start == -1:
                break
                
            tow_end = text.find(self.tow_end_token, tow_start)
            if tow_end == -1:
                break
                
            # Convert character positions to token positions
            before_tow = text[:tow_start]
            before_tokens = len(self.tokenizer.tokenize(before_tow))
            
            after_tow = text[:tow_end + len(self.tow_end_token)]
            after_tokens = len(self.tokenizer.tokenize(after_tow))
            
            positions.append((before_tokens, after_tokens))
            start_idx = tow_end + len(self.tow_end_token)
        
        return positions
    
    def smart_truncate(self, text: str, max_length: int) -> List[str]:
        """Smart truncation that preserves ToW tokens"""
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= max_length:
            return [text]  # No truncation needed
        
        if not self.config.preserve_tow_tokens:
            # Simple truncation
            truncated_tokens = tokens[:max_length]
            return [self.tokenizer.convert_tokens_to_string(truncated_tokens)]
        
        # Find ToW positions
        tow_positions = self.find_tow_positions(text)
        
        if not tow_positions:
            # No ToW tokens, simple truncation
            truncated_tokens = tokens[:max_length]
            return [self.tokenizer.convert_tokens_to_string(truncated_tokens)]
        
        chunks = []
        
        if self.config.enable_chunking:
            # Create chunks that preserve ToW tokens
            for tow_start, tow_end in tow_positions:
                # Calculate chunk boundaries to include ToW token
                chunk_start = max(0, tow_start - max_length // 2)
                chunk_end = min(len(tokens), tow_end + max_length // 2)
                
                # Adjust if chunk is too long
                if chunk_end - chunk_start > max_length:
                    chunk_end = chunk_start + max_length
                
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
        else:
            # Single chunk preserving first ToW token
            tow_start, tow_end = tow_positions[0]
            chunk_start = max(0, tow_start - max_length // 2)
            chunk_end = min(len(tokens), chunk_start + max_length)
            
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def load_tow_data(self, data_path: str) -> List[Dict]:
        """Load ToW-augmented Korean dataset"""
        logger.info(f"Loading ToW data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} ToW-augmented entries")
        return data
    
    def create_training_dataset(self, data: List[Dict]) -> Dataset:
        """Create dataset with smart text handling"""
        logger.info("Creating training dataset with smart processing...")
        
        # Analyze data and set adaptive max length
        if self.config.adaptive_max_length:
            optimal_length = self.analyze_data_lengths(data)
            self.config.max_sequence_length = optimal_length
        
        # Process all entries
        formatted_texts = []
        
        for entry in tqdm(data, desc="Smart processing ToW examples"):
            if entry['tow_count'] > 0:
                text = entry['augmented_text']
                
                # Smart truncation with ToW preservation
                text_chunks = self.smart_truncate(text, self.config.max_sequence_length)
                formatted_texts.extend(text_chunks)
        
        # Limit dataset size for initial testing
        formatted_texts = formatted_texts[:1000]
        logger.info(f"Created {len(formatted_texts)} training examples (after smart processing)")
        
        # Create and tokenize dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        return tokenized_dataset

# 나머지 클래스들은 동일 (ToWTrainer, main 등)
class ToWTrainer:
    """Custom trainer for ToW fine-tuning"""
    
    def __init__(self, model_config: ModelConfig, training_config: ToWTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(training_config.output_base_dir) / model_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_config.model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True,
            torch_dtype=self.model_config.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            seed=42,
            data_seed=42,
            report_to=[],
        )
    
    def train(self):
        """Execute smart ToW fine-tuning"""
        logger.info(f"Starting smart ToW training for {self.model_config.name}")
        
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Use smart data processor
        data_processor = SmartToWDataProcessor(tokenizer, self.training_config)
        tow_data = data_processor.load_tow_data(self.training_config.tow_data_path)
        
        train_dataset = data_processor.create_training_dataset(tow_data)
        
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        logger.info(f"Using max sequence length: {self.training_config.max_sequence_length}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        training_args = self.create_training_arguments()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            ]
        )
        
        logger.info("Starting smart training...")
        train_result = trainer.train()
        
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(self.output_dir))
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training completed for {self.model_config.name}")
        logger.info(f"Model saved to: {self.output_dir}")
        
        return train_result

def main():
    """Main function with smart processing"""
    logger.info("ToW Training with Smart Text Handling")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
    
    training_config = ToWTrainingConfig()
    
    if not Path(training_config.tow_data_path).exists():
        logger.error(f"ToW data not found: {training_config.tow_data_path}")
        return
    
    model_config = MODEL_CONFIGS[0]
    
    try:
        logger.info(f"Training model with smart processing: {model_config.name}")
        
        trainer = ToWTrainer(model_config, training_config)
        result = trainer.train()
        
        logger.info("✅ Smart training completed successfully!")
        logger.info(f"Model saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()