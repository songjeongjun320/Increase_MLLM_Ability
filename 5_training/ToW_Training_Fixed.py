#!/usr/bin/env python3
"""
ToW (Thoughts of Words) Fine-tuning Implementation - FIXED VERSION
Fixed tensor size mismatch and DataLoader issues
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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
import wandb
from tqdm import tqdm

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ModelConfig:
    name: str
    model_id: str
    use_quantization: bool = False
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

@dataclass
class ToWTrainingConfig:
    """ToW-specific training configuration - FIXED"""
    # Data paths
    tow_data_path: str = "ToW_koconovel_complete.json"
    output_base_dir: str = "ToW_Models"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Reduced for stability
    per_device_eval_batch_size: int = 4   # Reduced for stability
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    
    # ToW-specific settings
    max_sequence_length: int = 512       # Reduced from 1024 for stability
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Memory optimization - FIXED
    dataloader_num_workers: int = 0      # Changed to 0 to avoid multiprocessing issues
    remove_unused_columns: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True

MODEL_CONFIGS = [
    # Start with one model for testing
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),
]

class ToWDataProcessor:
    """FIXED - Process ToW-augmented data for training"""
    
    def __init__(self, tokenizer, config: ToWTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
        
        # Add ToW tokens to tokenizer
        special_tokens = [self.tow_start_token, self.tow_end_token]
        new_tokens = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new ToW tokens to tokenizer")
    
    def load_tow_data(self, data_path: str) -> List[Dict]:
        """Load ToW-augmented Korean dataset"""
        logger.info(f"Loading ToW data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} ToW-augmented entries")
        return data
    
    def format_tow_example(self, entry: Dict) -> str:
        """Format ToW entry for causal language modeling - SIMPLIFIED"""
        # Use only the augmented text without additional instruction formatting
        return entry['augmented_text']
    
    def create_training_dataset(self, data: List[Dict]) -> Dataset:
        """Create HuggingFace Dataset for training - FIXED"""
        logger.info("Creating training dataset...")
        
        # Format all examples
        formatted_texts = []
        for entry in tqdm(data, desc="Formatting ToW examples"):
            if entry['tow_count'] > 0:
                formatted_text = self.format_tow_example(entry)
                formatted_texts.append(formatted_text)
        
        # Limit dataset size for testing
        formatted_texts = formatted_texts[:500]  # Use first 500 for testing
        logger.info(f"Created {len(formatted_texts)} training examples")
        
        # Create dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        
        # Tokenize with proper settings - FIXED
        def tokenize_function(examples):
            # Tokenize with consistent padding and truncation
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,  # Enable padding
                max_length=self.config.max_sequence_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        return tokenized_dataset

class ToWTrainer:
    """Custom trainer for ToW fine-tuning - FIXED"""
    
    def __init__(self, model_config: ModelConfig, training_config: ToWTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(training_config.output_base_dir) / model_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration"""
        logger.info(f"Loading model: {self.model_config.model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.model_config.torch_dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            **model_kwargs
        )
        
        # Resize token embeddings for new ToW tokens
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments - FIXED"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # Training hyperparameters
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            
            # Optimization
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            
            # Logging and evaluation
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            
            # Saving
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Memory optimization - FIXED
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,  # Now 0
            remove_unused_columns=self.training_config.remove_unused_columns,
            
            # Miscellaneous
            seed=42,
            data_seed=42,
            report_to=[],  # Disable wandb for now
            run_name=f"tow-{self.model_config.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    def train(self):
        """Execute ToW fine-tuning - FIXED"""
        logger.info(f"Starting ToW training for {self.model_config.name}")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Process data
        data_processor = ToWDataProcessor(tokenizer, self.training_config)
        tow_data = data_processor.load_tow_data(self.training_config.tow_data_path)
        
        # Create training dataset
        train_dataset = data_processor.create_training_dataset(tow_data)
        
        # Split dataset for evaluation
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # Data collator - FIXED
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            ]
        )
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(self.output_dir))
        
        # Save training metrics
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training completed for {self.model_config.name}")
        logger.info(f"Model saved to: {self.output_dir}")
        
        return train_result

def main():
    """Main training function - FIXED"""
    logger.info("ToW (Thoughts of Words) Fine-tuning for Korean Models - FIXED VERSION")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Create training configuration
    training_config = ToWTrainingConfig()
    
    # Check ToW data
    if not Path(training_config.tow_data_path).exists():
        logger.error(f"ToW data not found: {training_config.tow_data_path}")
        return
    
    # Train single model for testing
    model_config = MODEL_CONFIGS[0]
    
    try:
        logger.info(f"Training model: {model_config.name}")
        
        trainer = ToWTrainer(model_config, training_config)
        result = trainer.train()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Model saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()