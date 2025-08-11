#!/usr/bin/env python3
"""
ToW (Thoughts of Words) Fine-tuning Implementation
Based on Zhikun Xu et al. (2024) methodology for Korean language models

This script fine-tunes multiple base models using ToW-augmented Korean text data
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
    EarlyStoppingCallback, TrainerCallback
)
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import wandb
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str                             # Unique name for this run (used for filenames)
    model_id: str                         # Hugging Face model identifier
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

@dataclass
class ToWTrainingConfig:
    """ToW-specific training configuration based on the paper methodology"""
    # Data paths
    tow_data_path: str = "ToW_koconovel_complete.json"
    output_base_dir: str = "ToW_Models"
    
    # Training hyperparameters (based on ToW paper)
    learning_rate: float = 2e-5          # Standard fine-tuning rate for ToW
    num_train_epochs: int = 3            # ToW paper uses 3 epochs
    per_device_train_batch_size: int = 4 # Adjust based on GPU memory
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    
    # ToW-specific settings
    max_sequence_length: int = 1024      # Max context length for ToW reasoning
    tow_token_weight: float = 1.2        # Slightly higher weight for ToW tokens
    warmup_ratio: float = 0.1            # Warmup steps
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
    
    # Memory optimization
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True

MODEL_CONFIGS = [
    ModelConfig(
        name="Qwen2.5-7B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3:1_8B_Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),
]

class ToWDataProcessor:
    """Process ToW-augmented data for training following the paper methodology"""
    
    def __init__(self, tokenizer, config: ToWTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
        
        # Add ToW tokens to tokenizer if not present
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
        """
        Format ToW entry for causal language modeling
        Following ToW paper methodology:
        - Use the augmented text with ToW tokens embedded
        - The model learns to predict next tokens including reasoning
        """
        augmented_text = entry['augmented_text']
        
        # Add instruction format for better learning (optional)
        formatted_text = f"다음 한국어 텍스트를 이어서 작성하세요:\n\n{augmented_text}"
        
        return formatted_text
    
    def create_training_dataset(self, data: List[Dict]) -> Dataset:
        """Create HuggingFace Dataset for training"""
        logger.info("Creating training dataset...")
        
        # Format all examples
        formatted_texts = []
        for entry in tqdm(data, desc="Formatting ToW examples"):
            if entry['tow_count'] > 0:  # Only use entries with ToW tokens
                formatted_text = self.format_tow_example(entry)
                formatted_texts.append(formatted_text)
        
        logger.info(f"Created {len(formatted_texts)} training examples")
        
        # Tokenize texts
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_sequence_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        return tokenized_dataset

class ToWTrainer:
    """Custom trainer for ToW fine-tuning"""
    
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
            padding_side='right'  # Important for causal LM
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
        
        if self.model_config.use_quantization:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            **model_kwargs
        )
        
        # Resize token embeddings for new ToW tokens
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments based on ToW methodology"""
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
            adam_beta1=0.9,
            adam_beta2=0.999,
            
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
            
            # Memory optimization
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            
            # Miscellaneous
            seed=42,
            data_seed=42,
            report_to=["wandb"] if wandb.api.api_key else [],
            run_name=f"tow-{self.model_config.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    def train(self):
        """Execute ToW fine-tuning"""
        logger.info(f"Starting ToW training for {self.model_config.name}")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Process data
        data_processor = ToWDataProcessor(tokenizer, self.training_config)
        tow_data = data_processor.load_tow_data(self.training_config.tow_data_path)
        
        # Create training dataset
        train_dataset = data_processor.create_training_dataset(tow_data)
        
        # Split dataset for evaluation (90% train, 10% eval)
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8,
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
            tokenizer=tokenizer,
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

class ToWTrainingPipeline:
    """Main pipeline for training all models with ToW"""
    
    def __init__(self, training_config: ToWTrainingConfig):
        self.training_config = training_config
        self.results = {}
        
    def run_all_trainings(self):
        """Train all models in sequence"""
        logger.info("Starting ToW training pipeline for all models")
        
        # Create output base directory
        Path(self.training_config.output_base_dir).mkdir(parents=True, exist_ok=True)
        
        for model_config in MODEL_CONFIGS:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training model: {model_config.name}")
                logger.info(f"{'='*60}")
                
                # Initialize W&B run if available
                if wandb.api.api_key:
                    wandb.init(
                        project="tow-korean-finetuning",
                        name=f"tow-{model_config.name}",
                        config={
                            **vars(self.training_config),
                            **vars(model_config)
                        }
                    )
                
                # Create trainer and start training
                trainer = ToWTrainer(model_config, self.training_config)
                result = trainer.train()
                
                self.results[model_config.name] = {
                    "status": "success",
                    "metrics": result.metrics,
                    "output_dir": str(trainer.output_dir)
                }
                
                logger.info(f"✅ Successfully trained {model_config.name}")
                
                # Cleanup
                torch.cuda.empty_cache()
                if wandb.api.api_key:
                    wandb.finish()
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_config.name}: {str(e)}")
                self.results[model_config.name] = {
                    "status": "failed",
                    "error": str(e)
                }
                
                if wandb.api.api_key:
                    wandb.finish()
                continue
        
        # Save overall results
        results_path = Path(self.training_config.output_base_dir) / "training_summary.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("ToW Training Pipeline Completed")
        logger.info(f"{'='*60}")
        
        # Print summary
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        total = len(self.results)
        logger.info(f"Successfully trained: {successful}/{total} models")
        
        for model_name, result in self.results.items():
            status = "✅" if result["status"] == "success" else "❌"
            logger.info(f"{status} {model_name}: {result['status']}")
        
        return self.results

def main():
    """Main training function"""
    # Setup
    logger.info("ToW (Thoughts of Words) Fine-tuning for Korean Models")
    logger.info("Based on Zhikun Xu et al. (2024) methodology")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available, using CPU (training will be very slow)")
    
    # Create training configuration
    training_config = ToWTrainingConfig()
    
    # Check if ToW data exists
    if not Path(training_config.tow_data_path).exists():
        logger.error(f"ToW data not found: {training_config.tow_data_path}")
        logger.error("Please run the ToW dataset generation script first")
        return
    
    # Initialize W&B if available
    try:
        import wandb
        if wandb.api.api_key:
            logger.info("W&B available for experiment tracking")
        else:
            logger.info("W&B not configured, skipping experiment tracking")
    except ImportError:
        logger.info("W&B not installed, skipping experiment tracking")
    
    # Start training pipeline
    pipeline = ToWTrainingPipeline(training_config)
    results = pipeline.run_all_trainings()
    
    logger.info("Training pipeline completed!")
    return results

if __name__ == "__main__":
    main()