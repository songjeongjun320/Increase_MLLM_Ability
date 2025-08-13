#!/usr/bin/env python3
"""
ToW Training with Smart Text Handling - Fixed Version
- Fixed batch_size mismatch error
- Adaptive max length based on data analysis
- ToW token preservation
- Smart chunking for long texts
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
from tqdm import tqdm

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Custom data collator that ensures input_ids and labels have the same size"""
    
    def __call__(self, features, return_tensors=None):
        # 먼저 부모 클래스의 __call__ 메서드를 호출
        batch = super().__call__(features, return_tensors=return_tensors)
        
        # input_ids와 labels의 크기를 동일하게 맞춤
        if 'labels' in batch and 'input_ids' in batch:
            # labels를 input_ids와 동일한 크기로 만듦
            batch['labels'] = batch['input_ids'].clone()
        
        return batch


class ToWTrainerWithPinMemory(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Override to set pin_memory=True for the train dataloader"""
        train_dataloader = super().get_train_dataloader()
        
        if hasattr(train_dataloader, 'pin_memory'):
            train_dataloader.pin_memory = True
            
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Override to set pin_memory=True for the eval dataloader"""
        eval_dataloader = super().get_eval_dataloader(eval_dataset)
        
        if hasattr(eval_dataloader, 'pin_memory'):
            eval_dataloader.pin_memory = True
            
        return eval_dataloader


@dataclass
class ModelConfig:
    name: str
    model_id: str
    use_quantization: bool = False
    torch_dtype: torch.dtype = field(default=torch.float16)


@dataclass
class ToWTrainingConfig:
    """ToW training config with smart text handling"""
    tow_data_path: str = "../4_tow_generation/tow_data/koconovel_tow_gemini_2.0-flash-lite.json"
    output_base_dir: str = "ToW_Models"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    
    # Smart text handling
    adaptive_max_length: bool = True
    preserve_tow_tokens: bool = True
    enable_chunking: bool = True
    min_chunk_overlap: int = 50
    
    # Default settings
    max_sequence_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    
    # Other settings
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    logging_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True
    fp16: bool = False
    bf16: bool = True
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
    # ModelConfig(
    #     name="Llama-3.1-8B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3:1_8B_Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-0528-Qwen3-8B-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
    #     use_quantization=False
    # ),
]


class SmartToWDataProcessor:
    """Smart data processor with adaptive length and ToW preservation"""
    
    def __init__(self, tokenizer, config: ToWTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
    
    def analyze_data_lengths(self, data: List[Dict]) -> int:
        """Analyze data to determine optimal max length"""
        logger.info("Analyzing dataset lengths...")
        
        lengths = []
        
        for entry in tqdm(data, desc="Analyzing lengths"):
            context = entry.get('context', '')
            tow = entry.get('tow', '')
            gold_label = entry.get('gold_label', '')
            text = f"{context}{tow}{gold_label}{self.tokenizer.eos_token}"
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
        
        optimal_length = int(np.percentile(lengths, 98)) # Use 98th percentile for better coverage
        optimal_length = max(256, min(optimal_length, self.tokenizer.model_max_length or 4096))
        
        logger.info(f"Setting adaptive max length to: {optimal_length} tokens")
        return optimal_length
    
    def find_tow_positions(self, text: str) -> List[Tuple[int, int]]:
        """This method is no longer needed in the new format but kept for compatibility."""
        return []
    
    def smart_truncate(self, text: str, max_length: int) -> List[str]:
        """Simplified truncation for context + tow + gold_label format"""
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= max_length:
            return [text]
        
        # Simple truncation from the beginning if text is too long
        truncated_tokens = tokens[-max_length:]
        return [self.tokenizer.convert_tokens_to_string(truncated_tokens)]

    def load_tow_data(self, data_path: str) -> List[Dict]:
        """Load ToW-augmented Korean dataset"""
        logger.info(f"Loading ToW data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} ToW-augmented entries")
        return data
    
    def create_training_dataset(self, data: List[Dict]) -> Dataset:
        """Create dataset for context -> tow + gold_label training"""
        logger.info("Creating training dataset for context-based ToW training...")
        
        if self.config.adaptive_max_length:
            optimal_length = self.analyze_data_lengths(data)
            self.config.max_sequence_length = optimal_length
        
        processed_data = []
        for entry in tqdm(data, desc="Processing context/ToW examples"):
            context = entry.get('context', '')
            tow = entry.get('tow', '')
            gold_label = entry.get('gold_label', '')

            if not context or not tow or not gold_label:
                continue

            # Format: context -> tow -> gold_label -> eos
            full_text = f"{context}{tow}{gold_label}{self.tokenizer.eos_token}"
            
            processed_data.append({
                "context": context,
                "full_text": full_text
            })

        logger.info(f"Created {len(processed_data)} training examples.")
        
        dataset = Dataset.from_list(processed_data)
        
        def tokenize_function(examples):
            # The function now correctly handles batches of examples
            
            # Temporarily set truncation side to 'left' if supported
            original_truncation_side = getattr(self.tokenizer, 'truncation_side', None)
            if hasattr(self.tokenizer, 'truncation_side'):
                self.tokenizer.truncation_side = 'left'
            
            context_tokens = self.tokenizer(
                examples['context'],
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
            # Iterate over each example in the batch
            for i in range(len(full_tokens['input_ids'])):
                input_ids = full_tokens['input_ids'][i]
                attention_mask = full_tokens['attention_mask'][i]
                
                # Get the length of the context for this specific example
                context_len = len(context_tokens['input_ids'][i])
                
                labels = input_ids.copy()
                
                # Determine the actual number of context tokens to mask.
                # This prevents index errors if the context itself was truncated.
                mask_len = min(context_len, len(labels))

                # Mask context tokens by setting them to -100
                labels[:mask_len] = [-100] * mask_len
                
                # Also mask padding tokens
                for j in range(len(labels)):
                    if attention_mask[j] == 0:
                        labels[j] = -100
                
                all_labels.append(labels)
            
            full_tokens['labels'] = all_labels
            return full_tokens
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,  # Explicitly set batched=True
            remove_columns=['context', 'full_text'],
            desc="Tokenizing and creating labels"
        )

        logger.info("Checking tokenized sample lengths...")
        for i in range(min(5, len(tokenized_dataset))):
            input_ids_len = len(tokenized_dataset[i]['input_ids'])
            labels_len = len(tokenized_dataset[i]['labels'])
            # Count non-masked labels
            actual_labels = sum(1 for label in tokenized_dataset[i]['labels'] if label != -100)
            logger.info(f"Sample {i} - input_ids: {input_ids_len}, labels: {labels_len}, actual_labels: {actual_labels}")

        return tokenized_dataset


class ToWTrainer:
    """Custom trainer for ToW fine-tuning"""
    
    def __init__(self, model_config: ModelConfig, training_config: ToWTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(training_config.output_base_dir) / model_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer, add special tokens, and resize embeddings."""
        logger.info(f"Loading model: {self.model_config.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id,
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

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True,
            torch_dtype=self.model_config.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        logger.info(f"Final tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
        assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]

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
            bf16=self.training_config.bf16,
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
        
        data_processor = SmartToWDataProcessor(tokenizer, self.training_config)
        tow_data = data_processor.load_tow_data(self.training_config.tow_data_path)
        
        train_dataset = data_processor.create_training_dataset(tow_data)
        
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        logger.info(f"Using max sequence length: {self.training_config.max_sequence_length}")
        
        # Use custom data collator
        data_collator = CustomDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None,  # 패딩은 이미 tokenize_function에서 처리
            return_tensors="pt"
        )
        
        training_args = self.create_training_arguments()
        
        trainer = ToWTrainerWithPinMemory(
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
    
    for model_config in MODEL_CONFIGS:
        try:
            logger.info("=================================================================")
            logger.info(f"Starting training for model: {model_config.name}")
            logger.info("=================================================================")
            
            trainer = ToWTrainer(model_config, training_config)
            result = trainer.train()
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"Model saved to: {trainer.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for model {model_config.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue 


if __name__ == "__main__":
    main()