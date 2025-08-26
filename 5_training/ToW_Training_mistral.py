#!/usr/bin/env python3
"""
ToW Training with Smart Text Handling - Fixed Version
- Fixed batch_size mismatch error
- Adaptive max length based on data analysis
- ToW token preservation
- Smart chunking for long texts

# 사용 가능한 CUDA 모듈 확인
# CUDA 모듈 로드 (가장 최신 버전)
# 환경변수 확인
module avail cuda
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
deepspeed --num_gpus=4 ToW_Training_deepseek.py
torchrun --nproc_per_node=4 ToW_Training_deepseek.py
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
    EarlyStoppingCallback, BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainerCallback

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class JsonLoggingCallback(TrainerCallback):
    """Callback to log metrics to a JSON file."""

    def __init__(self, log_file):
        self.log_file = log_file
        # Clear the log file at the beginning of training
        with open(self.log_file, 'w') as f:
            f.write("[\n")
        self.first_log = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Append logs to the JSON file
            with open(self.log_file, 'a') as f:
                if not self.first_log:
                    f.write(",\n")
                json.dump(logs, f, indent=2)
                self.first_log = False
    
    def on_train_end(self, args, state, control, **kwargs):
        # Close the JSON array
        with open(self.log_file, 'a') as f:
            f.write("\n]\n")


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
    tow_data_paths: List[str] = field(default_factory=lambda: [
        "../4_tow_generation/tow_data/training_dataset_over_6_words.json"
    ])
    output_base_dir: str = "ToW_Models_2"
    
    # Training hyperparameters
    learning_rate: float = 2e-5  # This will be a fallback
    max_grad_norm = 1.0
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8  # Reduced for memory efficiency with DeepSpeed
    per_device_eval_batch_size: int = 8  # Reduced for memory efficiency
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    lr_scheduler_type: str = "cosine" 
    
    # Smart text handling
    adaptive_max_length: bool = True
    preserve_tow_tokens: bool = True
    enable_chunking: bool = True
    min_chunk_overlap: int = 50
    
    # Default settings
    max_sequence_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Other settings
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 500
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False


MODEL_CONFIGS = [
    ModelConfig(
        name="Mistral-7B-Instruct-v0.3-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-7B-Instruct-v0.3",
        use_quantization=True,
    ),
]


# Import the new caching utilities
from dataset_cache_utils import SmartToWDataProcessor


class ToWTrainer:
    """Custom trainer for ToW fine-tuning"""
    
    def __init__(self, model_config: ModelConfig, training_config: ToWTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(training_config.output_base_dir) / model_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Set up file logging for the training process."""
        log_file_path = self.output_dir / "training_log.log"
        file_handler = logging.FileHandler(log_file_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Get the root logger and add the file handler
        root_logger = logging.getLogger()
        
        # Avoid adding duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in root_logger.handlers):
            root_logger.addHandler(file_handler)
            logger.info(f"File-based logging is set up at: {log_file_path}")

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

        quantization_config = None
        if self.model_config.use_quantization:
            logger.info("Using 4-bit quantization (NF4) to reduce memory usage.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.model_config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # device_map 설정 수정
        local_rank = -1
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            
        device_map = {"": f"cuda:{local_rank}"} if local_rank != -1 else "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True,
            torch_dtype=self.model_config.torch_dtype,
            device_map=device_map, # 수정된 device_map 적용
            quantization_config=quantization_config,
        )

        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        # Prepare for quantization if enabled
        if self.model_config.use_quantization:
            logger.info("Preparing quantized model for k-bit training with LoRA.")
            model = prepare_model_for_kbit_training(model)
            
        # Always apply LoRA for fine-tuning in this script
        logger.info("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"], # Important for new tokens
        )
        
        model = get_peft_model(model, lora_config)
        logger.info("LoRA setup complete. Trainable parameters:")
        model.print_trainable_parameters()
        
        logger.info(f"Final tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
        assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]

        return model, tokenizer
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            optim="paged_adamw_8bit",
            overwrite_output_dir=False,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            max_grad_norm=self.training_config.max_grad_norm,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=3,
            ddp_find_unused_parameters=False, # DeepSeek 모델과 LoRA 사용 시 이 옵션이 필요할 수 있습니다.
            load_best_model_at_end=True,
            local_rank=int(os.getenv('LOCAL_RANK', -1)),
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
            deepspeed="./deepspeed_config_mistral.json",
        )
    
    def train(self):
        """Execute smart ToW fine-tuning"""
        self.setup_logging() # Setup file logging right at the start
        logger.info(f"Starting smart ToW training for {self.model_config.name}")
        
        model, tokenizer = self.load_model_and_tokenizer()
        
        data_processor = SmartToWDataProcessor(tokenizer, self.training_config, self.model_config.model_id)
        
        train_dataset = data_processor.create_training_dataset(self.training_config.tow_data_paths)
        
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
        
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            logger.info("No checkpoint found. Starting a new training.")

        # Setup custom JSON logging callback
        json_log_path = self.output_dir / "training_metrics.json"
        json_logging_callback = JsonLoggingCallback(log_file=json_log_path)
        logger.info(f"Metrics will be logged to {json_log_path}")

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
                ),
                json_logging_callback  # Add our custom callback here
            ]
        )
        
        logger.info("Starting smart training...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(self.output_dir))
        
        logger.info("Saving training parameters...")
        model_params = self.model_config.__dict__.copy()
        if 'torch_dtype' in model_params:
            model_params['torch_dtype'] = str(model_params['torch_dtype'])

        all_params = {
            "model_config": model_params,
            "training_config": self.training_config.__dict__,
            "environment_config": {
                "gpus_used": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        params_path = self.output_dir / "training_parameters.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(all_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Training parameters saved to {params_path}")
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training completed for {self.model_config.name}")
        logger.info(f"Model saved to: {self.output_dir}")
        
        return train_result


def main():
    """Main function with smart processing"""
    logger.info("ToW Training with Smart Text Handling")

    # 분산 훈련 설정 추가
    local_rank = -1
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
    
    training_config = ToWTrainingConfig()
    
    if not training_config.tow_data_paths:
        logger.error("No ToW data paths specified in the configuration.")
        return

    # Check if at least one data file exists
    if not any(Path(p).exists() for p in training_config.tow_data_paths):
        logger.error("None of the specified ToW data files were found.")
        for p in training_config.tow_data_paths:
            logger.error(f"  - Checked path: {p}")
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