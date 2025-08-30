#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ToW (Thought-of-Words) Training Script with DeepSpeed Support
module avail cuda
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
Run with: deepspeed --num_gpus=2 ToW_Training_llama.py
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ToW (Thought-of-Words) Training Script with DeepSpeed Support
module avail cuda
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
Run with: deepspeed --num_gpus=2 ToW_Training_llama.py
"""

import os
import sys
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import timedelta  # 추가된 import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from tqdm.auto import tqdm
import deepspeed
from deepspeed import DeepSpeedConfig

# ================================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# ================================================================================

# Model Configuration
MODEL_NAME_OR_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./tow_trained_models/llama-3.2-3b-tow"
CACHE_DIR = "./cache"

# Dataset Configuration
DATASET_PATH = "../4_tow_generation/tow_data/final_multiple_tow.jsonl"
VALIDATION_SPLIT = 0.1  # Use 10% of data for validation

# Training Hyperparameters
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Optimization Settings
LR_SCHEDULER_TYPE = "cosine"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Saving and Logging
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"

# Quantization Settings
USE_QUANTIZATION = False  # Set to True for 4-bit quantization
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# DeepSpeed Configuration
USE_DEEPSPEED = True
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": True
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }
}

# Misc Settings
SEED = 42
PREPROCESSING_NUM_WORKERS = 4
DISABLE_TQDM = False
PUSH_TO_HUB = False
HUB_MODEL_ID = None
HUB_TOKEN = None

# ================================================================================
# SETUP LOGGING (초기 로깅만)
# ================================================================================

# 초기 Python 로거 설정
initial_logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# ================================================================================
# TOW TOKEN DEFINITIONS
# ================================================================================

TOW_START_TOKEN = "<ToW>"
TOW_END_TOKEN = "</ToW>"
SPECIAL_TOKENS = {
    "additional_special_tokens": [TOW_START_TOKEN, TOW_END_TOKEN]
}

# ================================================================================
# DATASET CLASS
# ================================================================================

class ToWDataset(Dataset):
    """Custom dataset for ToW training"""
    
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.tow_start_id = tokenizer.convert_tokens_to_ids(TOW_START_TOKEN)
        self.tow_end_id = tokenizer.convert_tokens_to_ids(TOW_END_TOKEN)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct the full text with ToW tokens
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        
        # Format: prompt <ToW> completion </ToW>
        full_text = f"{prompt} {TOW_START_TOKEN} {completion} {TOW_END_TOKEN}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        # Create labels - mask out the prompt part
        labels = input_ids.clone()
        
        # Find ToW token positions
        tow_start_pos = (input_ids == self.tow_start_id).nonzero(as_tuple=True)[0]
        
        if len(tow_start_pos) > 0:
            # Mask everything before ToW token (set to -100)
            labels[:tow_start_pos[0]] = -100
        
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ================================================================================
# DATA LOADING FUNCTIONS
# ================================================================================

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                initial_logger.warning(f"Skipping invalid JSON line: {e}")
    return data

def prepare_datasets(tokenizer, data_path, validation_split=0.1, max_seq_length=512):
    """Prepare train and validation datasets"""
    
    # Load data
    initial_logger.info(f"Loading data from {data_path}")
    data = load_jsonl_data(data_path)
    
    if not data:
        raise ValueError(f"No valid data found in {data_path}")
    
    initial_logger.info(f"Loaded {len(data)} examples")
    
    # Split into train and validation
    random.shuffle(data)
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    initial_logger.info(f"Train examples: {len(train_data)}, Validation examples: {len(val_data)}")
    
    # Create datasets
    train_dataset = ToWDataset(train_data, tokenizer, max_seq_length)
    val_dataset = ToWDataset(val_data, tokenizer, max_seq_length)
    
    return train_dataset, val_dataset

# ================================================================================
# MODEL INITIALIZATION
# ================================================================================

def setup_model_and_tokenizer():
    """Initialize model and tokenizer with ToW tokens"""
    
    initial_logger.info(f"Loading tokenizer from {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    # Add special tokens
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    initial_logger.info(f"Added {num_added_tokens} special tokens: {SPECIAL_TOKENS}")
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with optional quantization
    model_kwargs = {
        "cache_dir": CACHE_DIR,
        "trust_remote_code": True,
    }
    
    if USE_QUANTIZATION:
        initial_logger.info("Loading model with 4-bit quantization")
        bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    initial_logger.info(f"Loading model from {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        **model_kwargs
    )
    
    # Resize embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing if needed
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        initial_logger.info("Enabled gradient checkpointing")
    
    return model, tokenizer

# ================================================================================
# TRAINING FUNCTION
# ================================================================================

def train():
    """Main training function"""
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Initialize accelerator with DeepSpeed
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    
    if USE_DEEPSPEED:
        # Save DeepSpeed config
        ds_config_path = "ds_config.json"
        with open(ds_config_path, 'w') as f:
            json.dump(DEEPSPEED_CONFIG, f, indent=2)

        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            mixed_precision="bf16" if DEEPSPEED_CONFIG["bf16"]["enabled"] else "fp16",
            log_with="tensorboard",
            project_dir=OUTPUT_DIR,
            kwargs_handlers=[kwargs],
            deepspeed_plugin=deepspeed_plugin
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            mixed_precision="bf16",
            log_with="tensorboard",
            project_dir=OUTPUT_DIR,
            kwargs_handlers=[kwargs]
        )
    
    # accelerator가 초기화된 후 logger 설정
    logger = get_logger(__name__)
    
    # Setup logging
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    
    # Load model and tokenizer
    with accelerator.main_process_first():
        model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare datasets
    with accelerator.main_process_first():
        train_dataset, val_dataset = prepare_datasets(
            tokenizer, 
            DATASET_PATH, 
            VALIDATION_SPLIT, 
            MAX_SEQ_LENGTH
        )
    
    # Create data loaders
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=PREPROCESSING_NUM_WORKERS,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=PREPROCESSING_NUM_WORKERS,
        pin_memory=True
    )
    
    # Setup optimizer
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
    # num_warmup_steps = int(WARMUP_RATIO * max_train_steps) # 이 계산은 DeepSpeed가 알아서 합니다.
    # Prepare everything with accelerator
    # 옵티마이저와 스케줄러 자리에 None을 전달하면, DeepSpeed가 config를 보고 자동으로 생성합니다.
    # 하지만 이 코드에서는 이미 optimizer 객체를 사용하고 있으므로, Dummy 객체를 사용하는 것이 더 안전합니다.
    # accelerate.prepare는 DeepSpeed가 optimizer를 생성한 후, 생성된 optimizer를 다시 반환해줍니다.
    
    # 먼저 weight_decay를 적용할 파라미터 그룹을 만듭니다.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Dummy Optimizer와 Scheduler를 사용합니다.
    # 이 방식은 accelerate에게 "옵티마이저와 스케줄러는 DeepSpeed 설정에 따라 만들어줘" 라고 알려주는 신호입니다.
    from accelerate.utils import DummyOptim, DummyScheduler
    
    optimizer = DummyOptim(
            optimizer_grouped_parameters, 
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2), # betas 값을 튜플 형태로 전달
            eps=ADAM_EPSILON               # eps 값도 전달
        )
    lr_scheduler = DummyScheduler(optimizer, num_warmup_steps=int(WARMUP_RATIO * max_train_steps), num_training_steps=max_train_steps)

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Training info
    total_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {PER_DEVICE_TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    
    # Initialize tracking variables
    global_step = 0
    best_eval_loss = float('inf')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training loop
    for epoch in range(NUM_TRAIN_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Training Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % LOGGING_STEPS == 0:
                    avg_loss = total_loss / LOGGING_STEPS
                    logger.info(f"Step {global_step}: avg_loss={avg_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")
                    
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": avg_loss,
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )
                    total_loss = 0
                
                # Evaluation
                if global_step % EVAL_STEPS == 0:
                    eval_loss = evaluate(model, val_dataloader, accelerator)
                    logger.info(f"Step {global_step}: eval_loss={eval_loss:.4f}")
                    
                    if accelerator.is_main_process:
                        accelerator.log({"eval_loss": eval_loss}, step=global_step)
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint(
                                model, tokenizer, accelerator, 
                                os.path.join(OUTPUT_DIR, "best_model")
                            )
                            logger.info(f"Saved best model with eval_loss={eval_loss:.4f}")
                
                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(
                        model, tokenizer, accelerator,
                        os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    )
        
        progress_bar.close()
        
        # End of epoch evaluation
        eval_loss = evaluate(model, val_dataloader, accelerator)
        logger.info(f"Epoch {epoch + 1} finished: eval_loss={eval_loss:.4f}")
        
        if accelerator.is_main_process:
            accelerator.log({"epoch_eval_loss": eval_loss}, step=global_step)
    
    # Save final model
    save_checkpoint(
        model, tokenizer, accelerator,
        os.path.join(OUTPUT_DIR, "final_model")
    )
    
    logger.info("Training completed!")
    
    # Cleanup
    accelerator.end_training()

# ================================================================================
# EVALUATION FUNCTION
# ================================================================================

def evaluate(model, dataloader, accelerator):
    """Evaluate the model on validation set"""
    model.eval()
    losses = []
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(PER_DEVICE_EVAL_BATCH_SIZE)))
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    
    model.train()
    return eval_loss.item()

# ================================================================================
# CHECKPOINT SAVING
# ================================================================================

def save_checkpoint(model, tokenizer, accelerator, output_dir):
    """Save model checkpoint"""
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        unwrapped_model = accelerator.unwrap_model(model)
        if USE_DEEPSPEED:
            # DeepSpeed saving
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
        else:
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save training config
        config = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "learning_rate": LEARNING_RATE,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "tow_tokens": SPECIAL_TOKENS,
        }
        
        with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # accelerator 초기화 후의 logger 사용
        logger = get_logger(__name__)
        logger.info(f"Saved checkpoint to {output_dir}")

# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        # 표준 로거 사용
        print("Training interrupted by user")
        initial_logger.info("Training interrupted by user")
    except Exception as e:
        # 표준 로거 사용
        print(f"Training failed with error: {e}")
        initial_logger.error(f"Training failed with error: {e}")
        raise