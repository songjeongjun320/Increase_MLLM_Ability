#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ToW Training Script with Memory-Optimized DeepSpeed Configuration
Training entire sequence in "completion" values in dataset.
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
deepspeed --num_gpus=2 ToW_Training_llama_v2.py
torchrun --nproc_per_node=4 ToW_Training_llama_v2.py
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
from datetime import timedelta

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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

# ================================================================================
# CONFIGURATION SECTION - MEMORY OPTIMIZED
# ================================================================================

# Model Configuration
MODEL_NAME_OR_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./tow_trained_models/llama-3.2-3b-tow-lora"
CACHE_DIR = "./cache"

# Dataset Configuration
DATASET_PATH = "../4_tow_generation/tow_data/final_multiple_tow.jsonl"
VALIDATION_SPLIT = 0.1

# Training Hyperparameters - OPTIMIZED FOR LORA
LEARNING_RATE = 3e-4  # Higher LR for LoRA training
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Can increase due to memory efficiency of LoRA
PER_DEVICE_EVAL_BATCH_SIZE = 4   # Can increase due to memory efficiency of LoRA
GRADIENT_ACCUMULATION_STEPS = 8   # Reduced due to increased batch size
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Optimization Settings
LR_SCHEDULER_TYPE = "cosine"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Saving and Logging
SAVE_STEPS = 250  # Increased to reduce I/O
EVAL_STEPS = 250  # Increased to reduce evaluation overhead
EVAL_ON_EPOCH_END = False
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 5
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"

# Quantization Settings
USE_QUANTIZATION = False
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# LoRA Configuration
USE_LORA = True
LORA_CONFIG = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,  # LoRA scaling parameter
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"
    ],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# DeepSpeed Configuration - HEAVILY OPTIMIZED FOR MEMORY
USE_DEEPSPEED = True
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_optimization": {
        "stage": 2,  # Changed to Stage 2 for better stability
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False,
            "buffer_count": 2,  # Reduced further
            "fast_init": True   # Changed to True for faster init
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": False,
            "buffer_count": 2,  # Reduced further
            "buffer_size": 5e7,  # Smaller buffer size
            "max_in_cpu": 5e8   # Reduced
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e7,  # Significantly reduced
        "reduce_bucket_size": 1e7,  # Significantly reduced
        "stage3_prefetch_bucket_size": 1e7,  # Reduced
        "stage3_param_persistence_threshold": 1e5,  # Reduced
        "stage3_max_live_parameters": 1e7,  # Significantly reduced
        "stage3_max_reuse_distance": 1e7,  # Significantly reduced
        "stage3_gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True,
        "memory_efficient_linear": True  # Enable memory efficient linear layers
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
            "lr": LEARNING_RATE,
            "betas": [ADAM_BETA1, ADAM_BETA2],
            "eps": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY
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
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    },
    "wall_clock_breakdown": False,
    "memory_breakdown": False
}

# Misc Settings
SEED = 42
PREPROCESSING_NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues
DISABLE_TQDM = False
PUSH_TO_HUB = False
HUB_MODEL_ID = None
HUB_TOKEN = None

# ================================================================================
# SETUP LOGGING
# ================================================================================

initial_logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# ================================================================================
# TOW TOKEN DEFINITIONS (using HCoT training approach)
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
        
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        
        full_text = f"{prompt} {completion}"
        full_text = full_text + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ================================================================================
# IMPROVED CHECKPOINT UTILITIES
# ================================================================================

def save_training_state(accelerator, model, tokenizer, optimizer, lr_scheduler, 
                        epoch, global_step, best_eval_loss, checkpoint_dir):
    """Save complete training state including all necessary files"""
    
    accelerator.wait_for_everyone()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save accelerator state (includes model, optimizer, scheduler, RNG states)
    accelerator.save_state(checkpoint_dir)
    
    # Additionally save tokenizer and configs on main process
    if accelerator.is_main_process:
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save model config
        unwrapped_model = accelerator.unwrap_model(model)
        if USE_LORA:
            # For PEFT models, save the adapter config and weights
            unwrapped_model.save_pretrained(checkpoint_dir)
        else:
            unwrapped_model.config.save_pretrained(checkpoint_dir)
        
        # Save training state info
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_eval_loss": best_eval_loss,
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
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Create a marker file to indicate checkpoint is complete
        with open(os.path.join(checkpoint_dir, "checkpoint_complete.marker"), 'w') as f:
            f.write(f"Checkpoint saved at step {global_step}")
        
        initial_logger.info(f"Complete checkpoint saved to {checkpoint_dir}")
    
    accelerator.wait_for_everyone()


def load_training_state(checkpoint_dir):
    """Load training state from checkpoint"""
    
    state_file = os.path.join(checkpoint_dir, "training_state.json")
    if not os.path.exists(state_file):
        return None
    
    with open(state_file, 'r') as f:
        return json.load(f)


# ================================================================================
# CHECKPOINT UTILITIES
# ================================================================================

def get_last_checkpoint(output_dir):
    """Find the most recent COMPLETE checkpoint"""
    
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(output_dir, item)
            # Check if checkpoint is complete by looking for marker file
            marker_file = os.path.join(checkpoint_path, "checkpoint_complete.marker")
            if os.path.exists(marker_file):
                try:
                    step_num = int(item.split('-')[1])
                    checkpoints.append((step_num, checkpoint_path))
                except (ValueError, IndexError):
                    continue
    
    if checkpoints:
        # Return path of the checkpoint with highest step number
        return max(checkpoints, key=lambda x: x[0])[1]
    return None

# ================================================================================
# DATA LOADING UTILITIES
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
    
    initial_logger.info(f"Loading data from {data_path}")
    data = load_jsonl_data(data_path)
    
    if not data:
        raise ValueError(f"No valid data found in {data_path}")
    
    initial_logger.info(f"Loaded {len(data)} examples")
    
    random.shuffle(data)
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    initial_logger.info(f"Train examples: {len(train_data)}, Validation examples: {len(val_data)}")
    
    train_dataset = ToWDataset(train_data, tokenizer, max_seq_length)
    val_dataset = ToWDataset(val_data, tokenizer, max_seq_length)
    
    return train_dataset, val_dataset

def cleanup_old_checkpoints(output_dir, keep_last=3):
    """Clean up old checkpoints, keeping only the most recent ones"""
    
    if not os.path.exists(output_dir):
        return
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(output_dir, item)
            # Only consider complete checkpoints
            marker_file = os.path.join(checkpoint_path, "checkpoint_complete.marker")
            if os.path.exists(marker_file):
                try:
                    step_num = int(item.split('-')[1])
                    checkpoints.append((step_num, checkpoint_path))
                except (ValueError, IndexError):
                    continue
    
    if len(checkpoints) > keep_last:
        checkpoints.sort(key=lambda x: x[0])
        checkpoints_to_remove = checkpoints[:-keep_last]
        
        for _, checkpoint_path in checkpoints_to_remove:
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                initial_logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                initial_logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")


def save_best_model(accelerator, model, tokenizer, best_eval_loss, output_dir):
    """Save the best model separately"""
    
    best_model_dir = os.path.join(output_dir, "best_model")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        os.makedirs(best_model_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        if USE_LORA:
            # For PEFT models, save only the adapter weights
            unwrapped_model.save_pretrained(best_model_dir)
        else:
            unwrapped_model.save_pretrained(
                best_model_dir,
                save_function=accelerator.save,
                safe_serialization=True
            )
        
        tokenizer.save_pretrained(best_model_dir)
        
        # Save best model info
        with open(os.path.join(best_model_dir, "best_model_info.json"), 'w') as f:
            json.dump({"best_eval_loss": best_eval_loss}, f, indent=2)
        
        initial_logger.info(f"Best model saved with eval_loss={best_eval_loss:.4f}")


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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.train()
    return eval_loss.item()


# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================

def train():
    """Main training function with single global progress bar"""
    
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    set_seed(SEED)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="tensorboard",
        project_dir=OUTPUT_DIR,
        kwargs_handlers=[kwargs],
        mixed_precision="bf16"
    )
    
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        tokenizer, 
        DATASET_PATH, 
        VALIDATION_SPLIT, 
        MAX_SEQ_LENGTH
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True, 
        pad_to_multiple_of=8
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator, 
        num_workers=PREPROCESSING_NUM_WORKERS,
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=data_collator, 
        num_workers=PREPROCESSING_NUM_WORKERS,
        pin_memory=False
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(WARMUP_RATIO * max_train_steps)
    
    # Load model
    with accelerator.main_process_first():
        logger.info(f"Loading model from {MODEL_NAME_OR_PATH}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        # Initialize ToW token embeddings
        logger.info("Initializing ToW token embeddings...")
        embeddings = model.get_input_embeddings()
        
        dash_token_ids = tokenizer.encode('---', add_special_tokens=False)
        if len(dash_token_ids) > 0:
            dash_embedding = embeddings.weight.data[dash_token_ids[0], :].clone()
            embeddings.weight.data[len(tokenizer)-2, :] = dash_embedding
            embeddings.weight.data[len(tokenizer)-1, :] = dash_embedding
            logger.info("Initialized ToW tokens with '---' token embedding")
        
        # Apply LoRA if enabled
        if USE_LORA:
            logger.info("Applying LoRA configuration...")
            lora_config = LoraConfig(**LORA_CONFIG)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            logger.info("LoRA adapters applied successfully")
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    
    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    
    from torch.optim import AdamW
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=LEARNING_RATE, 
        betas=(ADAM_BETA1, ADAM_BETA2), 
        eps=ADAM_EPSILON
    )
    
    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Initialize training state
    global_step = 0
    starting_epoch = 0
    best_eval_loss = float('inf')
    resumed_from_checkpoint = False
    total_loss = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        logger.info(f"Found checkpoint: {last_checkpoint}")
        logger.info("Resuming training from checkpoint...")
        
        # Load accelerator state
        accelerator.load_state(last_checkpoint)
        
        # Load training state
        training_state = load_training_state(last_checkpoint)
        if training_state:
            global_step = training_state["global_step"]
            best_eval_loss = training_state.get("best_eval_loss", float('inf'))
            
            current_epoch_float = global_step / num_update_steps_per_epoch
            logger.info(f"Resumed from step {global_step}, epoch {current_epoch_float:.2f}")
            logger.info(f"Best eval loss so far: {best_eval_loss:.4f}")
        else:
            logger.warning("Could not load training state, starting fresh")
    else:
        logger.info("No checkpoint found, starting fresh training")
    
    # Log training info
    total_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {PER_DEVICE_TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    
    # Create single progress bar for entire training
    progress_bar = tqdm(
        total=max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Training Progress",
        ncols=120  # Make progress bar wider to show more info
    )
    
    # Training loop - single loop over all steps
    current_dataloader_iter = None
    current_epoch = starting_epoch
    steps_in_current_epoch = 0
    
    # Calculate how many steps to skip in current epoch if resuming
    if resumed_from_checkpoint:
        steps_completed_in_current_epoch = global_step % num_update_steps_per_epoch
        if steps_completed_in_current_epoch > 0:
            logger.info(f"Will skip {steps_completed_in_current_epoch * GRADIENT_ACCUMULATION_STEPS} batches in current epoch")
    
    while global_step < max_train_steps:
        # Start new epoch if needed
        if current_dataloader_iter is None:
            model.train()
            current_dataloader_iter = iter(train_dataloader)
            steps_in_current_epoch = 0
            
            # Skip batches if resuming from checkpoint in middle of epoch
            if resumed_from_checkpoint and current_epoch == starting_epoch:
                steps_completed_in_current_epoch = global_step % num_update_steps_per_epoch
                batches_to_skip = steps_completed_in_current_epoch * GRADIENT_ACCUMULATION_STEPS
                
                if batches_to_skip > 0:
                    logger.info(f"Skipping {batches_to_skip} batches in epoch {current_epoch}")
                    for _ in range(batches_to_skip):
                        try:
                            next(current_dataloader_iter)
                        except StopIteration:
                            break
                    steps_in_current_epoch = steps_completed_in_current_epoch
                
                resumed_from_checkpoint = False  # Only skip once
        
        # Get next batch
        try:
            batch = next(current_dataloader_iter)
        except StopIteration:
            # End of epoch reached
            current_dataloader_iter = None
            current_epoch += 1
            
            # Perform end-of-epoch evaluation
            if EVAL_ON_EPOCH_END and current_epoch <= NUM_TRAIN_EPOCHS:  # Don't evaluate after last epoch
                eval_loss = evaluate(model, val_dataloader, accelerator)
                
                # Calculate progress info
                epoch_progress = f"Epoch {current_epoch}/{NUM_TRAIN_EPOCHS} completed"
                step_progress = f"Step {global_step}/{max_train_steps}"
                
                logger.info(f"{epoch_progress} | {step_progress} | eval_loss={eval_loss:.4f}")
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "epoch": current_epoch,
                        "epoch_eval_loss": eval_loss,
                    }, step=global_step)
                
                # Update progress bar description
                progress_bar.set_description(f"Training | {epoch_progress} | eval_loss={eval_loss:.4f}")
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_best_model(accelerator, model, tokenizer, best_eval_loss, OUTPUT_DIR)
            
            continue
        
        # Forward pass
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Clear cache periodically
        if global_step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if accelerator.sync_gradients:
            global_step += 1
            steps_in_current_epoch += 1
            
            # Update progress bar
            current_epoch_float = global_step / num_update_steps_per_epoch
            progress_bar.update(1)
            progress_bar.set_postfix({
                'epoch': f'{current_epoch_float:.2f}',
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if global_step % LOGGING_STEPS == 0:
                avg_loss = total_loss / LOGGING_STEPS
                current_lr = lr_scheduler.get_last_lr()[0]
                
                current_epoch_float = global_step / num_update_steps_per_epoch

                logger.info(f"Step {global_step}/{max_train_steps} | Epoch {current_epoch_float:.2f} | loss={avg_loss:.4f} | lr={current_lr:.2e}")

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": current_epoch_float,
                        },
                        step=global_step,
                    )
                total_loss = 0
            
            # Evaluation
            if global_step % EVAL_STEPS == 0:
                eval_loss = evaluate(model, val_dataloader, accelerator)
                logger.info(f"Step {global_step}/{max_train_steps} | eval_loss={eval_loss:.4f}")
                
                if accelerator.is_main_process:
                    accelerator.log({"eval_loss": eval_loss}, step=global_step)
                
                # Update progress bar description
                progress_bar.set_description(f"Training | Step {global_step}/{max_train_steps} | eval_loss={eval_loss:.4f}")
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_best_model(accelerator, model, tokenizer, best_eval_loss, OUTPUT_DIR)
            
            # Save checkpoint
            if global_step % SAVE_STEPS == 0 and global_step > 0:
                checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                logger.info(f"Saving checkpoint at step {global_step}")
                
                save_training_state(
                    accelerator, model, tokenizer, optimizer, lr_scheduler,
                    current_epoch, global_step, best_eval_loss, checkpoint_dir
                )
                
                # Cleanup old checkpoints
                if accelerator.is_main_process:
                    cleanup_old_checkpoints(OUTPUT_DIR, keep_last=SAVE_TOTAL_LIMIT)
                
                accelerator.wait_for_everyone()
        
        # Break if we've reached max steps
        if global_step >= max_train_steps:
            break
    
    progress_bar.close()
    
    # Save final model
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_model")
    logger.info("Saving final model...")
    
    save_training_state(
        accelerator, model, tokenizer, optimizer, lr_scheduler,
        NUM_TRAIN_EPOCHS, global_step, best_eval_loss, final_checkpoint_dir
    )
    
    logger.info("Training completed!")
    accelerator.end_training()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        initial_logger.info("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        initial_logger.error(f"Training failed with error: {e}", exc_info=True)
        raise