#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ToW Training Script with Memory-Optimized DeepSpeed Configuration
Training entire sequence in "completion" values in dataset.
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
deepspeed --num_gpus=2 ToW_Training_gemma_v2.py
torchrun --nproc_per_node=4 ToW_Training_gemma_v2.py
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

# ================================================================================
# CONFIGURATION SECTION - MEMORY OPTIMIZED
# ================================================================================

# Model Configuration
MODEL_NAME_OR_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it"
OUTPUT_DIR = "./tow_trained_models/google_gemma-3-4b-it-tow"
CACHE_DIR = "./cache"

# Dataset Configuration
DATASET_PATH = "../4_tow_generation/tow_data/final_multiple_tow.jsonl"
VALIDATION_SPLIT = 0.1

# Training Hyperparameters - REDUCED FOR MEMORY
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Reduced from 4
PER_DEVICE_EVAL_BATCH_SIZE = 2   # Reduced from 4
GRADIENT_ACCUMULATION_STEPS = 16  # Increased from 8 to maintain effective batch size
MAX_SEQ_LENGTH = 1024  # Reduced from 2048
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Optimization Settings
LR_SCHEDULER_TYPE = "cosine"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Saving and Logging
SAVE_STEPS = 500  # Increased to reduce I/O
EVAL_STEPS = 500  # Increased to reduce evaluation overhead
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3  # Reduced from 5
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

# DeepSpeed Configuration - HEAVILY OPTIMIZED FOR MEMORY
USE_DEEPSPEED = False  # Temporarily disable DeepSpeed to test
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
    """Custom dataset for ToW training using HCoT-style approach"""
    
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
        
        # completion에 이미 ToW 토큰이 포함되어 있으므로 그대로 사용
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
        
        # HCoT-style: predict all tokens (no masking of prompt)
        labels = input_ids.clone()
        
        # Only mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ================================================================================
# CHECKPOINT UTILITIES
# ================================================================================

def get_last_checkpoint(output_dir):
    """Find the most recent checkpoint in the output directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            try:
                step_num = int(item.split('-')[1])
                checkpoint_path = os.path.join(output_dir, item)
                # Check if checkpoint is complete
                if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(checkpoint_path, "model.safetensors")) or \
                   os.path.exists(os.path.join(checkpoint_path, "zero_pp_rank_0_mp_rank_00_optim_states.pt")):
                    checkpoints.append((step_num, checkpoint_path))
            except (ValueError, IndexError):
                continue
    
    if checkpoints:
        # Return path of the checkpoint with highest step number
        return max(checkpoints, key=lambda x: x[0])[1]
    return None

def extract_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint path"""
    try:
        return int(os.path.basename(checkpoint_path).split('-')[1])
    except (ValueError, IndexError):
        return 0

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
    try:
        if not os.path.exists(output_dir):
            return
        
        checkpoints = []
        for item in os.listdir(output_dir):
            if item.startswith('checkpoint-'):
                try:
                    step_num = int(item.split('-')[1])
                    checkpoint_path = os.path.join(output_dir, item)
                    checkpoints.append((step_num, checkpoint_path))
                except (ValueError, IndexError):
                    continue
        
        if len(checkpoints) > keep_last:
            # Sort by step number and keep only the most recent
            checkpoints.sort(key=lambda x: x[0])
            checkpoints_to_remove = checkpoints[:-keep_last]
            
            for _, checkpoint_path in checkpoints_to_remove:
                try:
                    if os.path.exists(checkpoint_path):
                        import shutil
                        shutil.rmtree(checkpoint_path)
                        logger = get_logger(__name__)
                        logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
                    
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Checkpoint cleanup failed: {e}")
        # Don't let cleanup failure stop training
        pass

# ================================================================================
# TRAINING FUNCTION - WITH MEMORY OPTIMIZATIONS
# ================================================================================

def train():
    """Main training function with memory optimizations"""
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    set_seed(SEED)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    
    # Load tokenizer and prepare datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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
        pin_memory=False  # Disable pin_memory to save memory
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

    # Setup DeepSpeed
    deepspeed_plugin = None
    if USE_DEEPSPEED:
        final_deepspeed_config = DEEPSPEED_CONFIG.copy()
        final_deepspeed_config["scheduler"]["params"]["total_num_steps"] = max_train_steps
        final_deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
        final_deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = LEARNING_RATE
        final_deepspeed_config["scheduler"]["params"]["warmup_min_lr"] = 0.0
        
        ds_config_path = "ds_config.json"
        with open(ds_config_path, 'w') as f:
            json.dump(final_deepspeed_config, f, indent=2)
        
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config_path,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            gradient_clipping=MAX_GRAD_NORM,
            zero_stage=2,  # Match the config change
            offload_optimizer_device="cpu",
            offload_param_device="none"  # Disable param offloading initially
        )
    
    # Initialize Accelerator - FIXED: Initialize before using
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="tensorboard",  # FIXED: Removed conditional that caused the error
        project_dir=OUTPUT_DIR,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[kwargs],
        mixed_precision="bf16"  # Explicitly set mixed precision
    )
    
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # Load model with memory optimization
    with accelerator.main_process_first():
        logger.info(f"Loading model from {MODEL_NAME_OR_PATH}")
        
        # Use device_map for better memory management
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Important for memory optimization
            device_map=None  # Let DeepSpeed handle device placement
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        # ToW-style embedding initialization (using HCoT approach)
        # Initialize new token embeddings with existing token embeddings (like '---')
        logger.info("Initializing ToW token embeddings...")
        embeddings = model.get_input_embeddings()
        
        # Use '---' token embedding as initialization for new tokens
        dash_token_ids = tokenizer.encode('---', add_special_tokens=False)
        if len(dash_token_ids) > 0:
            dash_embedding = embeddings.weight.data[dash_token_ids[0], :].clone()
            # Initialize both ToW tokens with the same embedding
            embeddings.weight.data[len(tokenizer)-2, :] = dash_embedding  # </ToW>
            embeddings.weight.data[len(tokenizer)-1, :] = dash_embedding  # <ToW>
            logger.info("Initialized ToW tokens with '---' token embedding")
        else:
            logger.warning("Could not find '---' token for embedding initialization")
        
        # Enable gradient checkpointing for memory savings
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Enable memory efficient attention if available
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False  # Disable KV cache during training

    # Setup optimizer and scheduler
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

    if USE_DEEPSPEED:
        from accelerate.utils import DummyOptim, DummyScheduler
        optimizer = DummyOptim(
            optimizer_grouped_parameters, 
            lr=LEARNING_RATE, 
            betas=(ADAM_BETA1, ADAM_BETA2), 
            eps=ADAM_EPSILON
        )
        lr_scheduler = DummyScheduler(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=max_train_steps
        )
    else:
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

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Log training info
    total_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {PER_DEVICE_TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    
    # Training loop with checkpoint resumption
    global_step = 0
    starting_epoch = 0
    best_eval_loss = float('inf')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for existing checkpoints and resume if found
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        logger.info(f"Found checkpoint: {last_checkpoint}")
        logger.info("Resuming training from checkpoint...")
        
        # Load the checkpoint state
        accelerator.load_state(last_checkpoint)
        
        # Extract step information
        resumed_step = extract_step_from_checkpoint(last_checkpoint)
        global_step = resumed_step
        
        # Calculate which epoch we're in
        steps_per_epoch = num_update_steps_per_epoch
        starting_epoch = resumed_step // steps_per_epoch
        steps_completed_in_epoch = resumed_step % steps_per_epoch
        
        logger.info(f"Resumed from step {resumed_step}, epoch {starting_epoch}")
        logger.info(f"Completed {steps_completed_in_epoch}/{steps_per_epoch} steps in current epoch")
    else:
        logger.info("No checkpoint found, starting fresh training")
    
    for epoch in range(starting_epoch, NUM_TRAIN_EPOCHS):
        model.train()
        total_loss = 0
        
        # 올바른 계산
        steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
        
        # 재시작 시 스킵할 스텝 계산
        steps_to_skip = 0
        if epoch == starting_epoch and last_checkpoint:
            steps_completed_in_epoch = global_step % steps_per_epoch
            steps_to_skip = steps_completed_in_epoch
        
        # 수정된 progress bar
        progress_bar = tqdm(
            total=steps_per_epoch - steps_to_skip,  # 실제 업데이트 스텝 수
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS} (Steps: {steps_per_epoch})"
        )
        
        # 또는 배치 기준으로 하려면:
        # progress_bar = tqdm(
        #     total=len(train_dataloader),  # 955 (배치 수)
        #     disable=not accelerator.is_local_main_process,
        #     desc=f"Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS} (Batches: {len(train_dataloader)})"
        # )
        
        # 스킵할 배치 계산
        dataloader_to_use = train_dataloader
        if steps_to_skip > 0:
            batches_to_skip = steps_to_skip * GRADIENT_ACCUMULATION_STEPS
            logger.info(f"Skipping {batches_to_skip} batches to resume from checkpoint")
            dataloader_to_use = accelerator.skip_first_batches(train_dataloader, batches_to_skip)
        
        for step, batch in enumerate(dataloader_to_use):
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
            
            # Progress bar 업데이트 - 배치마다 또는 스텝마다
            if accelerator.sync_gradients:
                progress_bar.update(1)  # 스텝 기준
                global_step += 1
                
                if global_step % LOGGING_STEPS == 0:
                    avg_loss = total_loss / LOGGING_STEPS
                    current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else LEARNING_RATE
                    
                    # Log memory usage
                    if torch.cuda.is_available() and accelerator.is_local_main_process:
                        memory_used = torch.cuda.max_memory_allocated() / 1024**3
                        logger.info(f"Step {global_step}: avg_loss={avg_loss:.4f}, lr={current_lr:.2e}, GPU mem={memory_used:.2f}GB")
                    else:
                        logger.info(f"Step {global_step}: avg_loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": avg_loss,
                                "learning_rate": current_lr,
                                "epoch": epoch,
                            },
                            step=global_step,
                        )
                    total_loss = 0
                
                if global_step % EVAL_STEPS == 0:
                    eval_loss = evaluate(model, val_dataloader, accelerator)
                    logger.info(f"Step {global_step}: eval_loss={eval_loss:.4f}")
                    
                    if accelerator.is_main_process:
                        accelerator.log({"eval_loss": eval_loss}, step=global_step)
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint(
                                model, tokenizer, accelerator, 
                                os.path.join(OUTPUT_DIR, "best_model")
                            )
                            logger.info(f"Saved best model with eval_loss={eval_loss:.4f}")
                
                if global_step % SAVE_STEPS == 0:
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    logger.info(f"Saving checkpoint at step {global_step}")
                    
                    try:
                        # Add timeout to prevent hanging
                        accelerator.save_state(checkpoint_dir)
                        logger.info(f"Checkpoint saved to {checkpoint_dir}")
                        
                        # Only cleanup on main process and add error handling
                        if accelerator.is_main_process:
                            try:
                                cleanup_old_checkpoints(OUTPUT_DIR, keep_last=3)
                            except Exception as e:
                                logger.warning(f"Checkpoint cleanup failed: {e}")
                        
                        # Force synchronization
                        accelerator.wait_for_everyone()
                        
                    except Exception as e:
                        logger.error(f"Checkpoint saving failed: {e}")
                        # Continue training even if checkpoint fails
                        pass
        
        progress_bar.close()
        
        eval_loss = evaluate(model, val_dataloader, accelerator)
        logger.info(f"Epoch {epoch + 1} finished: eval_loss={eval_loss:.4f}")
        
        if accelerator.is_main_process:
            accelerator.log({"epoch_eval_loss": eval_loss}, step=global_step)
    
    # Save final checkpoint
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_model")
    logger.info("Saving final model...")
    accelerator.save_state(final_checkpoint_dir)
    save_checkpoint(
        model, tokenizer, accelerator,
        final_checkpoint_dir
    )
    
    logger.info("Training completed!")
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
    
    # Clear cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save with DeepSpeed state dict if using DeepSpeed
        if USE_DEEPSPEED:
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True  # Use safetensors for better memory efficiency
            )
        else:
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True
            )
        
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
        
        logger = get_logger(__name__)
        logger.info(f"Saved checkpoint to {output_dir}")

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