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
deepspeed --num_gpus=2 ToW_Training_llama.py
torchrun --nproc_per_node=4 ToW_Training_llama.py
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
from datetime import datetime, timedelta
import time
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
import torch.nn.functional as F

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_scheduler, BitsAndBytesConfig
)
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functools import partial
import json
from pathlib import Path

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_from_jsonl(raw_file):
    """Read data from JSONL file in prompt-completion format"""
    outputs = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Convert to prompt-completion format if needed
                    if 'prompt' in data and 'completion' in data:
                        outputs.append({"prompt": data['prompt'], "completion": data['completion']})
                    elif 'input' in data and 'output' in data:
                        outputs.append({"prompt": data['input'], "completion": data['output']})
                    else:
                        logger.warning(f"Skipping line with unexpected format: {line[:100]}...")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    return outputs

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    """Encode example with prompt-completion format and proper masking"""
    prompt = example['prompt']
    completion = example['completion']
    
    # Format the conversation
    if add_bos:
        full_text = tokenizer.bos_token + prompt + completion + tokenizer.eos_token
    else:
        full_text = prompt + completion + tokenizer.eos_token
    
    # Tokenize the full text
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors="pt"
    )
    
    # Tokenize just the prompt to find where to start learning
    if add_bos:
        prompt_text = tokenizer.bos_token + prompt
    else:
        prompt_text = prompt
    
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors="pt"
    )
    
    input_ids = tokenized_full['input_ids'].squeeze()
    labels = input_ids.clone()
    
    # Mask the prompt part in labels (set to -100 to ignore in loss calculation)
    prompt_length = len(tokenized_prompt['input_ids'].squeeze())
    labels[:prompt_length] = -100
    
    # Also mask padding tokens
    labels[input_ids == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': tokenized_full['attention_mask'].squeeze()
    }

def compute_loss_function(outputs, batch, reduce_loss="mean"):
    """Compute loss with sum or mean reduction"""
    if reduce_loss == "mean":
        return outputs.loss
    else:
        # sum reduction implementation
        logits = outputs.logits
        labels = batch["labels"]
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Calculate cross entropy with sum reduction
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='sum')
        return loss

def save_with_accelerate(accelerator, model, tokenizer, output_dir):
    """Save model and tokenizer using accelerate"""
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")

def get_last_checkpoint_path(output_dir):
    """Get the last checkpoint path for resuming"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    return os.path.join(output_dir, checkpoint_dirs[-1])


class FlatArguments:
    """Simple arguments class for training configuration"""
    def __init__(self):
        # Model and tokenizer
        self.model_name = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
        self.tokenizer_name = None  # Use model_name by default
        
        # Training data
        self.data_path = "../4_tow_generation/tow_data/final_multiple_tow.jsonl"
        self.max_seq_length = 256
        
        # Training hyperparameters
        self.learning_rate = 1e-5
        self.num_train_epochs = 10
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.gradient_accumulation_steps = 8
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.lr_scheduler_type = "cosine"
        
        # Output and logging
        self.output_dir = "ToW_Models_3/Llama-3.2-3B-Instruct-ToW"
        self.logging_steps = 250
        self.eval_steps = 250
        self.save_steps = 250
        self.save_total_limit = 6
        
        # System settings
        self.fp16 = False
        self.bf16 = True
        self.dataloader_num_workers = 1
        self.gradient_checkpointing = False
        
        # LoRA settings
        self.use_lora = True
        self.lora_r = 64
        self.lora_alpha = 16
        self.lora_dropout = 0.2
        
        # Quantization
        self.use_quantization = True
        
        # Special tokens
        self.tow_start_token = "<ToW>"
        self.tow_end_token = "</ToW>"
        
        # Loss configuration
        self.reduce_loss = "mean"  # or "sum"


def default_data_collator(features):
    """Simple data collator for the manual training loop"""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    
    # Extract features
    input_ids = [torch.tensor(f['input_ids']) for f in features]
    labels = [torch.tensor(f['labels']) for f in features]
    attention_mask = [torch.tensor(f['attention_mask']) for f in features]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


def create_optimizer_and_scheduler(model, args, num_training_steps):
    """Create optimizer and learning rate scheduler"""
    # Get parameters that require gradients
    params_with_grad = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        params_with_grad,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create learning rate scheduler
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, lr_scheduler

def load_and_prepare_model(args):
    """Load and prepare the model with LoRA and quantization"""
    logger.info(f"Loading model: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': [args.tow_start_token, args.tow_end_token]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added_tokens} special tokens. New vocab size: {len(tokenizer)}")
    
    # Quantization config
    quantization_config = None
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto" if not args.use_quantization else None
    )
    
    # Resize embeddings
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        logger.info(f"Resized embeddings to: {model.get_input_embeddings().weight.shape[0]}")
    
    # Initialize new token embeddings using DeepSpeed compatible approach
    try:
        import deepspeed
        with deepspeed.zero.GatheredParameters(model.get_input_embeddings().weight, modifier_rank=None):
            embeddings = model.get_input_embeddings()
            # Use a base token for initialization (like '---')
            base_token_ids = tokenizer.encode('---', add_special_tokens=False)
            if base_token_ids:
                base_token_id = base_token_ids[0]
                init_embeddings = embeddings.weight.data[base_token_id, :]
                # Initialize the last two tokens (our special tokens)
                embeddings.weight.data[len(tokenizer)-1, :] = init_embeddings
                embeddings.weight.data[len(tokenizer)-2, :] = init_embeddings
                logger.info("Initialized special token embeddings")
    except ImportError:
        # Fallback without DeepSpeed
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            base_token_ids = tokenizer.encode('---', add_special_tokens=False)
            if base_token_ids:
                base_token_id = base_token_ids[0]
                init_embeddings = embeddings.weight.data[base_token_id, :]
                embeddings.weight.data[len(tokenizer)-1, :] = init_embeddings
                embeddings.weight.data[len(tokenizer)-2, :] = init_embeddings
                logger.info("Initialized special token embeddings (fallback)")
    
    # Prepare for quantization training
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA setup complete. Trainable parameters:")
        model.print_trainable_parameters()
    
    return model, tokenizer


def create_dataset(args, tokenizer):
    """Create training and evaluation datasets"""
    # Read data from JSONL file
    logger.info(f"Reading data from: {args.data_path}")
    raw_data = read_from_jsonl(args.data_path)
    logger.info(f"Loaded {len(raw_data)} examples")
    
    # Create encoding function
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        add_bos=False
    )
    
    # Convert to dataset and encode
    dataset = Dataset.from_list(raw_data)
    encoded_dataset = dataset.map(
        encode_function,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Encoding dataset"
    )
    
    # Split into train and eval
    train_test_split = encoded_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def evaluate_model(model, eval_dataloader, tokenizer, accelerator):
    """Evaluate the model and calculate metrics"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gather losses across all processes
            losses = accelerator.gather(loss.repeat(batch['input_ids'].shape[0]))
            total_loss += losses.mean().item()
            
            # Calculate token accuracy
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch['labels']
            
            # Mask padding tokens
            mask = (labels != -100)
            correct = ((predictions == labels) & mask).sum()
            total = mask.sum()
            
            # Gather across processes
            correct_gathered = accelerator.gather(correct)
            total_gathered = accelerator.gather(total)
            
            correct_tokens += correct_gathered.sum().item()
            total_tokens += total_gathered.sum().item()
    
    avg_loss = total_loss / len(eval_dataloader)
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        'eval_loss': avg_loss,
        'eval_accuracy': token_accuracy,
        'eval_perplexity': perplexity
    }
    
    model.train()
    return metrics


# Simple argument parser
class ArgumentParserPlus:
    def __init__(self, args_class):
        self.args_class = args_class
    
    def parse(self):
        return self.args_class()


def train_model(args):
    """Main training function using Accelerate"""
    # Initialize accelerator
    init_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
        kwargs_handlers=[init_kwargs]
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(output_dir / "training.log")
            ]
        )
    
    logger.info(f"Starting training with accelerator: {accelerator}")
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(args)
    
    # Create datasets
    train_dataset, eval_dataset = create_dataset(args, tokenizer)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    
    # Calculate training steps
    num_training_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Check for resume checkpoint
    last_checkpoint = get_last_checkpoint_path(args.output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        accelerator.load_state(last_checkpoint)
    
    # Training loop
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    
    completed_steps = 0
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = compute_loss_function(outputs, batch, args.reduce_loss)
                
                # Scale loss for gradient accumulation
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                
                # Logging
                if completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.logging_steps
                    logger.info(f"Step: {completed_steps}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0
                
                # Evaluation
                if completed_steps % args.eval_steps == 0:
                    eval_metrics = evaluate_model(model, eval_dataloader, tokenizer, accelerator)
                    logger.info(f"Evaluation at step {completed_steps}: {eval_metrics}")
                
                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{completed_steps}"
                    accelerator.save_state(checkpoint_dir)
                    logger.info(f"Checkpoint saved at step {completed_steps}")
            
            if completed_steps >= num_training_steps:
                break
        
        if completed_steps >= num_training_steps:
            break
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader, tokenizer, accelerator)
    logger.info(f"Final evaluation: {final_metrics}")
    
    # Save final model
    logger.info("Saving final model...")
    save_with_accelerate(accelerator, model, tokenizer, args.output_dir)
    
    # Save training arguments
    if accelerator.is_main_process:
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        with open(output_dir / "training_args.json", 'w') as f:
            json.dump(args_dict, f, indent=2)
        
        with open(output_dir / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    logger.info("Training completed!")
    
def main(args: FlatArguments):
    """Main function with Accelerate-based training"""
    logger.info("ToW Training with Accelerate - finetune.py style")
    
    # Validate data path
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    # Run training
    train_model(args)
    
    logger.info(f"Training completed successfully! Model saved to: {args.output_dir}")





if __name__ == "__main__":
    parser = ArgumentParserPlus(FlatArguments)
    args = parser.parse()
    main(args) 


