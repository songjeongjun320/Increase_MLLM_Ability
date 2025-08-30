import re
import logging
import math
import os
import random
import subprocess
import time
import json
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import List, Optional, Union

import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    FalconForCausalLM,
    get_scheduler,
)
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
        "../4_tow_generation/tow_data/final_multiple_tow.jsonl"
    ])
    output_base_dir: str = "ToW_Models_3"
    
    # Training hyperparameters
    learning_rate: float = 1e-5  # This will be a fallback
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
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 250
    logging_steps: int = 250
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    dataloader_num_workers: int = 1
    remove_unused_columns: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False


MODEL_CONFIGS = [
    ModelConfig(
        name="Llama-3.2-3B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        use_quantization=True,
    ),
]

MODEL_OUTPUT_PATH = "tow_models"
