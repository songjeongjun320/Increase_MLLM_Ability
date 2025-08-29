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
from datetime import datetime
import time
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
import torch.nn.functional as F

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


class EnhancedJsonLoggingCallback(TrainerCallback):
    """Enhanced callback to log comprehensive metrics to a JSON file."""

    def __init__(self, log_file):
        self.log_file = log_file
        # Clear the log file at the beginning of training
        with open(self.log_file, 'w') as f:
            f.write("[\n")
        self.first_log = True
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is not None:
            enhanced_logs = logs.copy()
            
            # Add timestamp
            enhanced_logs['timestamp'] = datetime.now().isoformat()
            enhanced_logs['training_time_elapsed'] = time.time() - self.start_time
            
            # Add perplexity if we have loss
            if 'train_loss' in enhanced_logs:
                enhanced_logs['train_perplexity'] = float(np.exp(enhanced_logs['train_loss']))
            if 'eval_loss' in enhanced_logs:
                enhanced_logs['eval_perplexity'] = float(np.exp(enhanced_logs['eval_loss']))
            
            # Add GPU memory information
            if torch.cuda.is_available():
                enhanced_logs['gpu_memory'] = {
                    'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                    'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
                }
                
                # Add GPU utilization if GPUtil is available
                if GPUtil:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            enhanced_logs['gpu_utilization'] = {
                                'gpu_percent': gpus[0].load * 100,
                                'memory_percent': gpus[0].memoryUtil * 100,
                                'temperature': gpus[0].temperature
                            }
                    except Exception:
                        pass
            
            # Add system memory
            try:
                memory = psutil.virtual_memory()
                enhanced_logs['system_memory'] = {
                    'used_gb': memory.used / 1e9,
                    'available_gb': memory.available / 1e9,
                    'percent': memory.percent
                }
            except Exception:
                pass
            
            # Add detailed learning rate information
            if hasattr(state, 'log_history') and state.log_history:
                last_lr = None
                for log_entry in reversed(state.log_history):
                    if 'learning_rate' in log_entry:
                        last_lr = log_entry['learning_rate']
                        break
                if last_lr is not None:
                    enhanced_logs['current_learning_rate'] = last_lr
            
            # Append logs to the JSON file
            with open(self.log_file, 'a') as f:
                if not self.first_log:
                    f.write(",\n")
                json.dump(enhanced_logs, f, indent=2, default=str)
                self.first_log = False
    
    def on_train_end(self, args, state, control, **kwargs):
        # Close the JSON array
        with open(self.log_file, 'a') as f:
            f.write("\n]\n")


class ToWMetricsCallback(TrainerCallback):
    """Callback to track ToW-specific metrics during training."""
    
    def __init__(self, tokenizer, log_file_prefix="tow_metrics"):
        self.tokenizer = tokenizer
        self.tow_start_id = tokenizer.convert_tokens_to_ids("<ToW>")
        self.tow_end_id = tokenizer.convert_tokens_to_ids("</ToW>")
        self.log_file_prefix = log_file_prefix
        
    def on_evaluate(self, args, state, control, logs=None, model=None, eval_dataloader=None, **kwargs):
        """Calculate ToW-specific metrics during evaluation."""
        if model is None or eval_dataloader is None:
            return
            
        model.eval()
        total_tow_tokens = 0
        correct_tow_predictions = 0
        total_tow_sequences = 0
        tow_losses = []
        
        with torch.no_grad():
            # Limit evaluation to first 50 batches to avoid excessive evaluation time
            for i, batch in enumerate(eval_dataloader):
                if i >= 50:  # Limit to 50 batches
                    break
                    
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Get model predictions
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits
                
                # Calculate predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Find ToW tokens in this batch
                for seq_idx in range(input_ids.size(0)):
                    sequence_labels = labels[seq_idx]
                    sequence_predictions = predicted_ids[seq_idx]
                    
                    # Find ToW regions
                    tow_start_positions = (sequence_labels == self.tow_start_id).nonzero(as_tuple=True)[0]
                    tow_end_positions = (sequence_labels == self.tow_end_id).nonzero(as_tuple=True)[0]
                    
                    if len(tow_start_positions) > 0 and len(tow_end_positions) > 0:
                        total_tow_sequences += 1
                        
                        # For each ToW region, calculate accuracy
                        for start_pos in tow_start_positions:
                            # Find corresponding end position
                            end_positions = tow_end_positions[tow_end_positions > start_pos]
                            if len(end_positions) > 0:
                                end_pos = end_positions[0]
                                
                                # Count tokens in ToW region
                                tow_region_length = end_pos - start_pos + 1
                                total_tow_tokens += tow_region_length
                                
                                # Count correct predictions in ToW region
                                tow_labels = sequence_labels[start_pos:end_pos+1]
                                tow_preds = sequence_predictions[start_pos:end_pos+1]
                                correct_tow_predictions += (tow_labels == tow_preds).sum().item()
                                
                                # Calculate ToW-specific loss
                                tow_logits = logits[seq_idx, start_pos:end_pos+1]
                                tow_loss = F.cross_entropy(tow_logits, tow_labels, reduction='mean')
                                tow_losses.append(tow_loss.item())
        
        # Calculate ToW metrics
        tow_metrics = {}
        
        if total_tow_tokens > 0:
            tow_accuracy = correct_tow_predictions / total_tow_tokens
            tow_metrics['tow_token_accuracy'] = tow_accuracy
            tow_metrics['total_tow_tokens_evaluated'] = total_tow_tokens
            tow_metrics['correct_tow_predictions'] = correct_tow_predictions
            
        if tow_losses:
            avg_tow_loss = np.mean(tow_losses)
            tow_metrics['tow_loss'] = avg_tow_loss
            tow_metrics['tow_perplexity'] = np.exp(avg_tow_loss)
            
        tow_metrics['total_tow_sequences'] = total_tow_sequences
        
        # Add individual token tracking
        tow_start_correct = 0
        tow_end_correct = 0
        tow_start_total = 0
        tow_end_total = 0
        
        # Count individual <ToW> and </ToW> token accuracy
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= 50:  # Same limit as above
                    break
                    
                input_ids = batch['input_ids']
                labels = batch['labels']
                outputs = model(input_ids=input_ids, labels=labels)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                
                # Count <ToW> tokens
                tow_start_mask = (labels == self.tow_start_id)
                tow_start_total += tow_start_mask.sum().item()
                tow_start_correct += ((predicted_ids == labels) & tow_start_mask).sum().item()
                
                # Count </ToW> tokens
                tow_end_mask = (labels == self.tow_end_id)
                tow_end_total += tow_end_mask.sum().item()
                tow_end_correct += ((predicted_ids == labels) & tow_end_mask).sum().item()
        
        if tow_start_total > 0:
            tow_metrics['tow_start_accuracy'] = tow_start_correct / tow_start_total
            tow_metrics['tow_start_total'] = tow_start_total
            tow_metrics['tow_start_correct'] = tow_start_correct
            
        if tow_end_total > 0:
            tow_metrics['tow_end_accuracy'] = tow_end_correct / tow_end_total
            tow_metrics['tow_end_total'] = tow_end_total
            tow_metrics['tow_end_correct'] = tow_end_correct
        
        # Add timestamp
        tow_metrics['timestamp'] = datetime.now().isoformat()
        
        # Update logs with ToW metrics
        if logs is not None:
            logs.update(tow_metrics)
            
        # Save ToW metrics to separate file
        tow_log_file = Path(args.output_dir) / f"{self.log_file_prefix}_{state.epoch or 'final'}.json"
        with open(tow_log_file, 'w') as f:
            json.dump(tow_metrics, f, indent=2, default=str)
            
        logger.info(f"ToW Metrics: {tow_metrics}")
        
        model.train()
        return tow_metrics


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


class EnhancedToWTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs.get('tokenizer')
        
        # Add token-level accuracy tracking
        self.token_correct = 0
        self.token_total = 0

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
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to add detailed loss tracking."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Calculate token-level accuracy
            logits = outputs.get('logits')
            if logits is not None:
                predictions = torch.argmax(logits, dim=-1)
                
                # Mask out padding tokens for accuracy calculation
                mask = (labels != -100)
                correct_tokens = ((predictions == labels) & mask).sum().item()
                total_tokens = mask.sum().item()
                
                self.token_correct += correct_tokens
                self.token_total += total_tokens
        
        loss = outputs.get('loss')
        
        # Add gradient norm to the loss tracking
        if hasattr(model, 'parameters'):
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Store gradient norm for logging
            if not hasattr(self, '_gradient_norms'):
                self._gradient_norms = []
            self._gradient_norms.append(total_norm)
        
        return (loss, outputs) if return_outputs else loss
        
    def log(self, logs):
        """Override log to add token-level accuracy."""
        # Add token-level accuracy if we have tracked tokens
        if self.token_total > 0:
            logs['token_accuracy'] = self.token_correct / self.token_total
            logs['tokens_correct'] = self.token_correct
            logs['tokens_total'] = self.token_total
            
            # Reset counters for next logging period
            self.token_correct = 0
            self.token_total = 0
        
        # Add gradient norm if available
        if hasattr(self, '_gradient_norms') and self._gradient_norms:
            logs['gradient_norm'] = np.mean(self._gradient_norms)
            logs['gradient_norm_std'] = np.std(self._gradient_norms)
            self._gradient_norms = []  # Reset for next period
        
        super().log(logs)


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
            padding_side='right',
            local_files_only=True  # 로컬 파일만 사용
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
            r=64,
            lora_alpha=16,
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
            deepspeed="./deepspeed_config_llama.json",
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

        # Setup enhanced logging callbacks
        json_log_path = self.output_dir / "enhanced_training_metrics.json"
        json_logging_callback = EnhancedJsonLoggingCallback(log_file=json_log_path)
        logger.info(f"Enhanced metrics will be logged to {json_log_path}")
        
        # Setup ToW-specific metrics callback
        tow_metrics_callback = ToWMetricsCallback(
            tokenizer=tokenizer, 
            log_file_prefix="tow_detailed_metrics"
        )
        logger.info("ToW-specific metrics callback initialized")

        trainer = EnhancedToWTrainer(
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
                json_logging_callback,  # Enhanced JSON logging
                tow_metrics_callback    # ToW-specific metrics
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
        
        # Generate comprehensive training report
        logger.info("Generating comprehensive training report...")
        try:
            comprehensive_report = generate_comprehensive_training_report(
                self.output_dir, train_result, self.model_config, self.training_config
            )
            report_path = self.output_dir / "comprehensive_training_report.json"
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            logger.info(f"Comprehensive training report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
        
        logger.info(f"Training completed for {self.model_config.name}")
        logger.info(f"Model saved to: {self.output_dir}")
        
        return train_result


def generate_comprehensive_training_report(output_dir, train_result, model_config, training_config):
    """Generate a comprehensive training report with all metrics and analysis."""
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_name": model_config.name,
            "model_id": model_config.model_id,
            "report_version": "1.0"
        },
        "training_summary": {
            "model_config": {
                "name": model_config.name,
                "model_id": model_config.model_id,
                "use_quantization": model_config.use_quantization,
                "torch_dtype": str(model_config.torch_dtype)
            },
            "training_config": training_config.__dict__,
            "training_results": train_result.metrics if hasattr(train_result, 'metrics') else {}
        }
    }
    
    # Load and analyze enhanced metrics
    enhanced_metrics_path = Path(output_dir) / "enhanced_training_metrics.json"
    if enhanced_metrics_path.exists():
        try:
            with open(enhanced_metrics_path, 'r') as f:
                enhanced_metrics = json.load(f)
            
            report["training_analysis"] = {
                "total_log_entries": len(enhanced_metrics),
                "training_progression": {
                    "initial_loss": None,
                    "final_loss": None,
                    "best_eval_loss": None,
                    "loss_reduction": None,
                    "perplexity_improvement": None
                },
                "learning_dynamics": {
                    "learning_rate_schedule": [],
                    "gradient_norms": [],
                    "token_accuracy_progression": []
                },
                "computational_efficiency": {
                    "average_gpu_utilization": None,
                    "peak_gpu_memory_gb": None,
                    "average_gpu_memory_gb": None,
                    "training_time_hours": None
                }
            }
            
            # Analyze training progression
            train_losses = [entry.get('train_loss') for entry in enhanced_metrics if entry.get('train_loss')]
            eval_losses = [entry.get('eval_loss') for entry in enhanced_metrics if entry.get('eval_loss')]
            
            if train_losses:
                report["training_analysis"]["training_progression"]["initial_loss"] = train_losses[0]
                report["training_analysis"]["training_progression"]["final_loss"] = train_losses[-1]
                report["training_analysis"]["training_progression"]["loss_reduction"] = train_losses[0] - train_losses[-1]
            
            if eval_losses:
                report["training_analysis"]["training_progression"]["best_eval_loss"] = min(eval_losses)
                
                # Calculate perplexity improvement
                initial_perplexity = np.exp(eval_losses[0]) if eval_losses else None
                final_perplexity = np.exp(min(eval_losses)) if eval_losses else None
                if initial_perplexity and final_perplexity:
                    report["training_analysis"]["training_progression"]["perplexity_improvement"] = initial_perplexity - final_perplexity
            
            # Analyze computational efficiency
            gpu_utils = [entry.get('gpu_utilization', {}).get('gpu_percent') for entry in enhanced_metrics if entry.get('gpu_utilization')]
            gpu_memories = [entry.get('gpu_memory', {}).get('allocated_gb') for entry in enhanced_metrics if entry.get('gpu_memory')]
            training_times = [entry.get('training_time_elapsed') for entry in enhanced_metrics if entry.get('training_time_elapsed')]
            
            if gpu_utils:
                report["training_analysis"]["computational_efficiency"]["average_gpu_utilization"] = np.mean(gpu_utils)
            if gpu_memories:
                report["training_analysis"]["computational_efficiency"]["peak_gpu_memory_gb"] = max(gpu_memories)
                report["training_analysis"]["computational_efficiency"]["average_gpu_memory_gb"] = np.mean(gpu_memories)
            if training_times:
                report["training_analysis"]["computational_efficiency"]["training_time_hours"] = max(training_times) / 3600
            
            # Extract learning dynamics
            learning_rates = [entry.get('current_learning_rate') for entry in enhanced_metrics if entry.get('current_learning_rate')]
            gradient_norms = [entry.get('gradient_norm') for entry in enhanced_metrics if entry.get('gradient_norm')]
            token_accuracies = [entry.get('token_accuracy') for entry in enhanced_metrics if entry.get('token_accuracy')]
            
            report["training_analysis"]["learning_dynamics"]["learning_rate_schedule"] = learning_rates
            report["training_analysis"]["learning_dynamics"]["gradient_norms"] = gradient_norms
            report["training_analysis"]["learning_dynamics"]["token_accuracy_progression"] = token_accuracies
            
        except Exception as e:
            report["training_analysis"] = {"error": f"Failed to analyze enhanced metrics: {str(e)}"}
    
    # Analyze ToW-specific performance
    tow_metrics_files = list(Path(output_dir).glob("tow_detailed_metrics_*.json"))
    if tow_metrics_files:
        report["tow_performance_analysis"] = {
            "metrics_files_found": len(tow_metrics_files),
            "performance_progression": [],
            "final_performance": {}
        }
        
        try:
            all_tow_metrics = []
            for tow_file in sorted(tow_metrics_files):
                with open(tow_file, 'r') as f:
                    tow_data = json.load(f)
                    all_tow_metrics.append(tow_data)
            
            report["tow_performance_analysis"]["performance_progression"] = all_tow_metrics
            
            if all_tow_metrics:
                final_metrics = all_tow_metrics[-1]
                report["tow_performance_analysis"]["final_performance"] = {
                    "tow_token_accuracy": final_metrics.get('tow_token_accuracy'),
                    "tow_start_accuracy": final_metrics.get('tow_start_accuracy'),
                    "tow_end_accuracy": final_metrics.get('tow_end_accuracy'),
                    "tow_perplexity": final_metrics.get('tow_perplexity'),
                    "total_tow_sequences": final_metrics.get('total_tow_sequences')
                }
                
        except Exception as e:
            report["tow_performance_analysis"]["error"] = f"Failed to analyze ToW metrics: {str(e)}"
    
    # Resource usage analysis
    try:
        memory_info = psutil.virtual_memory()
        report["resource_usage_analysis"] = {
            "system_memory_gb": memory_info.total / 1e9,
            "available_memory_gb": memory_info.available / 1e9,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            report["resource_usage_analysis"]["gpu_info"] = {
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "current_gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "max_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
            }
            
    except Exception as e:
        report["resource_usage_analysis"] = {"error": f"Failed to collect resource info: {str(e)}"}
    
    return report


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