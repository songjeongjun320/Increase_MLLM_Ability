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
deepspeed --num_gpus=2 ToW_Training_deepseek.py
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
import time
import psutil
import GPUtil

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
import torch.nn.functional as F

# Setup logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EnhancedJsonLoggingCallback(TrainerCallback):
    """Enhanced callback to log comprehensive metrics to JSON file."""

    def __init__(self, log_file, tokenizer):
        self.log_file = log_file
        self.tokenizer = tokenizer
        self.tow_start_id = tokenizer.convert_tokens_to_ids("<ToW>") if "<ToW>" in tokenizer.get_vocab() else None
        self.tow_end_id = tokenizer.convert_tokens_to_ids("</ToW>") if "</ToW>" in tokenizer.get_vocab() else None
        self.start_time = time.time()
        
        # Clear the log file at the beginning of training
        with open(self.log_file, 'w') as f:
            f.write("[\n")
        self.first_log = True

    def _get_gpu_memory_info(self):
        """Get GPU memory usage information."""
        gpu_info = {}
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    gpu_info[f'gpu_{i}'] = {
                        'memory_allocated_gb': round(allocated, 2),
                        'memory_reserved_gb': round(reserved, 2),
                        'memory_utilization_%': round((allocated / (reserved + 1e-8)) * 100, 2)
                    }
        except Exception as e:
            gpu_info['error'] = str(e)
        return gpu_info

    def _calculate_perplexity(self, loss_value):
        """Calculate perplexity from loss."""
        try:
            return round(torch.exp(torch.tensor(loss_value)).item(), 4)
        except:
            return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            enhanced_logs = logs.copy()
            
            # Add timestamp
            enhanced_logs['timestamp'] = datetime.now().isoformat()
            enhanced_logs['elapsed_time_minutes'] = round((time.time() - self.start_time) / 60, 2)
            
            # Add perplexity calculations
            if 'train_loss' in logs:
                enhanced_logs['train_perplexity'] = self._calculate_perplexity(logs['train_loss'])
            if 'eval_loss' in logs:
                enhanced_logs['eval_perplexity'] = self._calculate_perplexity(logs['eval_loss'])
            
            # Add detailed learning rate info
            if 'learning_rate' in logs:
                enhanced_logs['learning_rate_scientific'] = f"{logs['learning_rate']:.2e}"
                enhanced_logs['learning_rate_scaled'] = logs['learning_rate'] * 1e6  # for easier viewing
            
            # Add GPU memory information
            enhanced_logs['gpu_memory'] = self._get_gpu_memory_info()
            
            # Add system resource info
            try:
                enhanced_logs['system_resources'] = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'available_memory_gb': round(psutil.virtual_memory().available / 1024**3, 2)
                }
            except:
                pass
            
            # Add training state info
            enhanced_logs['training_state'] = {
                'global_step': state.global_step,
                'epoch': round(state.epoch, 4),
                'max_steps': state.max_steps,
                'num_train_epochs': args.num_train_epochs
            }
            
            # Append logs to the JSON file
            with open(self.log_file, 'a') as f:
                if not self.first_log:
                    f.write(",\n")
                json.dump(enhanced_logs, f, indent=2, ensure_ascii=False)
                self.first_log = False
    
    def on_train_end(self, args, state, control, **kwargs):
        # Close the JSON array
        with open(self.log_file, 'a') as f:
            f.write("\n]\n")


class ToWMetricsCallback(TrainerCallback):
    """Callback to track ToW-specific metrics during training."""
    
    def __init__(self, tokenizer, output_dir):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.tow_start_id = tokenizer.convert_tokens_to_ids("<ToW>") if "<ToW>" in tokenizer.get_vocab() else None
        self.tow_end_id = tokenizer.convert_tokens_to_ids("</ToW>") if "</ToW>" in tokenizer.get_vocab() else None
        self.tow_metrics_log = []
        
        logger.info(f"ToW tokens - Start ID: {self.tow_start_id}, End ID: {self.tow_end_id}")
    
    def on_evaluate(self, args, state, control, logs=None, model=None, eval_dataloader=None, **kwargs):
        """Calculate ToW-specific metrics during evaluation."""
        if model is None or eval_dataloader is None:
            return
            
        model.eval()
        tow_correct_start = 0
        tow_correct_end = 0
        tow_total_start = 0
        tow_total_end = 0
        tow_sequence_loss = 0.0
        total_tow_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                if num_batches >= 50:  # Limit to avoid too long evaluation
                    break
                    
                input_ids = batch['input_ids'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits
                
                # Find ToW token positions
                if self.tow_start_id is not None:
                    start_positions = (labels == self.tow_start_id)
                    tow_total_start += start_positions.sum().item()
                    
                    if start_positions.any():
                        start_predictions = torch.argmax(logits[start_positions], dim=-1)
                        tow_correct_start += (start_predictions == self.tow_start_id).sum().item()
                
                if self.tow_end_id is not None:
                    end_positions = (labels == self.tow_end_id)
                    tow_total_end += end_positions.sum().item()
                    
                    if end_positions.any():
                        end_predictions = torch.argmax(logits[end_positions], dim=-1)
                        tow_correct_end += (end_predictions == self.tow_end_id).sum().item()
                
                # Calculate ToW-specific loss
                tow_positions = torch.zeros_like(labels, dtype=torch.bool)
                if self.tow_start_id is not None:
                    tow_positions |= (labels == self.tow_start_id)
                if self.tow_end_id is not None:
                    tow_positions |= (labels == self.tow_end_id)
                
                if tow_positions.any():
                    tow_logits = logits[tow_positions]
                    tow_labels = labels[tow_positions]
                    tow_loss = F.cross_entropy(tow_logits, tow_labels)
                    tow_sequence_loss += tow_loss.item()
                    total_tow_tokens += tow_positions.sum().item()
                
                num_batches += 1
        
        # Calculate metrics
        tow_start_accuracy = (tow_correct_start / max(tow_total_start, 1)) * 100
        tow_end_accuracy = (tow_correct_end / max(tow_total_end, 1)) * 100
        tow_avg_loss = tow_sequence_loss / max(num_batches, 1)
        tow_perplexity = np.exp(tow_avg_loss) if tow_avg_loss > 0 else float('inf')
        
        # Log ToW metrics
        tow_metrics = {
            'step': state.global_step,
            'epoch': round(state.epoch, 4),
            'tow_start_accuracy': round(tow_start_accuracy, 4),
            'tow_end_accuracy': round(tow_end_accuracy, 4),
            'tow_overall_accuracy': round((tow_start_accuracy + tow_end_accuracy) / 2, 4),
            'tow_loss': round(tow_avg_loss, 6),
            'tow_perplexity': round(tow_perplexity, 4),
            'tow_token_counts': {
                'start_tokens': tow_total_start,
                'end_tokens': tow_total_end,
                'total_tow_tokens': total_tow_tokens
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.tow_metrics_log.append(tow_metrics)
        
        # Save ToW metrics to file
        tow_metrics_file = self.output_dir / "tow_metrics.json"
        with open(tow_metrics_file, 'w') as f:
            json.dump(self.tow_metrics_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ToW Metrics - Start Acc: {tow_start_accuracy:.2f}%, End Acc: {tow_end_accuracy:.2f}%, Loss: {tow_avg_loss:.6f}, Perplexity: {tow_perplexity:.4f}")
        
        model.train()  # Return to training mode


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
        self.tow_start_id = None
        self.tow_end_id = None
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.tow_start_id = self.tokenizer.convert_tokens_to_ids("<ToW>") if "<ToW>" in self.tokenizer.get_vocab() else None
            self.tow_end_id = self.tokenizer.convert_tokens_to_ids("</ToW>") if "</ToW>" in self.tokenizer.get_vocab() else None

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
        """
        Enhanced loss computation with token-level accuracy tracking.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard loss computation
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None

        # Calculate additional metrics during evaluation
        if not model.training and labels is not None:
            with torch.no_grad():
                # Token-level accuracy
                predictions = torch.argmax(shift_logits, dim=-1)
                valid_tokens = (shift_labels != -100)
                
                if valid_tokens.sum() > 0:
                    token_accuracy = (predictions == shift_labels)[valid_tokens].float().mean()
                    
                    # Log token accuracy periodically
                    if hasattr(self.state, 'global_step') and self.state.global_step % 100 == 0:
                        logger.info(f"Step {self.state.global_step} - Token Accuracy: {token_accuracy:.4f}")
                
                # ToW-specific accuracy
                if self.tow_start_id is not None or self.tow_end_id is not None:
                    tow_positions = torch.zeros_like(shift_labels, dtype=torch.bool)
                    if self.tow_start_id is not None:
                        tow_positions |= (shift_labels == self.tow_start_id)
                    if self.tow_end_id is not None:
                        tow_positions |= (shift_labels == self.tow_end_id)
                    
                    if tow_positions.sum() > 0:
                        tow_accuracy = (predictions == shift_labels)[tow_positions].float().mean()
                        
                        if hasattr(self.state, 'global_step') and self.state.global_step % 100 == 0:
                            logger.info(f"Step {self.state.global_step} - ToW Token Accuracy: {tow_accuracy:.4f}")

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Enhanced training step with detailed loss breakdown.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.deepspeed:
            loss = self.deepspeed.backward(loss)
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # Log detailed loss info periodically
        if hasattr(self.state, 'global_step') and self.state.global_step % 250 == 0:
            logger.info(f"Step {self.state.global_step} - Detailed Loss: {loss.item():.6f}")
            
            # Log gradient norms
            if hasattr(model, 'parameters'):
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                logger.info(f"Step {self.state.global_step} - Gradient Norm: {total_norm:.6f}")

        return loss.detach()


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
    learning_rate: float = 5e-6  # This will be a fallback
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
    max_sequence_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
    
    # Other settings
    eval_strategy: str = "steps"
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 250
    logging_steps: int = 100
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.005
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False


MODEL_CONFIGS = [
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        use_quantization=True,
    ),
]


# Import the new caching utilities
from dataset_cache_utils import SmartToWDataProcessor


def generate_comprehensive_training_report(output_dir, model_config, training_config, train_result=None):
    """Generate a comprehensive training report with all metrics."""
    output_dir = Path(output_dir)
    
    # Load all available metrics files
    metrics_data = {}
    
    # Load main training metrics
    training_metrics_file = output_dir / "training_metrics.json"
    if training_metrics_file.exists():
        with open(training_metrics_file, 'r') as f:
            metrics_data['training_metrics'] = json.load(f)
    
    # Load ToW metrics
    tow_metrics_file = output_dir / "tow_metrics.json"
    if tow_metrics_file.exists():
        with open(tow_metrics_file, 'r') as f:
            metrics_data['tow_metrics'] = json.load(f)
    
    # Load training results
    training_results_file = output_dir / "training_results.json"
    if training_results_file.exists():
        with open(training_results_file, 'r') as f:
            metrics_data['training_results'] = json.load(f)
    
    # Analyze training progression
    training_analysis = {}
    if 'training_metrics' in metrics_data and metrics_data['training_metrics']:
        train_losses = [log.get('train_loss') for log in metrics_data['training_metrics'] if log.get('train_loss')]
        eval_losses = [log.get('eval_loss') for log in metrics_data['training_metrics'] if log.get('eval_loss')]
        learning_rates = [log.get('learning_rate') for log in metrics_data['training_metrics'] if log.get('learning_rate')]
        
        if train_losses:
            training_analysis['loss_progression'] = {
                'initial_train_loss': round(train_losses[0], 6),
                'final_train_loss': round(train_losses[-1], 6),
                'loss_improvement': round((train_losses[0] - train_losses[-1]) / train_losses[0] * 100, 2),
                'min_train_loss': round(min(train_losses), 6),
                'avg_train_loss': round(np.mean(train_losses), 6)
            }
        
        if eval_losses:
            training_analysis['eval_progression'] = {
                'initial_eval_loss': round(eval_losses[0], 6),
                'final_eval_loss': round(eval_losses[-1], 6),
                'best_eval_loss': round(min(eval_losses), 6),
                'avg_eval_loss': round(np.mean(eval_losses), 6)
            }
        
        if learning_rates:
            training_analysis['learning_rate_analysis'] = {
                'initial_lr': learning_rates[0],
                'final_lr': learning_rates[-1],
                'max_lr': max(learning_rates),
                'min_lr': min(learning_rates)
            }
    
    # Analyze ToW performance
    tow_analysis = {}
    if 'tow_metrics' in metrics_data and metrics_data['tow_metrics']:
        tow_data = metrics_data['tow_metrics']
        
        if tow_data:
            start_accuracies = [entry.get('tow_start_accuracy', 0) for entry in tow_data]
            end_accuracies = [entry.get('tow_end_accuracy', 0) for entry in tow_data]
            overall_accuracies = [entry.get('tow_overall_accuracy', 0) for entry in tow_data]
            tow_losses = [entry.get('tow_loss', 0) for entry in tow_data]
            tow_perplexities = [entry.get('tow_perplexity', 0) for entry in tow_data if entry.get('tow_perplexity', 0) != float('inf')]
            
            tow_analysis = {
                'accuracy_progression': {
                    'initial_start_accuracy': round(start_accuracies[0], 2) if start_accuracies else 0,
                    'final_start_accuracy': round(start_accuracies[-1], 2) if start_accuracies else 0,
                    'best_start_accuracy': round(max(start_accuracies), 2) if start_accuracies else 0,
                    'initial_end_accuracy': round(end_accuracies[0], 2) if end_accuracies else 0,
                    'final_end_accuracy': round(end_accuracies[-1], 2) if end_accuracies else 0,
                    'best_end_accuracy': round(max(end_accuracies), 2) if end_accuracies else 0,
                    'initial_overall_accuracy': round(overall_accuracies[0], 2) if overall_accuracies else 0,
                    'final_overall_accuracy': round(overall_accuracies[-1], 2) if overall_accuracies else 0,
                    'best_overall_accuracy': round(max(overall_accuracies), 2) if overall_accuracies else 0
                },
                'loss_and_perplexity': {
                    'initial_tow_loss': round(tow_losses[0], 6) if tow_losses else 0,
                    'final_tow_loss': round(tow_losses[-1], 6) if tow_losses else 0,
                    'best_tow_loss': round(min(tow_losses), 6) if tow_losses else 0,
                    'avg_tow_perplexity': round(np.mean(tow_perplexities), 4) if tow_perplexities else 0,
                    'best_tow_perplexity': round(min(tow_perplexities), 4) if tow_perplexities else 0
                }
            }
    
    # Resource usage analysis
    resource_analysis = {}
    if 'training_metrics' in metrics_data and metrics_data['training_metrics']:
        gpu_memory_data = []
        cpu_usage_data = []
        memory_usage_data = []
        
        for log in metrics_data['training_metrics']:
            if 'gpu_memory' in log and log['gpu_memory']:
                for gpu_id, gpu_info in log['gpu_memory'].items():
                    if isinstance(gpu_info, dict) and 'memory_allocated_gb' in gpu_info:
                        gpu_memory_data.append(gpu_info['memory_allocated_gb'])
            
            if 'system_resources' in log:
                sys_res = log['system_resources']
                if 'cpu_percent' in sys_res:
                    cpu_usage_data.append(sys_res['cpu_percent'])
                if 'memory_percent' in sys_res:
                    memory_usage_data.append(sys_res['memory_percent'])
        
        resource_analysis = {
            'gpu_memory_usage': {
                'avg_gpu_memory_gb': round(np.mean(gpu_memory_data), 2) if gpu_memory_data else 0,
                'max_gpu_memory_gb': round(max(gpu_memory_data), 2) if gpu_memory_data else 0,
                'min_gpu_memory_gb': round(min(gpu_memory_data), 2) if gpu_memory_data else 0
            },
            'system_resources': {
                'avg_cpu_usage_%': round(np.mean(cpu_usage_data), 2) if cpu_usage_data else 0,
                'max_cpu_usage_%': round(max(cpu_usage_data), 2) if cpu_usage_data else 0,
                'avg_memory_usage_%': round(np.mean(memory_usage_data), 2) if memory_usage_data else 0,
                'max_memory_usage_%': round(max(memory_usage_data), 2) if memory_usage_data else 0
            }
        }
    
    # Create comprehensive report
    comprehensive_report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'model_name': model_config.name,
            'model_id': model_config.model_id,
            'training_config_summary': {
                'learning_rate': training_config.learning_rate,
                'num_epochs': training_config.num_train_epochs,
                'batch_size': training_config.per_device_train_batch_size,
                'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
                'max_sequence_length': training_config.max_sequence_length
            }
        },
        'training_summary': training_analysis,
        'tow_performance_analysis': tow_analysis,
        'resource_usage_analysis': resource_analysis,
        'raw_metrics_summary': {
            'total_training_logs': len(metrics_data.get('training_metrics', [])),
            'total_tow_evaluations': len(metrics_data.get('tow_metrics', [])),
            'has_training_results': 'training_results' in metrics_data
        }
    }
    
    # Add final training results if available
    if train_result:
        comprehensive_report['final_training_results'] = train_result.metrics
    
    # Save comprehensive report
    report_file = output_dir / "comprehensive_training_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Comprehensive training report saved to: {report_file}")
    
    # Generate summary text report
    summary_lines = [
        "=== COMPREHENSIVE TRAINING REPORT SUMMARY ===",
        f"Model: {model_config.name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    if training_analysis and 'loss_progression' in training_analysis:
        loss_prog = training_analysis['loss_progression']
        summary_lines.extend([
            "TRAINING PROGRESSION:",
            f"  Initial Loss: {loss_prog.get('initial_train_loss', 'N/A')}",
            f"  Final Loss: {loss_prog.get('final_train_loss', 'N/A')}",
            f"  Improvement: {loss_prog.get('loss_improvement', 'N/A')}%",
            ""
        ])
    
    if tow_analysis and 'accuracy_progression' in tow_analysis:
        acc_prog = tow_analysis['accuracy_progression']
        summary_lines.extend([
            "ToW PERFORMANCE:",
            f"  Start Token Accuracy: {acc_prog.get('initial_start_accuracy', 'N/A')}% → {acc_prog.get('final_start_accuracy', 'N/A')}% (Best: {acc_prog.get('best_start_accuracy', 'N/A')}%)",
            f"  End Token Accuracy: {acc_prog.get('initial_end_accuracy', 'N/A')}% → {acc_prog.get('final_end_accuracy', 'N/A')}% (Best: {acc_prog.get('best_end_accuracy', 'N/A')}%)",
            f"  Overall Accuracy: {acc_prog.get('initial_overall_accuracy', 'N/A')}% → {acc_prog.get('final_overall_accuracy', 'N/A')}% (Best: {acc_prog.get('best_overall_accuracy', 'N/A')}%)",
            ""
        ])
    
    if resource_analysis:
        summary_lines.extend([
            "RESOURCE USAGE:",
            f"  GPU Memory: {resource_analysis.get('gpu_memory_usage', {}).get('avg_gpu_memory_gb', 'N/A')} GB (avg), {resource_analysis.get('gpu_memory_usage', {}).get('max_gpu_memory_gb', 'N/A')} GB (max)",
            f"  CPU Usage: {resource_analysis.get('system_resources', {}).get('avg_cpu_usage_%', 'N/A')}% (avg), {resource_analysis.get('system_resources', {}).get('max_cpu_usage_%', 'N/A')}% (max)",
            f"  System Memory: {resource_analysis.get('system_resources', {}).get('avg_memory_usage_%', 'N/A')}% (avg)",
            ""
        ])
    
    summary_text = "\n".join(summary_lines)
    
    summary_file = output_dir / "training_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    logger.info(f"Training summary saved to: {summary_file}")
    logger.info("Summary:\n" + summary_text)
    
    return comprehensive_report


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
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.2,
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
            save_total_limit=6,
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
            deepspeed="./deepspeed_config_deepseek.json",
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
        json_log_path = self.output_dir / "training_metrics.json"
        enhanced_json_callback = EnhancedJsonLoggingCallback(log_file=json_log_path, tokenizer=tokenizer)
        tow_metrics_callback = ToWMetricsCallback(tokenizer=tokenizer, output_dir=self.output_dir)
        
        logger.info(f"Enhanced metrics will be logged to {json_log_path}")
        logger.info(f"ToW-specific metrics will be logged to {self.output_dir / 'tow_metrics.json'}")

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
                enhanced_json_callback,  # Enhanced logging with GPU memory, perplexity, etc.
                tow_metrics_callback     # ToW-specific metrics tracking
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
                output_dir=self.output_dir,
                model_config=self.model_config,
                training_config=self.training_config,
                train_result=train_result
            )
            logger.info("✅ Comprehensive training report generated successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to generate comprehensive training report: {str(e)}")
        
        logger.info(f"Training completed for {self.model_config.name}")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info(f"All training logs and reports available in: {self.output_dir}")
        
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