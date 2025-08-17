#!/usr/bin/env python3
"""
ToW Training with Smart Text Handling - Fixed Version
- Fixed batch_size mismatch error
- Adaptive max length based on data analysis
- ToW token preservation
- Smart chunking for long texts
- torchrun --nproc_per_node=[사용할 GPU 개수] ToW_Training_llama.py
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
    learning_rate: float = 2e-5  # Add learning_rate to ModelConfig


@dataclass
class ToWTrainingConfig:
    """ToW training config with smart text handling"""
    tow_data_paths: List[str] = field(default_factory=lambda: [
        "../4_tow_generation/tow_data/klue_tow_gemini_2.0-flash-lite.json",
        "../4_tow_generation/tow_data/koconovel_tow_gemini_2.0-flash-lite.json"
    ])
    output_base_dir: str = "ToW_Models"
    
    # Training hyperparameters
    learning_rate: float = 5e-5  # This will be a fallback
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    
    # Smart text handling
    adaptive_max_length: bool = True
    preserve_tow_tokens: bool = True
    enable_chunking: bool = True
    min_chunk_overlap: int = 50
    
    # Default settings
    max_sequence_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Other settings
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_steps: int = 200
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    dataloader_num_workers: int = 8
    remove_unused_columns: bool = True
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False


MODEL_CONFIGS = [
    # ModelConfig(
    #     name="Qwen2.5-7B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
    #     use_quantization=True,
    #     learning_rate=2e-5  # Stable learning rate for Qwen
    # ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
        use_quantization=True,
        learning_rate=5e-6  # Lower learning rate for stability
    ),
    # ModelConfig(
    #     name="Mistral-8B-Instruct-2410-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
    #     use_quantization=True,
    #     learning_rate=5e-6  # Lower learning rate for stability
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-0528-Qwen3-8B-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
    #     use_quantization=True,
    #     learning_rate=5e-6  # Lower learning rate for stability
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

    def load_tow_data(self, data_paths: List[str]) -> List[Dict]:
        """Load ToW-augmented Korean dataset from multiple files"""
        logger.info(f"Loading ToW data from {len(data_paths)} files...")
        
        all_data = []
        for data_path in data_paths:
            path = Path(data_path)
            if not path.exists():
                logger.warning(f"Data file not found, skipping: {data_path}")
                continue
            
            logger.info(f"  - Loading from {data_path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        logger.warning(f"Data in {data_path} is not a list, skipping.")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {data_path}. Skipping.")
        
        logger.info(f"Loaded a total of {len(all_data)} ToW-augmented entries")
        return all_data
    
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
            lora_dropout=0.05,
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
        # Use the learning rate from the specific model_config, or fall back to the training_config
        lr = self.model_config.learning_rate if self.model_config.learning_rate else self.training_config.learning_rate
        logger.info(f"Using model-specific learning rate: {lr}")

        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=False,
            learning_rate=lr,
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
            ddp_find_unused_parameters=False, # Llama 모델과 LoRA 사용 시 이 옵션이 필요할 수 있습니다.
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
        )
    
    def train(self):
        """Execute smart ToW fine-tuning"""
        self.setup_logging() # Setup file logging right at the start
        logger.info(f"Starting smart ToW training for {self.model_config.name}")
        
        model, tokenizer = self.load_model_and_tokenizer()
        
        data_processor = SmartToWDataProcessor(tokenizer, self.training_config)
        tow_data = data_processor.load_tow_data(self.training_config.tow_data_paths)
        
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