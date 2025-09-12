# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""module avail cuda
module load cuda-12.6.1-gcc-12.1.0
echo $CUDA_HOME
llama
deepspeed --num_gpus=2 finetune.py --model_name_or_path /scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt --train_file /scratch/jsong132/Increase_MLLM_Ability/4_tow_generation/tow_data/tow_09_05.jsonl --output_dir ./tow_trained_models/llama-3.2-3b-pt-tow-09_05_allenai --exp_name "llama-3.2-3b-pt-tow-sft" --num_train_epochs 10 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --max_seq_length 2048 --use_flash_attn False --gradient_checkpointing True --logging_steps 10 --checkpointing_steps 500 --with_tracking True --report_to "wandb" --seed 42 --use_qlora False --keep_last_n_checkpoints 3
qwen 
deepspeed --num_gpus=2 finetune.py --model_name_or_path /scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt --train_file /scratch/jsong132/Increase_MLLM_Ability/4_tow_generation/tow_data/tow_09_05.jsonl --output_dir ./tow_trained_models/qwem-2.5-3b-pt-tow-09_05_allenai --exp_name "qwem-2.5-3b-pt-tow-sft" --num_train_epochs 10 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --max_seq_length 2048 --use_flash_attn False --gradient_checkpointing True --logging_steps 10 --checkpointing_steps 500 --with_tracking True --report_to "wandb" --seed 42 --use_qlora False --keep_last_n_checkpoints 3
gemma
deepspeed --num_gpus=2 finetune.py --model_name_or_path /scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt --train_file /scratch/jsong132/Increase_MLLM_Ability/4_tow_generation/tow_data/tow_09_05.jsonl --output_dir ./tow_trained_models/gemma-3-4b-pt-tow-09_05_allenai --exp_name "gemma-3-4b-pt-tow-sft" --num_train_epochs 10 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --max_seq_length 2048 --use_flash_attn False --gradient_checkpointing True --logging_steps 10 --checkpointing_steps 500 --with_tracking True --report_to "wandb" --seed 42 --use_qlora False --keep_last_n_checkpoints 3
olmo
deepspeed --num_gpus=2 finetune.py --model_name_or_path /scratch/jsong132/Increase_MLLM_Ability/Base_Models/olmo-2-0425-1b --train_file /scratch/jsong132/Increase_MLLM_Ability/4_tow_generation/tow_data/tow_09_05.jsonl --output_dir ./tow_trained_models/olmo-2-0425-1b-tow-09_05_allenai --exp_name "olmo-2-0425-1b-tow-sft" --num_train_epochs 10 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --max_seq_length 2048 --use_flash_attn False --gradient_checkpointing True --logging_steps 10 --checkpointing_steps 500 --with_tracking True --report_to "wandb" --seed 42 --use_qlora False --keep_last_n_checkpoints 3
"""
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

from fine_nwp.model_utils import push_folder_to_hub, save_with_accelerate
from fine_nwp.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_datasets,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)

logger = get_logger(__name__)


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """
    local_rank: Optional[int] = field(default=None)

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    use_slow_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the slow tokenizer or not (which is then fast tokenizer)."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                "This option should only be set to `True` for repositories you trust and in which you "
                "have read the code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None, metadata={"help": "A list of datasets (local or HF) to sample from."}
    )
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    add_bos: bool = field(
        default=False,
        metadata={
            "help": "Forcibly add bos token to the beginning of the input sequence."
            " Use only when tokenizer does not add bos token by default."
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=25,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=42,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=True,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    with_tracking: bool = field(
        default=True,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: Optional[str] = field(
        default=42,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Means that resumption will always start from scratch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."},
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = False
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None) and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")

        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main(args: FlatArguments):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = (
                f"{args.exp_name}__{args.model_name_or_path.replace('/', '_')}__{args.seed}__{int(time.time())}"
            )
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # use_seedable_sampler=True,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )
    
    # 모델별 로그 파일 설정
    if accelerator.is_main_process:
        # 모델 이름에서 파일명으로 사용할 수 없는 문자 제거
        model_name = os.path.basename(args.model_name_or_path).replace('/', '_').replace('\\', '_')
        log_filename = f"{model_name}_{args.exp_name}_log.txt"
        log_filepath = os.path.join(args.output_dir, log_filename)
        
        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 파일과 콘솔 모두에 로그 출력
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                logging.FileHandler(log_filepath, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Model training logs will be saved to: {log_filepath}")
        logger.info(f"Starting training for model: {args.model_name_or_path}")
    else:
        # 다른 프로세스는 콘솔만 사용
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = args.model_revision if args.tokenizer_revision is None else args.tokenizer_revision

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert (
                args.add_bos
            ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:

        if isinstance(model, FalconForCausalLM):
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "bos_token": "<|endoftext|>", # do we need this?
                    "eos_token": "<|endoftext|>",
                    "unk_token": "<|unk|>",
                    "pad_token": "<|pad|>",
                }
            )
            assert num_added_tokens == 2, "check special tokens mapping for Falcom model"
        else:
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    tokenizer.add_special_tokens({'additional_special_tokens': ['<ToW>', '</ToW>']})


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)



    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        # Initialize <ToW> token with "Let's think step by step in English." average embedding
        start_tokens = tokenizer.encode("Let's think step by step in English.", add_special_tokens=False)
        start_embeddings = embeddings.weight.data[start_tokens, :]
        tow_start_embedding = start_embeddings.mean(dim=0)
        
        # Initialize </ToW> token with "What is the proper next word?" average embedding
        end_tokens = tokenizer.encode("What is the proper next word?", add_special_tokens=False)
        end_embeddings = embeddings.weight.data[end_tokens, :]
        tow_end_embedding = end_embeddings.mean(dim=0)
        
        # Apply initializations
        embeddings.weight.data[len(tokenizer)-2, :] = tow_start_embedding  # <ToW>
        embeddings.weight.data[len(tokenizer)-1, :] = tow_end_embedding    # </ToW>
    # tow_init_embeddings = torch.unsqueeze(model.model.embed_tokens.weight.data[tokenizer.encode('---', add_special_tokens=False)[0], :], dim=0)
    # model.model.embed_tokens.weight.data[-2:, :] = torch.concat([tow_init_embeddings, tow_init_embeddings], dim=0)


    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # # set the tokenizer chat template to the tulu format
    # # this makes evaluation/etc easier down the line.
    # chat_template = (
    #     "{% for message in messages %}\n"
    #     "{% if message['role'] == 'system' %}\n"
    #     "{{ '<|system|>\n' + message['content'] }}\n"
    #     "{% elif message['role'] == 'user' %}\n"
    #     "{{ '<|user|>\n' + message['content'] }}\n"
    #     "{% elif message['role'] == 'assistant' %}\n"
    #     "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    #     "{% endif %}\n"
    #     "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n"
    #     "{% endif %}\n"
    #     "{% endfor %}"
    # )
    # tokenizer.chat_template = chat_template
    if args.add_bos:
        # also add bos in the chat template
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze vision tower for Gemma 3 multimodal models
    def freeze_vision_tower(model):
        """Freeze vision tower parameters to avoid unused parameter errors"""
        vision_modules = []
        
        # Check various possible paths for vision tower
        if hasattr(model, 'vision_tower'):
            vision_modules.append(model.vision_tower)
        if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            vision_modules.append(model.model.vision_tower)
        if hasattr(model, 'base_model'):
            base_model = model.base_model
            if hasattr(base_model, 'model') and hasattr(base_model.model, 'vision_tower'):
                vision_modules.append(base_model.model.vision_tower)
        
        # Freeze all vision tower parameters
        for vision_module in vision_modules:
            for param in vision_module.parameters():
                param.requires_grad = False
                
        if vision_modules:
            logger.info(f"Frozen {len(vision_modules)} vision tower module(s) to avoid unused parameter errors")
        else:
            logger.info("No vision tower found - proceeding with text-only training")
    
    # Apply vision tower freezing
    freeze_vision_tower(model)


    # Preprocessing the datasets.

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    elif args.dataset_mixer is not None:
        # mixing datasets via config
        raw_datasets = get_datasets(
            args.dataset_mixer,
            configs=args.dataset_config_name,
            splits=["train"],
            save_data_dir=args.dataset_mix_dir,
            columns_to_keep=["messages"],
        )
    elif args.dataset_mixer_list is not None:
        # mixing datasets via config
        raw_datasets = get_datasets(
            args.dataset_mixer_list,
            configs=args.dataset_config_name,
            splits=["train"],
            save_data_dir=args.dataset_mix_dir,
            columns_to_keep=["messages"],
        )
    else:
        def read_from_jsonl(raw_file):
            outputs = []
            with open(raw_file) as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    outputs.append({"prompt": data['prompt'], "completion": data['completion']})
            return outputs

        raw_datasets = DatasetDict({"train": Dataset.from_list(read_from_jsonl(args.train_file))})
    

    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    train_dataset = raw_datasets["train"]

    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        train_dataset.set_format(type="pt")
        train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # For gemma, we added
    if hasattr(accelerator.state, 'ddp_kwargs'):
        accelerator.state.ddp_kwargs['find_unused_parameters'] = True
    else:
        accelerator.state.ddp_kwargs = {'find_unused_parameters': True}

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    beaker_config = None
    if is_beaker_job():
        try:
            beaker_config = maybe_get_beaker_config()
        except Exception as e:
            logger.warning(f"Failed to get beaker config: {e}")
            beaker_config = None
    
    wandb_tracker = None
    if args.with_tracking:
        try:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

            # (Optional) Ai2 internal tracking
            if args.wandb_entity is None:
                args.wandb_entity = maybe_use_ai2_wandb_entity()
            if beaker_config:
                experiment_config.update(vars(beaker_config))
            
            accelerator.init_trackers(
                "open_instruct_internal",
                experiment_config,
                init_kwargs={"wandb": {"entity": args.wandb_entity, "tags": [args.exp_name] + get_wandb_tags()}},
            )
            
            # Try to get wandb tracker with fallback
            try:
                wandb_tracker = accelerator.get_tracker("wandb")
            except:
                # Try alternative approaches to get wandb tracker
                for tracker in accelerator.trackers:
                    if hasattr(tracker, 'run') or 'wandb' in str(type(tracker)).lower():
                        wandb_tracker = tracker
                        break
                
        except Exception as e:
            logger.warning(f"Failed to initialize tracking: {e}")
            wandb_tracker = None

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                if args.load_balancing_loss:
                    outputs = model(**batch, use_cache=False, output_router_logits=True)
                else:
                    outputs = model(**batch, use_cache=False)
                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    if args.load_balancing_loss:
                        aux_loss = args.load_balancing_weight * outputs.aux_loss
                        loss += aux_loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                    }
                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}"
                        )
                    if args.with_tracking:
                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps,
                        )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(
                            os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w"
                        ) as f:
                            f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
                        if accelerator.is_local_main_process:
                            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
            args.use_lora,
        )
        
        # LoRA 훈련이 끝난 후 기본 모델과 LoRA adapter를 합치기
        if args.use_lora and accelerator.is_main_process:
            logger.info("Merging LoRA adapter with base model...")
            
            try:
                # Wait for all processes to complete model saving
                accelerator.wait_for_everyone()
                
                # LoRA adapter가 저장된 경로
                lora_adapter_path = os.path.join(args.output_dir, "final_model")
                merged_model_path = os.path.join(args.output_dir, "merged_model")
                
                # Ensure adapter path exists
                if not os.path.exists(lora_adapter_path):
                    raise FileNotFoundError(f"LoRA adapter path does not exist: {lora_adapter_path}")
                
                # Clear GPU memory and gradients before merge
                if hasattr(model, 'zero_grad'):
                    model.zero_grad()
                torch.cuda.empty_cache()
                
                # Move training model to CPU and clear CUDA cache
                try:
                    if hasattr(model, 'cpu'):
                        model = model.cpu()
                except Exception as move_error:
                    logger.warning(f"Could not move training model to CPU: {move_error}")
                
                # Clear all CUDA cached memory
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Import required modules
                import gc
                from peft import PeftModel, PeftConfig
                
                # Force garbage collection
                gc.collect()
                
                # 기본 모델 다시 로드 (quantization 없이)
                logger.info(f"Loading base model from: {args.model_name_or_path}")
                
                # Load PEFT config first to understand the setup
                peft_config = PeftConfig.from_pretrained(lora_adapter_path)
                
                # Load base model with minimal memory usage
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # Use auto device mapping
                    trust_remote_code=args.trust_remote_code,
                    low_cpu_mem_usage=True,
                )
                
                # vocab size 확인 및 조정 (merge_lora.py 로직 추가)
                def get_adapter_vocab_size(adapter_path):
                    """어댑터에서 실제 vocab size 추출"""
                    import glob
                    from safetensors import safe_open
                    try:
                        # safetensors 파일 우선 검색
                        safetensor_files = glob.glob(os.path.join(adapter_path, "final_model", "*.safetensors"))
                        if not safetensor_files:
                            safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
                        pytorch_files = glob.glob(os.path.join(adapter_path, "final_model", "*.bin"))
                        if not pytorch_files:
                            pytorch_files = glob.glob(os.path.join(adapter_path, "*.bin"))
                        
                        if safetensor_files:
                            with safe_open(safetensor_files[0], framework="pt") as f:
                                for key in f.keys():
                                    if any(target in key for target in ['embed_tokens.weight', 'lm_head.weight']):
                                        return f.get_tensor(key).shape[0]
                        
                        elif pytorch_files:
                            checkpoint = torch.load(pytorch_files[0], map_location='cpu')
                            state_dict = checkpoint.get('state_dict', checkpoint)
                            for key, tensor in state_dict.items():
                                if any(target in key for target in ['embed_tokens.weight', 'lm_head.weight']):
                                    return tensor.shape[0]
                            del checkpoint
                                
                    except Exception as e:
                        logger.warning(f"Error extracting vocab size: {e}")
                    return None

                # 현재 모델과 어댑터의 vocab size 확인
                base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
                adapter_vocab_size = get_adapter_vocab_size(lora_adapter_path)
                
                logger.info(f"Base model vocab size: {base_vocab_size}")
                logger.info(f"Adapter vocab size: {adapter_vocab_size}")
                
                # vocab size가 다르면 베이스 모델을 어댑터에 맞춰 조정
                if adapter_vocab_size and adapter_vocab_size != base_vocab_size:
                    logger.info(f"Vocab size mismatch resolved: {base_vocab_size} -> {adapter_vocab_size}")
                    base_model.resize_token_embeddings(adapter_vocab_size)
                    logger.info("Base model embedding size adjusted successfully")
                else:
                    logger.info("Vocab size match - no adjustment needed")
                
                # Force garbage collection after base model load
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # LoRA adapter 로드
                logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
                peft_model = PeftModel.from_pretrained(
                    base_model, 
                    lora_adapter_path,
                    torch_dtype=torch.bfloat16,
                    is_trainable=False  # Ensure it's not trainable during merge
                )
                
                # LoRA와 기본 모델 합치기
                logger.info("Merging LoRA adapter with base model...")
                # Set recursion limit higher temporarily
                import sys
                old_recursion_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(10000)
                
                try:
                    merged_model = peft_model.merge_and_unload()
                finally:
                    # Restore original recursion limit
                    sys.setrecursionlimit(old_recursion_limit)
                
                # 합쳐진 모델 저장
                logger.info(f"Saving merged model to: {merged_model_path}")
                os.makedirs(merged_model_path, exist_ok=True)
                merged_model.save_pretrained(
                    merged_model_path, 
                    safe_serialization=True,
                    max_shard_size="5GB"  # Prevent huge single files
                )
                tokenizer.save_pretrained(merged_model_path)
                
                logger.info("✅ Successfully merged LoRA adapter with base model!")
                logger.info(f"Merged model saved at: {merged_model_path}")
                
                # 메모리 정리
                del base_model
                del peft_model
                del merged_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ Failed to merge LoRA adapter with base model: {e}")
                logger.error("The LoRA adapter is still saved separately and can be merged later.")
                logger.error("You can merge it manually using the separate merge script.")
                # Don't re-raise the exception to allow training completion

    # Conditional checkpoint cleanup - preserve steps for restart capability
    if accelerator.is_local_main_process:
        try:
            # Wait for all processes to complete before cleanup
            accelerator.wait_for_everyone()
            
            # Only clean checkpoints if both conditions are met:
            # 1. Training completed successfully (final or merged model exists)
            # 2. User explicitly wants cleanup (keep_last_n_checkpoints >= 0)
            # 3. keep_last_n_checkpoints is 0 (full cleanup) - be extra careful
            final_model_path = os.path.join(args.output_dir, "final_model")
            merged_model_path = os.path.join(args.output_dir, "merged_model")
            
            training_completed_successfully = os.path.exists(final_model_path) or os.path.exists(merged_model_path)
            
            if training_completed_successfully and args.keep_last_n_checkpoints == 0:
                # Full cleanup only if training completed AND explicitly requested
                logger.info("Training completed successfully. Performing full checkpoint cleanup.")
                clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)
            elif training_completed_successfully and args.keep_last_n_checkpoints > 0:
                # Partial cleanup - keep specified number of checkpoints
                logger.info(f"Training completed successfully. Keeping last {args.keep_last_n_checkpoints} checkpoints.")
                clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=args.keep_last_n_checkpoints)
            else:
                # Preserve all checkpoints if training failed or cleanup disabled
                logger.info("Preserving all checkpoints to enable restart from failure point.")
                if not training_completed_successfully:
                    logger.warning("Training may have failed - all checkpoints preserved for restart")
                else:
                    logger.info("Cleanup disabled (keep_last_n_checkpoints < 0) - all checkpoints preserved")
                    
        except Exception as e:
            logger.error(f"Error during checkpoint evaluation: {e}")
            logger.info("Defaulting to preserving all checkpoints for safety")

    if accelerator.is_main_process:
        # dpo script only supports these two options right now for datasets
        if args.dataset_mixer:
            dataset_list = args.dataset_mixer.keys()
        elif args.dataset_mixer_list:
            dataset_list = args.dataset_mixer_list[::2]  # even indices
        elif args.dataset_name:
            dataset_list = [args.dataset_name]
        else:
            dataset_list = [args.train_file]
        # mainly just focussing here on what would be useful for the leaderboard.
        # wandb will have even more useful information.
        metadata_blob = {
            "model_name": args.exp_name,
            "model_type": "sft",
            "datasets": dataset_list,
            "base_model": args.model_name_or_path,
        }
        
        # Safely add wandb URL if tracker is available
        if wandb_tracker is not None:
            try:
                # Handle both GeneralTracker and direct WandB tracker
                if hasattr(wandb_tracker, 'tracker') and hasattr(wandb_tracker.tracker, 'url'):
                    metadata_blob["wandb_path"] = wandb_tracker.tracker.url
                elif hasattr(wandb_tracker, 'run') and hasattr(wandb_tracker.run, 'get_url'):
                    metadata_blob["wandb_path"] = wandb_tracker.run.get_url()
                elif hasattr(wandb_tracker, 'run') and hasattr(wandb_tracker.run, 'url'):
                    metadata_blob["wandb_path"] = wandb_tracker.run.url
                else:
                    logger.warning("WandB tracker available but cannot extract URL")
                    metadata_blob["wandb_path"] = "unavailable"
            except Exception as e:
                logger.warning(f"Failed to get WandB URL: {e}")
                metadata_blob["wandb_path"] = "error"
        else:
            metadata_blob["wandb_path"] = "not_configured"
            
        # Safely add beaker info if available
        if beaker_config is not None:
            try:
                metadata_blob["beaker_experiment"] = getattr(beaker_config, 'beaker_experiment_url', 'unavailable')
                metadata_blob["beaker_datasets"] = getattr(beaker_config, 'beaker_dataset_id_urls', 'unavailable')
            except Exception as e:
                logger.warning(f"Failed to get beaker config info: {e}")
                metadata_blob["beaker_experiment"] = "error"
                metadata_blob["beaker_datasets"] = "error"
        else:
            metadata_blob["beaker_experiment"] = "not_beaker_job"
            metadata_blob["beaker_datasets"] = "not_beaker_job"
        # save metadata to the output directory. then it should also get pushed to HF.
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata_blob, f)
        print(f"Metadata saved to {args.output_dir}/metadata.json")

        # upload metadata to the dataset if set
        if args.hf_metadata_dataset:
            upload_metadata_to_hf(
                metadata_blob,
                "metadata.json",
                args.hf_metadata_dataset,
                "results/" + args.hf_repo_revision,  # to match what the auto-evals name as.
            )

        if args.try_launch_beaker_eval_jobs and beaker_config is not None:
            try:
                beaker_workload_id = getattr(beaker_config, 'beaker_workload_id', None)
                if beaker_workload_id:
                    command = f"""\
                    python mason.py  \
                        --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/s2-cirrascale ai2/general-cirrascale \
                        --priority low \
                        --preemptible \
                        --budget ai2/allennlp \
                        --workspace ai2/tulu-2-improvements \
                        --image nathanl/open_instruct_auto \
                        --pure_docker_mode \
                        --gpus 0 -- python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py \
                        --beaker_workload_id {beaker_workload_id} \
                        --model_name {args.hf_repo_revision}
                    """
                    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
                    print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
                    print(f"Submit jobs after model training is finished - process return code: {process.returncode}")
                else:
                    logger.warning("Beaker workload ID not available, skipping evaluation job launch")
            except Exception as e:
                logger.error(f"Failed to launch beaker evaluation jobs: {e}")
        elif args.try_launch_beaker_eval_jobs:
            logger.warning("Beaker evaluation jobs requested but beaker_config is not available")

    if args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )
    accelerator.wait_for_everyone()
    if args.with_tracking:
        try:
            accelerator.end_training()
        except Exception as e:
            logger.warning(f"Error ending tracking: {e}")
            # Try to finish wandb run manually if possible
            if wandb_tracker is not None:
                try:
                    if hasattr(wandb_tracker, 'tracker'):
                        wandb_tracker.tracker.finish()
                    elif hasattr(wandb_tracker, 'run'):
                        wandb_tracker.run.finish()
                except Exception as wandb_error:
                    logger.warning(f"Could not manually finish wandb run: {wandb_error}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()
    main(args)