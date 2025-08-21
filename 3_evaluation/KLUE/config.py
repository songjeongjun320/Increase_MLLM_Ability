#!/usr/bin/env python3
"""
KLUE Benchmark Configuration
Common configuration and model definitions for KLUE evaluation
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)

@dataclass
class ModelConfig:
    name: str                             # Unique name for this run
    model_id: str                         # Hugging Face model identifier or local path
    adapter_path: str = None              # Path to the LoRA adapter
    use_quantization: bool = True         # Whether to use quantization
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3:1_8B_Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),

    # TOW Trained Models
    ModelConfig(
        name="Qwen2.5-7B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-7B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Mistral-8B-Instruct-2410-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3:1_8B_Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.1-8B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-0528-Qwen3-8B-ToW",
        use_quantization=False
    ),
]

# General Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"
BASE_OUTPUT_DIR = "klue_evaluation_results"
DATA_DIR = "../klue_all_tasks_json"

# KLUE Task Configuration
KLUE_TASKS = {
    'tc': {
        'name': 'Topic Classification',
        'data_files': ['klue_tc_train.json', 'klue_tc_validation.json'],
        'metric': 'f1_macro'
    },
    'sts': {
        'name': 'Sentence Textual Similarity',
        'data_files': ['klue_sts_train.json', 'klue_sts_validation.json'],
        'metric': 'pearson'
    },
    'nli': {
        'name': 'Natural Language Inference',
        'data_files': ['klue_nli_train.json', 'klue_nli_validation.json'],
        'metric': 'accuracy'
    },
    'ner': {
        'name': 'Named Entity Recognition',
        'data_files': ['klue_ner_train.json', 'klue_ner_validation.json'],
        'metric': 'f1'
    },
    're': {
        'name': 'Relation Extraction',
        'data_files': ['klue_re_train.json', 'klue_re_validation.json'],
        'metric': 'f1_micro'
    },
    'dp': {
        'name': 'Dependency Parsing',
        'data_files': ['klue_dp_train.json', 'klue_dp_validation.json'],
        'metric': 'uas_las'
    },
    'mrc': {
        'name': 'Machine Reading Comprehension',
        'data_files': ['klue_mrc_train.json', 'klue_mrc_validation.json'],
        'metric': 'f1'
    },
    'dst': {
        'name': 'Dialogue State Tracking',
        'data_files': ['klue_dst_train.json', 'klue_dst_validation.json'],
        'metric': 'joint_goal_accuracy'
    }
}

# Label mappings for classification tasks
LABEL_MAPPINGS = {
    'tc': {
        0: 'IT과학', 1: '경제', 2: '사회', 3: '생활문화', 4: '세계', 5: '스포츠', 6: '정치'
    },
    'nli': {
        0: 'entailment', 1: 'contradiction', 2: 'neutral'
    }
}

# Prompt templates for different tasks
PROMPT_TEMPLATES = {
    'tc': """다음 뉴스 제목을 분류하세요. 가능한 카테고리는 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치입니다.

제목: {title}

카테고리:""",

    'sts': """두 문장의 의미적 유사도를 0부터 5까지의 점수로 평가하세요. 0은 완전히 다르고, 5는 의미가 동일함을 나타냅니다.

문장 1: {sentence1}
문장 2: {sentence2}

유사도 점수:""",

    'nli': """주어진 전제와 가설의 관계를 판단하세요. 가능한 답은 entailment(함의), contradiction(모순), neutral(중립)입니다.

전제: {premise}
가설: {hypothesis}

관계:""",

    'ner': """다음 문장에서 개체명을 찾고 그 유형을 분류하세요.

문장: {sentence}

개체명과 유형:""",

    're': """다음 문장에서 주어진 두 개체 사이의 관계를 분류하세요.

문장: {sentence}
개체1: {entity1}
개체2: {entity2}

관계:""",

    'dp': """다음 문장의 구문 분석을 수행하세요.

문장: {sentence}

구문 분석 결과:""",

    'mrc': """다음 지문을 읽고 질문에 답하세요.

지문: {context}

질문: {question}

답:""",

    'dst': """대화에서 사용자의 의도와 슬롯을 추출하세요.

대화: {dialogue}

의도와 슬롯:"""
}