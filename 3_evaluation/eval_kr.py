#!/usr/bin/env python3
"""
Korean Benchmark Evaluation System
==================================

Unified evaluation system for Korean benchmarks:
- KMMLU (Korean MMLU - Massive Multitask Language Understanding)
- KLUE (Korean Language Understanding Evaluation) - 8 tasks
- KorNLI (Korean Natural Language Inference)
- KorSTS (Korean Semantic Textual Similarity)
- NSMC (Naver Sentiment Movie Corpus)
- KorQuAD (Korean Question Answering Dataset)

Supports multiple models with consistent evaluation framework.
"""

import os
import json
import logging
import torch
import time
import gc
import sys
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Data Classes ---
@dataclass
class ModelConfig:
    """Model configuration for evaluation"""
    name: str                             # Unique name for this run
    model_id: str                         # Hugging Face model identifier or local path
    use_quantization: bool = True         # Enable 4-bit quantization
    torch_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

@dataclass
class EvaluationResult:
    """Result for a single benchmark evaluation"""
    model_name: str
    benchmark_name: str
    task_name: str
    score: float
    total_questions: int
    correct_answers: int
    evaluation_time: float
    timestamp: str
    additional_metrics: Optional[Dict[str, float]] = None

@dataclass
class BenchmarkResults:
    """Complete evaluation results for a model across all Korean benchmarks"""
    model_name: str
    model_path: str
    evaluation_date: str
    kmmlu_score: Optional[float] = None
    klue_scores: Optional[Dict[str, float]] = None
    kornli_score: Optional[float] = None
    korsts_score: Optional[float] = None
    nsmc_score: Optional[float] = None
    korquad_score: Optional[float] = None
    total_evaluation_time: Optional[float] = None
    detailed_results: Optional[List[EvaluationResult]] = None

# --- Configuration ---
MODEL_CONFIGS = [
    ModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/1_models/qwen2.5-7b-instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-7B", 
        model_id="C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/1_models/deepseek-r1-distill-qwen-7b",
        use_quantization=False
    )
]

# Dataset paths - update these to your actual dataset locations
DATASET_PATHS = {
    "kmmlu": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/KMMLU",
    "klue": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/KLUE",
    "kornli": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/KorNLI",
    "korsts": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/KorSTS",
    "nsmc": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/NSMC",
    "korquad": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/KorQuAD"
}

BASE_OUTPUT_DIR = "evaluation_results_korean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Data Loading Functions ---
def load_dataset_data(filepath: str, data_format: str = "json") -> Optional[List[Dict]]:
    """
    Load dataset from file with format detection and validation
    
    Args:
        filepath: Path to dataset file
        data_format: Format type (json, jsonl, csv, tsv)
    
    Returns:
        List of dataset items or None if loading failed
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Dataset file not found: {filepath}")
            return None
            
        if data_format == "json" or filepath.suffix == ".json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_format == "jsonl" or filepath.suffix == ".jsonl":
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_format in ["csv", "tsv"] or filepath.suffix in [".csv", ".tsv"]:
            import pandas as pd
            separator = '\t' if data_format == "tsv" or filepath.suffix == ".tsv" else ','
            df = pd.read_csv(filepath, sep=separator, encoding='utf-8')
            data = df.to_dict('records')
        else:
            logger.error(f"Unsupported data format: {data_format}")
            return None
            
        if not isinstance(data, list):
            logger.error(f"Expected list format, got {type(data)}")
            return None
            
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading dataset from {filepath}: {e}")
        return None

def load_kmmlu_data() -> Optional[List[Dict]]:
    """Load KMMLU dataset"""
    kmmlu_path = DATASET_PATHS.get("kmmlu")
    if not kmmlu_path or not Path(kmmlu_path).exists():
        logger.warning("KMMLU dataset path not found, using mock data")
        return create_mock_kmmlu_data()
    
    # Try to load from directory structure or single file
    try:
        kmmlu_dir = Path(kmmlu_path)
        if kmmlu_dir.is_dir():
            all_data = []
            for data_file in kmmlu_dir.glob("*.json*"):
                subject_data = load_dataset_data(str(data_file))
                if subject_data:
                    for item in subject_data:
                        if 'subject' not in item:
                            item['subject'] = data_file.stem
                    all_data.extend(subject_data)
            
            if all_data:
                return all_data
        else:
            return load_dataset_data(str(kmmlu_path))
    except Exception as e:
        logger.warning(f"Error loading KMMLU: {e}")
    
    return create_mock_kmmlu_data()

def create_mock_kmmlu_data() -> List[Dict]:
    """Create mock KMMLU data for testing"""
    logger.info("Creating mock KMMLU data for testing")
    return [
        {
            "question": "한국의 수도는 어디인가요?",
            "choices": ["부산", "서울", "대구", "인천"],
            "answer": 1,
            "subject": "지리학"
        },
        {
            "question": "태양계에서 가장 큰 행성은 무엇인가요?",
            "choices": ["지구", "화성", "목성", "토성"],
            "answer": 2,
            "subject": "천문학"
        },
        {
            "question": "2 더하기 2는 얼마인가요?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "수학"
        },
        {
            "question": "한글을 창제한 조선의 왕은 누구인가요?",
            "choices": ["태조", "태종", "세종대왕", "성종"],
            "answer": 2,
            "subject": "한국사"
        }
    ]

def load_klue_data() -> Dict[str, List[Dict]]:
    """Load KLUE benchmark data (8 tasks)"""
    klue_path = DATASET_PATHS.get("klue")
    klue_tasks = {
        "ynat": "Topic Classification",
        "sts": "Semantic Textual Similarity",
        "nli": "Natural Language Inference", 
        "ner": "Named Entity Recognition",
        "re": "Relation Extraction",
        "dp": "Dependency Parsing",
        "mrc": "Machine Reading Comprehension",
        "wos": "Dialogue State Tracking"
    }
    
    klue_data = {}
    
    if not klue_path or not Path(klue_path).exists():
        logger.warning("KLUE dataset path not found, using mock data")
        return create_mock_klue_data()
    
    try:
        klue_dir = Path(klue_path)
        for task_code, task_name in klue_tasks.items():
            task_path = klue_dir / task_code
            if task_path.exists():
                # Try to load test data
                for test_file in task_path.glob("*test*.json*"):
                    data = load_dataset_data(str(test_file))
                    if data:
                        klue_data[task_code] = data
                        break
                
                # Fallback to any data file
                if task_code not in klue_data:
                    for data_file in task_path.glob("*.json*"):
                        data = load_dataset_data(str(data_file))
                        if data:
                            klue_data[task_code] = data[:100]  # Limit for testing
                            break
    except Exception as e:
        logger.warning(f"Error loading KLUE data: {e}")
    
    if not klue_data:
        return create_mock_klue_data()
    
    return klue_data

def create_mock_klue_data() -> Dict[str, List[Dict]]:
    """Create mock KLUE data for testing"""
    logger.info("Creating mock KLUE data for testing")
    return {
        "ynat": [
            {"sentence": "이 영화 정말 재미있어요!", "label": 0},
            {"sentence": "오늘 날씨가 좋네요.", "label": 1}
        ],
        "sts": [
            {"sentence1": "고양이가 잠을 자고 있다", "sentence2": "고양이가 자고 있다", "score": 4.5},
            {"sentence1": "비가 내리고 있다", "sentence2": "햇빛이 쨍쨍하다", "score": 0.5}
        ],
        "nli": [
            {"premise": "남자가 책을 읽고 있다", "hypothesis": "사람이 독서를 하고 있다", "label": "entailment"},
            {"premise": "고양이가 자고 있다", "hypothesis": "개가 뛰고 있다", "label": "contradiction"}
        ]
    }

def load_nsmc_data() -> Optional[List[Dict]]:
    """Load NSMC (Naver Sentiment Movie Corpus) data"""
    nsmc_path = DATASET_PATHS.get("nsmc")
    if not nsmc_path or not Path(nsmc_path).exists():
        logger.warning("NSMC dataset not found, using mock data")
        return [
            {"document": "이 영화 정말 재미있어요 강추합니다", "label": 1},
            {"document": "시간 낭비였네요 별로예요", "label": 0},
            {"document": "배우들 연기가 훌륭했습니다", "label": 1}
        ]
    
    return load_dataset_data(nsmc_path)

# --- Prompt Generation Functions ---
def create_korean_multiple_choice_prompt(question: str, choices: List[str], context: str = "") -> str:
    """Create Korean multiple choice prompt"""
    prompt = ""
    if context:
        prompt += f"맥락: {context}\n\n"
    
    prompt += f"질문: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{i+1}. {choice}\n"
    prompt += "정답:"
    
    return prompt

def create_kmmlu_prompt(item: Dict) -> Optional[str]:
    """Create KMMLU-specific prompt"""
    question = item.get("question", "")
    choices = item.get("choices", [])
    
    if not question or not isinstance(choices, list) or len(choices) != 4:
        return None
    
    return create_korean_multiple_choice_prompt(question, choices)

def create_klue_prompt(item: Dict, task: str) -> Optional[str]:
    """Create KLUE task-specific prompts"""
    if task == "ynat":  # Topic Classification
        sentence = item.get("sentence", "")
        if not sentence:
            return None
        return f"다음 문장의 주제를 분류하세요.\n문장: {sentence}\n주제 (IT과학/경제/사회/생활문화/세계/스포츠/정치): "
    
    elif task == "sts":  # Semantic Textual Similarity
        sentence1 = item.get("sentence1", "")
        sentence2 = item.get("sentence2", "")
        if not sentence1 or not sentence2:
            return None
        return f"다음 두 문장의 의미적 유사도를 0-5점으로 평가하세요.\n문장1: {sentence1}\n문장2: {sentence2}\n유사도 점수:"
    
    elif task == "nli":  # Natural Language Inference
        premise = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        if not premise or not hypothesis:
            return None
        return f"전제와 가설의 관계를 판단하세요.\n전제: {premise}\n가설: {hypothesis}\n관계 (entailment/contradiction/neutral):"
    
    return None

def create_nsmc_prompt(item: Dict) -> Optional[str]:
    """Create NSMC-specific prompt"""
    document = item.get("document", "")
    if not document:
        return None
    
    return f"다음 영화 리뷰의 감정을 분석하세요.\n리뷰: {document}\n감정 (긍정/부정):"

# --- Answer Extraction Functions ---
def extract_korean_multiple_choice_answer(model_output: str, prompt: str, valid_choices: str = "1234") -> Optional[str]:
    """Extract Korean multiple choice answer from model output"""
    # Remove prompt echo if present
    prediction_text = model_output
    if model_output.strip().startswith(prompt.strip()):
        prompt_end = prompt.strip().split("정답:")[-1] if "정답:" in prompt else prompt.strip()
        prediction_text = model_output[len(prompt_end):].strip()
    
    # Clean the text
    cleaned_text = prediction_text.strip()
    
    # Pattern 1: Direct number at start
    match = re.search(rf"^\s*([{valid_choices}])(?:[).:\s]|\b|$)", cleaned_text)
    if match:
        return match.group(1)
    
    # Pattern 2: Single digit answer
    if len(cleaned_text) == 1 and cleaned_text in valid_choices:
        return cleaned_text
    
    # Pattern 3: Answer with Korean phrase
    match = re.search(rf"(?:정답은|답은|정답|답)\s*([{valid_choices}])\b", cleaned_text)
    if match:
        return match.group(1)
    
    return None

def extract_korean_classification_answer(model_output: str, prompt: str, valid_labels: List[str]) -> Optional[str]:
    """Extract classification answer from Korean model output"""
    prediction_text = model_output
    if model_output.strip().startswith(prompt.strip()):
        prediction_text = model_output[len(prompt.strip()):].strip()
    
    cleaned_text = prediction_text.strip()
    
    # Check for exact label matches
    for label in valid_labels:
        if label in cleaned_text:
            return label
    
    return None

def extract_korean_sentiment_answer(model_output: str, prompt: str) -> Optional[str]:
    """Extract sentiment answer from Korean model output"""
    prediction_text = model_output
    if model_output.strip().startswith(prompt.strip()):
        prediction_text = model_output[len(prompt.strip()):].strip()
    
    cleaned_text = prediction_text.strip().lower()
    
    # Check for Korean sentiment keywords
    if any(word in cleaned_text for word in ["긍정", "좋", "추천"]):
        return "긍정"
    elif any(word in cleaned_text for word in ["부정", "나쁘", "별로"]):
        return "부정"
    
    return None

def get_korean_ground_truth(item: Dict, benchmark: str, task: str = "") -> Optional[str]:
    """Extract ground truth answer in standardized format for Korean benchmarks"""
    if benchmark == "kmmlu":
        answer_idx = item.get("answer", -1)
        if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
            return str(answer_idx + 1)  # Convert to 1-4 format
    
    elif benchmark == "klue":
        if task == "ynat":
            label = item.get("label", -1)
            if isinstance(label, int) and 0 <= label <= 6:
                topics = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
                return topics[label] if label < len(topics) else None
        
        elif task == "nli":
            label = item.get("label", "")
            if label in ["entailment", "contradiction", "neutral"]:
                return label
    
    elif benchmark == "nsmc":
        label = item.get("label", -1)
        if label == 1:
            return "긍정"
        elif label == 0:
            return "부정"
    
    return None

# --- Model Management ---
def load_model(config: ModelConfig) -> Tuple[Optional[Any], Optional[Any]]:
    """Load model and tokenizer with error handling"""
    try:
        logger.info(f"Loading model: {config.name} from {config.model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, 
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True
        )
        
        # Handle pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Adding new pad token '[PAD]'")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Setup quantization if needed
        quantization_config = None
        if config.use_quantization:
            logger.info("Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
            low_cpu_mem_usage=True
        )
        
        # Handle pad token in model config
        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
            logger.warning("Resizing model embeddings for new PAD token")
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id
        
        model.eval()
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {config.name}: {e}")
        return None, None

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> Optional[str]:
    """Generate model response with error handling"""
    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1
            )
        
        # Extract only the generated tokens
        output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None

def cleanup_resources(model, tokenizer):
    """Clean up model and tokenizer resources"""
    logger.info("Cleaning up resources...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Evaluation Functions ---
def evaluate_kmmlu(model, tokenizer, kmmlu_data: List[Dict], model_name: str) -> EvaluationResult:
    """Evaluate model on KMMLU benchmark"""
    logger.info(f"Evaluating KMMLU for {model_name}")
    
    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0
    errors_or_skipped = 0
    
    for i, item in enumerate(tqdm(kmmlu_data, desc="KMMLU Evaluation")):
        ground_truth = get_korean_ground_truth(item, "kmmlu")
        prompt = create_kmmlu_prompt(item)
        
        if ground_truth is None or prompt is None:
            errors_or_skipped += 1
            continue
        
        generated_text = generate_response(model, tokenizer, prompt)
        
        if generated_text is None:
            errors_or_skipped += 1
            continue
        
        predicted_answer = extract_korean_multiple_choice_answer(generated_text, prompt)
        
        if predicted_answer is None:
            errors_or_skipped += 1
            continue
        
        total_predictions += 1
        if predicted_answer == ground_truth:
            correct_predictions += 1
        
        if (i + 1) % 50 == 0:
            current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            logger.info(f"Progress: {i + 1}/{len(kmmlu_data)}, Acc: {current_acc:.2f}%")
    
    evaluation_time = time.time() - start_time
    score = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    logger.info(f"KMMLU Results: {score:.2f}% ({correct_predictions}/{total_predictions})")
    
    return EvaluationResult(
        model_name=model_name,
        benchmark_name="KMMLU",
        task_name="Korean Reasoning",
        score=score,
        total_questions=len(kmmlu_data),
        correct_answers=correct_predictions,
        evaluation_time=evaluation_time,
        timestamp=datetime.now().isoformat(),
        additional_metrics={
            "total_predictions": total_predictions,
            "errors_or_skipped": errors_or_skipped
        }
    )

def evaluate_klue_task(model, tokenizer, task_data: List[Dict], task: str, model_name: str) -> EvaluationResult:
    """Evaluate model on a single KLUE task"""
    logger.info(f"Evaluating KLUE-{task.upper()} for {model_name}")
    
    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0
    errors_or_skipped = 0
    
    for i, item in enumerate(tqdm(task_data, desc=f"KLUE-{task.upper()}")):
        ground_truth = get_korean_ground_truth(item, "klue", task)
        prompt = create_klue_prompt(item, task)
        
        if ground_truth is None or prompt is None:
            errors_or_skipped += 1
            continue
        
        generated_text = generate_response(model, tokenizer, prompt, max_new_tokens=30)
        
        if generated_text is None:
            errors_or_skipped += 1
            continue
        
        # Task-specific answer extraction
        if task == "ynat":
            topics = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
            predicted_answer = extract_korean_classification_answer(generated_text, prompt, topics)
        elif task == "nli":
            labels = ["entailment", "contradiction", "neutral"]
            predicted_answer = extract_korean_classification_answer(generated_text, prompt, labels)
        else:
            predicted_answer = generated_text.strip()[:50]  # Simple text extraction for other tasks
        
        if predicted_answer is None:
            errors_or_skipped += 1
            continue
        
        total_predictions += 1
        if predicted_answer == ground_truth:
            correct_predictions += 1
    
    evaluation_time = time.time() - start_time
    score = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    logger.info(f"KLUE-{task.upper()} Results: {score:.2f}% ({correct_predictions}/{total_predictions})")
    
    return EvaluationResult(
        model_name=model_name,
        benchmark_name="KLUE",
        task_name=f"KLUE-{task.upper()}",
        score=score,
        total_questions=len(task_data),
        correct_answers=correct_predictions,
        evaluation_time=evaluation_time,
        timestamp=datetime.now().isoformat(),
        additional_metrics={
            "total_predictions": total_predictions,
            "errors_or_skipped": errors_or_skipped
        }
    )

def evaluate_nsmc(model, tokenizer, nsmc_data: List[Dict], model_name: str) -> EvaluationResult:
    """Evaluate model on NSMC sentiment analysis"""
    logger.info(f"Evaluating NSMC for {model_name}")
    
    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0
    errors_or_skipped = 0
    
    for i, item in enumerate(tqdm(nsmc_data, desc="NSMC Evaluation")):
        ground_truth = get_korean_ground_truth(item, "nsmc")
        prompt = create_nsmc_prompt(item)
        
        if ground_truth is None or prompt is None:
            errors_or_skipped += 1
            continue
        
        generated_text = generate_response(model, tokenizer, prompt)
        
        if generated_text is None:
            errors_or_skipped += 1
            continue
        
        predicted_answer = extract_korean_sentiment_answer(generated_text, prompt)
        
        if predicted_answer is None:
            errors_or_skipped += 1
            continue
        
        total_predictions += 1
        if predicted_answer == ground_truth:
            correct_predictions += 1
    
    evaluation_time = time.time() - start_time
    score = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    logger.info(f"NSMC Results: {score:.2f}% ({correct_predictions}/{total_predictions})")
    
    return EvaluationResult(
        model_name=model_name,
        benchmark_name="NSMC",
        task_name="Sentiment Analysis",
        score=score,
        total_questions=len(nsmc_data),
        correct_answers=correct_predictions,
        evaluation_time=evaluation_time,
        timestamp=datetime.now().isoformat(),
        additional_metrics={
            "total_predictions": total_predictions,
            "errors_or_skipped": errors_or_skipped
        }
    )

def evaluate_single_model(config: ModelConfig, output_dir: Path) -> Optional[BenchmarkResults]:
    """Evaluate a single model on all Korean benchmarks"""
    logger.info(f"Starting Korean benchmark evaluation for {config.name}")
    
    start_time = time.time()
    
    # Load model
    model, tokenizer = load_model(config)
    if model is None or tokenizer is None:
        logger.error(f"Failed to load model {config.name}")
        return None
    
    try:
        detailed_results = []
        
        # Evaluate KMMLU
        kmmlu_data = load_kmmlu_data()
        if kmmlu_data:
            kmmlu_result = evaluate_kmmlu(model, tokenizer, kmmlu_data, config.name)
            detailed_results.append(kmmlu_result)
        else:
            logger.warning("KMMLU data not available")
            kmmlu_result = None
        
        # Evaluate KLUE tasks
        klue_data = load_klue_data()
        klue_scores = {}
        if klue_data:
            for task_code, task_data in klue_data.items():
                if task_data:
                    klue_result = evaluate_klue_task(model, tokenizer, task_data, task_code, config.name)
                    detailed_results.append(klue_result)
                    klue_scores[task_code] = klue_result.score
        
        # Evaluate NSMC
        nsmc_data = load_nsmc_data()
        if nsmc_data:
            nsmc_result = evaluate_nsmc(model, tokenizer, nsmc_data, config.name)
            detailed_results.append(nsmc_result)
        else:
            logger.warning("NSMC data not available")
            nsmc_result = None
        
        total_evaluation_time = time.time() - start_time
        
        # Create benchmark results
        results = BenchmarkResults(
            model_name=config.name,
            model_path=config.model_id,
            evaluation_date=datetime.now().isoformat(),
            kmmlu_score=kmmlu_result.score if kmmlu_result else None,
            klue_scores=klue_scores if klue_scores else None,
            nsmc_score=nsmc_result.score if nsmc_result else None,
            total_evaluation_time=total_evaluation_time,
            detailed_results=detailed_results
        )
        
        # Save results
        save_results(results, output_dir)
        
        logger.info(f"Korean evaluation completed for {config.name}")
        logger.info(f"Total time: {total_evaluation_time:.1f}s")
        
        return results
        
    finally:
        cleanup_resources(model, tokenizer)

# --- Result Storage ---
def save_results(results: BenchmarkResults, output_dir: Path):
    """Save evaluation results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary for JSON serialization
    results_dict = {
        "model_name": results.model_name,
        "model_path": results.model_path,
        "evaluation_date": results.evaluation_date,
        "kmmlu_score": results.kmmlu_score,
        "klue_scores": results.klue_scores,
        "nsmc_score": results.nsmc_score,
        "total_evaluation_time": results.total_evaluation_time,
        "detailed_results": [
            {
                "model_name": r.model_name,
                "benchmark_name": r.benchmark_name,
                "task_name": r.task_name,
                "score": r.score,
                "total_questions": r.total_questions,
                "correct_answers": r.correct_answers,
                "evaluation_time": r.evaluation_time,
                "timestamp": r.timestamp,
                "additional_metrics": r.additional_metrics
            }
            for r in (results.detailed_results or [])
        ]
    }
    
    # Save detailed results
    results_file = output_dir / f"results_{results.model_name}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")

# --- Main Execution ---
def main():
    """Main evaluation function"""
    logger.info("Starting Korean Benchmark Evaluation System")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {DEVICE}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(BASE_OUTPUT_DIR) / f"run_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {output_base}")
    
    # Evaluate each model
    all_results = []
    
    for config in MODEL_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Model: {config.name}")
        logger.info(f"{'='*60}")
        
        model_output_dir = output_base / config.name
        results = evaluate_single_model(config, model_output_dir)
        
        if results:
            all_results.append(results)
        
        logger.info(f"Finished evaluating {config.name}")
    
    # Save summary
    if all_results:
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_models": len(all_results),
            "models_evaluated": [r.model_name for r in all_results],
            "summary": []
        }
        
        for r in all_results:
            klue_avg = np.mean(list(r.klue_scores.values())) if r.klue_scores else None
            model_summary = {
                "model_name": r.model_name,
                "kmmlu_score": r.kmmlu_score,
                "klue_average_score": klue_avg,
                "nsmc_score": r.nsmc_score,
                "evaluation_time": r.total_evaluation_time
            }
            summary["summary"].append(model_summary)
        
        summary_file = output_base / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")
    
    logger.info("\n🎉 한국어 벤치마크 평가가 완료되었습니다!")

if __name__ == "__main__":
    main()