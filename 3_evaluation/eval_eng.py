#!/usr/bin/env python3
"""
English Benchmark Evaluation System
===================================

Unified evaluation system for English benchmarks:
- MMLU (Massive Multitask Language Understanding)
- GLUE (General Language Understanding Evaluation) 
- SuperGLUE (Advanced Language Understanding)
- HellaSwag (Commonsense Reasoning)
- ARC (AI2 Reasoning Challenge)
- TruthfulQA (Truthfulness in QA)

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
    """Complete evaluation results for a model across all English benchmarks"""
    model_name: str
    model_path: str
    evaluation_date: str
    mmlu_score: Optional[float] = None
    glue_scores: Optional[Dict[str, float]] = None
    superglue_scores: Optional[Dict[str, float]] = None
    hellaswag_score: Optional[float] = None
    arc_scores: Optional[Dict[str, float]] = None
    truthfulqa_score: Optional[float] = None
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
    "mmlu": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/MMLU/test",
    "glue": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/GLUE", 
    "superglue": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/SuperGLUE",
    "hellaswag": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/HellaSwag/hellaswag_val.json",
    "arc": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/ARC",
    "truthfulqa": "C:/Users/songj/OneDrive/Desktop/Increase_MLLM_Ability/2_datasets/benchmarks/TruthfulQA/TruthfulQA.csv"
}

BASE_OUTPUT_DIR = "evaluation_results_english"
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
        data_format: Format type (json, jsonl, csv)
    
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
        elif data_format == "csv" or filepath.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(filepath)
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

def load_mmlu_data() -> Optional[List[Dict]]:
    """Load MMLU dataset"""
    mmlu_path = DATASET_PATHS.get("mmlu")
    if not mmlu_path or not Path(mmlu_path).exists():
        logger.warning("MMLU dataset path not found, using mock data")
        return create_mock_mmlu_data()
    
    # Try to load from directory structure
    try:
        all_data = []
        mmlu_dir = Path(mmlu_path)
        for subject_file in mmlu_dir.glob("*.csv"):
            subject_data = load_dataset_data(str(subject_file), "csv")
            if subject_data:
                for item in subject_data:
                    item['subject'] = subject_file.stem
                all_data.extend(subject_data)
        
        if all_data:
            return all_data
    except Exception as e:
        logger.warning(f"Error loading MMLU from directory: {e}")
    
    # Fallback to single file
    return load_dataset_data(mmlu_path) or create_mock_mmlu_data()

def create_mock_mmlu_data() -> List[Dict]:
    """Create mock MMLU data for testing"""
    logger.info("Creating mock MMLU data for testing")
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2,
            "subject": "geography"
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Mercury", "Earth", "Mars"], 
            "answer": 1,
            "subject": "astronomy"
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "mathematics"
        }
    ]

def load_hellaswag_data() -> Optional[List[Dict]]:
    """Load HellaSwag dataset"""
    hellaswag_path = DATASET_PATHS.get("hellaswag")
    if not hellaswag_path or not Path(hellaswag_path).exists():
        logger.warning("HellaSwag dataset not found, using mock data")
        return [
            {
                "ctx": "A person is trying to cut a tree branch.",
                "endings": [
                    "The person uses a chainsaw to cut the branch.",
                    "The person uses a spoon to cut the branch.", 
                    "The branch cuts the person.",
                    "The tree flies away."
                ],
                "label": "0",
                "activity_label": "cutting"
            }
        ]
    
    return load_dataset_data(hellaswag_path)

# --- Prompt Generation Functions ---
def create_multiple_choice_prompt(question: str, choices: List[str], context: str = "") -> str:
    """Create standardized multiple choice prompt"""
    prompt = ""
    if context:
        prompt += f"Context: {context}\n\n"
    
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}) {choice}\n"
    prompt += "Answer:"
    
    return prompt

def create_mmlu_prompt(item: Dict) -> Optional[str]:
    """Create MMLU-specific prompt"""
    question = item.get("question", "")
    choices = item.get("choices", [])
    
    if not question or not isinstance(choices, list) or len(choices) != 4:
        return None
    
    return create_multiple_choice_prompt(question, choices)

def create_hellaswag_prompt(item: Dict) -> Optional[str]:
    """Create HellaSwag-specific prompt"""
    context = item.get("ctx", "")
    endings = item.get("endings", [])
    
    if not context or not isinstance(endings, list) or len(endings) != 4:
        return None
    
    return create_multiple_choice_prompt(
        "Which ending makes the most sense?", 
        endings, 
        context
    )

def create_arc_prompt(item: Dict) -> Optional[str]:
    """Create ARC-specific prompt"""
    question = item.get("question", {})
    choices = question.get("choices", []) if isinstance(question, dict) else []
    stem = question.get("stem", "") if isinstance(question, dict) else item.get("question", "")
    
    if not stem:
        return None
    
    if isinstance(choices, list) and choices:
        choice_texts = [choice.get("text", "") if isinstance(choice, dict) else str(choice) for choice in choices]
        return create_multiple_choice_prompt(stem, choice_texts)
    
    return f"Question: {stem}\nAnswer:"

# --- Answer Extraction Functions ---
def extract_multiple_choice_answer(model_output: str, prompt: str, valid_choices: str = "ABCD") -> Optional[str]:
    """Extract multiple choice answer from model output"""
    # Remove prompt echo if present
    prediction_text = model_output
    if model_output.strip().startswith(prompt.strip()):
        prompt_end = prompt.strip().split("Answer:")[-1] if "Answer:" in prompt else prompt.strip()
        prediction_text = model_output[len(prompt_end):].strip()
    
    # Clean the text
    cleaned_text = prediction_text.upper().strip()
    
    # Pattern 1: Direct answer at start
    match = re.search(rf"^\s*([{valid_choices}])(?:[).:\s]|\b|$)", cleaned_text)
    if match:
        return match.group(1)
    
    # Pattern 2: Single character answer
    if len(cleaned_text) == 1 and cleaned_text in valid_choices:
        return cleaned_text
    
    # Pattern 3: Answer with phrase
    match = re.search(rf"(?:ANSWER\s*IS|:\s*)\s*([{valid_choices}])\b", cleaned_text)
    if match:
        return match.group(1)
    
    # Pattern 4: Answer followed by explanation
    match = re.search(rf"\b([{valid_choices}])\b.*(?:correct|answer|choice)", cleaned_text)
    if match:
        return match.group(1)
    
    return None

def get_ground_truth(item: Dict, benchmark: str) -> Optional[str]:
    """Extract ground truth answer in standardized format"""
    if benchmark == "mmlu":
        answer_idx = item.get("answer", -1)
        if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
            return chr(ord('A') + answer_idx)
    
    elif benchmark == "hellaswag":
        label = item.get("label", "")
        if isinstance(label, str) and label.isdigit():
            idx = int(label)
            if 0 <= idx <= 3:
                return chr(ord('A') + idx)
    
    elif benchmark == "arc":
        answer_key = item.get("answerKey", "")
        if isinstance(answer_key, str) and answer_key.upper() in "ABCD":
            return answer_key.upper()
    
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

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 15) -> Optional[str]:
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
def evaluate_benchmark(model, tokenizer, benchmark_data: List[Dict], benchmark_name: str, model_name: str) -> EvaluationResult:
    """Evaluate model on a specific benchmark"""
    logger.info(f"Evaluating {benchmark_name} for {model_name}")
    
    start_time = time.time()
    correct_predictions = 0
    total_predictions = 0
    errors_or_skipped = 0
    
    # Select appropriate prompt and ground truth functions
    if benchmark_name == "mmlu":
        create_prompt = create_mmlu_prompt
    elif benchmark_name == "hellaswag":
        create_prompt = create_hellaswag_prompt
    elif benchmark_name == "arc":
        create_prompt = create_arc_prompt
    else:
        create_prompt = lambda item: create_multiple_choice_prompt(
            item.get("question", ""), 
            item.get("choices", [])
        )
    
    for i, item in enumerate(tqdm(benchmark_data, desc=f"Evaluating {benchmark_name}")):
        # Get ground truth and create prompt
        ground_truth = get_ground_truth(item, benchmark_name)
        prompt = create_prompt(item)
        
        if ground_truth is None or prompt is None:
            errors_or_skipped += 1
            continue
        
        # Generate response
        generated_text = generate_response(model, tokenizer, prompt)
        
        if generated_text is None:
            errors_or_skipped += 1
            continue
        
        # Extract answer
        predicted_answer = extract_multiple_choice_answer(generated_text, prompt)
        
        if predicted_answer is None:
            errors_or_skipped += 1
            continue
        
        total_predictions += 1
        if predicted_answer == ground_truth:
            correct_predictions += 1
        
        # Log progress every 100 items
        if (i + 1) % 100 == 0:
            current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            logger.info(f"Progress: {i + 1}/{len(benchmark_data)}, Acc: {current_acc:.2f}%")
    
    evaluation_time = time.time() - start_time
    score = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    logger.info(f"{benchmark_name} Results: {score:.2f}% ({correct_predictions}/{total_predictions})")
    
    return EvaluationResult(
        model_name=model_name,
        benchmark_name=benchmark_name,
        task_name=benchmark_name.upper(),
        score=score,
        total_questions=len(benchmark_data),
        correct_answers=correct_predictions,
        evaluation_time=evaluation_time,
        timestamp=datetime.now().isoformat(),
        additional_metrics={
            "total_predictions": total_predictions,
            "errors_or_skipped": errors_or_skipped
        }
    )

def evaluate_single_model(config: ModelConfig, output_dir: Path) -> Optional[BenchmarkResults]:
    """Evaluate a single model on all English benchmarks"""
    logger.info(f"Starting evaluation for {config.name}")
    
    start_time = time.time()
    
    # Load model
    model, tokenizer = load_model(config)
    if model is None or tokenizer is None:
        logger.error(f"Failed to load model {config.name}")
        return None
    
    try:
        detailed_results = []
        
        # Load and evaluate MMLU
        mmlu_data = load_mmlu_data()
        if mmlu_data:
            mmlu_result = evaluate_benchmark(model, tokenizer, mmlu_data, "mmlu", config.name)
            detailed_results.append(mmlu_result)
        else:
            logger.warning("MMLU data not available")
            mmlu_result = None
        
        # Load and evaluate HellaSwag
        hellaswag_data = load_hellaswag_data()
        if hellaswag_data:
            hellaswag_result = evaluate_benchmark(model, tokenizer, hellaswag_data, "hellaswag", config.name)
            detailed_results.append(hellaswag_result)
        else:
            logger.warning("HellaSwag data not available")
            hellaswag_result = None
        
        # Additional benchmarks can be added here
        # arc_data = load_arc_data()
        # glue_data = load_glue_data()
        # etc.
        
        total_evaluation_time = time.time() - start_time
        
        # Create benchmark results
        results = BenchmarkResults(
            model_name=config.name,
            model_path=config.model_id,
            evaluation_date=datetime.now().isoformat(),
            mmlu_score=mmlu_result.score if mmlu_result else None,
            hellaswag_score=hellaswag_result.score if hellaswag_result else None,
            total_evaluation_time=total_evaluation_time,
            detailed_results=detailed_results
        )
        
        # Save results
        save_results(results, output_dir)
        
        logger.info(f"Evaluation completed for {config.name}")
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
        "mmlu_score": results.mmlu_score,
        "hellaswag_score": results.hellaswag_score,
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
    logger.info("Starting English Benchmark Evaluation System")
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
            "summary": [
                {
                    "model_name": r.model_name,
                    "mmlu_score": r.mmlu_score,
                    "hellaswag_score": r.hellaswag_score,
                    "evaluation_time": r.total_evaluation_time
                }
                for r in all_results
            ]
        }
        
        summary_file = output_base / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")
    
    logger.info("\n🎉 English benchmark evaluation completed!")

if __name__ == "__main__":
    main()