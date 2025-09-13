#!/usr/bin/env python3
"""
GSM8K (HRM8K) Evaluation Script
- Evaluates mathematical reasoning capability on Korean translated GSM8K dataset
- Extracts numerical answers from model outputs
- Saves detailed results per model and creates final summary
"""

import os
import json
import logging
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from dataclasses import dataclass, field
import gc
import sys
from pathlib import Path
from datetime import datetime
import time
import random
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "../cache"  # Cache directory for models
DATASET_PATH = "../../2_datasets/HRM8K_TEXT/GSM8K-test.json"
BASE_OUTPUT_DIR = "../4_evaluation_results/GSM8K_8shot"  # Output directory

# Import performance analyzer
try:
    import sys
    sys.path.append('../')
    from performance_analyzer import create_enhanced_summary
except ImportError:
    logger.warning("Performance analyzer not available. Using basic summary.")
    create_enhanced_summary = None

# GSM8K 8-shot examples (placeholder - these should be defined with actual examples)
GSM8K_8SHOT_KOR_COT_EXAMPLES = [
    {
        "question": "연필 3자루를 각각 2달러, 공책 2권을 각각 5달러에 산다면 총 얼마를 쓰게 되나요?",
        "cot_content": "연필 값은 3 × 2달러 = 6달러. 공책 값은 2 × 5달러 = 10달러. 합하면 6 + 10 = 16달러.",
        "answer": "16"
    },
    {
        "question": "기차가 시속 60마일로 달린다면, 3시간 동안 몇 마일을 갈까요?",
        "cot_content": "기차는 1시간에 60마일을 간다. 3시간 동안은 60 × 3 = 180마일.",
        "answer": "180"
    },
    {
        "question": "사라는 사과 24개를 가지고 있다. 6명의 친구에게 똑같이 나누어 준다면, 친구 한 명당 몇 개를 받게 될까요?",
        "cot_content": "24 ÷ 6 = 4개씩 받는다.",
        "answer": "4"
    },
    {
        "question": "톰은 50달러를 가지고 있었다. 점심에 18달러, 책에 12달러를 썼다면, 남은 돈은 얼마인가요?",
        "cot_content": "쓴 돈은 18 + 12 = 30달러. 남은 돈은 50 - 30 = 20달러.",
        "answer": "20"
    },
    {
        "question": "상자 하나에 초콜릿이 8개 들어 있다. 7상자에는 총 몇 개의 초콜릿이 있을까요?",
        "cot_content": "상자당 8개이므로 7 × 8 = 56개.",
        "answer": "56"
    },
    {
        "question": "농부가 소 36마리를 가지고 있다. 이 중 9마리를 팔았다면, 몇 마리가 남을까요?",
        "cot_content": "36 - 9 = 27마리.",
        "answer": "27"
    },
    {
        "question": "리사는 하루에 책 12쪽을 읽는다. 5일 동안 총 몇 쪽을 읽게 될까요?",
        "cot_content": "12 × 5 = 60쪽.",
        "answer": "60"
    },
    {
        "question": "피자 한 판은 8조각이다. 피자를 3판 주문하면 총 몇 조각이 될까요?",
        "cot_content": "피자 한 판에 8조각이므로 3 × 8 = 24조각.",
        "answer": "24"
    }
]


GSM8K_8SHOT_COT_EXAMPLES = [
    {
        "question": "If you buy 3 pencils for $2 each and 2 notebooks for $5 each, how much do you spend in total?",
        "cot_content": "The pencils cost 3 × $2 = $6. The notebooks cost 2 × $5 = $10. Adding them gives $6 + $10 = $16.",
        "answer": "16"
    },
    {
        "question": "A train travels 60 miles per hour. How far will it go in 3 hours?",
        "cot_content": "The train travels 60 miles each hour. In 3 hours, it will go 60 × 3 = 180 miles.",
        "answer": "180"
    },
    {
        "question": "Sarah has 24 apples. She shares them equally among 6 friends. How many apples does each friend get?",
        "cot_content": "24 apples ÷ 6 friends = 4 apples each.",
        "answer": "4"
    },
    {
        "question": "Tom had $50. He spent $18 on lunch and $12 on a book. How much money does he have left?",
        "cot_content": "He spent $18 + $12 = $30. Subtracting gives $50 - $30 = $20 left.",
        "answer": "20"
    },
    {
        "question": "A box holds 8 chocolates. How many chocolates are there in 7 boxes?",
        "cot_content": "Each box has 8 chocolates. 7 × 8 = 56 chocolates in total.",
        "answer": "56"
    },
    {
        "question": "A farmer has 36 cows. If he sells 9 of them, how many are left?",
        "cot_content": "36 - 9 = 27 cows left.",
        "answer": "27"
    },
    {
        "question": "Lisa reads 12 pages of a book each day. How many pages does she read in 5 days?",
        "cot_content": "12 pages/day × 5 days = 60 pages.",
        "answer": "60"
    },
    {
        "question": "A pizza has 8 slices. If 3 pizzas are ordered, how many slices are there in total?",
        "cot_content": "Each pizza has 8 slices. 3 × 8 = 24 slices in total.",
        "answer": "24"
    }
]


# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_id: str
    adapter_path: str = None
    use_quantization: bool = True
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

MODEL_CONFIGS = [
    # Base Models (commented out for now)
    # ModelConfig(
    #     name="qwem-2.5-3b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
    #     use_quantization=False
    # ),
    ModelConfig(
        name="llama-3.2-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-org-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-org-merged",
        use_quantization=False
    ),

    # ModelConfig(
    #     name="qwem-2.5-3b-pt-tow",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/Qwen2.5-3B-Instruct-tow",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-pt-tow",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/gemma-3-4b-it-tow",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="llama-3.2-3b-pt-tow",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/llama-3.2-3b-tow/checkpoint-5750",
    #     use_quantization=False
    # ),
]


# --- Helper Functions ---
def load_gsm8k_data(filepath):
    """Load GSM8K dataset from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Data is not a list format")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Not all items in list are dictionaries")
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_gsm8k_prompt(text, few_shot_examples, is_korean=False):
    """
    (개선된 최종 버전)
    딕셔너리 리스트 형태의 few-shot 예제를 사용하여
    GSM8K 8-shot CoT 평가 프롬프트를 동적으로 생성합니다.
    """
    prompt_parts = []

    # 1. 루프를 통해 8개의 예제를 동적으로 구성합니다.
    for example in few_shot_examples:
        question = example["question"]
        cot_content = example["cot_content"] # 예제 딕셔너리의 cot_content를 사용
        answer = example["answer"]
        
        if is_korean:
            full_answer_block = f"응답: {cot_content} #### 따라서 정답은 {{{answer}}}. #### {{{answer}}}"
            example_block = f"문제: {question}\n {full_answer_block}"
        else:
            full_answer_block = f"Response: {cot_content} #### So the answer is {{{answer}}}. #### {{{answer}}}"
            example_block = f"Question: {question}\n {full_answer_block}"

        # 최종 예제 블록을 조립합니다.
        prompt_parts.append(example_block)

    # 2. 모든 예제를 하나의 문자열로 합칩니다. (예제 사이에 두 줄 띄어쓰기)
    final_examples_str = "\n\n".join(prompt_parts)

    # 3. 최종 프롬프트를 완성합니다: [8개 예제] + [실제 문제] + [CoT 시작 유도]
    if is_korean:
        final_prompt = f"""{final_examples_str}

문제: {text}
응답: 단계별로 생각해봅시다."""
    else:
        final_prompt = f"""{final_examples_str}

Question: {text}
Response: Let's think step by step."""
    
    return final_prompt

def extract_numerical_answer(model_output):
    """
    Extract numerical answer from model output with strict priority:
    1) #### <number>
    2) { <number> }  # boxed answer
    3) Other English/Korean phrasings
    """
    cleaned_output = model_output.strip()

    patterns = [
        # 1) #### 18  (최우선)
        r'####\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',

        # 2) { 18 }   (둘째 우선)
        r'\{\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*\}',

        # 3) 그 외 일반 패턴
        r'답[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'The answer is\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:정답|Answer)[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:답|정답|Answer)\s*(?:은|는|is)?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'(?:따라서|그러므로|그래서|결론적으로|최종적으로|Hence|Therefore)\s*(?:답|정답|answer)?[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?)(?:\s*(?:입니다|이다|\.|\s*$))',
        r'(?:총|합계|전체|Total)\s*[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',
        r'=\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)(?:\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?))?(?:\s*$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer_str = matches[-1].replace(',', '').strip()
            try:
                return float(answer_str)
            except ValueError:
                continue

    # Fallback: 마지막 줄에서 숫자 스캔
    for line in reversed(cleaned_output.split('\n')):
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)', line)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                continue

    return None


def check_numerical_match(predicted, ground_truth, tolerance=1e-6):
    """
    Check if predicted answer matches ground truth with tolerance
    """
    if predicted is None or ground_truth is None:
        return False
    
    try:
        pred_float = float(predicted)
        gt_float = float(ground_truth)
        return abs(pred_float - gt_float) < tolerance
    except (ValueError, TypeError):
        return False

def process_single_with_retry(model, tokenizer, prompt, max_retries=5):
    """
    Process a single prompt with retry logic for answer extraction failures
    Only retries when answer extraction fails (not on genuine model errors)
    """
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                )
            
            input_lengths = inputs['input_ids'].shape[1]
            output_only_tokens = outputs[:, input_lengths:]
            generated_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
            
            # Try to extract answer
            extracted_answer = extract_numerical_answer(generated_text)
            if extracted_answer is not None:
                return generated_text, extracted_answer
            else:
                # Answer extraction failed - try again if we have retries left
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Failed to extract answer, retrying...")
                    # Small delay before retry
                    time.sleep(0.1 + random.random() * 0.1)
                    continue
                else:
                    logger.warning(f"Final attempt failed - could not extract answer after {max_retries} attempts")
                    return generated_text, None
                    
        except Exception as e:
            logger.error(f"Retry {attempt + 1}/{max_retries}: Model inference error: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.2 + random.random() * 0.2)
                continue
            else:
                # Return error info after all retries exhausted
                return f"ERROR after {max_retries} attempts: {str(e)}", None
    
    return f"EXTRACTION_FAILED after {max_retries} attempts", None

def evaluate_single_model(config: ModelConfig, gsm8k_data: list, model_output_dir: str):
    """
    Evaluate single model on GSM8K dataset
    """
    os.makedirs(model_output_dir, exist_ok=True)
    results_korean_filepath = os.path.join(model_output_dir, f"results_korean_{config.name}.json")
    results_english_filepath = os.path.join(model_output_dir, f"results_english_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")
    raw_gen_korean_filepath = os.path.join(model_output_dir, f"raw_generations_korean_{config.name}.json")
    raw_gen_english_filepath = os.path.join(model_output_dir, f"raw_generations_english_{config.name}.json")

    # Setup logging for this model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler is not file_handler:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)

    logger.info(f"--- Starting GSM8K Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Korean results will be saved to: {results_korean_filepath}")
    logger.info(f"English results will be saved to: {results_english_filepath}")
    logger.info(f"Korean raw generations will be saved to: {raw_gen_korean_filepath}")
    logger.info(f"English raw generations will be saved to: {raw_gen_english_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations_korean_list = []
    raw_generations_english_list = []

    try:
        # Load Model and Tokenizer
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding new pad token '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        logger.info(f"Loading model {config.model_id}...")
        quantization_config_bnb = None
        if config.use_quantization:
            logger.info("Applying 4-bit quantization.")
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            quantization_config=quantization_config_bnb,
            device_map=DEVICE,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        if config.adapter_path:
            # LoRA 어댑터가 있는 경우, 먼저 LoRA의 실제 vocab size를 확인
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")
            
            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")
            
            # LoRA 어댑터의 실제 vocab size 확인
            try:
                import glob
                pytorch_files = glob.glob(os.path.join(absolute_adapter_path, "*.bin")) + \
                            glob.glob(os.path.join(absolute_adapter_path, "*.safetensors"))
                
                target_vocab_size = None
                if pytorch_files:
                    if pytorch_files[0].endswith('.safetensors'):
                        from safetensors import safe_open
                        with safe_open(pytorch_files[0], framework="pt") as f:
                            for key in f.keys():
                                if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                                    target_vocab_size = f.get_tensor(key).shape[0]
                                    break
                    else:
                        checkpoint = torch.load(pytorch_files[0], map_location='cpu')
                        for key, tensor in checkpoint.items():
                            if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                                target_vocab_size = tensor.shape[0]
                                break
                
                if target_vocab_size:
                    current_vocab_size = model.get_input_embeddings().weight.shape[0]
                    if current_vocab_size != target_vocab_size:
                        logger.info(f"Resizing model from {current_vocab_size} to {target_vocab_size} for LoRA compatibility")
                        model.resize_token_embeddings(target_vocab_size)
                else:
                    # fallback: tokenizer 길이로 리사이즈
                    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                        model.resize_token_embeddings(len(tokenizer))
                        
            except Exception as e:
                logger.warning(f"Could not determine LoRA vocab size: {e}. Using tokenizer length.")
                if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                    model.resize_token_embeddings(len(tokenizer))
            
            try:
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            # 베이스 모델인 경우 기존 로직 사용
            if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
                model.resize_token_embeddings(len(tokenizer))
            logger.info("No LoRA adapter path specified. Using the base model directly.")

        # Configure tokenizer padding
        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
            logger.warning("Resizing model embeddings after load due to added PAD token.")
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
        
        # Gemma 모델에서만 컴파일 비활성화
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")


        # Run Evaluation
        correct_predictions_korean = 0
        total_predictions_korean = 0
        errors_or_skipped_korean = 0
        results_details_korean = []
        
        correct_predictions_english = 0
        total_predictions_english = 0
        errors_or_skipped_english = 0
        results_details_english = []

        logger.info("Starting GSM8K inference loop...")
        logger.info(f"Dataset size: {len(gsm8k_data)}")

        # Process each item for both Korean and English versions
        for idx, item in enumerate(tqdm(gsm8k_data, desc=f"Evaluating {config.name} (GSM8K)")):
            ground_truth = item.get("answer", None)
            if ground_truth is None:
                logger.warning(f"Item with no ground truth found at index {idx}. Skipping.")
                errors_or_skipped_korean += 1
                errors_or_skipped_english += 1
                continue

            question = item.get("question", "")
            original = item.get("original", "")
            
            # Check if we have both Korean and English versions
            has_korean = question and original and question != original
            
            # Process Korean version (translated question)
            if has_korean:
                korean_prompt = None
                korean_gen_text = None
                korean_answer = None
                is_correct_korean = False
                exception_info = None

                try:
                    korean_prompt = create_gsm8k_prompt(question, GSM8K_8SHOT_KOR_COT_EXAMPLES, is_korean=True)

                    korean_gen_text, korean_answer = process_single_with_retry(model, tokenizer, korean_prompt)

                    if korean_answer is not None:
                        total_predictions_korean += 1
                        if check_numerical_match(korean_answer, ground_truth):
                            correct_predictions_korean += 1
                            is_correct_korean = True
                    else:
                        # 추출 실패도 에러로 집계(정책 유지)
                        errors_or_skipped_korean += 1
                        if korean_gen_text and not korean_gen_text.startswith("ERROR"):
                            korean_gen_text = f"EXTRACTION_FAILED: {korean_gen_text}"

                except Exception as e:
                    errors_or_skipped_korean += 1
                    exception_info = f"{e}\n{traceback.format_exc()}"
                    # model 출력이 전혀 없었어도, RAW에 남길 수 있도록 메시지 구성
                    if not korean_gen_text:
                        korean_gen_text = f"ERROR: {exception_info}"

                finally:
                    # details는 항상 남긴다
                    results_details_korean.append({
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_raw_output": korean_gen_text,
                        "extracted_answer": korean_answer,
                        "is_correct": is_correct_korean,
                        "exception": exception_info,           # 추가: 예외 스택까지 기록
                        "prompt_snapshot": korean_prompt       # 추가: 사용한 프롬프트 저장(디버깅 유용)
                    })

                    raw_generations_korean_list.append({
                        "index": idx,
                        "language": "Korean",
                        "question": question,
                        "original": original,
                        "ground_truth": ground_truth,
                        "prompt": korean_prompt,               # 추가
                        "raw_output": korean_gen_text,
                        "extracted_answer": korean_answer,
                        "is_correct": is_correct_korean,       # 추가
                        "exception": exception_info            # 추가
                    })


            # Process English version (original question)  
            if original:
                english_prompt = None
                english_gen_text = None
                english_answer = None
                is_correct_english = False
                exception_info_en = None

                try:
                    english_prompt = create_gsm8k_prompt(original, GSM8K_8SHOT_COT_EXAMPLES, is_korean=False)

                    english_gen_text, english_answer = process_single_with_retry(model, tokenizer, english_prompt)

                    if english_answer is not None:
                        total_predictions_english += 1
                        if check_numerical_match(english_answer, ground_truth):
                            correct_predictions_english += 1
                            is_correct_english = True
                    else:
                        errors_or_skipped_english += 1
                        if english_gen_text and not english_gen_text.startswith("ERROR"):
                            english_gen_text = f"EXTRACTION_FAILED: {english_gen_text}"

                except Exception as e:
                    errors_or_skipped_english += 1
                    exception_info_en = f"{e}\n{traceback.format_exc()}"
                    if not english_gen_text:
                        english_gen_text = f"ERROR: {exception_info_en}"

                finally:
                    results_details_english.append({
                        "index": idx,
                        "question": original,
                        "ground_truth": ground_truth,
                        "model_raw_output": english_gen_text,
                        "extracted_answer": english_answer,
                        "is_correct": is_correct_english,
                        "exception": exception_info_en,        # 추가
                        "prompt_snapshot": english_prompt      # 추가
                    })

                    raw_generations_english_list.append({
                        "index": idx,
                        "language": "English",
                        "question": question,
                        "original": original,
                        "ground_truth": ground_truth,
                        "prompt": english_prompt,              # 추가
                        "raw_output": english_gen_text,
                        "extracted_answer": english_answer,
                        "is_correct": is_correct_english,      # 추가
                        "exception": exception_info_en         # 추가
                    })


        # Final Results
        logger.info(f"Inference loop finished for {config.name}.")
        
        # Calculate accuracies for Korean
        accuracy_standard_korean = (correct_predictions_korean / total_predictions_korean * 100) if total_predictions_korean > 0 else 0
        accuracy_strict_korean = (correct_predictions_korean / len(gsm8k_data) * 100) if len(gsm8k_data) > 0 else 0

        # Calculate accuracies for English
        accuracy_standard_english = (correct_predictions_english / total_predictions_english * 100) if total_predictions_english > 0 else 0
        accuracy_strict_english = (correct_predictions_english / len(gsm8k_data) * 100) if len(gsm8k_data) > 0 else 0

        logger.info(f"--- GSM8K Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Questions: {len(gsm8k_data)}")
        logger.info(f"=== Korean Results ===")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions_korean}")
        logger.info(f"Correct Predictions: {correct_predictions_korean}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped_korean}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard_korean:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_questions): {accuracy_strict_korean:.2f}%")
        logger.info(f"=== English Results ===")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions_english}")
        logger.info(f"Correct Predictions: {correct_predictions_english}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped_english}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard_english:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_questions): {accuracy_strict_english:.2f}%")

        # Save Results
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "evaluation_type": "GSM8K (HRM8K Korean and English Separate)",
            "total_questions": len(gsm8k_data),
            "korean_results": {
                "valid_predictions": total_predictions_korean,
                "correct_predictions": correct_predictions_korean,
                "errors_or_skipped": errors_or_skipped_korean,
                "accuracy_standard": accuracy_standard_korean,
                "accuracy_strict": accuracy_strict_korean,
                "details": results_details_korean
            },
            "english_results": {
                "valid_predictions": total_predictions_english,
                "correct_predictions": correct_predictions_english,
                "errors_or_skipped": errors_or_skipped_english,
                "accuracy_standard": accuracy_standard_english,
                "accuracy_strict": accuracy_strict_english,
                "details": results_details_english
            }
        }

        # Save Korean results
        try:
            with open(results_korean_filepath, 'w', encoding='utf-8') as f:
                korean_results = {
                    "model_config": config_dict_serializable,
                    "dataset_path": DATASET_PATH,
                    "evaluation_type": "GSM8K (HRM8K Korean 8-shot)",
                    "language": "Korean",
                    "results": final_summary["korean_results"]
                }
                json.dump(korean_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Korean results saved to {results_korean_filepath}")
        except Exception as e:
            logger.error(f"Failed to save Korean results file {results_korean_filepath}: {e}")
            
        # Save English results
        try:
            with open(results_english_filepath, 'w', encoding='utf-8') as f:
                english_results = {
                    "model_config": config_dict_serializable,
                    "dataset_path": DATASET_PATH,
                    "evaluation_type": "GSM8K (HRM8K English 8-shot)",
                    "language": "English",
                    "results": final_summary["english_results"]
                }
                json.dump(english_results, f, indent=2, ensure_ascii=False)
            logger.info(f"English results saved to {results_english_filepath}")
        except Exception as e:
            logger.error(f"Failed to save English results file {results_english_filepath}: {e}")

        # Save Raw Generations for Korean
        try:
            with open(raw_gen_korean_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_korean_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Korean raw generations saved to {raw_gen_korean_filepath}")
        except Exception as e:
            logger.error(f"Failed to save Korean raw generations file {raw_gen_korean_filepath}: {e}")
            
        # Save Raw Generations for English
        try:
            with open(raw_gen_english_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_english_list, f, indent=2, ensure_ascii=False)
            logger.info(f"English raw generations saved to {raw_gen_english_filepath}")
        except Exception as e:
            logger.error(f"Failed to save English raw generations file {raw_gen_english_filepath}: {e}")

        return final_summary

    except Exception as e:
        logger.exception(f"Critical error during evaluation for {config.name}: {e}")
        return None

    finally:
        # Clean up resources
        logger.info(f"Cleaning up resources for {config.name}...")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Resources cleaned up for {config.name}.")
        
        # Remove file handler
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            try:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            except Exception as e:
                logger.debug(f"Error closing/removing file handler: {e}")

def create_final_summary(all_results: list, base_output_dir: str):
    """Create final summary JSON with all model results"""
    final_results_korean = []
    final_results_english = []
    
    for result in all_results:
        if result is not None:
            # Korean results
            korean_summary = {
                "model_name": result["model_config"]["name"],
                "model_id": result["model_config"]["model_id"],
                "adapter_path": result["model_config"].get("adapter_path", None),
                "total_questions": result["total_questions"],
                "correct_predictions": result["korean_results"]["correct_predictions"],
                "valid_predictions": result["korean_results"]["valid_predictions"],
                "errors_or_skipped": result["korean_results"]["errors_or_skipped"],
                "accuracy_standard": result["korean_results"]["accuracy_standard"],
                "accuracy_strict": result["korean_results"]["accuracy_strict"],
                "evaluation_date": result.get("evaluation_date", "N/A")
            }
            final_results_korean.append(korean_summary)
            
            # English results
            english_summary = {
                "model_name": result["model_config"]["name"],
                "model_id": result["model_config"]["model_id"],
                "adapter_path": result["model_config"].get("adapter_path", None),
                "total_questions": result["total_questions"],
                "correct_predictions": result["english_results"]["correct_predictions"],
                "valid_predictions": result["english_results"]["valid_predictions"],
                "errors_or_skipped": result["english_results"]["errors_or_skipped"],
                "accuracy_standard": result["english_results"]["accuracy_standard"],
                "accuracy_strict": result["english_results"]["accuracy_strict"],
                "evaluation_date": result.get("evaluation_date", "N/A")
            }
            final_results_english.append(english_summary)
    
    # Sort by accuracy (strict) descending
    final_results_korean.sort(key=lambda x: x["accuracy_strict"], reverse=True)
    final_results_english.sort(key=lambda x: x["accuracy_strict"], reverse=True)
    
    final_summary = {
        "evaluation_type": "GSM8K (HRM8K Korean and English Separate Evaluation)",
        "dataset_info": {
            "name": "GSM8K-test (Korean translated and English original)",
            "path": DATASET_PATH,
            "total_questions": final_results_korean[0]["total_questions"] if final_results_korean else 0
        },
        "evaluation_summary": {
            "models_evaluated": len(final_results_korean),
            "best_model_korean": final_results_korean[0]["model_name"] if final_results_korean else "N/A",
            "best_accuracy_korean": final_results_korean[0]["accuracy_strict"] if final_results_korean else 0.0,
            "best_model_english": final_results_english[0]["model_name"] if final_results_english else "N/A",
            "best_accuracy_english": final_results_english[0]["accuracy_strict"] if final_results_english else 0.0
        },
        "korean_results": final_results_korean,
        "english_results": final_results_english
    }
    
    final_json_path = os.path.join(base_output_dir, "final_gsm8k_results.json")
    try:
        # Enhanced summary with performance analysis
        if create_enhanced_summary and final_results_korean and final_results_english:
            evaluation_info = {
                "evaluation_type": "GSM8K (HRM8K Korean and English Separate Evaluation)",
                "evaluation_date": datetime.now().isoformat(),
                "dataset_path": DATASET_PATH,
                "total_models_evaluated": len(final_results_korean)
            }
            
            # Analyze Korean results
            korean_enhanced = create_enhanced_summary(
                model_results=final_results_korean,
                evaluation_info=evaluation_info,
                primary_metric="accuracy_strict",
                subject_metric=None  # GSM8K doesn't have subject breakdown
            )
            
            # Analyze English results
            english_enhanced = create_enhanced_summary(
                model_results=final_results_english,
                evaluation_info=evaluation_info,
                primary_metric="accuracy_strict",
                subject_metric=None
            )
            
            # Combine analyses
            enhanced_summary = {
                "evaluation_info": evaluation_info,
                "korean_analysis": korean_enhanced,
                "english_analysis": english_enhanced,
                "original_summary": final_summary,
                "language_comparison": {
                    "korean_avg_score": korean_enhanced["performance_analysis"]["average_score"],
                    "english_avg_score": english_enhanced["performance_analysis"]["average_score"],
                    "korean_best_model": korean_enhanced["performance_analysis"]["best_model"],
                    "english_best_model": english_enhanced["performance_analysis"]["best_model"],
                    "performance_difference": english_enhanced["performance_analysis"]["average_score"] - korean_enhanced["performance_analysis"]["average_score"]
                }
            }
            
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)
                
            # Log key insights
            logger.info(f"🏆 Best Korean model: {korean_enhanced['performance_analysis']['best_model']} ({korean_enhanced['performance_analysis']['average_score']:.2f}%)")
            logger.info(f"🏆 Best English model: {english_enhanced['performance_analysis']['best_model']} ({english_enhanced['performance_analysis']['average_score']:.2f}%)")
            logger.info(f"📈 Language performance gap: {abs(enhanced_summary['language_comparison']['performance_difference']):.2f}%p")
            
        else:
            # Fallback to basic summary
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Summary saved to {final_json_path}")
        
        # Also create separate CSV files for Korean and English results
        korean_csv_path = os.path.join(base_output_dir, "gsm8k_results_korean.csv")
        with open(korean_csv_path, 'w', encoding='utf-8') as f:
            f.write("Model Name,Accuracy Standard (%),Accuracy Strict (%),Correct,Valid,Total\n")
            for result in final_results_korean:
                f.write(f"{result['model_name']},{result['accuracy_standard']:.2f},{result['accuracy_strict']:.2f},{result['correct_predictions']},{result['valid_predictions']},{result['total_questions']}\n")
        logger.info(f"Korean CSV summary saved to {korean_csv_path}")
        
        english_csv_path = os.path.join(base_output_dir, "gsm8k_results_english.csv")
        with open(english_csv_path, 'w', encoding='utf-8') as f:
            f.write("Model Name,Accuracy Standard (%),Accuracy Strict (%),Correct,Valid,Total\n")
            for result in final_results_english:
                f.write(f"{result['model_name']},{result['accuracy_standard']:.2f},{result['accuracy_strict']:.2f},{result['correct_predictions']},{result['valid_predictions']},{result['total_questions']}\n")
        logger.info(f"English CSV summary saved to {english_csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to save final summary: {e}")

def main():
    """Main execution function"""
    logger.info(f"Loading GSM8K data from: {DATASET_PATH}")
    gsm8k_data = load_gsm8k_data(DATASET_PATH)
    if gsm8k_data is None:
        logger.error("Could not load GSM8K data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    all_results = []

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====\n")
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_output_dir}")

        result = evaluate_single_model(config, gsm8k_data, model_output_dir)
        if result is not None:
            result["evaluation_date"] = str(Path().resolve()).split('/')[-1]  # Simple timestamp
            all_results.append(result)

        logger.info(f"\n===== Finished Evaluation for Model: {config.name} =====")
        print("-" * 80)

    # Create final summary
    logger.info("Creating final summary of all results...")
    create_final_summary(all_results, BASE_OUTPUT_DIR)
    
    logger.info("All GSM8K evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()