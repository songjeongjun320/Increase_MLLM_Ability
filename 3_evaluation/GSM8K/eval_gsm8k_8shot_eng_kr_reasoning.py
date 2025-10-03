#!/usr/bin/env python3
"""
module load cuda-12.6.1-gcc-12.1.0

GSM8K Evaluation Script - English Input with Korean Reasoning
- Evaluates mathematical reasoning capability using English questions with Korean reasoning
- Uses English questions from 'original' field with Korean CoT reasoning
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

# Import ToW token checker
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from check_tokenizer import check_tow_tokens_for_eval

torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

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
BASE_OUTPUT_DIR = "./eng_input_kr_reasoning"  # Output directory

# Batch Processing Configuration
BATCH_SIZE = 16  # A100 optimized batch size

# Import performance analyzer
try:
    import sys
    sys.path.append('../')
    from performance_analyzer import create_enhanced_summary
except ImportError:
    logger.warning("Performance analyzer not available. Using basic summary.")
    create_enhanced_summary = None

# GSM8K 8-shot examples
# English questions for few-shot examples
GSM8K_8SHOT_ENG_EXAMPLES = [
    {
        "question": "If you buy 3 pencils for $2 each and 2 notebooks for $5 each, how much do you spend in total?",
        "answer": "16"
    },
    {
        "question": "A train travels 60 miles per hour. How far will it go in 3 hours?",
        "answer": "180"
    },
    {
        "question": "Sarah has 24 apples. She shares them equally among 6 friends. How many apples does each friend get?",
        "answer": "4"
    },
    {
        "question": "Tom had $50. He spent $18 on lunch and $12 on a book. How much money does he have left?",
        "answer": "20"
    },
    {
        "question": "A box holds 8 chocolates. How many chocolates are there in 7 boxes?",
        "answer": "56"
    },
    {
        "question": "A farmer has 36 cows. If he sells 9 of them, how many are left?",
        "answer": "27"
    },
    {
        "question": "Lisa reads 12 pages of a book each day. How many pages does she read in 5 days?",
        "answer": "60"
    },
    {
        "question": "A pizza has 8 slices. If 3 pizzas are ordered, how many slices are there in total?",
        "answer": "24"
    }
]

# Korean reasoning (cot_content) for few-shot examples
GSM8K_8SHOT_KOR_COT_EXAMPLES = [
    {
        "cot_content": "연필 값은 3 × 2달러 = 6달러. 공책 값은 2 × 5달러 = 10달러. 합하면 6 + 10 = 16달러.",
        "answer": "16"
    },
    {
        "cot_content": "기차는 1시간에 60마일을 간다. 3시간 동안은 60 × 3 = 180마일.",
        "answer": "180"
    },
    {
        "cot_content": "24 ÷ 6 = 4개씩 받는다.",
        "answer": "4"
    },
    {
        "cot_content": "쓴 돈은 18 + 12 = 30달러. 남은 돈은 50 - 30 = 20달러.",
        "answer": "20"
    },
    {
        "cot_content": "상자당 8개이므로 7 × 8 = 56개.",
        "answer": "56"
    },
    {
        "cot_content": "36 - 9 = 27마리.",
        "answer": "27"
    },
    {
        "cot_content": "12 × 5 = 60쪽.",
        "answer": "60"
    },
    {
        "cot_content": "피자 한 판에 8조각이므로 3 × 8 = 24조각.",
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
    ModelConfig(
        name="llama-3.2-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
        use_quantization=False
    ),
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

def create_gsm8k_prompt(text):
    """
    Creates 8-shot GSM8K prompt with English questions and Korean reasoning.
    - Uses GSM8K_8SHOT_ENG_EXAMPLES for English questions
    - Uses GSM8K_8SHOT_KOR_COT_EXAMPLES for Korean reasoning
    - Uses Korean trigger: "단계별로 생각해봅시다."
    """
    prompt_parts = []

    # 1. Create 8 few-shot examples combining English questions with Korean reasoning
    for eng_example, kor_example in zip(GSM8K_8SHOT_ENG_EXAMPLES, GSM8K_8SHOT_KOR_COT_EXAMPLES):
        question = eng_example["question"]
        cot_content = kor_example["cot_content"]
        answer = kor_example["answer"]

        # Use English question format with Korean reasoning
        full_answer_block = f"Response: {cot_content} #### So the answer is {{{answer}}}. #### {{{answer}}}"
        example_block = f"Question: {question}\n {full_answer_block}"

        prompt_parts.append(example_block)

    # 2. Combine all examples with double line breaks
    final_examples_str = "\n\n".join(prompt_parts)

    # 3. Add the actual test question with Korean reasoning trigger
    final_prompt = f"""{final_examples_str}

Question: {text}
Response: 단계별로 생각해봅시다."""

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
        # 1) { 18 }   (최우선 - few-shot 예제와 일치)
        r'\{([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\}',

        # 2) #### 18  (둘째 우선)
        r'####\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',

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

def create_data_batches(data, batch_size):
    """
    Split data into batches for processing
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def process_batch_inference(model, tokenizer, prompts_batch, max_retries=3):
    """
    Process a batch of prompts with optimized inference
    Returns list of (generated_text, extracted_answer) tuples
    """
    batch_size = len(prompts_batch)
    results = []

    for attempt in range(max_retries):
        try:
            # Tokenize all prompts in batch
            inputs = tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                )

            # Process each output in the batch
            L_in = inputs["input_ids"].shape[1]
            gen_only = outputs[:, L_in:]  # [B, L_gen] 생성 토큰만

            # 한 번에 디코딩
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for i, generated_text in enumerate(texts):
                try:
                    generated_text = generated_text.strip()
                    extracted_answer = extract_numerical_answer(generated_text)
                    results.append((generated_text, extracted_answer))
                except Exception as e:
                    logger.warning(f"Error processing batch item {i}: {e}")
                    results.append((f"ERROR: {str(e)}", None))


            return results

        except Exception as e:
            logger.error(f"Batch inference attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.2 + random.random() * 0.2)
                # Clear results for retry
                results = []
                continue
            else:
                # Return error results for all items in batch
                error_msg = f"BATCH_ERROR after {max_retries} attempts: {str(e)}"
                return [(error_msg, None)] * batch_size

    return results

def evaluate_single_model(config: ModelConfig, gsm8k_data: list, model_output_dir: str):
    """
    Evaluate single model on GSM8K dataset with English input and Korean reasoning
    """
    os.makedirs(model_output_dir, exist_ok=True)
    results_filepath = os.path.join(model_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_output_dir, f"raw_generations_{config.name}.json")

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
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # Load Model and Tokenizer
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding new pad token '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # === TOKENIZER VERIFICATION ===
        tokenizer_status = check_tow_tokens_for_eval(
            tokenizer=tokenizer,
            model_path=tokenizer_load_path,
            model_name=config.name,
            logger=logger
        )

        if not tokenizer_status.is_valid:
            logger.warning(f"⚠️ ToW tokens not properly configured for {config.name}")
            for issue in tokenizer_status.issues:
                logger.warning(f"   - {issue}")
        # ===============================

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
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        if config.adapter_path:
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")

            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")

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
        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []

        logger.info("Starting GSM8K inference loop...")
        logger.info(f"Dataset size: {len(gsm8k_data)}")

        # Create data batches for processing
        data_batches = create_data_batches(gsm8k_data, BATCH_SIZE)
        logger.info(f"Processing {len(gsm8k_data)} items in {len(data_batches)} batches of size {BATCH_SIZE}")

        # Process each batch
        for batch_idx, batch_items in enumerate(tqdm(data_batches, desc=f"Evaluating {config.name} (GSM8K)")):
            logger.info(f"Processing batch {batch_idx + 1}/{len(data_batches)} with {len(batch_items)} items")

            # Prepare prompts for the batch using 'original' field (English questions)
            prompts = []
            batch_indices = []
            questions = []
            ground_truths = []

            # Prepare batch data
            for local_idx, item in enumerate(batch_items):
                global_idx = batch_idx * BATCH_SIZE + local_idx
                ground_truth = item.get("answer", None)
                if ground_truth is None:
                    logger.warning(f"Item with no ground truth found at index {global_idx}. Skipping.")
                    errors_or_skipped += 1
                    continue

                # Use 'original' field for English question
                question = item.get("original", "")

                batch_indices.append(global_idx)
                ground_truths.append(ground_truth)

                # Create prompt with English question and Korean reasoning
                if question:
                    prompt = create_gsm8k_prompt(question)
                    prompts.append(prompt)
                    questions.append(question)
                else:
                    prompts.append(None)
                    questions.append(None)

            # Process batch
            valid_prompts = [p for p in prompts if p is not None]
            if valid_prompts:
                try:
                    batch_results = process_batch_inference(model, tokenizer, valid_prompts)
                    result_idx = 0

                    for i, (idx, gt, question) in enumerate(zip(batch_indices, ground_truths, questions)):
                        if prompts[i] is not None:
                            gen_text, answer = batch_results[result_idx]
                            result_idx += 1

                            is_correct = False
                            exception_info = None

                            if answer is not None:
                                total_predictions += 1
                                if check_numerical_match(answer, gt):
                                    correct_predictions += 1
                                    is_correct = True
                            else:
                                errors_or_skipped += 1
                                if gen_text and not gen_text.startswith("ERROR"):
                                    gen_text = f"EXTRACTION_FAILED: {gen_text}"

                            # Store results
                            results_details.append({
                                "index": idx,
                                "question": question,
                                "ground_truth": gt,
                                "model_raw_output": gen_text,
                                "extracted_answer": answer,
                                "is_correct": is_correct,
                                "exception": exception_info,
                                "prompt_snapshot": prompts[i]
                            })

                            raw_generations_list.append({
                                "index": idx,
                                "language": "English (Korean reasoning)",
                                "question": question,
                                "ground_truth": gt,
                                "prompt": prompts[i],
                                "raw_output": gen_text,
                                "extracted_answer": answer,
                                "is_correct": is_correct,
                                "exception": exception_info
                            })

                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    # Handle batch error
                    for i, (idx, gt, question) in enumerate(zip(batch_indices, ground_truths, questions)):
                        if prompts[i] is not None:
                            errors_or_skipped += 1
                            exception_info = f"BATCH_ERROR: {e}\n{traceback.format_exc()}"

                            results_details.append({
                                "index": idx,
                                "question": question,
                                "ground_truth": gt,
                                "model_raw_output": f"BATCH_ERROR: {exception_info}",
                                "extracted_answer": None,
                                "is_correct": False,
                                "exception": exception_info,
                                "prompt_snapshot": prompts[i]
                            })

                            raw_generations_list.append({
                                "index": idx,
                                "language": "English (Korean reasoning)",
                                "question": question,
                                "ground_truth": gt,
                                "prompt": prompts[i],
                                "raw_output": f"BATCH_ERROR: {exception_info}",
                                "extracted_answer": None,
                                "is_correct": False,
                                "exception": exception_info
                            })

            # Periodic memory cleanup during batch processing
            if (batch_idx + 1) % 5 == 0:  # Every 5 batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Memory cleanup performed after batch {batch_idx + 1}")


        # Final Results
        logger.info(f"Inference loop finished for {config.name}.")

        # Calculate accuracies
        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        accuracy_strict = (correct_predictions / len(gsm8k_data) * 100) if len(gsm8k_data) > 0 else 0

        logger.info(f"--- GSM8K Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Questions: {len(gsm8k_data)}")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_questions): {accuracy_strict:.2f}%")

        # Save Results
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "evaluation_type": "GSM8K (English input with Korean reasoning, 8-shot)",
            "total_questions": len(gsm8k_data),
            "results": {
                "valid_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "errors_or_skipped": errors_or_skipped,
                "accuracy_standard": accuracy_standard,
                "accuracy_strict": accuracy_strict,
                "details": results_details
            }
        }

        # Save results
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

        # Save Raw Generations
        try:
            with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw generations saved to {raw_gen_filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw generations file {raw_gen_filepath}: {e}")

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
    final_results = []

    for result in all_results:
        if result is not None:
            summary = {
                "model_name": result["model_config"]["name"],
                "model_id": result["model_config"]["model_id"],
                "adapter_path": result["model_config"].get("adapter_path", None),
                "total_questions": result["total_questions"],
                "correct_predictions": result["results"]["correct_predictions"],
                "valid_predictions": result["results"]["valid_predictions"],
                "errors_or_skipped": result["results"]["errors_or_skipped"],
                "accuracy_standard": result["results"]["accuracy_standard"],
                "accuracy_strict": result["results"]["accuracy_strict"],
                "evaluation_date": result.get("evaluation_date", "N/A")
            }
            final_results.append(summary)

    # Sort by accuracy (strict) descending
    final_results.sort(key=lambda x: x["accuracy_strict"], reverse=True)

    final_summary = {
        "evaluation_type": "GSM8K (English input with Korean reasoning, 8-shot)",
        "dataset_info": {
            "name": "GSM8K-test (English input, Korean reasoning)",
            "path": DATASET_PATH,
            "total_questions": final_results[0]["total_questions"] if final_results else 0
        },
        "evaluation_summary": {
            "models_evaluated": len(final_results),
            "best_model": final_results[0]["model_name"] if final_results else "N/A",
            "best_accuracy": final_results[0]["accuracy_strict"] if final_results else 0.0
        },
        "results": final_results
    }

    final_json_path = os.path.join(base_output_dir, "final_gsm8k_results.json")
    try:
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {final_json_path}")

        # Create CSV file
        csv_path = os.path.join(base_output_dir, "gsm8k_results.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Model Name,Accuracy Standard (%),Accuracy Strict (%),Correct,Valid,Total\n")
            for result in final_results:
                f.write(f"{result['model_name']},{result['accuracy_standard']:.2f},{result['accuracy_strict']:.2f},{result['correct_predictions']},{result['valid_predictions']},{result['total_questions']}\n")
        logger.info(f"CSV summary saved to {csv_path}")

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
