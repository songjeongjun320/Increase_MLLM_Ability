import os
import re
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, field
import gc
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Model Configuration (from mmlu.py) ---
@dataclass
class ModelConfig:
    name: str                             # Unique name for this run (used for filenames)
    model_id: str                         # Hugging Face model identifier
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

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
]

# --- General Configuration ---
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_validation.json"
BASE_OUTPUT_DIR = "evaluation_results_klue_mrc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = 2000

# --- Helper Functions ---
def load_klue_mrc_data(filepath):
    """Load KLUE MRC data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Data is not in list format.")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Not all items in the list are dictionaries.")
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_mrc_prompt(item):
    """Create prompt for MRC task."""
    title = item.get("title", "")
    context = item.get("context", "")
    question = item.get("question", "")
    
    if not context or not question:
        return None
    
    prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
    return prompt

def extract_mrc_answer(generated_text, prompt):
    """Extract answer from model output."""
    # Remove prompt from generated text
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        answer = generated_text.strip()
    
    # Extract only the first line or sentence as answer
    lines = answer.split('\n')
    if lines:
        answer = lines[0].strip()
    
    # Remove common prefixes
    answer = re.sub(r'^(답:|Answer:|정답:|답변:)\s*', '', answer, flags=re.IGNORECASE)
    
    return answer.strip()

# MRC evaluation metrics
def compute_exact_match(prediction, ground_truth):
    """Calculate exact match score between prediction and ground truth."""
    prediction = prediction.strip().lower()
    if isinstance(ground_truth, list):
        ground_truths = [gt.strip().lower() for gt in ground_truth]
    else:
        ground_truths = [ground_truth.strip().lower()]
    return max(int(prediction == gt) for gt in ground_truths)

def compute_f1_score(prediction, ground_truth):
    """Calculate token-level F1 score between prediction and ground truth."""
    def get_tokens(s):
        return s.strip().lower().split()
    
    prediction_tokens = get_tokens(prediction)
    if isinstance(ground_truth, list):
        gt_list = ground_truth
    else:
        gt_list = [ground_truth]
    
    f1_scores = []
    
    for gt in gt_list:
        gt_tokens = get_tokens(gt)
        
        common_tokens = set(prediction_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        precision = len(common_tokens) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return max(f1_scores) if f1_scores else 0.0

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, mrc_data: list, model_specific_output_dir: str):
    """Evaluate a single model on KLUE MRC task."""
    # Setup file paths
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}.json")

    # Setup logging for this specific model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove existing file handlers
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler is not file_handler:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)

    logger.info(f"--- Starting KLUE MRC Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Output directory: {model_specific_output_dir}")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Logs will be saved to: {log_filepath}")
    logger.info(f"Raw generations will be saved to: {raw_gen_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # --- Load Model and Tokenizer ---
        logger.info(f"Loading tokenizer for {config.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=CACHE_DIR)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding a new pad token '[PAD]'.")
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

        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
            logger.warning("Resizing model embeddings after load due to added PAD token.")
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # --- Run Evaluation ---
        exact_match_scores = []
        f1_scores = []
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []

        # Limit evaluation samples
        eval_data = mrc_data[:MAX_EVAL_SAMPLES]
        logger.info(f"Evaluating on {len(eval_data)} samples (limited from {len(mrc_data)} total)")

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {config.name}")):
            item_index_for_log = item.get("index", i)
            prompt = create_mrc_prompt(item)
            
            # Get ground truth answers
            answers = item.get("answers", {})
            ground_truth_answers = answers.get("text", []) if isinstance(answers.get("text"), list) else [answers.get("text", "")]
            
            # Default values
            generated_text_log = "SKIPPED"
            predicted_answer_log = None
            em_score = 0
            f1_score = 0

            if not prompt:
                logger.warning(f"Item {item_index_for_log}: Failed to create prompt. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Prompt Creation Failed"
            elif not ground_truth_answers or not ground_truth_answers[0]:
                logger.warning(f"Item {item_index_for_log}: No ground truth answers. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - No Ground Truth"
            else:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024).to(DEVICE)

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,  # Limit answer length
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            temperature=0.1
                        )
                    
                    output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                    predicted_answer_log = extract_mrc_answer(generated_text_log, prompt)

                    if predicted_answer_log:
                        total_predictions += 1
                        em_score = compute_exact_match(predicted_answer_log, ground_truth_answers)
                        f1_score = compute_f1_score(predicted_answer_log, ground_truth_answers)
                        exact_match_scores.append(em_score)
                        f1_scores.append(f1_score)
                    else:
                        logger.warning(f"Item {item_index_for_log}: Failed to extract answer from output")
                        errors_or_skipped += 1
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                except Exception as e:
                    logger.error(f"Item {item_index_for_log}: Inference error: {e}", exc_info=False)
                    errors_or_skipped += 1
                    generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

            # Record results
            results_details.append({
                "index": item_index_for_log,
                "title": item.get("title", ""),
                "question": item.get("question", ""),
                "ground_truth": ground_truth_answers,
                "model_raw_output": generated_text_log,
                "predicted_answer": predicted_answer_log,
                "exact_match": em_score,
                "f1_score": f1_score
            })
            
            raw_generations_list.append({
                "index": item_index_for_log,
                "title": item.get("title", ""),
                "context": item.get("context", "")[:200] + "..." if len(item.get("context", "")) > 200 else item.get("context", ""),
                "question": item.get("question", ""),
                "ground_truth": ground_truth_answers,
                "raw_output": generated_text_log,
                "extracted_answer": predicted_answer_log,
                "exact_match": em_score,
                "f1_score": f1_score
            })

            if (i + 1) % 50 == 0:
                current_em = (sum(exact_match_scores) / len(exact_match_scores) * 100) if exact_match_scores else 0
                current_f1 = (sum(f1_scores) / len(f1_scores) * 100) if f1_scores else 0
                logger.info(f"Progress ({config.name}): {i + 1}/{len(eval_data)}, EM: {current_em:.2f}%, F1: {current_f1:.2f}%, Errors/Skipped: {errors_or_skipped}")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        avg_em = (sum(exact_match_scores) / len(exact_match_scores) * 100) if exact_match_scores else 0
        avg_f1 = (sum(f1_scores) / len(f1_scores) * 100) if f1_scores else 0

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(eval_data)}")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Exact Match Score: {avg_em:.2f}%")
        logger.info(f"F1 Score: {avg_f1:.2f}%")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": JSON_VAL_DATASET_PATH,
            "total_items": len(eval_data),
            "valid_predictions": total_predictions,
            "errors_or_skipped": errors_or_skipped,
            "exact_match": avg_em,
            "f1_score": avg_f1,
            "details": results_details
        }
        
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

        # --- Save Raw Generations ---
        logger.info(f"Saving raw model generations to {raw_gen_filepath}...")
        try:
            with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw generations saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save raw generations file {raw_gen_filepath}: {e}")

    except Exception as e:
        logger.exception(f"A critical error occurred during evaluation for {config.name}: {e}")

    finally:
        # --- Clean up resources ---
        logger.info(f"Cleaning up resources for {config.name}...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Resources cleaned up for {config.name}.")
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            try:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            except Exception as e:
                logger.debug(f"Error closing/removing file handler: {e}")

# --- Main Execution Logic ---
def main():
    logger.info(f"Loading KLUE MRC data from: {JSON_VAL_DATASET_PATH}")
    mrc_data = load_klue_mrc_data(JSON_VAL_DATASET_PATH)
    if mrc_data is None:
        logger.error("Could not load KLUE MRC data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting KLUE MRC Evaluation for Model: {config.name} =====\n")
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # Run evaluation
        evaluate_single_model(config, mrc_data, model_specific_output_dir)

        logger.info(f"\n===== Finished KLUE MRC Evaluation for Model: {config.name} =====")
        print("-" * 80)

    logger.info("All KLUE MRC evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()