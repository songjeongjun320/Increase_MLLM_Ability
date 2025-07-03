import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
import gc
import sys
import re

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
DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_validation.json"
BASE_OUTPUT_DIR = "evaluation_results_klue_sts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = 1000

# --- Helper Functions ---
def load_klue_sts_data(filepath):
    """Load KLUE STS data from JSON file."""
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
        logger.error(f"Data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def extract_score_from_prompt(item):
    """Extract the similarity score from the prompt format."""
    input_text = item.get("input", "")
    if not input_text:
        return None, None, None
    
    # Extract sentences from the input prompt
    pattern1 = r"Sentence 1:\s*(.+?)\s*Sentence 2:\s*(.+?)$"
    pattern2 = r"Sentence 1:\s*(.+?)\s*Sentence 2:\s*(.+?)\s*Score:"
    
    match = re.search(pattern1, input_text, re.DOTALL) or re.search(pattern2, input_text, re.DOTALL)
    if match:
        sentence1 = match.group(1).strip()
        sentence2 = match.group(2).strip()
        return sentence1, sentence2, input_text
    else:
        return None, None, input_text

def create_sts_prompt(item):
    """Create truly zero-shot prompt for STS task without providing scoring criteria."""
    sentence1, sentence2, original_prompt = extract_score_from_prompt(item)
    
    if not sentence1 or not sentence2:
        return None
    
    prompt = f"""How similar are these two sentences?

Sentence 1: {sentence1}
Sentence 2: {sentence2}

Answer:"""
    return prompt

def extract_sts_score(model_output, prompt):
    """Extract similarity score from model output."""
    # Remove prompt from output if present
    if prompt.strip() in model_output:
        prediction_text = model_output.replace(prompt.strip(), "").strip()
    else:
        prediction_text = model_output.strip()
    
    # Look for numeric score in the first line
    first_line = prediction_text.split('\n')[0].strip()
    
    # Try to find a number between 0 and 5
    number_pattern = r'([0-5](?:\.[0-9]+)?)'
    match = re.search(number_pattern, first_line)
    
    if match:
        try:
            score = float(match.group(1))
            # Clamp to valid range
            return max(0.0, min(5.0, score))
        except ValueError:
            pass
    
    return None

def calculate_sts_metrics(true_scores, pred_scores):
    """Calculate STS metrics."""
    # Filter out None predictions
    valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores) if p is not None and t is not None]
    
    if not valid_pairs:
        return {
            "num_valid_samples": 0,
            "pearson_r": None,
            "rmse": None,
            "mae": None
        }
    
    true_clean, pred_clean = zip(*valid_pairs)
    num_valid_samples = len(true_clean)
    
    # Calculate Pearson correlation
    pearson_r = None
    if len(set(true_clean)) > 1 and len(set(pred_clean)) > 1:
        try:
            pearson_r, _ = pearsonr(true_clean, pred_clean)
            if np.isnan(pearson_r):
                pearson_r = None
        except Exception:
            pearson_r = None
    
    # Calculate RMSE and MAE
    try:
        mse = mean_squared_error(true_clean, pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_clean, pred_clean)
    except Exception:
        rmse = mae = None
    
    return {
        "num_valid_samples": num_valid_samples,
        "pearson_r": float(pearson_r) if pearson_r is not None else None,
        "rmse": float(rmse) if rmse is not None else None,
        "mae": float(mae) if mae is not None else None
    }

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, sts_data: list, model_specific_output_dir: str):
    """
    Evaluate a single model on KLUE STS using zero-shot approach and save results to model_specific_output_dir.
    """
    # Result and log file paths
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}.json")

    # Setup logging for this specific model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to prevent duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler is not file_handler:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)

    logger.info(f"--- Starting Zero-Shot STS Evaluation for Model: {config.name} ({config.model_id}) ---")
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

        # --- Prepare evaluation data ---
        eval_data = sts_data[:MAX_EVAL_SAMPLES] if MAX_EVAL_SAMPLES else sts_data
        logger.info(f"Evaluating on {len(eval_data)} samples")

        # --- Run Evaluation ---
        valid_predictions = 0
        errors_or_skipped = 0
        results_details = []

        true_scores = []
        pred_scores = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {config.name}")):
            ground_truth = item.get("output", None)
            
            # Default values
            generated_text_log = "SKIPPED"
            predicted_score = None

            if ground_truth is None or not isinstance(ground_truth, (int, float)):
                logger.warning(f"Item {i}: Invalid/missing ground truth score: {ground_truth}. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Invalid Ground Truth"
            else:
                prompt = create_sts_prompt(item)
                if not prompt:
                    logger.warning(f"Item {i}: Failed to create prompt. Skipping.")
                    errors_or_skipped += 1
                    generated_text_log = "SKIPPED - Prompt Creation Failed"
                else:
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=20,  # Short answer needed for score
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=False,
                                temperature=0.1
                            )
                        
                        output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                        generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                        predicted_score = extract_sts_score(generated_text_log, prompt)

                        if predicted_score is not None:
                            valid_predictions += 1
                            true_scores.append(ground_truth)
                            pred_scores.append(predicted_score)
                        else:
                            logger.warning(f"Item {i}: Failed to extract score from output: '{generated_text_log[:50]}...'")
                            errors_or_skipped += 1
                            generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                    except Exception as e:
                        logger.error(f"Item {i}: Inference error: {e}", exc_info=False)
                        errors_or_skipped += 1
                        generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

            # Record detailed results
            results_details.append({
                "index": i,
                "ground_truth": ground_truth,
                "predicted_score": predicted_score,
                "raw_output": generated_text_log
            })
            
            raw_generations_list.append({
                "index": i,
                "ground_truth": ground_truth,
                "raw_output": generated_text_log,
                "extracted_score": predicted_score
            })

            if (i + 1) % 50 == 0:
                logger.info(f"Progress ({config.name}): {i + 1}/{len(eval_data)}, Valid: {valid_predictions}, Errors/Skipped: {errors_or_skipped}")

        # --- Calculate Final Metrics ---
        logger.info(f"Inference loop finished for {config.name}.")
        
        # Calculate metrics
        metrics = calculate_sts_metrics(true_scores, pred_scores)

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items: {len(eval_data)}")
        logger.info(f"Valid Predictions: {valid_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Pearson Correlation: {metrics['pearson_r']:.4f}" if metrics['pearson_r'] is not None else "Pearson Correlation: N/A")
        logger.info(f"RMSE: {metrics['rmse']:.4f}" if metrics['rmse'] is not None else "RMSE: N/A")
        logger.info(f"MAE: {metrics['mae']:.4f}" if metrics['mae'] is not None else "MAE: N/A")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(eval_data),
            "valid_predictions": valid_predictions,
            "errors_or_skipped": errors_or_skipped,
            "pearson_r": metrics['pearson_r'],
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "num_valid_samples": metrics['num_valid_samples'],
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
    logger.info(f"Loading KLUE STS data from: {DATASET_PATH}")
    sts_data = load_klue_sts_data(DATASET_PATH)
    if sts_data is None:
        logger.error("Could not load KLUE STS data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Zero-Shot STS Evaluation for Model: {config.name} =====\n")
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # Call evaluation function
        evaluate_single_model(config, sts_data, model_specific_output_dir)

        logger.info(f"\n===== Finished Zero-Shot STS Evaluation for Model: {config.name} =====")
        print("-" * 80)

    logger.info("All KLUE STS zero-shot evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()
