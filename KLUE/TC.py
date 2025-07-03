import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sklearn.metrics import precision_recall_f1_support, accuracy_score
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
import gc
import sys
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_tc_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE TC Label Definitions
TC_LABELS = [
    "IT/과학",  # 0
    "경제",     # 1
    "사회",     # 2
    "생활/문화", # 3
    "세계",     # 4
    "스포츠",    # 5
    "정치"      # 6
]
NUM_LABELS = len(TC_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(TC_LABELS)}
ID2LABEL = {idx: label for label, idx in enumerate(TC_LABELS)}
logger.info(f"Total number of KLUE-TC labels: {NUM_LABELS}")

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
DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_tc_validation.json"
BASE_OUTPUT_DIR = "evaluation_results_klue_tc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = 1000

# --- Helper Functions ---
def load_klue_tc_data(filepath):
    """Load KLUE TC data from JSON file."""
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

def create_tc_prompt(item):
    """Create truly zero-shot prompt for TC (Text Classification) task without providing categories."""
    title = item.get("title", "")
    
    if not title:
        return None
    
    prompt = f"""What topic category is this Korean news headline about?

News headline: {title}

Answer:"""
    return prompt

def extract_tc_answer(model_output, prompt):
    """Extract TC category from model output."""
    # Remove prompt from output if present
    if prompt.strip() in model_output:
        prediction_text = model_output.replace(prompt.strip(), "").strip()
    else:
        prediction_text = model_output.strip()
    
    # Clean the prediction text
    cleaned_text = prediction_text.strip()
    
    # Remove common prefixes
    prefixes = ["카테고리:", "Category:", "답:", "Answer:", "분류:"]
    for prefix in prefixes:
        if cleaned_text.startswith(prefix):
            cleaned_text = cleaned_text[len(prefix):].strip()
    
    # Look for exact matches first
    for label in TC_LABELS:
        if label in cleaned_text[:20]:  # Check first 20 characters
            return LABEL2ID[label]
    
    # Look for partial matches
    partial_matches = {
        "IT": LABEL2ID["IT/과학"],
        "과학": LABEL2ID["IT/과학"],
        "경제": LABEL2ID["경제"],
        "사회": LABEL2ID["사회"],
        "생활": LABEL2ID["생활/문화"],
        "문화": LABEL2ID["생활/문화"],
        "세계": LABEL2ID["세계"],
        "스포츠": LABEL2ID["스포츠"],
        "정치": LABEL2ID["정치"]
    }
    
    for keyword, label_id in partial_matches.items():
        if keyword in cleaned_text[:20]:
            return label_id
    
    return None

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, tc_data: list, model_specific_output_dir: str):
    """
    Evaluate a single model on KLUE TC using zero-shot approach and save results to model_specific_output_dir.
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

    logger.info(f"--- Starting Zero-Shot TC Evaluation for Model: {config.name} ({config.model_id}) ---")
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
        eval_data = tc_data[:MAX_EVAL_SAMPLES] if MAX_EVAL_SAMPLES else tc_data
        logger.info(f"Evaluating on {len(eval_data)} samples")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []

        true_labels = []
        pred_labels = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {config.name}")):
            item_index_for_log = item.get("guid", i)
            ground_truth = item.get("label", None)
            
            # Default values
            generated_text_log = "SKIPPED"
            model_answer_log = None
            is_correct_log = False

            if ground_truth is None or not isinstance(ground_truth, int) or ground_truth not in range(NUM_LABELS):
                logger.warning(f"Item {item_index_for_log}: Invalid/missing ground truth (label: {ground_truth}). Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Invalid Ground Truth"
            else:
                title = item.get("title", "")
                
                if not title:
                    logger.warning(f"Item {item_index_for_log}: Missing title. Skipping.")
                    errors_or_skipped += 1
                    generated_text_log = "SKIPPED - Missing Title"
                else:
                    prompt = create_tc_prompt(item)
                    if not prompt:
                        logger.warning(f"Item {item_index_for_log}: Failed to create prompt. Skipping.")
                        errors_or_skipped += 1
                        generated_text_log = "SKIPPED - Prompt Creation Failed"
                    else:
                        try:
                            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)

                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=30,  # Enough for category name
                                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    do_sample=False,
                                    temperature=0.1
                                )
                            
                            output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                            generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                            model_answer_log = extract_tc_answer(generated_text_log, prompt)

                            if model_answer_log is not None:
                                total_predictions += 1
                                true_labels.append(ground_truth)
                                pred_labels.append(model_answer_log)
                                
                                if model_answer_log == ground_truth:
                                    correct_predictions += 1
                                    is_correct_log = True
                            else:
                                logger.warning(f"Item {item_index_for_log}: Failed to extract category from output: '{generated_text_log[:50]}...'")
                                errors_or_skipped += 1
                                generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                        except Exception as e:
                            logger.error(f"Item {item_index_for_log}: Inference error: {e}", exc_info=False)
                            errors_or_skipped += 1
                            generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

            # Record detailed results for all cases
            results_details.append({
                "index": item_index_for_log,
                "title": item.get("title", ""),
                "ground_truth": ground_truth,
                "ground_truth_label": ID2LABEL.get(ground_truth, "UNKNOWN") if ground_truth is not None else "UNKNOWN",
                "model_raw_output": generated_text_log,
                "predicted_answer": model_answer_log,
                "predicted_label": ID2LABEL.get(model_answer_log, "UNKNOWN") if model_answer_log is not None else "UNKNOWN",
                "is_correct": is_correct_log
            })
            
            raw_generations_list.append({
                "index": item_index_for_log,
                "title": item.get("title", ""),
                "ground_truth": ground_truth,
                "ground_truth_label": ID2LABEL.get(ground_truth, "UNKNOWN") if ground_truth is not None else "UNKNOWN",
                "raw_output": generated_text_log,
                "extracted_answer": model_answer_log,
                "extracted_label": ID2LABEL.get(model_answer_log, "UNKNOWN") if model_answer_log is not None else "UNKNOWN"
            })

            if (i + 1) % 50 == 0:
                current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                logger.info(f"Progress ({config.name}): {i + 1}/{len(eval_data)}, Acc: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors/Skipped: {errors_or_skipped}")

        # --- Calculate Final Metrics ---
        logger.info(f"Inference loop finished for {config.name}.")
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # Calculate additional metrics if we have valid predictions
        precision = recall = f1 = 0.0
        per_class_metrics = {}
        
        if len(true_labels) > 0 and len(pred_labels) > 0:
            precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="macro", zero_division=0)
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_f1_support(
                true_labels, pred_labels, average=None, zero_division=0
            )
            per_class_metrics = {
                ID2LABEL[i]: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class))
                if i < len(ID2LABEL)
            }

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(eval_data)}")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Final Accuracy: {accuracy:.2f}%")
        logger.info(f"Macro Precision: {precision:.4f}")
        logger.info(f"Macro Recall: {recall:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        logger.info("Per-class metrics:")
        logger.info(json.dumps(per_class_metrics, indent=2))

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(eval_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors_or_skipped,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_metrics": per_class_metrics,
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
    logger.info(f"Loading KLUE TC data from: {DATASET_PATH}")
    tc_data = load_klue_tc_data(DATASET_PATH)
    if tc_data is None:
        logger.error("Could not load KLUE TC data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Zero-Shot TC Evaluation for Model: {config.name} =====\n")
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # Call evaluation function
        evaluate_single_model(config, tc_data, model_specific_output_dir)

        logger.info(f"\n===== Finished Zero-Shot TC Evaluation for Model: {config.name} =====")
        print("-" * 80)

    logger.info("All KLUE TC zero-shot evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()