import os
import re
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score
import logging
from tqdm import tqdm
import gc
import sys
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_dp_evaluation.log")
    ]
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
DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dp_validation.json"
BASE_OUTPUT_DIR = "klue_dp_evaluation_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = None  # Set to None to evaluate all samples

# --- Helper Functions ---
def load_klue_dp_data(filepath):
    """Load KLUE DP evaluation data from JSON file."""
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

def create_dp_prompt(item):
    """Create prompt for dependency parsing task."""
    tokens = item.get("word_form", [])
    if not tokens:
        return None
    
    sentence = " ".join(tokens)
    prompt = f"""Perform dependency parsing on the following Korean sentence. For each token, provide its head index (0-based, where 0 means ROOT) and dependency relation.

Sentence: {sentence}

For each token, respond in the format: "token: head=X, relation=Y"
Answer:"""
    return prompt

def extract_dp_predictions(model_output, tokens):
    """Extract dependency parsing predictions from model output."""
    heads = [-1] * len(tokens)
    relations = [""] * len(tokens)
    
    # Remove prompt if it appears in the output
    lines = model_output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        # Parse format: "token: head=X, relation=Y"
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
            
        token = parts[0].strip().strip('"\'')
        info = parts[1].strip()
        
        # Find token index
        token_idx = -1
        for i, t in enumerate(tokens):
            if t == token:
                token_idx = i
                break
        
        if token_idx == -1:
            continue
            
        # Extract head
        head_match = re.search(r'head\s*=\s*(\d+)', info)
        if head_match:
            try:
                heads[token_idx] = int(head_match.group(1))
            except ValueError:
                pass
        
        # Extract relation
        rel_match = re.search(r'relation\s*=\s*([A-Z_]+)', info, re.IGNORECASE)
        if rel_match:
            relations[token_idx] = rel_match.group(1).upper()
    
    return heads, relations

def calculate_dp_metrics(pred_heads, pred_relations, gold_heads, gold_relations):
    """Calculate UAS and LAS for dependency parsing."""
    if len(pred_heads) != len(gold_heads) or len(pred_relations) != len(gold_relations):
        return 0, 0, 0
    
    total_tokens = len(gold_heads)
    uas_correct = 0
    las_correct = 0
    
    for i in range(total_tokens):
        if pred_heads[i] == gold_heads[i]:
            uas_correct += 1
            if pred_relations[i] == gold_relations[i]:
                las_correct += 1
    
    uas = uas_correct / total_tokens if total_tokens > 0 else 0
    las = las_correct / total_tokens if total_tokens > 0 else 0
    
    return uas, las, total_tokens

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, klue_dp_data: list, model_specific_output_dir: str):
    """
    Evaluate a single model on KLUE DP dataset and save results to model_specific_output_dir.
    """
    # Set up file paths
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

    logger.info(f"--- Starting Evaluation for Model: {config.name} ({config.model_id}) ---")
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
        total_uas = 0
        total_las = 0
        total_tokens = 0
        errors_or_skipped = 0
        results_details = []

        # Limit evaluation samples if specified
        eval_data = klue_dp_data[:MAX_EVAL_SAMPLES] if MAX_EVAL_SAMPLES else klue_dp_data
        
        logger.info(f"Starting inference loop on {len(eval_data)} samples...")
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {config.name}")):
            item_index = i
            tokens = item.get("word_form", [])
            gold_heads = item.get("head", [])
            gold_relations = item.get("deprel", [])
            
            # Default values
            generated_text_log = "SKIPPED"
            pred_heads_log = []
            pred_relations_log = []
            uas_log = 0
            las_log = 0
            tokens_count = 0

            if not tokens or not gold_heads or not gold_relations:
                logger.warning(f"Item {item_index}: Missing required fields. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Missing Fields"
            elif len(tokens) != len(gold_heads) or len(tokens) != len(gold_relations):
                logger.warning(f"Item {item_index}: Mismatched lengths. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Length Mismatch"
            else:
                prompt = create_dp_prompt(item)
                if prompt is None:
                    logger.warning(f"Item {item_index}: Failed to create prompt. Skipping.")
                    errors_or_skipped += 1
                    generated_text_log = "SKIPPED - Prompt Creation Failed"
                else:
                    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)

                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=150,  # Enough for dependency parsing output
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=False,
                            )
                        output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                        generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                        
                        pred_heads_log, pred_relations_log = extract_dp_predictions(generated_text_log, tokens)
                        uas_log, las_log, tokens_count = calculate_dp_metrics(pred_heads_log, pred_relations_log, gold_heads, gold_relations)
                        
                        total_uas += uas_log * tokens_count
                        total_las += las_log * tokens_count
                        total_tokens += tokens_count

                    except Exception as e:
                        logger.error(f"Item {item_index}: Inference error: {e}", exc_info=False)
                        errors_or_skipped += 1
                        generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

            # Record detailed results
            results_details.append({
                "index": item_index,
                "tokens": tokens,
                "gold_heads": gold_heads,
                "gold_relations": gold_relations,
                "pred_heads": pred_heads_log,
                "pred_relations": pred_relations_log,
                "uas": uas_log,
                "las": las_log,
                "raw_output": generated_text_log
            })
            
            raw_generations_list.append({
                "index": item_index,
                "sentence": " ".join(tokens) if tokens else "",
                "gold_heads": gold_heads,
                "gold_relations": gold_relations,
                "raw_output": generated_text_log,
                "pred_heads": pred_heads_log,
                "pred_relations": pred_relations_log
            })

            if (i + 1) % 50 == 0:
                current_uas = (total_uas / total_tokens * 100) if total_tokens > 0 else 0
                current_las = (total_las / total_tokens * 100) if total_tokens > 0 else 0
                logger.info(f"Progress ({config.name}): {i + 1}/{len(eval_data)}, UAS: {current_uas:.2f}%, LAS: {current_las:.2f}%, Errors/Skipped: {errors_or_skipped}")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        final_uas = (total_uas / total_tokens * 100) if total_tokens > 0 else 0
        final_las = (total_las / total_tokens * 100) if total_tokens > 0 else 0

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(eval_data)}")
        logger.info(f"Total Tokens Evaluated: {total_tokens}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Final UAS (Unlabeled Attachment Score): {final_uas:.2f}%")
        logger.info(f"Final LAS (Labeled Attachment Score): {final_las:.2f}%")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(eval_data),
            "total_tokens": total_tokens,
            "errors_or_skipped": errors_or_skipped,
            "uas": final_uas,
            "las": final_las,
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
    logger.info(f"Loading KLUE DP data from: {DATASET_PATH}")
    klue_dp_data = load_klue_dp_data(DATASET_PATH)
    if klue_dp_data is None:
        logger.error("Could not load KLUE DP data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====\n")
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # Call evaluation function
        evaluate_single_model(config, klue_dp_data, model_specific_output_dir)

        logger.info(f"\n===== Finished Evaluation for Model: {config.name} =====")
        print("-" * 80)

    logger.info("All evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()