import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sklearn.metrics import precision_recall_f1_support, classification_report
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
        logging.FileHandler("klue_ner_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE NER Label Definitions
NER_TAGS = [
    "B-LC",     # 0 - Location (Beginning)
    "I-LC",     # 1 - Location (Inside)
    "B-DT",     # 2 - Date/Time (Beginning)
    "I-DT",     # 3 - Date/Time (Inside)
    "B-OG",     # 4 - Organization (Beginning)
    "I-OG",     # 5 - Organization (Inside)
    "B-PS",     # 6 - Person (Beginning)
    "I-PS",     # 7 - Person (Inside)
    "B-QT",     # 8 - Quantity (Beginning)
    "I-QT",     # 9 - Quantity (Inside)
    "B-TI",     # 10 - Time (Beginning)
    "I-TI",     # 11 - Time (Inside)
    "O"         # 12 - Outside
]
NUM_LABELS = len(NER_TAGS)
LABEL2ID = {label: idx for idx, label in enumerate(NER_TAGS)}
ID2LABEL = {idx: label for idx, label in enumerate(NER_TAGS)}
logger.info(f"Total number of KLUE-NER labels: {NUM_LABELS}")

# --- Model Configuration (from mmlu.py) ---
@dataclass
class ModelConfig:
    name: str                             # Unique name for this run
    model_id: str                         # Hugging Face model identifier or local path
    use_quantization: bool = True         # Default to quantization
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
DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_ner_validation.json"
BASE_OUTPUT_DIR = "klue_ner_evaluation_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = 500

# Helper functions for zero-shot NER evaluation
def create_ner_prompt(item):
    """Create zero-shot prompt for NER task."""
    tokens = item.get("tokens", [])
    if not tokens:
        return None
    
    sentence = "".join(tokens).strip()
    
    prompt = f"""Identify named entities in the following Korean sentence and label them with their entity types.

Entity types:
- PS: Person
- LC: Location  
- OG: Organization
- DT: Date/Time
- QT: Quantity
- TI: Time

Sentence: {sentence}

For each entity found, respond in the format: "entity_text: entity_type"
If no entities are found, respond with "No entities found"

Entities:"""
    
    return prompt

def extract_ner_predictions(model_output, tokens, prompt):
    """Extract NER predictions from model output."""
    # Initialize all tokens as 'O' (outside)
    predictions = ['O'] * len(tokens)
    
    # Remove prompt from output if present
    if prompt.strip() in model_output:
        prediction_text = model_output.replace(prompt.strip(), "").strip()
    else:
        prediction_text = model_output.strip()
    
    # Parse the output for entities
    lines = prediction_text.split('\n')
    found_entities = []
    
    for line in lines:
        line = line.strip()
        if ':' in line and line.lower() != "no entities found":
            parts = line.split(':', 1)
            if len(parts) == 2:
                entity_text = parts[0].strip().strip('"\'')
                entity_type = parts[1].strip().upper()
                
                # Map entity types
                type_mapping = {
                    'PS': 'PS', 'PERSON': 'PS',
                    'LC': 'LC', 'LOCATION': 'LC', 
                    'OG': 'OG', 'ORGANIZATION': 'OG',
                    'DT': 'DT', 'DATE': 'DT', 'TIME': 'DT',
                    'QT': 'QT', 'QUANTITY': 'QT',
                    'TI': 'TI'
                }
                
                if entity_type in type_mapping:
                    found_entities.append((entity_text, type_mapping[entity_type]))
    
    # Map found entities back to token positions
    sentence = "".join(tokens)
    for entity_text, entity_type in found_entities:
        # Find entity in the sentence
        entity_start = sentence.find(entity_text)
        if entity_start != -1:
            entity_end = entity_start + len(entity_text)
            
            # Find corresponding token positions
            current_pos = 0
            start_token_idx = None
            end_token_idx = None
            
            for i, token in enumerate(tokens):
                token_start = current_pos
                token_end = current_pos + len(token)
                
                if start_token_idx is None and token_start <= entity_start < token_end:
                    start_token_idx = i
                if entity_start < token_end <= entity_end:
                    end_token_idx = i
                
                current_pos = token_end
            
            # Apply BIO tagging
            if start_token_idx is not None:
                if end_token_idx is None:
                    end_token_idx = start_token_idx
                
                for token_idx in range(start_token_idx, min(end_token_idx + 1, len(tokens))):
                    if token_idx == start_token_idx:
                        predictions[token_idx] = f'B-{entity_type}'
                    else:
                        predictions[token_idx] = f'I-{entity_type}'
    
    # Convert to label IDs
    prediction_ids = []
    for pred in predictions:
        if pred in LABEL2ID:
            prediction_ids.append(LABEL2ID[pred])
        else:
            prediction_ids.append(LABEL2ID['O'])  # Default to 'O' if not found
    
    return prediction_ids

def load_klue_ner_data(filepath):
    """Load KLUE NER data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

# Single model evaluation function
def evaluate_single_model(config: ModelConfig, eval_data, model_specific_output_dir):
    """Evaluate a single model configuration using zero-shot NER."""
    logger.info(f"--- Starting Zero-Shot NER Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Output directory: {model_specific_output_dir}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")
    
    # Setup output files
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}.log")
    detailed_filepath = os.path.join(model_specific_output_dir, f"detailed_results_{config.name}.json")
    
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
    
    model = None
    tokenizer = None
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading tokenizer for {config.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=CACHE_DIR)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Adding [PAD] token")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Setup quantization if needed
        quantization_config = None
        if config.use_quantization:
            logger.info("Applying 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        
        logger.info(f"Loading model for causal language modeling...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        # Handle pad token for model
        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
            logger.warning("Resizing model embeddings after load due to added PAD token.")
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id
        
        model.eval()
        logger.info(f"Model loaded successfully: {config.name}")
        
        # Take subset for evaluation
        eval_subset = eval_data[:MAX_EVAL_SAMPLES]
        
        true_labels = []
        pred_labels = []
        detailed_logs = []
        
        correct_predictions = 0
        total_tokens = 0
        errors_or_skipped = 0
        
        for idx, item in enumerate(tqdm(eval_subset, desc=f"Evaluating {config.name}")):
            try:
                tokens = item["tokens"]
                gold_labels = item["ner_tags"]
                
                prompt = create_ner_prompt(item)
                if not prompt:
                    errors_or_skipped += 1
                    continue
                
                # Generate prediction
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,  # Allow more tokens for NER output
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=0.1
                    )
                
                output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                
                # Extract predictions
                pred_tags = extract_ner_predictions(generated_text, tokens, prompt)
                
                # Ensure same length
                min_len = min(len(pred_tags), len(gold_labels))
                pred_tags = pred_tags[:min_len]
                gold_tags = gold_labels[:min_len]
                
                # Calculate token-level accuracy
                for pred, gold in zip(pred_tags, gold_tags):
                    total_tokens += 1
                    if pred == gold:
                        correct_predictions += 1
                
                true_labels.extend(gold_tags)
                pred_labels.extend(pred_tags)
                
                # Log details
                detailed_logs.append({
                    "index": idx,
                    "sentence": "".join(tokens),
                    "tokens": tokens,
                    "gold_labels": [ID2LABEL.get(l, "O") for l in gold_tags],
                    "pred_labels": [ID2LABEL.get(p, "O") for p in pred_tags],
                    "gold_label_ids": gold_tags,
                    "pred_label_ids": pred_tags,
                    "raw_output": generated_text
                })
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                errors_or_skipped += 1
                continue
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="micro", zero_division=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="macro", zero_division=0)
        
        # Token-level accuracy
        token_accuracy = (correct_predictions / total_tokens * 100) if total_tokens > 0 else 0
        
        logger.info(f"Evaluation results for {config.name}:")
        logger.info(f"Token Accuracy: {token_accuracy:.4f}%")
        logger.info(f"Micro Precision: {precision:.4f}")
        logger.info(f"Micro Recall: {recall:.4f}")
        logger.info(f"Micro F1: {f1:.4f}")
        logger.info(f"Macro Precision: {macro_precision:.4f}")
        logger.info(f"Macro Recall: {macro_recall:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Total tokens evaluated: {total_tokens}")
        logger.info(f"Errors/Skipped: {errors_or_skipped}")
        
        # Classification report
        class_report = classification_report(true_labels, pred_labels, target_names=NER_TAGS, zero_division=0)
        logger.info(f"Classification Report:\n{class_report}")
        
        # Save results
        config_dict = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        results = {
            "model_config": config_dict,
            "dataset_path": DATASET_PATH,
            "total_samples": len(eval_subset),
            "total_tokens": total_tokens,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors_or_skipped,
            "token_accuracy": token_accuracy,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "classification_report": class_report
        }
        
        # Save main results
        try:
            with open(results_filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Save detailed logs
        try:
            with open(detailed_filepath, "w", encoding="utf-8") as f:
                json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed results saved to: {detailed_filepath}")
        except Exception as e:
            logger.error(f"Failed to save detailed results: {e}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error during evaluation for {config.name}: {e}")
        return None
        
    finally:
        # Clean up resources
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

# Main execution
def main():
    logger.info(f"Loading KLUE NER data from: {DATASET_PATH}")
    eval_data = load_klue_ner_data(DATASET_PATH)
    if eval_data is None:
        logger.error("Could not load KLUE NER data. Exiting.")
        return
    
    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")
    
    all_results = {}
    
    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Zero-Shot NER Evaluation for Model: {config.name} =====\n")
        
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")
        
        # Evaluate model
        results = evaluate_single_model(config, eval_data, model_specific_output_dir)
        if results is not None:
            all_results[config.name] = results
        
        logger.info(f"\n===== Finished Zero-Shot NER Evaluation for Model: {config.name} =====")
        print("-" * 80)
    
    # Save combined results
    combined_results_path = os.path.join(BASE_OUTPUT_DIR, "combined_results.json")
    try:
        with open(combined_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Combined results saved to: {combined_results_path}")
    except Exception as e:
        logger.error(f"Failed to save combined results: {e}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("KLUE NER ZERO-SHOT EVALUATION SUMMARY")
    logger.info("="*80)
    for model_name, results in all_results.items():
        if results:
            logger.info(f"{model_name}:")
            logger.info(f"  Token Accuracy: {results.get('token_accuracy', 0):.2f}%")
            logger.info(f"  Micro F1: {results.get('micro_f1', 0):.4f}")
            logger.info(f"  Macro F1: {results.get('macro_f1', 0):.4f}")
            logger.info(f"  Total Tokens: {results.get('total_tokens', 0)}")
    logger.info("="*80)
    
    logger.info("KLUE NER zero-shot evaluation completed for all models.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")
    
    main()