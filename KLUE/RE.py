import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sklearn.metrics import precision_recall_f1_support, accuracy_score
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import gc
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_re_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE RE Label Definitions
RE_LABELS = [
    "no_relation",
    "org:dissolved",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]
NUM_LABELS = len(RE_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(RE_LABELS)}
ID2LABEL = {idx: label for label, idx in enumerate(RE_LABELS)}
logger.info(f"Total number of KLUE-RE labels: {NUM_LABELS}")

# Model configuration class (from mmlu.py style)
@dataclass
class ModelConfig:
    name: str                             # Unique name for this run (used for filenames)
    model_id: str                         # Hugging Face model identifier
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

# Model configurations (from mmlu.py)
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

# Configuration parameters
BASE_OUTPUT_DIR = "klue_re_evaluation_results"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_validation.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"

# Model and tokenizer loading function
def load_model_and_tokenizer(model_config):
    """Load model and tokenizer based on model configuration."""
    logger.info(f"Loading model: {model_config.model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id, 
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            logger.warning("Tokenizer lacks both pad and eos tokens. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Add special tokens for entity markers
    special_tokens = {"additional_special_tokens": ["[SUBJ]", "[/SUBJ]", "[OBJ]", "[/OBJ]"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model with quantization if specified
    quantization_config_bnb = None
    if model_config.use_quantization:
        logger.info("Applying 4-bit quantization.")
        quantization_config_bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_config.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    logger.info(f"Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=model_config.torch_dtype,
        quantization_config=quantization_config_bnb,
        device_map=DEVICE,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    # Resize model embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Handle pad token configuration
    if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.pad_token == '[PAD]' and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    logger.info(f"Model loaded successfully: {model_config.name}")
    return model, tokenizer

# Evaluation function
def evaluate_model(model_config, model_specific_output_dir):
    """Evaluate the model on KLUE-RE metrics."""
    logger.info(f"Starting evaluation for {model_config.name}")
    
    # Setup file paths
    results_filepath = os.path.join(model_specific_output_dir, f"results_{model_config.name}.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{model_config.name}.log")
    raw_predictions_filepath = os.path.join(model_specific_output_dir, f"raw_predictions_{model_config.name}.json")
    
    # Setup logging for this specific model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler is not file_handler:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)
    
    logger.info(f"--- Starting Evaluation for Model: {model_config.name} ({model_config.model_id}) ---")
    logger.info(f"Output directory: {model_specific_output_dir}")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {model_config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if model_config.use_quantization else 'Disabled'}")
    
    model = None
    tokenizer = None
    raw_predictions_list = []
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        # Load validation data
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        
        val_subset = val_data[:MAX_EVAL_SAMPLES]
        logger.info(f"Evaluating on {len(val_subset)} samples")
        
        device = model.device
        
        true_labels = []
        pred_labels = []
        logs = []
        
        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        
        for i, item in enumerate(tqdm(val_subset, desc=f"Evaluating {model_config.name}")):
            try:
                sentence = item["sentence"]
                subject_entity = item["subject_entity"]
                object_entity = item["object_entity"]
                gold_label = item["label"]
                
                # Add entity markers to the sentence
                sub_start, sub_end = subject_entity["start_idx"], subject_entity["end_idx"]
                obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]
                
                if sub_start < obj_start:
                    marked_sentence = (
                        sentence[:sub_start] +
                        "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                        sentence[sub_end + 1:obj_start] +
                        "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                        sentence[obj_end + 1:]
                    )
                else:
                    marked_sentence = (
                        sentence[:obj_start] +
                        "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                        sentence[obj_end + 1:sub_start] +
                        "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                        sentence[sub_end + 1:]
                    )
                
                # Tokenize
                encoding = tokenizer(
                    marked_sentence,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(**encoding)
                    logits = outputs.logits
                
                prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
                
                true_labels.append(gold_label)
                pred_labels.append(prediction)
                
                is_correct = (prediction == gold_label)
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Log details
                log_entry = {
                    "index": i,
                    "sentence": sentence,
                    "marked_sentence": marked_sentence,
                    "subject_entity": subject_entity["word"],
                    "object_entity": object_entity["word"],
                    "gold_label": ID2LABEL[gold_label],
                    "gold_label_id": gold_label,
                    "pred_label": ID2LABEL[prediction],
                    "pred_label_id": prediction,
                    "is_correct": is_correct
                }
                logs.append(log_entry)
                raw_predictions_list.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                    logger.info(f"Progress ({model_config.name}): {i + 1}/{len(val_subset)}, Acc: {current_acc:.2f}% ({correct_predictions}/{total_predictions})")
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                errors_or_skipped += 1
                continue
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="macro", zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_f1_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        per_class_metrics = {
            ID2LABEL[i]: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class))
        }
        
        logger.info(f"--- Final Results for {model_config.name} ---")
        logger.info(f"Total Items Processed: {total_predictions}")
        logger.info(f"Errors or Skipped: {errors_or_skipped}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro Precision: {precision:.4f}")
        logger.info(f"Macro Recall: {recall:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        
        # Save detailed results
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in model_config.__dict__.items()}
        results = {
            "model_config": config_dict_serializable,
            "dataset_path": JSON_VAL_DATASET_PATH,
            "total_samples": len(val_subset),
            "processed_samples": total_predictions,
            "errors_or_skipped": errors_or_skipped,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "per_class_metrics": per_class_metrics,
            "detailed_predictions": logs
        }
        
        with open(results_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {results_filepath}")
        
        # Save raw predictions
        with open(raw_predictions_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_predictions_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Raw predictions saved to: {raw_predictions_filepath}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Critical error during evaluation for {model_config.name}: {e}")
        return None
        
    finally:
        # Clean up resources
        logger.info(f"Cleaning up resources for {model_config.name}...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Resources cleaned up for {model_config.name}.")
        
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            try:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            except Exception as e:
                logger.debug(f"Error closing/removing file handler: {e}")

# Main execution
def main():
    logger.info("Starting KLUE-RE evaluation")
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")
    
    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {model_config.name} =====\n")
        
        # Create model-specific output directory
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, model_config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {model_config.name} will be in: {model_specific_output_dir}")
        
        try:
            # Evaluate model
            results = evaluate_model(model_config, model_specific_output_dir)
            if results:
                all_results[model_config.name] = results
            
            logger.info(f"Completed processing for {model_config.name}")
            
        except Exception as e:
            logger.error(f"Error processing {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
        
        logger.info(f"\n===== Finished Evaluation for Model: {model_config.name} =====")
        print("-" * 80)
    
    # Save combined results
    combined_results_path = os.path.join(BASE_OUTPUT_DIR, "combined_results.json")
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-RE evaluation completed")

if __name__ == "__main__":
    main()