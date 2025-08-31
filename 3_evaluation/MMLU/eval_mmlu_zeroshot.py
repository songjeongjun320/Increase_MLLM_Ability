import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import gc
import sys # For version logging

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
    #     name="Qwen2.5-3B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="google_gemma-3-4b-it",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-Distill-Qwen-1.5B",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
    #     use_quantization=False
    # ),

    # ModelConfig(
    #     name="Qwen2.5-3B-Instruct-ToW-completion",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/Qwen2.5-3B-Instruct-tow",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="google_gemma-3-4b-it-ToW-completion",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/gemma-3-4b-it-tow",
    #     use_quantization=False
    # ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct-ToW-completion",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/llama-3.2-3b-tow",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B-ToW-completion",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/DeepSeek-R1-Distill-Qwen-1.5B-tow",
        use_quantization=False
    ),
]

# --- General Configuration ---
DATASET_PATH = "../../2_datasets/MMLU/MMLU_origin.json"
BASE_OUTPUT_DIR = "mmlu_model1_zeroshot" # 0-shot evaluation results
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions for 0-shot MMLU Evaluation ---
def create_0shot_prompt(item):
    """
    Creates a 0-shot MMLU prompt for a given test item.
    """
    subject = item.get("Subject", "unknown")  # MMLU uses 'Subject' field (uppercase)
    subject_display = subject.replace("_", " ").title()
    
    prompt_parts = [f"The following is a multiple choice question about {subject_display}."]
    prompt_parts.append("")
    
    question = item.get("Question", "")  # MMLU uses 'Question' field (uppercase)
    
    # Get choices from MMLU format (A, B, C, D fields)
    choice_a = item.get("A", "Option A")
    choice_b = item.get("B", "Option B")
    choice_c = item.get("C", "Option C")
    choice_d = item.get("D", "Option D")
    
    prompt_parts.append(question)
    prompt_parts.append(f"A. {choice_a}")
    prompt_parts.append(f"B. {choice_b}")
    prompt_parts.append(f"C. {choice_c}")
    prompt_parts.append(f"D. {choice_d}")

    prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)

def extract_answer_first_token(model_output):
    """
    Extract answer from model output using first token approach.
    """
    cleaned_output = model_output.strip().upper()
    for char in cleaned_output:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return None

def load_mmlu_data(filepath):
    """Loads MMLU data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth_origin(item):
    """Returns the ground truth answer letter from the MMLU data."""
    answer = item.get("Answer", None)
    if answer and answer.upper() in ["A", "B", "C", "D"]:
        return answer.upper()
    return None

# --- Batch Processing Function ---
def process_batch(model, tokenizer, batch_prompts, batch_indices):
    """Processes a batch of prompts efficiently."""
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        
        batch_results = []
        input_length = inputs['input_ids'].shape[1]
        for i, sequence in enumerate(outputs):
            output_tokens = sequence[input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            extracted_answer = extract_answer_first_token(generated_text)
            batch_results.append({
                'index': batch_indices[i],
                'raw_output': generated_text,
                'extracted_answer': extracted_answer
            })
        return batch_results
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=False)
        return [{'index': idx, 'raw_output': f"ERROR: {str(e)[:100]}", 'extracted_answer': None} for idx in batch_indices]

def process_single_with_retry(model, tokenizer, prompt, index, max_retries=5):
    """Process a single prompt with retry logic for answer extraction failures."""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            
            input_length = inputs['input_ids'].shape[1]
            output_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            extracted_answer = extract_answer_first_token(generated_text)
            
            if extracted_answer is not None:
                return {
                    'index': index,
                    'raw_output': generated_text,
                    'extracted_answer': extracted_answer,
                    'retry_count': attempt
                }
            else:
                logger.debug(f"Retry {attempt + 1}/{max_retries} for index {index}: Failed to extract answer from '{generated_text}'")
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1} for index {index}: {e}")
            if attempt == max_retries - 1:
                return {
                    'index': index,
                    'raw_output': f"ERROR after {max_retries} attempts: {str(e)[:100]}",
                    'extracted_answer': None,
                    'retry_count': attempt
                }
    
    # If all retries failed to extract answer
    return {
        'index': index,
        'raw_output': f"EXTRACTION_FAILED after {max_retries} attempts: {generated_text}",
        'extracted_answer': None,
        'retry_count': max_retries - 1
    }

# --- Single Model Evaluation Function with 0-shot Prompting ---
def evaluate_single_model(config: ModelConfig, mmlu_data: list, model_specific_output_dir: str):
    """
    Performs 0-shot MMLU evaluation for a single model.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_0shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_0shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_0shot.json")
    failure_cases_filepath = os.path.join(model_specific_output_dir, f"failure_cases_{config.name}_0shot.json")

    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # Remove previous file handlers to avoid duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 0-shot Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # --- Load Model and Tokenizer ---
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_load_path, 
                    cache_dir=CACHE_DIR,
                    padding_side='left',  # <--- 이 라인을 추가하세요!
                    trust_remote_code=True # Qwen 등 일부 모델은 이 옵션이 필요할 수 있습니다.
                )  

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model {config.model_id}...")
        quantization_config_bnb = None
        if config.use_quantization:
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

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

        if config.adapter_path:
            logger.info(f"Loading adapter from: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            logger.info("Successfully loaded LoRA adapter.")

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []
        raw_generations_list = []
        failure_cases_list = []  # New: Track failure cases separately

        logger.info("Starting 0-shot inference loop...")
        test_data = mmlu_data # Use all data for testing
        logger.info(f"Test data size: {len(test_data)}")
        
        pbar = tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Evaluating {config.name} (0-shot, errors: 0)")
        for i in pbar:
            batch_data = test_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []

            for j, item in enumerate(batch_data):
                current_index = i + j
                ground_truth = get_ground_truth_origin(item)
                prompt = create_0shot_prompt(item)

                if ground_truth is None or prompt is None:
                    errors_or_skipped += 1
                    failure_reason = "SKIPPED - Invalid GT/Prompt"
                    failure_type = "invalid_ground_truth" if ground_truth is None else "prompt_creation_failed"
                    
                    results_details.append({
                        "index": current_index, "ground_truth": ground_truth, "model_raw_output": failure_reason,
                        "predicted_answer": None, "is_correct": False
                    })
                    raw_generations_list.append({
                        "index": current_index, "subject": item.get("Subject", "unknown"), "ground_truth": ground_truth,
                        "raw_output": failure_reason, "extracted_answer": None
                    })
                    
                    # Add to failure cases
                    failure_cases_list.append({
                        "index": current_index,
                        "subject": item.get("Subject", "unknown"),
                        "question": item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": failure_type,
                        "failure_reason": failure_reason,
                        "raw_output": failure_reason
                    })
                    continue
                
                batch_prompts.append(prompt)
                batch_indices.append(current_index)
                batch_ground_truths.append(ground_truth)

            if not batch_prompts:
                continue

            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            # Retry logic for failed answer extractions
            retry_indices = []
            retry_prompts = []
            retry_ground_truths = []
            
            for i, result in enumerate(batch_results):
                if result['extracted_answer'] is None and not result['raw_output'].startswith("ERROR"):
                    # Need to retry this one
                    retry_indices.append(i)
                    retry_prompts.append(batch_prompts[i])
                    retry_ground_truths.append(batch_ground_truths[i])
            
            # Process retries individually
            if retry_indices:
                logger.info(f"Retrying {len(retry_indices)} failed extractions with individual processing...")
                for j, retry_idx in enumerate(retry_indices):
                    retry_result = process_single_with_retry(
                        model, tokenizer, retry_prompts[j], 
                        batch_results[retry_idx]['index']
                    )
                    # Update the original result
                    batch_results[retry_idx] = retry_result
            
            for result, ground_truth in zip(batch_results, batch_ground_truths):
                generated_text_log = result['raw_output']
                model_answer_log = result['extracted_answer']
                is_correct_log = False
                retry_info = f" (after {result.get('retry_count', 0) + 1} attempts)" if 'retry_count' in result else ""

                if model_answer_log:
                    total_predictions += 1
                    if model_answer_log == ground_truth:
                        correct_predictions += 1
                        is_correct_log = True
                else:
                    errors_or_skipped += 1
                    original_generated_text = generated_text_log
                    if not generated_text_log.startswith("ERROR"):
                        if generated_text_log.startswith("EXTRACTION_FAILED"):
                            failure_type = "answer_extraction_failed"
                        else:
                            generated_text_log = f"EXTRACTION_FAILED{retry_info}: {generated_text_log}"
                            failure_type = "answer_extraction_failed"
                    else:
                        failure_type = "model_error"

                    # Add to failure cases
                    original_item = test_data[result['index']]
                    failure_cases_list.append({
                        "index": result['index'],
                        "subject": original_item.get("Subject", "unknown"),
                        "question": original_item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": failure_type,
                        "failure_reason": generated_text_log,
                        "raw_output": original_generated_text,
                        "retry_count": result.get('retry_count', 0),
                        "choices": {
                            "A": original_item.get("A", ""),
                            "B": original_item.get("B", ""),
                            "C": original_item.get("C", ""),
                            "D": original_item.get("D", "")
                        }
                    })

                results_details.append({
                    "index": result['index'], "ground_truth": ground_truth, "model_raw_output": generated_text_log,
                    "predicted_answer": model_answer_log, "is_correct": is_correct_log, "retry_count": result.get('retry_count', 0)
                })
                
                original_item = test_data[result['index']]
                raw_generations_list.append({
                    "index": result['index'], "subject": original_item.get("Subject", "unknown"), "ground_truth": ground_truth,
                    "raw_output": generated_text_log, "extracted_answer": model_answer_log, "retry_count": result.get('retry_count', 0)
                })

            # Update progress bar with current error count
            pbar.set_description(f"Evaluating {config.name} (0-shot, errors: {errors_or_skipped})")

        # --- Final Results ---
        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        accuracy_strict = (correct_predictions / len(test_data) * 100) if len(test_data) > 0 else 0

        logger.info(f"--- 0-shot MMLU Results for {config.name} ---")
        logger.info(f"Test Items: {len(test_data)}")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")

        # --- Save Results ---
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "0-shot MMLU",
            "test_items": len(test_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy_standard": accuracy_standard,
            "accuracy_strict": accuracy_strict,
            "details": results_details
        }
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
        
        # --- Save Failure Cases ---
        if failure_cases_list:
            logger.info(f"Saving {len(failure_cases_list)} failure cases to {failure_cases_filepath}...")
            try:
                failure_summary = {
                    "total_failures": len(failure_cases_list),
                    "failure_types": {},
                    "failure_cases": failure_cases_list
                }
                
                # Count failure types  
                for case in failure_cases_list:
                    failure_type = case.get("failure_type", "unknown")
                    failure_summary["failure_types"][failure_type] = failure_summary["failure_types"].get(failure_type, 0) + 1
                
                with open(failure_cases_filepath, 'w', encoding='utf-8') as f:
                    json.dump(failure_summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Failure cases saved successfully. Types: {failure_summary['failure_types']}")
            except Exception as e:
                logger.error(f"Failed to save failure cases file {failure_cases_filepath}: {e}")
        else:
            logger.info("No failure cases to save.")

    except Exception as e:
        logger.exception(f"An critical error occurred during evaluation for {config.name}: {e}")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
             root_logger.removeHandler(file_handler)
             file_handler.close()

def main():
    mmlu_data = load_mmlu_data(DATASET_PATH)
    if not mmlu_data:
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    all_results_summary = []  # To store summaries for the final JSON

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        evaluate_single_model(config, mmlu_data, model_specific_output_dir)

    # --- Create a consolidated summary of all model results ---
    logger.info("--- Generating Consolidated Summary ---")
    for config in MODEL_CONFIGS:
        results_filepath = os.path.join(BASE_OUTPUT_DIR, config.name, f"results_{config.name}_0shot.json")
        if os.path.exists(results_filepath):
            try:
                with open(results_filepath, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                summary = {
                    "model_name": config.name,
                    "accuracy_standard": result_data.get("accuracy_standard"),
                    "accuracy_strict": result_data.get("accuracy_strict"),
                    "correct_predictions": result_data.get("correct_predictions"),
                    "valid_predictions": result_data.get("valid_predictions"),
                    "total_items": result_data.get("test_items")
                }
                all_results_summary.append(summary)
            except Exception as e:
                logger.error(f"Failed to read or parse result file for {config.name}: {e}")
        else:
            logger.warning(f"Result file not found for {config.name} at {results_filepath}")

    if all_results_summary:
        summary_filepath = os.path.join(BASE_OUTPUT_DIR, "summary.json")
        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_results_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Consolidated summary saved to {summary_filepath}")
        except Exception as e:
            logger.error(f"Failed to save consolidated summary: {e}")

if __name__ == "__main__":
    main()
