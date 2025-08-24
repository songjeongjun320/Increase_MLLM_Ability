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
    # ModelConfig(
    #     name="Qwen2.5-7B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Mistral-8B-Instruct-2410",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.1-8B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-0528-Qwen3-8B",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
    #     use_quantization=False # Adjust based on VRAM
    # ),

    # TOW Trained Model
    ModelConfig(
        name="Qwen2.5-7B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-7B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Mistral-8B-Instruct-2410-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.1-8B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-0528-Qwen3-8B-ToW",
        use_quantization=False
    ),


    # TOW Model 2
    # ModelConfig(
    #     name="Qwen2.5-7B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/Qwen2.5-7B-Instruct-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Mistral-8B-Instruct-2410-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/Mistral-8B-Instruct-2410-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.1-8B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/Llama-3.1-8B-Instruct-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-0528-Qwen3-8B-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/DeepSeek-R1-0528-Qwen3-8B-ToW",
    #     use_quantization=False
    # ),
]


# --- General Configuration ---
DATASET_PATH = "../../2_datasets/MMLU/MMLU_origin.json"
BASE_OUTPUT_DIR = "mmlu_tow_model1_zeroshot" # 0-shot evaluation results
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
    subject = item.get("subject", "unknown")
    subject_display = subject.replace("_", " ").title()
    
    prompt_parts = [f"The following is a multiple choice question about {subject_display}."]
    prompt_parts.append("")
    
    question = item.get("question", "")
    choices = item.get("choices", [])
    
    prompt_parts.append(question)
    if len(choices) >= 4:
        prompt_parts.append(f"A. {choices[0]}")
        prompt_parts.append(f"B. {choices[1]}")
        prompt_parts.append(f"C. {choices[2]}")
        prompt_parts.append(f"D. {choices[3]}")
    else:
        logger.warning(f"Insufficient choices for question in subject {subject}")
        prompt_parts.extend(["A. Option A", "B. Option B", "C. Option C", "D. Option D"])

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
    """Returns the ground truth answer letter from the original MMLU data."""
    answer_index = item.get("answer", -1)
    if isinstance(answer_index, int) and 0 <= answer_index <= 3:
        return chr(ord('A') + answer_index)
    return None

# --- Single Model Evaluation Function with 0-shot Prompting ---
def evaluate_single_model(config: ModelConfig, mmlu_data: list, model_specific_output_dir: str):
    """
    Performs 0-shot MMLU evaluation for a single model.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_0shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_0shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_0shot.json")

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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        
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

        logger.info("Starting 0-shot inference loop...")
        test_data = mmlu_data # Use all data for testing
        logger.info(f"Test data size: {len(test_data)}")
        
        pbar = tqdm(enumerate(test_data), desc=f"Evaluating {config.name} (0-shot, errors: 0)", total=len(test_data))
        for i, item in pbar:
            ground_truth = get_ground_truth_origin(item)
            prompt = create_0shot_prompt(item)
            
            generated_text_log = "SKIPPED"
            model_answer_log = None
            is_correct_log = False

            if ground_truth is None or prompt is None:
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Invalid Ground Truth or Prompt"
            else:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                        )
                    output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                    model_answer_log = extract_answer_first_token(generated_text_log)

                    if model_answer_log:
                        total_predictions += 1
                        if model_answer_log == ground_truth:
                            correct_predictions += 1
                            is_correct_log = True
                    else:
                        errors_or_skipped += 1
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"
                except Exception as e:
                    logger.error(f"Item {i}: Inference error: {e}", exc_info=False)
                    errors_or_skipped += 1
                    generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

            results_details.append({
                "index": i, "ground_truth": ground_truth, "model_raw_output": generated_text_log,
                "predicted_answer": model_answer_log, "is_correct": is_correct_log
            })
            raw_generations_list.append({
                "index": i, "subject": item.get("subject", "unknown"), "ground_truth": ground_truth,
                "raw_output": generated_text_log, "extracted_answer": model_answer_log
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

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        evaluate_single_model(config, mmlu_data, model_specific_output_dir)

if __name__ == "__main__":
    main()
