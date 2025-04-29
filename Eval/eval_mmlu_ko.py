import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import re
from dataclasses import dataclass, field # Use dataclass for config
import gc # For garbage collection

# --- Model Configuration (Removed output_dir) ---
@dataclass
class ModelConfig:
    name: str                             # Unique name for this run (used for filenames)
    model_id: str                         # Hugging Face model identifier
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

# !!! REPLACE PLACEHOLDER MODEL IDS WITH ACTUAL IDS WHEN AVAILABLE !!!
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="Llama-3.2-3B",
    #     model_id="meta-llama/Llama-3.2-3B",
    #     use_quantization=False # Smaller models might not *strictly* need it, but adjust based on your GPU VRAM
    # ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct", # Existing model
        use_quantization=False # 8B Instruct likely needs quantization on most systems
    ),
    # ModelConfig(
    #     name="Llama-3.1-8B-Instruct",
    #     model_id="meta-llama/Llama-3.1-8B-Instruct",
    #     use_quantization=False # Adjust based on VRAM
    # ),
]

# --- General Configuration (BASE_OUTPUT_DIR is now the primary output location) ---
DATASET_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/MMLU/MMLU_KO_Openai.json"
BASE_OUTPUT_DIR = "evaluation_results_mmlu_kr" # Base dir for ALL model results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Setup (Configured per model later) ---
# Basic setup, will add file handlers per model
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler()] # Initially just log to console
)
logger = logging.getLogger(__name__) # Get root logger

# --- Helper Functions (Unchanged) ---

def load_mmlu_data(filepath):
    """JSON 파일에서 MMLU 데이터를 로드합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("데이터가 리스트 형식이 아닙니다.")
        if not all(isinstance(item, dict) for item in data):
             raise ValueError("리스트의 모든 항목이 딕셔너리가 아닙니다.")
        logger.info(f"{filepath}에서 {len(data)}개의 항목을 로드했습니다.")
        return data
    except FileNotFoundError:
        logger.error(f"데이터 파일을 찾을 수 없습니다: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"JSON 파일을 디코딩하는 데 실패했습니다: {filepath}")
        return None
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {e}")
        return None

def create_prompt(item):
    """MMLU 항목에 대한 프롬프트를 생성합니다."""
    question = item.get("Question", "")
    choices = { k: item.get(k, "") for k in ["A", "B", "C", "D"] }
    if not question or not all(choices.values()):
        logger.warning(f"항목에 필수 필드(Question, A, B, C, D)가 없습니다: {item.get('id', 'N/A')}") # ID 등 식별자 추가
        return None
    prompt = f"""다음 질문에 가장 적절한 답을 선택하고, 선택한 답의 알파벳(A, B, C, D) 하나만 출력하세요.

질문: {question}
A: {choices['A']}
B: {choices['B']}
C: {choices['C']}
D: {choices['D']}

정답: """
    return prompt

def extract_answer(model_output, prompt):
    """모델 출력에서 답변(A, B, C, D)을 추출합니다."""
    # Remove the prompt part if the model echoes it
    # Handle cases where the prompt might be slightly modified (e.g., whitespace)
    normalized_output = model_output.strip()
    normalized_prompt = prompt.strip()
    if normalized_output.startswith(normalized_prompt):
        prediction_text = normalized_output[len(normalized_prompt):].strip()
    # Handle cases where model might just output the answer or have extra text before
    else:
        prediction_text = normalized_output # Assume the start might be the answer

    cleaned_text = prediction_text.upper()

    # More robust extraction: look for A/B/C/D possibly surrounded by common delimiters
    # Example: "정답: A", "A.", "(A)" etc.
    match = re.search(r"([(\[']*)?\b([ABCD])\b([.)\]']*)?", cleaned_text)
    if match:
        return match.group(2) # Return the letter itself

    # Fallback: check if the very first character is the answer
    if cleaned_text and cleaned_text[0] in ["A", "B", "C", "D"]:
        return cleaned_text[0]

    # logger.warning(f"모델 출력에서 유효한 답변(A,B,C,D)을 추출하지 못했습니다: '{prediction_text}'") # Too verbose
    return None


# --- Single Model Evaluation Function (Uses BASE_OUTPUT_DIR) ---

def evaluate_single_model(config: ModelConfig, mmlu_data: list, base_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 MMLU 평가를 수행하고,
    결과와 로그를 base_output_dir 아래 모델 이름의 하위 디렉토리에 저장합니다.
    """

    # Construct model-specific output directory and file paths
    model_output_dir = os.path.join(base_output_dir, config.name) # Subdirectory per model
    os.makedirs(model_output_dir, exist_ok=True)
    results_filepath = os.path.join(model_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")


    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w') # Overwrite log file each time
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    # Add handler to the root logger for this run
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    # Set level for root logger if needed (e.g., to capture DEBUG from libraries)
    # root_logger.setLevel(logging.DEBUG)

    logger.info(f"--- Starting Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Logs will be saved to: {log_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    try:
        # --- Load Model and Tokenizer ---
        logger.info(f"Loading tokenizer for {config.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                logger.info("Tokenizer does not have a pad token, setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if EOS is also missing (rare but possible)
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding a new pad token '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Important: If a new token is added, the model needs resizing.
                # This should ideally happen BEFORE loading weights, but we'll do it here
                # and hope the loaded model can handle it or has a resizable embedding layer.
                # model.resize_token_embeddings(len(tokenizer)) # Needs model loaded first

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
            device_map=DEVICE, # Assumes single device mapping
            trust_remote_code=True # Necessary for some models
        )

        # Handle tokenizer pad token ID config *after* model load
        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id

        # Resize if we added a pad token (best effort after loading)
        if tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
             logger.warning("Resizing model embeddings after load due to added PAD token.")
             model.resize_token_embeddings(len(tokenizer))
             if hasattr(model.config, "pad_token_id"):
                  model.config.pad_token_id = tokenizer.pad_token_id


        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors = 0
        results_details = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(mmlu_data, desc=f"Evaluating {config.name}")):
            ground_truth = item.get("Answer")
            if not ground_truth or ground_truth not in ["A", "B", "C", "D"]:
                logger.warning(f"Invalid/missing ground truth for item {i}. Skipping.")
                errors += 1
                results_details.append({
                    "index": i,
                    "ground_truth": ground_truth,
                    "model_raw_output": "SKIPPED - Invalid Ground Truth",
                    "predicted_answer": None,
                    "is_correct": False
                })
                continue

            prompt = create_prompt(item)
            if prompt is None:
                logger.warning(f"Could not create prompt for item {i}. Skipping.")
                errors += 1
                results_details.append({
                    "index": i,
                    "ground_truth": ground_truth,
                    "model_raw_output": "SKIPPED - Prompt Creation Failed",
                    "predicted_answer": None,
                    "is_correct": False
                })
                continue

            # Generate exactly one token - should be faster and sufficient for ABCD
            max_gen_tokens = 1 # Change this if more context/reasoning is needed before the letter

            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE) # Added truncation

            generated_text = "ERROR - Inference Failed" # Default in case of error
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens, # Generate very few tokens
                        pad_token_id=tokenizer.pad_token_id, # Use actual pad token ID
                        eos_token_id=tokenizer.eos_token_id, # Stop if EOS generated
                        do_sample=False, # Deterministic output
                        # temperature=1.0, # Not needed for do_sample=False
                        # top_k=50,        # Not needed for do_sample=False
                    )
                # Decode only the newly generated tokens
                output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

            except Exception as e:
                logger.error(f"Inference error on item {i}: {e}")
                errors += 1
                model_answer = None # Mark as error
                is_correct = False

            else: # If inference succeeded
                # Extract answer from the potentially very short generated text
                model_answer = extract_answer(generated_text, prompt)

                is_correct = False
                if model_answer:
                    total_predictions += 1 # Count as a valid prediction attempt
                    if model_answer == ground_truth:
                        correct_predictions += 1
                        is_correct = True
                    # logger.debug(f"Item {i}: GT='{ground_truth}', Pred='{model_answer}', Raw='{generated_text}', Correct={is_correct}")
                else:
                    errors += 1 # Count as an error if extraction fails
                    # logger.warning(f"Failed to extract answer for item {i}. Raw output: '{generated_text}'")

            results_details.append({
                "index": i,
                # "question": item.get("Question"), # Optional: Add back if needed
                # "choices": {k: item.get(k) for k in ["A", "B", "C", "D"]}, # Optional
                "ground_truth": ground_truth,
                # "prompt": prompt, # Optional
                "model_raw_output": generated_text,
                "predicted_answer": model_answer,
                "is_correct": is_correct
            })

            # Intermediate progress logging
            if (i + 1) % 200 == 0: # Log every 200 items
                 current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                 logger.info(f"Progress ({config.name}): {i + 1}/{len(mmlu_data)}, Accuracy: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors: {errors}")


        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        # Recalculate totals based on loop results
        total_processed_loop = len(results_details)
        valid_pred_loop = sum(1 for r in results_details if r['predicted_answer'] is not None and r['model_raw_output'] not in ["SKIPPED - Invalid Ground Truth", "SKIPPED - Prompt Creation Failed", "ERROR - Inference Failed"])
        correct_pred_loop = sum(1 for r in results_details if r['is_correct'])
        errors_loop = sum(1 for r in results_details if r['predicted_answer'] is None and r['model_raw_output'] not in ["SKIPPED - Invalid Ground Truth", "SKIPPED - Prompt Creation Failed"])
        skipped_loop = sum(1 for r in results_details if r['model_raw_output'].startswith("SKIPPED"))
        accuracy_loop = (correct_pred_loop / valid_pred_loop * 100) if valid_pred_loop > 0 else 0


        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(mmlu_data)}")
        logger.info(f"Items Processed (Attempted): {total_processed_loop}")
        logger.info(f"Valid Predictions Made: {valid_pred_loop}")
        logger.info(f"Correct Predictions: {correct_pred_loop}")
        logger.info(f"Errors (Inference/Extraction Failures): {errors_loop}")
        logger.info(f"Items Skipped (Invalid GT/Prompt): {skipped_loop}")
        logger.info(f"Final Accuracy (Correct/Valid): {accuracy_loop:.2f}%")

        # --- Save Results ---
        final_summary = {
            "model_config": config.__dict__, # Save config used
            "dataset_path": DATASET_PATH,
            "total_items": len(mmlu_data),
            "items_processed": total_processed_loop,
            "valid_predictions": valid_pred_loop,
            "correct_predictions": correct_pred_loop,
            "errors_or_failures": errors_loop,
            "items_skipped": skipped_loop,
            "accuracy": accuracy_loop,
            "details": results_details # Include detailed results
        }
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                # Handle non-serializable torch.dtype in config during dump
                def default_serializer(o):
                    if isinstance(o, torch.dtype):
                        return str(o) # Convert dtype to string
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                json.dump(final_summary, f, indent=2, ensure_ascii=False, default=default_serializer)
            logger.info(f"Detailed results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

    except Exception as e:
        logger.exception(f"An critical error occurred during evaluation for {config.name}: {e}") # Log full traceback

    finally:
        # --- CRITICAL: Clean up resources ---
        logger.info(f"Cleaning up resources for {config.name}...")
        del model
        del tokenizer
        gc.collect() # Explicitly call garbage collector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Resources cleaned up for {config.name}.")
        # Remove the file handler for this model from the root logger
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
             root_logger.removeHandler(file_handler)
             file_handler.close()
             logger.debug(f"Removed file handler for {log_filepath}")


# --- Main Execution Logic (Uses BASE_OUTPUT_DIR) ---
def main():
    # Ensure the base output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Load data once
    mmlu_data = load_mmlu_data(DATASET_PATH)
    if mmlu_data is None:
        logger.error("Could not load MMLU data. Exiting.")
        return

    # Evaluate each model in the list
    for config in MODEL_CONFIGS:
         logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====")
         # Pass BASE_OUTPUT_DIR to the evaluation function
         evaluate_single_model(config, mmlu_data, BASE_OUTPUT_DIR)
         logger.info(f"===== Finished Evaluation for Model: {config.name} =====")
         print("-" * 60) # Add visual separator in console

    logger.info("All model evaluations complete.")


if __name__ == "__main__":
    main()