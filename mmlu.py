import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from datasets import load_dataset # 직접 사용하지 않으므로 주석 처리 가능
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import gc
import sys # For version logging

# --- Model Configuration (output_dir 필드 제거) ---
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
        use_quantization=False # Adjust based on VRAM
    ),
]

# --- General Configuration ---
DATASET_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/MMLU/MMLU_origin.json"
BASE_OUTPUT_DIR = "evaluation_results_mmlu_origin_new" # 결과를 저장할 기본 디렉토리 이름 변경 (테스트용)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions (변경 없음) ---
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

def create_prompt_for_origin(item):
    """원본 MMLU 형식('question', 'choices' 리스트)에 맞는 프롬프트를 생성합니다."""
    question = item.get("question", "")
    choices_list = item.get("choices", [])
    if not question or not isinstance(choices_list, list) or len(choices_list) != 4:
        return None

    choices_dict = {chr(ord('A') + i): choice for i, choice in enumerate(choices_list)}
    prompt = f"""Question: {question}
A) {choices_dict.get('A', '')}
B) {choices_dict.get('B', '')}
C) {choices_dict.get('C', '')}
D) {choices_dict.get('D', '')}
Answer:"""
    return prompt

def get_ground_truth_origin(item):
    """원본 MMLU 데이터('answer' 정수 인덱스)에서 정답 문자를 반환합니다."""
    answer_index = item.get("answer", -1)
    if isinstance(answer_index, int) and 0 <= answer_index <= 3:
        return chr(ord('A') + answer_index)
    elif isinstance(answer_index, str) and answer_index.upper() in ["A", "B", "C", "D"]:
        return answer_index.upper()
    return None

def extract_answer(model_output, prompt): # prompt 인자 유지 (미래 사용 가능성)
    """모델 출력에서 답변(A, B, C, D)을 추출합니다."""
    # 프롬프트가 출력 시작 부분에 있으면 제거 (더 유연하게 처리)
    # model_output과 prompt 모두 strip()으로 앞뒤 공백 제거 후 비교
    stripped_output = model_output.strip()
    stripped_prompt_end = prompt.strip().split("Answer:")[0] + "Answer:" # "Answer:" 까지의 프롬프트
    
    prediction_text = stripped_output
    if stripped_output.startswith(stripped_prompt_end.strip()):
        prediction_text = stripped_output[len(stripped_prompt_end.strip()):].strip()
    
    cleaned_text = prediction_text.upper()
    match = re.search(r"^\s*([ABCD])(?:[).:\s]|\b)", cleaned_text)
    if match:
        return match.group(1)

    if len(cleaned_text) == 1 and cleaned_text in ["A", "B", "C", "D"]:
        return cleaned_text

    match_phrase = re.search(r"(?:ANSWER\s*IS|:\s*)\s*([ABCD])\b", cleaned_text)
    if match_phrase:
         return match_phrase.group(1)
    return None


# --- Single Model Evaluation Function (model_specific_output_dir 인자 사용) ---
def evaluate_single_model(config: ModelConfig, mmlu_data: list, model_specific_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 MMLU 평가를 수행하고,
    결과와 로그를 model_specific_output_dir에 저장합니다.
    """
    # 결과 및 로그 파일 경로 설정
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}.json")


    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # 기존 파일 핸들러 제거 (중복 로깅 방지)
    for handler in list(root_logger.handlers): # Iterate over a copy
        if isinstance(handler, logging.FileHandler) and handler is not file_handler : # 자기 자신은 제외
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    if file_handler not in root_logger.handlers: # 중복 추가 방지
        root_logger.addHandler(file_handler)

    logger.info(f"--- Starting Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Output directory: {model_specific_output_dir}") # 모델별 출력 디렉토리 로깅
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Logs will be saved to: {log_filepath}")
    logger.info(f"Raw generations will be saved to: {raw_gen_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations_list = [] # 리스트 이름 변경

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
        correct_predictions = 0
        total_predictions = 0 # 유효한 예측 시도 횟수
        errors_or_skipped = 0 # 데이터 문제, 프롬프트 생성 실패, 추론 오류, 답변 추출 실패 모두 포함
        results_details = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(mmlu_data, desc=f"Evaluating {config.name}")):
            item_index_for_log = item.get("index", i) # 데이터에 index 필드가 있다면 사용
            ground_truth = get_ground_truth_origin(item)
            prompt = create_prompt_for_origin(item)

            # 기본값 설정
            generated_text_log = "SKIPPED"
            model_answer_log = None
            is_correct_log = False

            if ground_truth is None:
                logger.warning(f"Item {item_index_for_log}: Invalid/missing ground truth (answer: {item.get('answer', 'N/A')}). Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Invalid Ground Truth"
            elif prompt is None:
                logger.warning(f"Item {item_index_for_log}: Failed to create prompt (check question/choices). Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Prompt Creation Failed"
            else:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE) # max_length 추가

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=15, # 답변(A,B,C,D)과 약간의 추가 텍스트를 고려한 길이
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            # early_stopping=True # Removed as it might stop too soon for some models
                        )
                    output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                    model_answer_log = extract_answer(generated_text_log, prompt) # prompt 전달

                    if model_answer_log:
                        total_predictions += 1
                        if model_answer_log == ground_truth:
                            correct_predictions += 1
                            is_correct_log = True
                    else:
                        logger.warning(f"Item {item_index_for_log}: Failed to extract answer from output: '{generated_text_log[:100]}...'")
                        errors_or_skipped += 1 # 답변 추출 실패도 오류로 카운트
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"


                except Exception as e:
                    logger.error(f"Item {item_index_for_log}: Inference error: {e}", exc_info=False)
                    errors_or_skipped += 1
                    generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}" # 오류 메시지 일부 저장

            # 모든 경우에 대해 상세 결과 및 원본 생성 결과 기록
            results_details.append({
                "index": item_index_for_log,
                "ground_truth": ground_truth,
                "model_raw_output": generated_text_log,
                "predicted_answer": model_answer_log,
                "is_correct": is_correct_log
            })
            raw_generations_list.append({
                "index": item_index_for_log,
                "subject": item.get("subject", "unknown"),
                "ground_truth": ground_truth,
                "raw_output": generated_text_log,
                "extracted_answer": model_answer_log
            })

            if (i + 1) % 200 == 0:
                 current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                 logger.info(f"Progress ({config.name}): {i + 1}/{len(mmlu_data)}, Acc: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors/Skipped: {errors_or_skipped}")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(mmlu_data)}")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Final Accuracy (Correct / Valid Predictions): {accuracy:.2f}%")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(mmlu_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors_or_skipped,
            "accuracy": accuracy,
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
        logger.exception(f"An critical error occurred during evaluation for {config.name}: {e}")

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
    logger.info(f"Loading MMLU data from: {DATASET_PATH}")
    mmlu_data = load_mmlu_data(DATASET_PATH)
    if mmlu_data is None:
        logger.error("Could not load MMLU data. Exiting.")
        return

    # 기본 출력 디렉토리 생성 (전체 실행에 대해 한 번만)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====\n")
        # 모델별 출력 디렉토리 경로 생성
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True) # 모델별 디렉토리 생성
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # 평가 함수 호출 (모델별 출력 디렉토리 전달)
        evaluate_single_model(config, mmlu_data, model_specific_output_dir)

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