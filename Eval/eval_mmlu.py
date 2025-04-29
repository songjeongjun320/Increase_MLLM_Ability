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

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str                             # Unique name for this run (used for filenames)
    model_id: str                         # Hugging Face model identifier
    output_dir: str                       # Directory to save results for THIS model
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

MODEL_CONFIGS = [
    # ModelConfig(
    #     name="Llama-3.2-3B",
    #     model_id="meta-llama/Llama-3.2-3B", # Placeholder - 실제 ID 확인 필요
    #     output_dir="placeholder_delete", # main 함수에서 재지정됨
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3B-Instruct",
    #     model_id="meta-llama/Llama-3.2-3B-Instruct", # Placeholder - 실제 ID 확인 필요
    #     output_dir="placeholder_delete",
    #     use_quantization=False # 3B 모델은 VRAM 충분하면 False 가능
    # ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", # 확인된 ID
        output_dir="placeholder_delete",
        use_quantization=False # 8B 모델은 양자화 권장
    ),
]

# --- General Configuration ---
DATASET_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/MMLU/MMLU_origin.json" # 원본 MMLU 경로
BASE_OUTPUT_DIR = "evaluation_results_mmlu_origin" # 결과를 저장할 기본 디렉토리
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface" # Hugging Face 캐시 디렉토리 (권장)

# --- Logging Setup ---
# 기본 로깅 설정, 파일 핸들러는 모델별로 추가/제거
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # 콘솔 출력 명시
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

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
        # logger.warning(f"항목 {item.get('index', 'N/A')}에 필수 필드(question, choices 리스트)가 없거나 형식이 다릅니다.")
        return None # 오류 로깅은 호출하는 쪽에서 처리하도록 변경 (중복 방지)

    choices_dict = {chr(ord('A') + i): choice for i, choice in enumerate(choices_list)}
    # 영어 프롬프트 사용 (모델이 영어 기반이므로)
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
        return answer_index.upper() # 혹시 이미 문자로 되어있는 경우 처리
    # logger.warning(f"항목 {item.get('index', 'N/A')}에서 유효한 정답 인덱스 또는 문자를 찾을 수 없습니다 (answer: {answer_index})")
    return None # 오류 로깅은 호출하는 쪽에서 처리

def extract_answer(model_output, prompt):
    """모델 출력에서 답변(A, B, C, D)을 추출합니다."""
    # 프롬프트가 출력 시작 부분에 있으면 제거
    if model_output.startswith(prompt):
        prediction_text = model_output[len(prompt):].strip()
    else:
        prediction_text = model_output.strip()

    # 답변 시작 부분에서 A, B, C, D 찾기 (더 다양한 형식 처리)
    cleaned_text = prediction_text.upper()
    match = re.search(r"^\s*([ABCD])(?:[).:\s]|\b)", cleaned_text) # 뒤에 구분자나 공백 허용
    if match:
        return match.group(1)

    # 단일 문자 답변
    if len(cleaned_text) == 1 and cleaned_text in ["A", "B", "C", "D"]:
        return cleaned_text

    # 매우 흔한 패턴 "The answer is A" 같은 경우
    match_phrase = re.search(r"(?:ANSWER\s*IS|:\s*)\s*([ABCD])\b", cleaned_text)
    if match_phrase:
         return match_phrase.group(1)

    return None

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, mmlu_data: list, results_filepath: str, log_filepath: str):
    """주어진 설정의 단일 모델에 대해 MMLU 평가를 수행합니다."""

    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8') # UTF-8 인코딩 명시
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # 기존 핸들러 제거 (이전 모델 로그 핸들러가 남아있을 수 있음)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
             root_logger.removeHandler(handler)
             handler.close()
    root_logger.addHandler(file_handler) # 새 파일 핸들러 추가

    logger.info(f"--- Starting Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Logs will be saved to: {log_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations = [] # 모델 원본 출력 저장용

    try:
        # --- Load Model and Tokenizer ---
        logger.info(f"Loading tokenizer for {config.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

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
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors = 0 # 오류 및 스킵 카운트 통합
        results_details = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(mmlu_data, desc=f"Evaluating {config.name}")):
            # 원본 MMLU 형식 처리 함수 사용
            ground_truth = get_ground_truth_origin(item)
            if ground_truth is None:
                logger.warning(f"Item {i}: Invalid/missing ground truth (answer: {item.get('answer', 'N/A')}). Skipping.")
                errors += 1
                continue

            prompt = create_prompt_for_origin(item)
            if prompt is None:
                logger.warning(f"Item {i}: Failed to create prompt (check question/choices). Skipping.")
                errors += 1
                continue

            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(DEVICE)
            generated_text = ""
            model_answer = None
            is_correct = False

            try:
                with torch.no_grad():
                    # 생성 길이 약간 늘림, eos 토큰 명시
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=15,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        early_stopping=True # 답변 찾으면 빨리 멈추도록 시도
                    )
                # 입력 프롬프트를 제외하고 디코딩
                output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                model_answer = extract_answer(generated_text, prompt) # 프롬프트 전달 불필요

            except Exception as e:
                logger.error(f"Item {i}: Inference error: {e}", exc_info=False) # 간단한 오류 로깅
                errors += 1
                generated_text = f"ERROR: {e}"
                # 오류 발생 시에도 결과 저장을 위해 아래 로직 계속 진행

            # 결과 집계
            raw_generations.append({
                "index": i,
                "subject": item.get("subject", "unknown"),
                "ground_truth": ground_truth,
                "raw_output": generated_text,
                "extracted_answer": model_answer
            })

            if generated_text.startswith("ERROR:"):
                 pass # 오류 발생 항목은 이미 errors에 카운트됨
            elif model_answer:
                total_predictions += 1 # 답변 추출 성공 시 유효 예측으로 카운트
                if model_answer == ground_truth:
                    correct_predictions += 1
                    is_correct = True
            else: # 답변 추출 실패
                logger.warning(f"Item {i}: Failed to extract answer from output: '{generated_text[:100]}...'")
                errors += 1 # 추출 실패도 오류로 카운트

            results_details.append({
                "index": i,
                "ground_truth": ground_truth,
                "model_raw_output": generated_text,
                "predicted_answer": model_answer,
                "is_correct": is_correct
            })

            # 중간 결과 로깅
            if (i + 1) % 200 == 0:
                 current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                 logger.info(f"Progress ({config.name}): {i + 1}/{len(mmlu_data)}, Acc: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors/Skipped: {errors}")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items: {len(mmlu_data)}")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors/Skipped (Invalid Data, Inference Error, Extraction Fail): {errors}")
        logger.info(f"Final Accuracy (Correct / Valid Predictions): {accuracy:.2f}%")

        # --- Save Results ---
        # config의 torch_dtype을 문자열로 변환하여 저장
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(mmlu_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_failures": errors,
            "accuracy": accuracy,
            "details": results_details # 개별 상세 결과 포함 여부 결정
        }
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

        # --- Save Raw Generations ---
        raw_gen_filepath = os.path.join(config.output_dir, f"raw_generations_{config.name}.json")
        logger.info(f"Saving raw model generations to {raw_gen_filepath}...")
        try:
            with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations, f, indent=2, ensure_ascii=False)
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
        # 파일 핸들러 닫고 제거
        if file_handler:
             root_logger.removeHandler(file_handler)
             file_handler.close()


# --- Main Execution Logic ---
def main():
    # Load data once
    logger.info(f"Loading MMLU data from: {DATASET_PATH}")
    mmlu_data = load_mmlu_data(DATASET_PATH)
    if mmlu_data is None:
        logger.error("Could not load MMLU data. Exiting.")
        return

    # 기본 출력 디렉토리 생성
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        # 모델별 출력 디렉토리 설정 및 생성
        config.output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Set model output directory: {config.output_dir}")

        results_file = os.path.join(config.output_dir, f"results_{config.name}.json")
        log_file = os.path.join(config.output_dir, f"eval_{config.name}.log")

        # 평가 함수 호출
        evaluate_single_model(config, mmlu_data, results_file, log_file)

        logger.info(f"===== Finished Evaluation for Model: {config.name} =====")
        print("-" * 80) # 콘솔 구분선

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    # 버전 정보 로깅
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    # 캐시 디렉토리 생성 확인
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()