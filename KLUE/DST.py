import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import gc
import sys

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
DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
BASE_OUTPUT_DIR = "evaluation_results_klue_dst"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/scratch/jsong132/.cache/huggingface"
MAX_EVAL_SAMPLES = 1000  # Set to None to evaluate all samples, or a number to limit

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions for DST Evaluation ---
def load_dst_data(filepath):
    """JSON 파일에서 KLUE DST 데이터를 로드합니다."""
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

def normalize_state(state):
    """상태 표현을 정규화합니다"""
    # 빈 상태 처리
    if not state or state == "없음":
        return []
    
    # 문자열인 경우 리스트로 분할
    if isinstance(state, str):
        # 앞의 공백 제거
        state = state.strip()
        if state.startswith(" "):
            state = state[1:]
        # 쉼표와 공백으로 구분된 항목들을 분할
        if not state or state == "없음":
            return []
        items = [item.strip() for item in state.split(",")]
        return [item for item in items if item]
    
    # 이미 리스트인 경우 그대로 반환
    return state

def calculate_joint_accuracy(true_states, pred_states):
    """조인트 정확도 계산 - 모든 슬롯이 정확히 일치해야 함"""
    correct = 0
    total = len(true_states)
    
    for true_state, pred_state in zip(true_states, pred_states):
        # 상태 정규화
        true_set = set(normalize_state(true_state))
        pred_set = set(normalize_state(pred_state))
        
        # 완전히 일치하는 경우에만 정답으로 간주
        if true_set == pred_set:
            correct += 1
    
    return correct / total if total > 0 else 0

def calculate_slot_f1(true_states, pred_states):
    """슬롯 F1 점수 계산 - 개별 슬롯 단위로 정확도 측정"""
    true_slots_all = []
    pred_slots_all = []
    
    for true_state, pred_state in zip(true_states, pred_states):
        true_slots = set(normalize_state(true_state))
        pred_slots = set(normalize_state(pred_state))
        
        true_slots_all.extend(list(true_slots))
        pred_slots_all.extend(list(pred_slots))
    
    # 정밀도, 재현율, F1 계산을 위한 TP, FP, FN 계산
    all_true_slots = set(true_slots_all)
    all_pred_slots = set(pred_slots_all)
    
    tp = len(all_true_slots & all_pred_slots)
    fp = len(all_pred_slots - all_true_slots)
    fn = len(all_true_slots - all_pred_slots)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def extract_dst_output(model_output, prompt):
    """모델 출력에서 DST 상태를 추출합니다."""
    # 프롬프트가 출력 시작 부분에 있으면 제거
    stripped_output = model_output.strip()
    
    prediction_text = stripped_output
    if stripped_output.startswith(prompt.strip()):
        prediction_text = stripped_output[len(prompt.strip()):].strip()
    
    return prediction_text

# --- Single Model Evaluation Function ---
def evaluate_single_model(config: ModelConfig, dst_data: list, model_specific_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 KLUE DST 평가를 수행하고,
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
        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []
        
        true_states = []
        pred_states = []

        logger.info("Starting inference loop...")
        for i, item in enumerate(tqdm(dst_data, desc=f"Evaluating {config.name}")):
            item_index_for_log = item.get("index", i)
            
            # 입력과 정답 추출
            prompt = item.get("input", "")
            ground_truth = item.get("output", "")
            
            # 기본값 설정
            generated_text_log = "SKIPPED"
            model_answer_log = None
            is_correct_log = False

            if not prompt:
                logger.warning(f"Item {item_index_for_log}: Missing input prompt. Skipping.")
                errors_or_skipped += 1
                generated_text_log = "SKIPPED - Missing Input"
            else:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,  # DST 상태가 더 길 수 있으므로 증가
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            temperature=0.1,
                        )
                    output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    generated_text_log = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                    model_answer_log = extract_dst_output(generated_text_log, prompt)

                    if model_answer_log is not None:
                        total_predictions += 1
                        
                        # DST 평가 로직
                        true_state_normalized = normalize_state(ground_truth)
                        pred_state_normalized = normalize_state(model_answer_log)
                        
                        true_states.append(ground_truth)
                        pred_states.append(model_answer_log)
                        
                        # 조인트 정확도 체크 (개별 항목)
                        if set(true_state_normalized) == set(pred_state_normalized):
                            correct_predictions += 1
                            is_correct_log = True
                    else:
                        logger.warning(f"Item {item_index_for_log}: Failed to extract DST state from output: '{generated_text_log[:100]}...'")
                        errors_or_skipped += 1
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                except Exception as e:
                    logger.error(f"Item {item_index_for_log}: Inference error: {e}", exc_info=False)
                    errors_or_skipped += 1
                    generated_text_log = f"ERROR_INFERENCE: {str(e)[:100]}"

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
                "ground_truth": ground_truth,
                "raw_output": generated_text_log,
                "extracted_answer": model_answer_log
            })

            if (i + 1) % 100 == 0:
                 current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                 logger.info(f"Progress ({config.name}): {i + 1}/{len(dst_data)}, Acc: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors/Skipped: {errors_or_skipped}")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        
        # DST 메트릭 계산
        joint_accuracy = calculate_joint_accuracy(true_states, pred_states)
        slot_metrics = calculate_slot_f1(true_states, pred_states)

        logger.info(f"--- Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Items in Dataset: {len(dst_data)}")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Joint Accuracy: {joint_accuracy:.4f}")
        logger.info(f"Slot F1: {slot_metrics['f1']:.4f}")
        logger.info(f"Slot Precision: {slot_metrics['precision']:.4f}")
        logger.info(f"Slot Recall: {slot_metrics['recall']:.4f}")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "total_items": len(dst_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors_or_skipped,
            "joint_accuracy": float(joint_accuracy),
            "slot_f1": float(slot_metrics["f1"]),
            "slot_precision": float(slot_metrics["precision"]),
            "slot_recall": float(slot_metrics["recall"]),
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
    logger.info(f"Loading KLUE DST data from: {DATASET_PATH}")
    dst_data = load_dst_data(DATASET_PATH)
    if dst_data is None:
        logger.error("Could not load KLUE DST data. Exiting.")
        return

    # 평가 샘플 수 제한 (설정된 경우)
    if MAX_EVAL_SAMPLES is not None and len(dst_data) > MAX_EVAL_SAMPLES:
        dst_data = dst_data[:MAX_EVAL_SAMPLES]
        logger.info(f"Limited evaluation to {MAX_EVAL_SAMPLES} samples")

    # 기본 출력 디렉토리 생성
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====\n")
        # 모델별 출력 디렉토리 경로 생성
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_specific_output_dir}")

        # 평가 함수 호출
        evaluate_single_model(config, dst_data, model_specific_output_dir)

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