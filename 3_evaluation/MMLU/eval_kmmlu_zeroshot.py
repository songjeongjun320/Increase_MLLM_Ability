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
import sys

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
    ModelConfig(
        name="Qwen2.5-3B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="google_gemma-3-4b-it",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        use_quantization=False
    ),

    # ToW Trained Models
    # ModelConfig(
    #     name="Qwen2.5-3B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-3B-Instruct-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="google_gemma-3-4b-it-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/google_gemma-3-4b-it-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3B-Instruct-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.2-3B-Instruct-ToW",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-Distill-Qwen-1.5B-ToW",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-Distill-Qwen-1.5B-ToW",
    #     use_quantization=False
    # ),
]

# --- General Configuration ---
DATASET_PATH = "../../2_datasets/MMLU/MMLU_KO_Openai.json"
BASE_OUTPUT_DIR = "kmmlu_model1_zeroshot" # 0-shot evaluation results
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

# --- Helper Functions for 0-shot Korean MMLU Evaluation ---
def create_0shot_korean_prompt(item):
    """
    Creates a 0-shot Korean MMLU prompt for a given test item.
    """
    subject = item.get("Subject", "unknown")
    subject_display_map = {
        "abstract_algebra": "추상대수학", "anatomy": "해부학", "astronomy": "천문학", "business_ethics": "경영 윤리",
        "clinical_knowledge": "임상 지식", "college_biology": "대학 생물학", "college_chemistry": "대학 화학",
        "college_computer_science": "대학 컴퓨터 과학", "college_mathematics": "대학 수학", "college_medicine": "대학 의학",
        "college_physics": "대학 물리학", "computer_security": "컴퓨터 보안", "conceptual_physics": "개념 물리학",
        "econometrics": "계량경제학", "electrical_engineering": "전기공학", "elementary_mathematics": "초등 수학",
        "formal_logic": "형식 논리학", "global_facts": "세계 사실", "high_school_biology": "고등학교 생물학",
        "high_school_chemistry": "고등학교 화학", "high_school_computer_science": "고등학교 컴퓨터 과학",
        "high_school_european_history": "고등학교 유럽사", "high_school_geography": "고등학교 지리학",
        "high_school_government_and_politics": "고등학교 정치학", "high_school_macroeconomics": "고등학교 거시경제학",
        "high_school_mathematics": "고등학교 수학", "high_school_microeconomics": "고등학교 미시경제학",
        "high_school_physics": "고등학교 물리학", "high_school_psychology": "고등학교 심리학",
        "high_school_statistics": "고등학교 통계학", "high_school_us_history": "고등학교 미국사",
        "high_school_world_history": "고등학교 세계사", "human_aging": "인간 노화", "human_sexuality": "인간 성학",
        "international_law": "국제법", "jurisprudence": "법학", "logical_fallacies": "논리적 오류",
        "machine_learning": "기계학습", "management": "경영학", "marketing": "마케팅",
        "medical_genetics": "의학 유전학", "miscellaneous": "기타", "moral_disputes": "도덕적 논쟁",
        "moral_scenarios": "도덕적 시나리오", "nutrition": "영양학", "philosophy": "철학", "prehistory": "선사학",
        "professional_accounting": "전문 회계학", "professional_law": "전문 법학", "professional_medicine": "전문 의학",
        "professional_psychology": "전문 심리학", "public_relations": "홍보학", "security_studies": "보안학",
        "sociology": "사회학", "us_foreign_policy": "미국 외교정책", "virology": "바이러스학", "world_religions": "세계 종교학"
    }
    subject_display = subject_display_map.get(subject, subject.replace("_", " "))
    
    prompt_parts = [f"다음은 {subject_display}에 관한 객관식 문제입니다."]
    prompt_parts.append("")
    
    question = item.get("Question", "")
    prompt_parts.append(question)
    
    prompt_parts.append(f"A. {item.get('A', '선택지 A')}")
    prompt_parts.append(f"B. {item.get('B', '선택지 B')}")
    prompt_parts.append(f"C. {item.get('C', '선택지 C')}")
    prompt_parts.append(f"D. {item.get('D', '선택지 D')}")
    
    prompt_parts.append("정답:")
    
    return "\n".join(prompt_parts)

def extract_korean_answer_first_token(model_output):
    """
    Extracts answer from Korean model output using first token approach.
    """
    cleaned_output = model_output.strip().upper()
    for char in cleaned_output:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return None

def load_kmmlu_data(filepath):
    """Loads Korean MMLU data from a JSON file."""
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
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth_korean(item):
    """Returns the ground truth answer letter from the Korean MMLU data."""
    answer = item.get("Answer", None)
    if answer in ["A", "B", "C", "D"]:
        return answer
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
            extracted_answer = extract_korean_answer_first_token(generated_text)
            batch_results.append({
                'index': batch_indices[i],
                'raw_output': generated_text,
                'extracted_answer': extracted_answer
            })
        return batch_results
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=False)
        return [{'index': idx, 'raw_output': f"ERROR: {str(e)[:100]}", 'extracted_answer': None} for idx in batch_indices]

# --- Single Model Evaluation Function with 0-shot Prompting ---
def evaluate_single_model(config: ModelConfig, kmmlu_data: list, model_specific_output_dir: str):
    """
    Performs 0-shot Korean MMLU evaluation for a single model.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_0shot_korean.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_0shot_korean.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_0shot_korean.json")
    failure_cases_filepath = os.path.join(model_specific_output_dir, f"failure_cases_{config.name}_0shot_korean.json")

    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 0-shot Korean MMLU Evaluation for Model: {config.name} ({config.model_id}) ---")
    
    model, tokenizer = None, None
    try:
        tokenizer_load_path = config.adapter_path or config.model_id
        tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_load_path, 
                    cache_dir=CACHE_DIR,
                    padding_side='left',  # <--- 이 라인을 추가하세요!
                    trust_remote_code=True # Qwen 등 일부 모델은 이 옵션이 필요할 수 있습니다.
                )  
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=config.torch_dtype) if config.use_quantization else None
        model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=config.torch_dtype, quantization_config=quantization_config_bnb, device_map=DEVICE, trust_remote_code=True, cache_dir=CACHE_DIR)

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        if config.adapter_path:
            model = PeftModel.from_pretrained(model, config.adapter_path)
        
        model.eval()

        correct_predictions, total_predictions, errors_or_skipped = 0, 0, 0
        results_details, raw_generations_list = [], []
        failure_cases_list = []  # New: Track failure cases separately

        test_data = kmmlu_data
        pbar = tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Evaluating {config.name} (0-shot Korean, errors: 0)")
        for i in pbar:
            batch_data = test_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []

            for j, item in enumerate(batch_data):
                current_index = i + j
                ground_truth = get_ground_truth_korean(item)
                prompt = create_0shot_korean_prompt(item)
                
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

            for result, ground_truth in zip(batch_results, batch_ground_truths):
                generated_text_log = result['raw_output']
                model_answer_log = result['extracted_answer']
                is_correct_log = False

                if model_answer_log:
                    total_predictions += 1
                    if model_answer_log == ground_truth:
                        correct_predictions += 1
                        is_correct_log = True
                else:
                    errors_or_skipped += 1
                    original_generated_text = generated_text_log
                    if not generated_text_log.startswith("ERROR"):
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"
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
                        "choices": {
                            "A": original_item.get("A", ""),
                            "B": original_item.get("B", ""),
                            "C": original_item.get("C", ""),
                            "D": original_item.get("D", "")
                        }
                    })

                results_details.append({
                    "index": result['index'], "ground_truth": ground_truth, "model_raw_output": generated_text_log,
                    "predicted_answer": model_answer_log, "is_correct": is_correct_log
                })
                original_item = test_data[result['index']]
                raw_generations_list.append({
                    "index": result['index'], "subject": original_item.get("Subject", "unknown"), "ground_truth": ground_truth,
                    "raw_output": generated_text_log, "extracted_answer": model_answer_log
                })

            pbar.set_description(f"Evaluating {config.name} (0-shot Korean, errors: {errors_or_skipped})")

        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        accuracy_strict = (correct_predictions / len(test_data) * 100) if len(test_data) > 0 else 0

        logger.info(f"--- 0-shot KMMLU Results for {config.name} ---")
        logger.info(f"Test Items: {len(test_data)}")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")

        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "0-shot Korean MMLU",
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

    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if 'file_handler' in locals():
            root_logger.removeHandler(file_handler)
            file_handler.close()

def main():
    kmmlu_data = load_kmmlu_data(DATASET_PATH)
    if not kmmlu_data: return
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    all_results_summary = [] # For consolidated summary

    for config in MODEL_CONFIGS:
        model_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_dir, exist_ok=True)
        evaluate_single_model(config, kmmlu_data, model_dir)

    # --- Create a consolidated summary of all model results ---
    logger.info("--- Generating Consolidated Summary for KMMLU ---")
    for config in MODEL_CONFIGS:
        results_filepath = os.path.join(BASE_OUTPUT_DIR, config.name, f"results_{config.name}_0shot_korean.json")
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
