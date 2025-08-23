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
DATASET_PATH = "../../DB/MMLU/MMLU_origin.json"
BASE_OUTPUT_DIR = "kmmlu_tow_model1_zeroshot" # 0-shot evaluation results
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
    """Loads KMMLU data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth_korean(item):
    """Returns the ground truth answer letter from the Korean MMLU data."""
    answer = item.get("Answer", None)
    if answer in ["A", "B", "C", "D"]:
        return answer
    return None

# --- Single Model Evaluation Function with 0-shot Prompting ---
def evaluate_single_model(config: ModelConfig, kmmlu_data: list, model_specific_output_dir: str):
    """
    Performs 0-shot Korean MMLU evaluation for a single model.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_0shot_korean.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_0shot_korean.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_0shot_korean.json")

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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
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

        test_data = kmmlu_data
        for i, item in enumerate(tqdm(test_data, desc=f"Evaluating {config.name} (0-shot Korean)")):
            ground_truth = get_ground_truth_korean(item)
            prompt = create_0shot_korean_prompt(item)
            
            model_answer_log, is_correct_log = None, False
            if ground_truth is None or prompt is None:
                errors_or_skipped += 1
                generated_text_log = "SKIPPED"
            else:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(DEVICE)
                try:
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id, do_sample=False)
                    generated_text_log = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    model_answer_log = extract_korean_answer_first_token(generated_text_log)
                    
                    if model_answer_log:
                        total_predictions += 1
                        if model_answer_log == ground_truth:
                            correct_predictions += 1
                            is_correct_log = True
                    else:
                        errors_or_skipped += 1
                except Exception as e:
                    errors_or_skipped += 1
                    generated_text_log = f"ERROR: {e}"

            results_details.append({"index": i, "ground_truth": ground_truth, "model_raw_output": generated_text_log, "predicted_answer": model_answer_log, "is_correct": is_correct_log})
            raw_generations_list.append({"index": i, "subject": item.get("Subject"), "ground_truth": ground_truth, "raw_output": generated_text_log, "extracted_answer": model_answer_log})

        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        logger.info(f"--- 0-shot KMMLU Results for {config.name} ---")
        logger.info(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

        final_summary = {"model_config": {k: str(v) for k, v in config.__dict__.items()}, "evaluation_type": "0-shot Korean MMLU", "accuracy": accuracy, "details": results_details}
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)

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
    for config in MODEL_CONFIGS:
        model_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_dir, exist_ok=True)
        evaluate_single_model(config, kmmlu_data, model_dir)

if __name__ == "__main__":
    main()
