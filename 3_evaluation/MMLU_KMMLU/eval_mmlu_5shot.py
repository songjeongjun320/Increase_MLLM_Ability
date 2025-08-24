import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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
    adapter_path: str = None              # Path to the LoRA adapter
    use_quantization: bool = True         # Default to quantization, especially for larger models
    # Default dtype, can be overridden per model if needed
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
BASE_OUTPUT_DIR = "mmlu_tow_model2_5shot" # 5-shot evaluation results
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions for 5-shot MMLU Evaluation ---
def prepare_mmlu_data_with_dev_split(data, dev_shots_per_subject=5):
    """
    Split MMLU data into development (few-shot examples) and test sets.
    Uses first N examples per subject as development set.
    """
    subjects_data = {}
    
    # Group by subject
    for item in data:
        subject = item.get("subject", "unknown")
        if subject not in subjects_data:
            subjects_data[subject] = []
        subjects_data[subject].append(item)
    
    dev_data = {}
    test_data = []
    
    for subject, items in subjects_data.items():
        if len(items) < dev_shots_per_subject:
            logger.warning(f"Subject {subject} has only {len(items)} items, less than required {dev_shots_per_subject} dev examples")
            # Use all available items as dev examples, no test items for this subject
            dev_data[subject] = items
        else:
            # First N items as dev examples
            dev_data[subject] = items[:dev_shots_per_subject]
            # Remaining items as test
            test_data.extend(items[dev_shots_per_subject:])
    
    logger.info(f"Split data: {len(dev_data)} subjects with dev examples, {len(test_data)} test items")
    return dev_data, test_data

def create_5shot_prompt(test_item, dev_examples):
    """
    Create standard 5-shot MMLU prompt using development examples.
    Follows the format: "The following are multiple choice questions (with answers) about [subject]."
    """
    subject = test_item.get("subject", "unknown")
    
    # Format subject name for display (replace underscores with spaces, capitalize)
    subject_display = subject.replace("_", " ").title()
    
    prompt_parts = [f"The following are multiple choice questions (with answers) about {subject_display}."]
    prompt_parts.append("")  # Empty line
    
    # Add development examples (few-shot examples)
    for i, example in enumerate(dev_examples):
        question = example.get("question", "")
        # Get choices from the standard MMLU format
        choices = example.get("choices", [])
        if not choices:
            logger.warning(f"No choices found for example in subject {subject}")
            choices = ["Option A", "Option B", "Option C", "Option D"]
        
        answer_idx = example.get("answer", 0)
        if isinstance(answer_idx, int):
            answer_letter = chr(ord('A') + answer_idx)
        else:
            answer_letter = str(answer_idx).upper()
        
        prompt_parts.append(question)
        if len(choices) >= 4:
            prompt_parts.append(f"A. {choices[0]}")
            prompt_parts.append(f"B. {choices[1]}")
            prompt_parts.append(f"C. {choices[2]}")
            prompt_parts.append(f"D. {choices[3]}")
        else:
            logger.warning(f"Insufficient choices for example in subject {subject}")
            prompt_parts.extend(["A. Option A", "B. Option B", "C. Option C", "D. Option D"])
        
        prompt_parts.append(f"Answer: {answer_letter}")
        prompt_parts.append("")  # Empty line between examples
    
    # Add test question
    test_question = test_item.get("question", "")
    prompt_parts.append(test_question)
    
    # Parse choices for test question
    test_choices = test_item.get("choices", [])
    if len(test_choices) >= 4:
        prompt_parts.append(f"A. {test_choices[0]}")
        prompt_parts.append(f"B. {test_choices[1]}")
        prompt_parts.append(f"C. {test_choices[2]}")
        prompt_parts.append(f"D. {test_choices[3]}")
    else:
        logger.warning(f"Insufficient choices for test question in subject {subject}")
        prompt_parts.extend(["A. Option A", "B. Option B", "C. Option C", "D. Option D"])
    
    prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)

def parse_choices_from_question(question):
    """
    Parse A, B, C, D choices from Korean MMLU question format.
    Korean format embeds choices within the question text with numbers.
    """
    import re
    
    # Look for numbered choices in the question
    choice_pattern = r'(\d+)\.\s*([^\n]+?)(?=\n\d+\.|$)'
    matches = re.findall(choice_pattern, question, re.MULTILINE)
    
    choices = []
    for i, (num, text) in enumerate(matches):
        if i < 4:  # Only take first 4 choices
            letter = chr(ord('A') + i)
            choices.append(f"{letter}. {text.strip()}")
    
    # If we couldn't parse choices, return placeholder
    if len(choices) < 4:
        choices = ["A. Option A", "B. Option B", "C. Option C", "D. Option D"]
    
    return choices

def extract_answer_first_token(model_output, tokenizer):
    """
    Extract answer from model output using first token approach.
    This follows the standard MMLU evaluation methodology.
    """
    # Clean and normalize output
    cleaned_output = model_output.strip().upper()
    
    # Look for first letter that is A, B, C, or D
    for char in cleaned_output:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return None

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

# Legacy 0-shot prompt function (replaced by 5-shot version)
def create_prompt_for_origin_legacy(item):
    """원본 MMLU 형식('question', 'choices' 리스트)에 맞는 프롬프트를 생성합니다. [LEGACY - 0-shot]"""
    question = item.get("question", "")
    choices_list = item.get("choices", [])
    if not question or not isinstance(choices_list, list) or len(choices_list) != 4:
        return None

    choices_dict = {chr(ord('A') + i): choice for i, choice in enumerate(choices_list)}
    prompt = f"""Question: {question}
A) {choices_dict.get('A', '')}
B) {choices_dict.get('B', '')}
C) {choices_dict.get('D', '')}
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

# Legacy answer extraction function (replaced by first-token approach)
def extract_answer_legacy(model_output, prompt): # prompt 인자 유지 (미래 사용 가능성)
    """모델 출력에서 답변(A, B, C, D)을 추출합니다. [LEGACY]"""
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

# --- Batch Processing Function ---
def process_batch(model, tokenizer, batch_prompts, batch_indices):
    """Processes a batch of prompts efficiently."""
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        
        batch_results = []
        input_length = inputs['input_ids'].shape[1]
        for i, sequence in enumerate(outputs):
            output_tokens = sequence[input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            extracted_answer = extract_answer_first_token(generated_text, tokenizer)
            batch_results.append({
                'index': batch_indices[i],
                'raw_output': generated_text,
                'extracted_answer': extracted_answer
            })
        return batch_results
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=False)
        return [{'index': idx, 'raw_output': f"ERROR: {str(e)[:100]}", 'extracted_answer': None} for idx in batch_indices]

# --- Single Model Evaluation Function with 5-shot Prompting ---
def evaluate_single_model(config: ModelConfig, mmlu_data: list, model_specific_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 5-shot MMLU 평가를 수행하고,
    결과와 로그를 model_specific_output_dir에 저장합니다.
    """
    # Split data into development (few-shot examples) and test sets
    dev_data, test_data = prepare_mmlu_data_with_dev_split(mmlu_data, dev_shots_per_subject=5)
    
    if not test_data:
        logger.error("No test data available after dev/test split. Check data size and dev_shots_per_subject setting.")
        return
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
        # 1. Determine the correct path for the tokenizer.
        # If an adapter is used, the updated tokenizer is saved with it.
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_load_path, 
                    cache_dir=CACHE_DIR,
                    padding_side='left',  # <--- 이 라인을 추가하세요!
                    trust_remote_code=True # Qwen 등 일부 모델은 이 옵션이 필요할 수 있습니다.
                )          
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

        # 3. Resize model embeddings to match the tokenizer's vocabulary size BEFORE loading the adapter.
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # 4. Load the LoRA adapter onto the correctly-sized base model.
        if config.adapter_path:
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")
            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")
            
            try:
                # Load the PEFT model by applying the adapter to the base model
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
                # If you want to merge the adapter into the model for faster inference:
                # model = model.merge_and_unload()
                # logger.info("LoRA adapter merged into the base model.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            logger.info("No LoRA adapter path specified. Using the base model directly.")

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

        logger.info("Starting 5-shot inference loop...")
        logger.info(f"Test data size: {len(test_data)}")
        
        pbar = tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Evaluating {config.name} (5-shot, errors: 0)")
        for i in pbar:
            batch_data = test_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            batch_original_items = []

            for j, item in enumerate(batch_data):
                current_index = i + j
                item_index_for_log = item.get("index", current_index)
                ground_truth = get_ground_truth_origin(item)
                subject = item.get("subject", "unknown")
                dev_examples = dev_data.get(subject, [])
                
                prompt = create_5shot_prompt(item, dev_examples) if dev_examples else None

                if ground_truth is None or prompt is None:
                    errors_or_skipped += 1
                    output_reason = "SKIPPED - Invalid Ground Truth" if ground_truth is None else "SKIPPED - Prompt Creation Failed"
                    results_details.append({
                        "index": item_index_for_log, "ground_truth": ground_truth, "model_raw_output": output_reason,
                        "predicted_answer": None, "is_correct": False
                    })
                    raw_generations_list.append({
                        "index": item_index_for_log, "subject": subject, "ground_truth": ground_truth,
                        "raw_output": output_reason, "extracted_answer": None
                    })
                    continue
                
                batch_prompts.append(prompt)
                batch_indices.append(item_index_for_log)
                batch_ground_truths.append(ground_truth)
                batch_original_items.append(item)

            if not batch_prompts:
                continue

            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)

            for result, ground_truth, original_item in zip(batch_results, batch_ground_truths, batch_original_items):
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
                    if not generated_text_log.startswith("ERROR"):
                        generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                results_details.append({
                    "index": result['index'], "ground_truth": ground_truth, "model_raw_output": generated_text_log,
                    "predicted_answer": model_answer_log, "is_correct": is_correct_log
                })
                raw_generations_list.append({
                    "index": result['index'], "subject": original_item.get("subject", "unknown"), "ground_truth": ground_truth,
                    "raw_output": generated_text_log, "extracted_answer": model_answer_log
                })

            pbar.set_description(f"Evaluating {config.name} (5-shot, errors: {errors_or_skipped})")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        
        # Calculate two types of accuracy
        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0  # correct / valid_predictions
        accuracy_strict = (correct_predictions / len(test_data) * 100) if len(test_data) > 0 else 0  # correct / total_test_items (including skipped/errors)

        # --- Calculate Category-wise Accuracy ---
        subject_stats = {}
        for i, item in enumerate(test_data):
            subject = item.get("subject", "unknown")
            result = results_details[i]
            
            if subject not in subject_stats:
                subject_stats[subject] = {
                    "total": 0,
                    "correct": 0,
                    "valid_predictions": 0,
                    "accuracy": 0.0
                }
            
            subject_stats[subject]["total"] += 1
            if result['predicted_answer'] is not None and not result['model_raw_output'].startswith(("SKIPPED", "ERROR")):
                subject_stats[subject]["valid_predictions"] += 1
                if result['is_correct']:
                    subject_stats[subject]["correct"] += 1
        
        # Calculate accuracy for each subject
        for subject in subject_stats:
            if subject_stats[subject]["valid_predictions"] > 0:
                subject_stats[subject]["accuracy"] = (subject_stats[subject]["correct"] / subject_stats[subject]["valid_predictions"]) * 100

        logger.info(f"--- 5-shot MMLU Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Original Dataset Size: {len(mmlu_data)}")
        logger.info(f"Test Items (after dev/test split): {len(test_data)}")
        logger.info(f"Development Examples per Subject: 5")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_test_items): {accuracy_strict:.2f}%")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "evaluation_type": "5-shot MMLU",
            "total_original_items": len(mmlu_data),
            "dev_examples_per_subject": 5,
            "test_items": len(test_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors_or_skipped,
            "accuracy_standard (correct / valid_predictions)": accuracy_standard,
            "accuracy_strict (correct / total_test_items)": accuracy_strict,
            "subjects_with_dev_examples": list(dev_data.keys()),
            "subject_wise_accuracy": subject_stats,  # Category-wise accuracy
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

    # --- Create a consolidated summary of all model results ---
    logger.info("--- Generating Consolidated Summary ---")
    all_results_summary = []
    for config in MODEL_CONFIGS:
        results_filepath = os.path.join(BASE_OUTPUT_DIR, config.name, f"results_{config.name}.json")
        if os.path.exists(results_filepath):
            try:
                with open(results_filepath, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                summary = {
                    "model_name": config.name,
                    "accuracy_standard": result_data.get("accuracy_standard (correct / valid_predictions)"),
                    "accuracy_strict": result_data.get("accuracy_strict (correct / total_test_items)"),
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
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()