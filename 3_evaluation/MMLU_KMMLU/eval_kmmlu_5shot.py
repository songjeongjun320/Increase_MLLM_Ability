import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import re
from dataclasses import dataclass, field # Use dataclass for config
import gc # For garbage collection

# --- Model Configuration (Removed output_dir) ---
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

# --- General Configuration (Updated for 5-shot evaluation) ---
DATASET_PATH = "../../2_datasets/MMLU/KO_MMLU.json"
BASE_OUTPUT_DIR = "kmmlu_tow_model2_5shot" # Base dir for ALL model results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup (Configured per model later) ---
# Basic setup, will add file handlers per model
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler()] # Initially just log to console
)
logger = logging.getLogger(__name__) # Get root logger

# --- Helper Functions for 5-shot Korean MMLU Evaluation ---
def prepare_kmmlu_data_with_dev_split(data, dev_shots_per_subject=5):
    """
    Split Korean MMLU data into development (few-shot examples) and test sets.
    Uses first N examples per subject as development set.
    """
    subjects_data = {}
    
    # Group by subject
    for item in data:
        subject = item.get("Subject", "unknown")  # Korean MMLU uses 'Subject' field
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
    
    logger.info(f"Split Korean MMLU data: {len(dev_data)} subjects with dev examples, {len(test_data)} test items")
    return dev_data, test_data

def create_5shot_korean_prompt(test_item, dev_examples):
    """
    Create standard 5-shot Korean MMLU prompt using development examples.
    Follows Korean format: "다음은 [과목]에 관한 객관식 문제입니다."
    """
    subject = test_item.get("Subject", "unknown")  # Korean MMLU uses 'Subject' field
    
    # Format subject name for Korean display (comprehensive mapping for all 57 subjects)
    subject_display_map = {
        "abstract_algebra": "추상대수학",
        "anatomy": "해부학",
        "astronomy": "천문학",
        "business_ethics": "경영 윤리",
        "clinical_knowledge": "임상 지식",
        "college_biology": "대학 생물학",
        "college_chemistry": "대학 화학",
        "college_computer_science": "대학 컴퓨터 과학",
        "college_mathematics": "대학 수학",
        "college_medicine": "대학 의학",
        "college_physics": "대학 물리학",
        "computer_security": "컴퓨터 보안",
        "conceptual_physics": "개념 물리학",
        "econometrics": "계량경제학",
        "electrical_engineering": "전기공학",
        "elementary_mathematics": "초등 수학",
        "formal_logic": "형식 논리학",
        "global_facts": "세계 사실",
        "high_school_biology": "고등학교 생물학",
        "high_school_chemistry": "고등학교 화학",
        "high_school_computer_science": "고등학교 컴퓨터 과학",
        "high_school_european_history": "고등학교 유럽사",
        "high_school_geography": "고등학교 지리학",
        "high_school_government_and_politics": "고등학교 정치학",
        "high_school_macroeconomics": "고등학교 거시경제학",
        "high_school_mathematics": "고등학교 수학",
        "high_school_microeconomics": "고등학교 미시경제학",
        "high_school_physics": "고등학교 물리학",
        "high_school_psychology": "고등학교 심리학",
        "high_school_statistics": "고등학교 통계학",
        "high_school_us_history": "고등학교 미국사",
        "high_school_world_history": "고등학교 세계사",
        "human_aging": "인간 노화",
        "human_sexuality": "인간 성학",
        "international_law": "국제법",
        "jurisprudence": "법학",
        "logical_fallacies": "논리적 오류",
        "machine_learning": "기계학습",
        "management": "경영학",
        "marketing": "마케팅",
        "medical_genetics": "의학 유전학",
        "miscellaneous": "기타",
        "moral_disputes": "도덕적 논쟁",
        "moral_scenarios": "도덕적 시나리오",
        "nutrition": "영양학",
        "philosophy": "철학",
        "prehistory": "선사학",
        "professional_accounting": "전문 회계학",
        "professional_law": "전문 법학",
        "professional_medicine": "전문 의학",
        "professional_psychology": "전문 심리학",
        "public_relations": "홍보학",
        "security_studies": "보안학",
        "sociology": "사회학",
        "us_foreign_policy": "미국 외교정책",
        "virology": "바이러스학",
        "world_religions": "세계 종교학"
    }
    subject_display = subject_display_map.get(subject, subject.replace("_", " "))
    
    prompt_parts = [f"다음은 {subject_display}에 관한 객관식 문제(정답 포함)입니다."]
    prompt_parts.append("")  # Empty line
    
    # Add development examples (few-shot examples)
    for i, example in enumerate(dev_examples):
        question = example.get("Question", "")  # Korean MMLU uses 'Question' field
        
        # Extract answer letter directly from Korean MMLU format
        answer_letter = example.get("Answer", "A")  # Already in letter format
        
        prompt_parts.append(question)
        
        # Get choices from Korean MMLU format (A, B, C, D fields)
        choice_a = example.get("A", "선택지 A")
        choice_b = example.get("B", "선택지 B")
        choice_c = example.get("C", "선택지 C")
        choice_d = example.get("D", "선택지 D")
        
        prompt_parts.append(f"A. {choice_a}")
        prompt_parts.append(f"B. {choice_b}")
        prompt_parts.append(f"C. {choice_c}")
        prompt_parts.append(f"D. {choice_d}")
        
        prompt_parts.append(f"정답: {answer_letter}")
        prompt_parts.append("")  # Empty line between examples
    
    # Add test question
    test_question = test_item.get("Question", "")  # Korean MMLU uses 'Question' field
    prompt_parts.append(test_question)
    
    # Get choices for test question from Korean MMLU format
    test_choice_a = test_item.get("A", "선택지 A")
    test_choice_b = test_item.get("B", "선택지 B")
    test_choice_c = test_item.get("C", "선택지 C")
    test_choice_d = test_item.get("D", "선택지 D")
    
    prompt_parts.append(f"A. {test_choice_a}")
    prompt_parts.append(f"B. {test_choice_b}")
    prompt_parts.append(f"C. {test_choice_c}")
    prompt_parts.append(f"D. {test_choice_d}")
    
    prompt_parts.append("정답:")
    
    return "\n".join(prompt_parts)

def parse_korean_choices_from_question(question):
    """
    Parse A, B, C, D choices from Korean MMLU question format.
    Korean format embeds choices within the question text with numbers.
    """
    import re
    
    # Look for numbered choices in the question (1. 2. 3. 4.)
    choice_pattern = r'(\d+)\.\s*([^\n\d]+?)(?=\n\d+\.|$)'
    matches = re.findall(choice_pattern, question, re.MULTILINE | re.DOTALL)
    
    choices = []
    for i, (num, text) in enumerate(matches):
        if i < 4:  # Only take first 4 choices
            letter = chr(ord('A') + i)
            # Clean up the choice text
            clean_text = text.strip().replace('\n', ' ').strip()
            choices.append(f"{letter}. {clean_text}")
    
    # If we couldn't parse choices, return placeholder
    if len(choices) < 4:
        choices = ["A. 선택지 A", "B. 선택지 B", "C. 선택지 C", "D. 선택지 D"]
    
    return choices

def extract_korean_answer_first_token(model_output, tokenizer):
    """
    Extract answer from Korean model output using first token approach.
    This follows the standard MMLU evaluation methodology adapted for Korean.
    """
    # Clean and normalize output
    cleaned_output = model_output.strip().upper()
    
    # Look for first letter that is A, B, C, or D
    for char in cleaned_output:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return None

def load_mmlu_data(filepath):
    """JSON 파일에서 KMMLU 데이터를 로드합니다."""
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
def create_prompt_legacy(item):
    """MMLU 항목에 대한 프롬프트를 생성합니다. [LEGACY - 0-shot]"""
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

# Legacy answer extraction function (replaced by first-token approach)
def extract_answer_legacy(model_output, prompt):
    """모델 출력에서 답변(A, B, C, D)을 추출합니다. [LEGACY]"""
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
    주어진 설정의 단일 모델에 대해 5-shot Korean MMLU 평가를 수행하고,
    결과와 로그를 base_output_dir 아래 모델 이름의 하위 디렉토리에 저장합니다.
    """
    # Split data into development (few-shot examples) and test sets
    dev_data, test_data = prepare_kmmlu_data_with_dev_split(mmlu_data, dev_shots_per_subject=5)
    
    if not test_data:
        logger.error("No test data available after dev/test split. Check data size and dev_shots_per_subject setting.")
        return

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
        # 1. Determine the correct path for the tokenizer.
        # If an adapter is used, the updated tokenizer is saved with it.
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

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
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
                # Optional: Merge the adapter for faster inference
                # model = model.merge_and_unload()
                # logger.info("LoRA adapter merged into the base model.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            logger.info("No LoRA adapter path specified. Using the base model directly.")
        # === END: LoRA Adapter Loading Logic ===

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
        logger.info("Starting 5-shot Korean MMLU inference loop...")
        logger.info(f"Test data size: {len(test_data)}")
        for i, item in enumerate(tqdm(test_data, desc=f"Evaluating {config.name} (5-shot Korean)")):
            # Extract ground truth from Korean MMLU format (already in letter format)
            ground_truth = item.get("Answer", None)  # Korean MMLU already has letter format (A, B, C, D)
            
            # Validate ground truth
            if not ground_truth or ground_truth not in ["A", "B", "C", "D"]:
                ground_truth = None
            
            if not ground_truth:
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

            # Get development examples for this subject
            subject = item.get("Subject", "unknown")  # Korean MMLU uses 'Subject' field
            dev_examples = dev_data.get(subject, [])
            
            if not dev_examples:
                logger.warning(f"No development examples available for subject: {subject}")
                prompt = None
            else:
                prompt = create_5shot_korean_prompt(item, dev_examples)
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

            # Generate exactly one token for standard MMLU evaluation
            max_gen_tokens = 1

            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE) # Added truncation

            generated_text = "ERROR - Inference Failed" # Default in case of error
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens, # Generate only first token for standard MMLU evaluation
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False, # Deterministic output
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
                # Extract answer using first-token approach
                model_answer = extract_korean_answer_first_token(generated_text, tokenizer)

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
            if (i + 1) % 100 == 0: # Log every 100 items for smaller test sets
                 current_acc = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                 logger.info(f"Progress ({config.name}): {i + 1}/{len(test_data)}, Accuracy: {current_acc:.2f}% ({correct_predictions}/{total_predictions}), Errors: {errors}")


        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        # Recalculate totals based on loop results
        total_processed_loop = len(results_details)
        valid_pred_loop = sum(1 for r in results_details if r['predicted_answer'] is not None and r['model_raw_output'] not in ["SKIPPED - Invalid Ground Truth", "SKIPPED - Prompt Creation Failed", "ERROR - Inference Failed"])
        correct_pred_loop = sum(1 for r in results_details if r['is_correct'])
        errors_loop = sum(1 for r in results_details if r['predicted_answer'] is None and r['model_raw_output'] not in ["SKIPPED - Invalid Ground Truth", "SKIPPED - Prompt Creation Failed"])
        skipped_loop = sum(1 for r in results_details if r['model_raw_output'].startswith("SKIPPED"))
        
        # Calculate two types of accuracy
        accuracy_standard = (correct_pred_loop / valid_pred_loop * 100) if valid_pred_loop > 0 else 0  # correct / valid_predictions
        accuracy_strict = (correct_pred_loop / total_processed_loop * 100) if total_processed_loop > 0 else 0  # correct / total_test_items (including skipped/errors)

        # --- Calculate Category-wise Accuracy ---
        subject_stats = {}
        for i, item in enumerate(test_data):
            subject = item.get("Subject", "unknown")
            result = results_details[i]
            
            if subject not in subject_stats:
                subject_stats[subject] = {
                    "total": 0,
                    "correct": 0,
                    "valid_predictions": 0,
                    "accuracy": 0.0
                }
            
            subject_stats[subject]["total"] += 1
            if result['predicted_answer'] is not None and result['model_raw_output'] not in ["SKIPPED - Invalid Ground Truth", "SKIPPED - Prompt Creation Failed", "ERROR - Inference Failed"]:
                subject_stats[subject]["valid_predictions"] += 1
                if result['is_correct']:
                    subject_stats[subject]["correct"] += 1
        
        # Calculate accuracy for each subject
        for subject in subject_stats:
            if subject_stats[subject]["valid_predictions"] > 0:
                subject_stats[subject]["accuracy"] = (subject_stats[subject]["correct"] / subject_stats[subject]["valid_predictions"]) * 100


        logger.info(f"--- 5-shot Korean MMLU Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Original Dataset Size: {len(mmlu_data)}")
        logger.info(f"Test Items (after dev/test split): {len(test_data)}")
        logger.info(f"Development Examples per Subject: 5")
        logger.info(f"Items Processed (Attempted): {total_processed_loop}")
        logger.info(f"Valid Predictions Made: {valid_pred_loop}")
        logger.info(f"Correct Predictions: {correct_pred_loop}")
        logger.info(f"Errors (Inference/Extraction Failures): {errors_loop}")
        logger.info(f"Items Skipped (Invalid GT/Prompt): {skipped_loop}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_test_items): {accuracy_strict:.2f}%")

        # --- Save Results ---
        final_summary = {
            "model_config": config.__dict__, # Save config used
            "dataset_path": DATASET_PATH,
            "evaluation_type": "5-shot Korean MMLU",
            "total_original_items": len(mmlu_data),
            "dev_examples_per_subject": 5,
            "test_items": len(test_data),
            "items_processed": total_processed_loop,
            "valid_predictions": valid_pred_loop,
            "correct_predictions": correct_pred_loop,
            "errors_or_failures": errors_loop,
            "items_skipped": skipped_loop,
            "accuracy_standard (correct / valid_predictions)": accuracy_standard,
            "accuracy_strict (correct / total_test_items)": accuracy_strict,
            "subjects_with_dev_examples": list(dev_data.keys()),
            "subject_wise_accuracy": subject_stats,  # Category-wise accuracy
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