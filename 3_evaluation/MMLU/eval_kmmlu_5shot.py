import os
import json
import logging
import torch
import warnings 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import re
from dataclasses import dataclass, field # Use dataclass for config
import gc # For garbage collection
from datetime import datetime
import time
import random

# Import ToW token checker
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from check_tokenizer import check_tow_tokens_for_eval

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")

# Import performance analyzer
try:
    import sys
    sys.path.append('../')
    from performance_analyzer import create_enhanced_summary
except ImportError:
    logger.warning("Performance analyzer not available. Using basic summary.")
    create_enhanced_summary = None

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
    #     name="llama-3.2-3b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="qwem-2.5-3b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="olmo-2-0425-1b",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/olmo-2-0425-1b",
    #     use_quantization=False
    # ),

    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="olmo-2-0425-1b-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),

    ModelConfig(
        name="llama-3.2-3b-tow-09_11_2epoch_org_initialize-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-tow-09_11_2epoch_org_initialize-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt-tow-09_11_2epoch_org_initialize-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_2epoch_org_initialize-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt-tow-09_11_2epoch_org_initialize-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_2epoch_org_initialize-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="olmo-2-0425-1b-tow-09_11_2epoch_org_initialize-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_org_initialize-merged",
        use_quantization=False
    ),

    ModelConfig(
        name="llama-3.2-3b-tow-09_11_2epoch_fix_tow-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-tow-09_11_2epoch_fix_tow-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-tow-09_11_2epoch_fix_tow-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-tow-09_11_2epoch_fix_tow-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-tow-09_11_2epoch_fix_tow-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-tow-09_11_2epoch_fix_tow-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="olmo-2-0425-1b-tow-09_11_2epoch_fix_tow-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_fix_tow-merged",
        use_quantization=False
    ),
]

# --- General Configuration (Updated for 5-shot evaluation) ---
DATASET_PATH = "../../2_datasets/MMLU/MMLU_KO_Openai.json"
BASE_OUTPUT_DIR = "kmmlu_5shot_results_tokenizer_added" # Base dir for ALL model results
BATCH_SIZE = 16
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
        subject = item.get("Subject", "unknown")  # Korean MMLU uses 'Subject' field (uppercase)
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

# --- Generic 5-shot Korean examples ---
GENERIC_5_SHOT_KOREAN_EXAMPLES = [
    {
        "question": "대중사회이론의 주장으로 옳은 것을 고르시오.",
        "choices": {
            "A": "미디어 내용은 시장의 힘에 의해 결정된다",
            "B": "하층계급은 지배계급의 이데올로기 지배를 받는다", 
            "C": "미디어는 '대중'을 취약하고 수동적인 소비자로 조작한다",
            "D": "청중은 미디어 메시지를 선택적으로 해석한다"
        },
        "answer": "C"
    },
    {
        "question": "2011년 물가를 기준으로 인플레이션과 구매력평가(PPP)를 조정했을 때 1850년 미국의 1인당 GDP?",
        "choices": {
            "A": "약 $300",
            "B": "약 $3000",
            "C": "약 $8000", 
            "D": "약 $15000"
        },
        "answer": "B"
    },
    {
        "question": "로비스트들이 국회의원들을 설득하는 방법에 해당하지 않는 것은 어느 것입니까?",
        "choices": {
            "A": "미디어에의 선전물 배포",
            "B": "공직 후보자 지지",
            "C": "정부 관료들과의 교류",
            "D": "후보자를 위한 기업 선거운동 기부금 획득"
        },
        "answer": "D"
    },
    {
        "question": "전위차법의 DC 전압 측정은 전압계를 사용한 직접 측정보다 더 정확한데, 다음 중 그 이유는 무엇입니까?",
        "choices": {
            "A": "회로에 적절한 부하를 주기 때문이다.",
            "B": "회로에 최대한의 부하를 주기 때문이다.",
            "C": "전압계 대신 센터 제로 검류계를 사용하기 때문이다.",
            "D": "회로에 부하를 전혀 주지 않기 때문이다."
        },
        "answer": "D"
    },
    {
        "question": "밀턴 프리드먼은 기업의 유일한 책임이 무엇이라고 생각하나요?",
        "choices": {
            "A": "기업의 유일한 사회적 책임은 주주에 대한 것이다.",
            "B": "관리자는 사회와 주주의 이익이 균형을 이루는 방식으로 행동해야 한다.",
            "C": "조직이 갖는 주요 책임은 직원에 대한 것이다.",
            "D": "조직이 갖는 주요 책임은 주주에 대한 것이다."
        },
        "answer": "A"
    }
]

def create_generic_5shot_korean_prompt(test_item):
    """
    Create improved 5-shot Korean MMLU prompt using generic examples from different subjects.
    """
    subject = test_item.get("Subject", "unknown")
    subject_display = subject.replace("_", " ").title()
    
    prompt_parts = [f"다음은 다양한 주제({subject_display} 포함)에 대한 객관식 문제입니다."]
    prompt_parts.append("")  # Empty line
    
    # Add the 5 generic Korean examples
    for i, example in enumerate(GENERIC_5_SHOT_KOREAN_EXAMPLES):
        prompt_parts.append(f"문제: {example['question']}")
        prompt_parts.append(f"A. {example['choices']['A']}")
        prompt_parts.append(f"B. {example['choices']['B']}")
        prompt_parts.append(f"C. {example['choices']['C']}")
        prompt_parts.append(f"D. {example['choices']['D']}")
        prompt_parts.append(f"정답: #### 따라서 정답은 {{{example['answer']}}}. #### {{{example['answer']}}}")
        prompt_parts.append("")  # Empty line between examples
    
    # Add test question
    test_question = test_item.get("Question", "")
    test_choice_a = test_item.get("A", "선택지 A")
    test_choice_b = test_item.get("B", "선택지 B") 
    test_choice_c = test_item.get("C", "선택지 C")
    test_choice_d = test_item.get("D", "선택지 D")
    
    prompt_parts.append(f"문제: {test_question}")
    prompt_parts.append(f"A. {test_choice_a}")
    prompt_parts.append(f"B. {test_choice_b}")
    prompt_parts.append(f"C. {test_choice_c}")
    prompt_parts.append(f"D. {test_choice_d}")
    
    # Clear instructions for answer format
    prompt_parts.append("")
    prompt_parts.append("정답을 A, B, C, D 중 오직 하나의 문자로만 선택해주세요.")
    prompt_parts.append("정답: 따라서 정답은")
    
    return "\n".join(prompt_parts)

def extract_korean_answer_first_token(model_output, tokenizer):
    """
    Extract answer from model output using STRICT validation.
    STRICT MODE: Only accepts {} format - unified across all evaluation scripts.
    """
    # Clean and normalize output
    cleaned_output = model_output.strip().upper()

    import re

    # STRICT: Only accept {} format for consistency across all evaluation scripts
    box_pattern = r'\{([A-D])\}'
    box_matches = re.findall(box_pattern, cleaned_output)
    if box_matches:
        return box_matches[0]  # Use last match (final answer)

    # No fallback patterns - forces models to use {} format only
    return None

def load_kmmlu_data(filepath):
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

def get_ground_truth_origin(item):
    """KMMLU 데이터에서 정답 문자를 반환합니다."""
    answer = item.get("Answer", None)
    if answer and answer.upper() in ["A", "B", "C", "D"]:
        return answer.upper()
    return None

# Update process_batch function to use 1024 max_length and Korean extraction
def process_batch(model, tokenizer, batch_prompts, batch_indices):
    """Processes a batch of prompts efficiently."""
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Reduced from 2048
        ).to(DEVICE)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        
        batch_results = []
        input_length = inputs['input_ids'].shape[1]
        for i, sequence in enumerate(outputs):
            output_tokens = sequence[input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            extracted_answer = extract_korean_answer_first_token(generated_text, tokenizer)
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
                max_length=2048
            ).to(DEVICE)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            
            input_length = inputs['input_ids'].shape[1]
            output_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            extracted_answer = extract_korean_answer_first_token(generated_text, tokenizer)
            
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


# --- Single Model Evaluation Function (Uses BASE_OUTPUT_DIR) ---
# Modified evaluation function for Korean MMLU with generic 5-shot
def evaluate_single_model(config: ModelConfig, mmlu_data: list, base_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 generic 5-shot Korean MMLU 평가를 수행하고,
    결과와 로그를 base_output_dir 아래 모델 이름의 하위 디렉토리에 저장합니다.
    """
    # Remove the data splitting part since we're using all data as test data
    test_data = mmlu_data
    
    if not test_data:
        logger.error("No test data available.")
        return

    # Sampling for testing (uncomment if needed)
    # test_data = test_data[:50]

    # Construct model-specific output directory and file paths
    model_output_dir = os.path.join(base_output_dir, config.name)
    os.makedirs(model_output_dir, exist_ok=True)
    results_filepath = os.path.join(model_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_output_dir, f"raw_generations_{config.name}.json")
    failure_cases_filepath = os.path.join(model_output_dir, f"failure_cases_{config.name}.json")

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
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Logs will be saved to: {log_filepath}")
    logger.info(f"Raw generations will be saved to: {raw_gen_filepath}")
    logger.info(f"Failure cases will be saved to: {failure_cases_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    
    try:
        # --- Load Model and Tokenizer (same as before) ---
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_load_path, 
                    cache_dir=CACHE_DIR,
                    padding_side='left',
                    trust_remote_code=True
                )

        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                logger.info("Tokenizer does not have a pad token, setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding a new pad token '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # === TOKENIZER VERIFICATION ===
        tokenizer_status = check_tow_tokens_for_eval(
            tokenizer=tokenizer,
            model_path=tokenizer_load_path,
            model_name=config.name,
            logger=logger
        )

        if not tokenizer_status.is_valid:
            logger.warning(f"⚠️ ToW tokens not properly configured for {config.name}")
            for issue in tokenizer_status.issues:
                logger.warning(f"   - {issue}")
        # ===============================

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

        if config.adapter_path:
            # LoRA 어댑터가 있는 경우, 먼저 LoRA의 실제 vocab size를 확인
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")
            
            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")
            
            # LoRA 어댑터의 실제 vocab size 확인
            try:
                import glob
                pytorch_files = glob.glob(os.path.join(absolute_adapter_path, "*.bin")) + \
                            glob.glob(os.path.join(absolute_adapter_path, "*.safetensors"))
                
                target_vocab_size = None
                if pytorch_files:
                    if pytorch_files[0].endswith('.safetensors'):
                        from safetensors import safe_open
                        with safe_open(pytorch_files[0], framework="pt") as f:
                            for key in f.keys():
                                if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                                    target_vocab_size = f.get_tensor(key).shape[0]
                                    break
                    else:
                        checkpoint = torch.load(pytorch_files[0], map_location='cpu')
                        for key, tensor in checkpoint.items():
                            if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                                target_vocab_size = tensor.shape[0]
                                break
                
                if target_vocab_size:
                    current_vocab_size = model.get_input_embeddings().weight.shape[0]
                    if current_vocab_size != target_vocab_size:
                        logger.info(f"Resizing model from {current_vocab_size} to {target_vocab_size} for LoRA compatibility")
                        model.resize_token_embeddings(target_vocab_size)
                else:
                    # fallback: tokenizer 길이로 리사이즈
                    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                        model.resize_token_embeddings(len(tokenizer))
                        
            except Exception as e:
                logger.warning(f"Could not determine LoRA vocab size: {e}. Using tokenizer length.")
                if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                    model.resize_token_embeddings(len(tokenizer))
            
            try:
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            # 베이스 모델인 경우 기존 로직 사용
            if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
                model.resize_token_embeddings(len(tokenizer))
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

        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors = 0
        results_details = []
        raw_generations_list = []
        failure_cases_list = []

        logger.info("Starting generic 5-shot Korean MMLU inference loop...")
        logger.info(f"Test data size: {len(test_data)}")
        
        pbar = tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Evaluating {config.name} (generic 5-shot Korean, errors: 0)")
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
                
                # Use generic 5-shot Korean prompt (no need for subject-specific dev examples)
                prompt = create_generic_5shot_korean_prompt(item)

                if ground_truth is None or prompt is None:
                    errors += 1
                    output_reason = "SKIPPED - Invalid Ground Truth" if ground_truth is None else "SKIPPED - Prompt Creation Failed"
                    failure_type = "invalid_ground_truth" if ground_truth is None else "prompt_creation_failed"
                    
                    results_details.append({
                        "index": item_index_for_log, "ground_truth": ground_truth, "model_raw_output": output_reason,
                        "predicted_answer": None, "is_correct": False
                    })
                    raw_generations_list.append({
                        "index": item_index_for_log, "subject": item.get("Subject", "unknown"), "ground_truth": ground_truth,
                        "raw_output": output_reason, "extracted_answer": None
                    })
                    
                    failure_cases_list.append({
                        "index": item_index_for_log,
                        "subject": item.get("Subject", "unknown"),
                        "question": item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": failure_type,
                        "failure_reason": output_reason,
                        "raw_output": output_reason
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
                model_answer = result['extracted_answer']
                generated_text = result['raw_output']
                is_correct = False

                if model_answer:
                    total_predictions += 1
                    if model_answer == ground_truth:
                        correct_predictions += 1
                        is_correct = True
                else:
                    errors += 1
                    original_generated_text = generated_text
                    if not generated_text.startswith("ERROR"):
                        generated_text = f"EXTRACTION_FAILED: {generated_text}"
                        failure_type = "answer_extraction_failed"
                    else:
                        failure_type = "model_error"
                    
                    # Add to failure cases
                    failure_cases_list.append({
                        "index": result['index'],
                        "subject": original_item.get("Subject", "unknown"),
                        "question": original_item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": failure_type,
                        "failure_reason": generated_text,
                        "raw_output": original_generated_text,
                        "choices": {
                            "A": original_item.get("A", ""),
                            "B": original_item.get("B", ""),
                            "C": original_item.get("C", ""),
                            "D": original_item.get("D", "")
                        }
                    })

                results_details.append({
                    "index": result['index'], "ground_truth": ground_truth, "model_raw_output": generated_text,
                    "predicted_answer": model_answer, "is_correct": is_correct
                })
                raw_generations_list.append({
                    "index": result['index'], "subject": original_item.get("Subject", "unknown"), "ground_truth": ground_truth,
                    "raw_output": generated_text, "extracted_answer": model_answer
                })

            pbar.set_description(f"Evaluating {config.name} (generic 5-shot Korean, errors: {errors})")

        # --- Final Results ---
        logger.info(f"Inference loop finished for {config.name}.")
        
        total_processed = len(test_data)
        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        accuracy_strict = (correct_predictions / total_processed * 100) if total_processed > 0 else 0

        # --- Calculate Category-wise Accuracy ---
        subject_stats = {}
        for idx, item in enumerate(test_data):
            subject = item.get("Subject", "unknown")
            result = results_details[idx]
            
            if subject not in subject_stats:
                subject_stats[subject] = {"total": 0, "correct": 0, "valid_predictions": 0, "accuracy": 0.0}
            
            subject_stats[subject]["total"] += 1
            if result['predicted_answer'] is not None and not result['model_raw_output'].startswith(("SKIPPED", "ERROR")):
                subject_stats[subject]["valid_predictions"] += 1
                if result['is_correct']:
                    subject_stats[subject]["correct"] += 1
        
        for subject in subject_stats:
            if subject_stats[subject]["valid_predictions"] > 0:
                subject_stats[subject]["accuracy"] = (subject_stats[subject]["correct"] / subject_stats[subject]["valid_predictions"]) * 100

        logger.info(f"--- Generic 5-shot Korean MMLU Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Original Dataset Size: {len(mmlu_data)}")
        logger.info(f"Test Items: {total_processed}")
        logger.info(f"Generic Korean Examples Used: 5")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Errors or Skipped: {errors}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_test_items): {accuracy_strict:.2f}%")

        # --- Save Results ---
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "evaluation_type": "Generic 5-shot Korean MMLU",
            "total_original_items": len(mmlu_data),
            "generic_korean_examples_used": 5,
            "test_items": total_processed,
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors,
            "accuracy_standard (correct / valid_predictions)": accuracy_standard,
            "accuracy_strict (correct / total_test_items)": accuracy_strict,
            "subject_wise_accuracy": subject_stats,
            "details": results_details
        }
        
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

        # Save raw generations and failure cases (same structure as before)
        logger.info(f"Saving raw model generations to {raw_gen_filepath}...")
        try:
            with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw generations saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save raw generations file {raw_gen_filepath}: {e}")

        if failure_cases_list:
            logger.info(f"Saving {len(failure_cases_list)} failure cases to {failure_cases_filepath}...")
            try:
                failure_summary = {
                    "total_failures": len(failure_cases_list),
                    "failure_types": {},
                    "failure_cases": failure_cases_list
                }
                
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
        logger.info(f"Cleaning up resources for {config.name}...")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
             try:
                root_logger.removeHandler(file_handler)
                file_handler.close()
             except Exception as e:
                logger.debug(f"Error closing/removing file handler: {e}")
# --- Main Execution Logic (Uses BASE_OUTPUT_DIR) ---
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    mmlu_data = load_kmmlu_data(DATASET_PATH)
    if mmlu_data is None:
        return

    for config in MODEL_CONFIGS:
         logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====")
         evaluate_single_model(config, mmlu_data, BASE_OUTPUT_DIR)
         logger.info(f"===== Finished Evaluation for Model: {config.name} =====")
         print("-" * 60)

    logger.info("All model evaluations complete.")
    
    # --- Create a consolidated summary of all model results ---
    logger.info("--- Generating Consolidated Summary for KMMLU 5-shot ---")
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
                    "total_items": result_data.get("test_items"),
                    # "subject_wise_accuracy": result_data.get("subject_wise_accuracy", {})
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