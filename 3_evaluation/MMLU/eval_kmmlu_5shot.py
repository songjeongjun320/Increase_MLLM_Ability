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
    # Base Models (commented out for now)
    # ModelConfig(
    #     name="Qwen2.5-3B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="google_gemma-3-4b-it",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3B-Instruct",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="DeepSeek-R1-Distill-Qwen-1.5B",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
    #     use_quantization=False
    # ),

    # ModelConfig(
    #     name="Qwen2.5-3B-Instruct-ToW-completion",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/Qwen2.5-3B-Instruct-tow",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="google_gemma-3-4b-it-ToW-completion",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    #     adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/gemma-3-4b-it-tow",
    #     use_quantization=False
    # ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct-ToW-completion",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/llama-3.2-3b-tow",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B-ToW-completion",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/tow_trained_models/DeepSeek-R1-Distill-Qwen-1.5B-tow",
        use_quantization=False
    ),
]

# --- General Configuration (Updated for 5-shot evaluation) ---
DATASET_PATH = "../../2_datasets/MMLU/MMLU_KO_Openai.json"
BASE_OUTPUT_DIR = "kmmlu_5shot_results" # Base dir for ALL model results
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

def create_5shot_korean_prompt(test_item, dev_examples):
    """
    Create improved 5-shot Korean MMLU prompt with clear answer format instructions.
    Based on patterns that work well for avoiding extraction issues.
    """  
    prompt_parts = [f"다음은 객관식 문제입니다."]
    prompt_parts.append("")  # Empty line
    
    # Add development examples (few-shot examples)
    for i, example in enumerate(dev_examples):
        question = example.get("Question", "")
        answer_letter = example.get("Answer", "A")
        
        # Get choices from Korean MMLU format
        choice_a = example.get("A", "선택지 A")
        choice_b = example.get("B", "선택지 B")
        choice_c = example.get("C", "선택지 C")
        choice_d = example.get("D", "선택지 D")
        
        # Format with clear structure
        prompt_parts.append(f"문제: {question}")
        prompt_parts.append(f"A. {choice_a}")
        prompt_parts.append(f"B. {choice_b}")
        prompt_parts.append(f"C. {choice_c}")
        prompt_parts.append(f"D. {choice_d}")
        
        # Use clear answer format that's easy to extract
        prompt_parts.append(f"정답: 정답은 {answer_letter}입니다.")
        prompt_parts.append("")  # Empty line between examples
    
    # Add test question with clear instructions
    test_question = test_item.get("Question", "")
    
    # Get choices for test question
    test_choice_a = test_item.get("A", "선택지 A")
    test_choice_b = test_item.get("B", "선택지 B")
    test_choice_c = test_item.get("C", "선택지 C")
    test_choice_d = test_item.get("D", "선택지 D")
    
    prompt_parts.append(f"문제: {test_question}")
    prompt_parts.append(f"A. {test_choice_a}")
    prompt_parts.append(f"B. {test_choice_b}")
    prompt_parts.append(f"C. {test_choice_c}")
    prompt_parts.append(f"D. {test_choice_d}")
    prompt_parts.append("")
    
    # Clear instructions for answer format
    prompt_parts.append("")
    prompt_parts.append("정답을 A, B, C, D 중 오직 하나의 문자로만 선택해주세요.")
    prompt_parts.append("정답: 따라서 정답은")
    
    return "\n".join(prompt_parts)

def extract_korean_answer_first_token(model_output, tokenizer):
    """
    Extract answer from Korean model output using first token approach.
    This follows the standard MMLU evaluation methodology adapted for Korean.
    Fixed to avoid false positives from common words.
    """
    # Clean and normalize output
    cleaned_output = model_output.strip().upper()
    
    # First, look for immediate A, B, C, or D at the start
    if cleaned_output and cleaned_output[0] in ['A', 'B', 'C', 'D']:
        return cleaned_output[0]
    
    # Look for patterns like "A.", "(A)", "A)", "답: A", "정답은 A" etc.
    import re
    patterns = [
        # Korean specific patterns (more comprehensive)
        r'정답은\s*([ABCD])',               # 정답은 A
        r'정답\s*:?\s*([ABCD])',           # 정답: A or 정답 A
        r'답은\s*([ABCD])',                # 답은 A
        r'답\s*:?\s*([ABCD])',             # 답: A or 답 A
        r'정답은\s*([ABCD])번',            # 정답은 A번
        r'정답은\s*([ABCD])\s*입니다',      # 정답은 A입니다
        r'정답은\s*([ABCD])\s*이다',        # 정답은 A이다
        r'답은\s*([ABCD])\s*입니다',        # 답은 A입니다
        r'답은\s*([ABCD])\s*이다',          # 답은 A이다
        r'([ABCD])가\s*정답',              # A가 정답
        r'([ABCD])이\s*정답',              # A이 정답
        r'([ABCD])번이\s*정답',            # A번이 정답
        r'([ABCD])\s*입니다',              # A 입니다
        
        # English patterns
        r'THE\s+CORRECT\s+ANSWER\s+IS\s+([ABCD])',  # The correct answer is A
        r'CORRECT\s+ANSWER\s+IS\s+([ABCD])',        # correct answer is A
        r'ANSWER\s+IS\s+([ABCD])',                  # answer is A
        r'Answer\s*:?\s*([ABCD])',                  # Answer: A
        
        # Generic patterns
        r'^\s*([ABCD])[\.\)\]\s]',         # A. or A) or A] at start
        r'^\s*\(?([ABCD])\)?\s*$',         # (A) or A with parentheses (whole line)
        r'^\s*([ABCD])\s*$'                # Just A, B, C, D (whole line)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[-1]  # Return the last match (most likely the final answer)
    
    # More strict approach: Only look for isolated letters at word boundaries
    # and exclude common English/Korean contexts
    word_boundary_matches = re.findall(r'\b([ABCD])\b', cleaned_output)
    
    if word_boundary_matches:
        for match in word_boundary_matches:
            # Find all positions of this match
            for m in re.finditer(r'\b' + match + r'\b', cleaned_output):
                # Get context around the match
                start_pos = max(0, m.start() - 20)
                end_pos = min(len(cleaned_output), m.end() + 20)
                context = cleaned_output[start_pos:end_pos]
                
                # Skip if it's clearly part of common words or phrases
                skip_contexts = [
                    # English words
                    'BECAUSE', 'BECAU', 'LOGICAL', 'LOGIC', 'ABSTRACT', 'ABOUT',
                    'ABOVE', 'ACCORDING', 'CONTEXT', 'CONTENT', 'CONSIDER', 
                    'CORRECT', 'CONCEPT', 'CALCULATE', 'CHOICE', 'CHART',
                    'BETWEEN', 'BEFORE', 'BASIC', 'BASED', 'DISCUSS', 
                    'DESCRIBE', 'DIFFERENT', 'DETERMINE', 'DECIDE', 'DATA',
                    'FOLLOWS', 'FOLLOW', 'LOGICALLY',
                    # Korean contexts where A,B,C,D might appear as part of words
                    '가나다라', '나다라마', '다라마바', '라마바사',
                    # Common Korean phrases where letters might appear
                    '과정에서', '근거로', '따라서', '그러나', '하지만'
                ]
                
                is_part_of_word = any(skip_word in context for skip_word in skip_contexts)
                
                if not is_part_of_word:
                    return match
    
    # If no valid answer pattern found, return None
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

# --- Batch Processing Function ---
def process_batch(model, tokenizer, batch_prompts, batch_indices):
    """Processes a batch of prompts efficiently."""
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 # Increased max length for 5-shot
        ).to(DEVICE)

        with torch.no_grad():
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

            with torch.no_grad():
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

def evaluate_single_model(config: ModelConfig, mmlu_data: list, base_output_dir: str):
    """
    주어진 설정의 단일 모델에 대해 5-shot Korean MMLU 평가를 수행하고,
    결과와 로그를 base_output_dir 아래 모델 이름의 하위 디렉토리에 저장합니다.
    """
    # Split data into development (few-shot examples) and test sets
    dev_data, test_data = prepare_kmmlu_data_with_dev_split(mmlu_data, dev_shots_per_subject=5)
    
    # Sampling
    # test_data = test_data[:10]

    if not test_data:
        logger.error("No test data available after dev/test split. Check data size and dev_shots_per_subject setting.")
        return

    # Construct model-specific output directory and file paths
    model_output_dir = os.path.join(base_output_dir, config.name) # Subdirectory per model
    os.makedirs(model_output_dir, exist_ok=True)
    results_filepath = os.path.join(model_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_output_dir, f"raw_generations_{config.name}.json")
    failure_cases_filepath = os.path.join(model_output_dir, f"failure_cases_{config.name}.json")


    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w') # Overwrite log file each time
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    # Add handler to the root logger for this run
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
        # --- Load Model and Tokenizer ---
        # 1. Determine the correct path for the tokenizer.
        # If an adapter is used, the updated tokenizer is saved with it.
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

        # Gemma 모델에서만 컴파일 비활성화
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")

        # --- Run Evaluation ---
        correct_predictions = 0
        total_predictions = 0
        errors = 0
        results_details = []
        raw_generations_list = []
        failure_cases_list = []  # New: Track failure cases separately

        logger.info("Starting inference loop...")
        logger.info("Starting 5-shot Korean MMLU inference loop...")
        logger.info(f"Test data size: {len(test_data)}")
        
        pbar = tqdm(range(0, len(test_data), BATCH_SIZE), desc=f"Evaluating {config.name} (5-shot Korean, errors: 0)")
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
                
                if not ground_truth or ground_truth not in ["A", "B", "C", "D"]:
                    errors += 1
                    failure_reason = "SKIPPED - Invalid Ground Truth"
                    results_details.append({"index": current_index, "ground_truth": None, "model_raw_output": failure_reason, "predicted_answer": None, "is_correct": False})
                    raw_generations_list.append({
                        "index": current_index, "subject": item.get("Subject", "unknown"), "ground_truth": None,
                        "raw_output": failure_reason, "extracted_answer": None
                    })
                    # Add to failure cases
                    failure_cases_list.append({
                        "index": current_index,
                        "subject": item.get("Subject", "unknown"),
                        "question": item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": "invalid_ground_truth",
                        "failure_reason": failure_reason,
                        "raw_output": failure_reason
                    })
                    continue
                
                subject = item.get("Subject", "unknown")
                dev_examples = dev_data.get(subject, [])
                prompt = create_5shot_korean_prompt(item, dev_examples) if dev_examples else None

                if prompt is None:
                    errors += 1
                    failure_reason = "SKIPPED - Prompt Creation Failed"
                    results_details.append({"index": current_index, "ground_truth": ground_truth, "model_raw_output": failure_reason, "predicted_answer": None, "is_correct": False})
                    raw_generations_list.append({
                        "index": current_index, "subject": subject, "ground_truth": ground_truth,
                        "raw_output": failure_reason, "extracted_answer": None
                    })
                    # Add to failure cases
                    failure_cases_list.append({
                        "index": current_index,
                        "subject": subject,
                        "question": item.get("Question", ""),
                        "ground_truth": ground_truth,
                        "failure_type": "prompt_creation_failed",
                        "failure_reason": failure_reason,
                        "raw_output": failure_reason
                    })
                    continue

                batch_prompts.append(prompt)
                batch_indices.append(current_index)
                batch_ground_truths.append(ground_truth)
                batch_original_items.append(item)

            if not batch_prompts:
                continue
            
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            # # Retry logic for failed answer extractions
            # retry_indices = []
            # retry_prompts = []
            # retry_ground_truths = []
            # retry_original_items = []
            
            # for i, result in enumerate(batch_results):
            #     if result['extracted_answer'] is None and not result['raw_output'].startswith("ERROR"):
            #         # Need to retry this one
            #         retry_indices.append(i)
            #         retry_prompts.append(batch_prompts[i])
            #         retry_ground_truths.append(batch_ground_truths[i])
            #         retry_original_items.append(batch_original_items[i])
            
            # # Process retries individually
            # if retry_indices:
            #     logger.info(f"Retrying {len(retry_indices)} failed extractions with individual processing...")
            #     for j, retry_idx in enumerate(retry_indices):
            #         retry_result = process_single_with_retry(
            #             model, tokenizer, retry_prompts[j], 
            #             batch_results[retry_idx]['index']
            #         )
            #         # Update the original result
            #         batch_results[retry_idx] = retry_result

            for result, ground_truth, original_item in zip(batch_results, batch_ground_truths, batch_original_items):
                model_answer = result['extracted_answer']
                generated_text = result['raw_output']
                is_correct = False
                retry_info = f" (after {result.get('retry_count', 0) + 1} attempts)" if 'retry_count' in result else ""

                if model_answer:
                    total_predictions += 1
                    if model_answer == ground_truth:
                        correct_predictions += 1
                        is_correct = True
                else:
                    errors += 1
                    original_generated_text = generated_text
                    if not generated_text.startswith("ERROR"):
                        if generated_text.startswith("EXTRACTION_FAILED"):
                            failure_type = "answer_extraction_failed"
                        else:
                            generated_text = f"EXTRACTION_FAILED{retry_info}: {generated_text}"
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
                        "retry_count": result.get('retry_count', 0),
                        "choices": {
                            "A": original_item.get("A", ""),
                            "B": original_item.get("B", ""),
                            "C": original_item.get("C", ""),
                            "D": original_item.get("D", "")
                        }
                    })

                results_details.append({
                    "index": result['index'], "ground_truth": ground_truth, "model_raw_output": generated_text,
                    "predicted_answer": model_answer, "is_correct": is_correct, "retry_count": result.get('retry_count', 0)
                })
                raw_generations_list.append({
                    "index": result['index'], "subject": original_item.get("Subject", "unknown"), "ground_truth": ground_truth,
                    "raw_output": generated_text, "extracted_answer": model_answer, "retry_count": result.get('retry_count', 0)
                })

            pbar.set_description(f"Evaluating {config.name} (5-shot Korean, errors: {errors})")

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

        logger.info(f"--- 5-shot Korean MMLU Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Test Items: {total_processed}")
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
            "evaluation_type": "5-shot Korean MMLU",
            "total_original_items": len(mmlu_data),
            "dev_examples_per_subject": 5,
            "test_items": total_processed,
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "errors_or_skipped": errors,
            "accuracy_standard (correct / valid_predictions)": accuracy_standard,
            "accuracy_strict (correct / total_test_items)": accuracy_strict,
            "subjects_with_dev_examples": list(dev_data.keys()),
            "subject_wise_accuracy": subject_stats,
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