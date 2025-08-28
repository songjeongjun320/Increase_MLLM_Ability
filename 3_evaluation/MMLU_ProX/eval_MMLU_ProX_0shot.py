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
from datetime import datetime
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "../cache"  # Cache directory for models
DATASET_PATH = "../../2_datasets/MMLU_ProX/test/"  # Path to MMLU-ProX dataset
KOREAN_DATASET_PATH = "../../2_datasets/MMLU_ProX/ko-test/"  # Path to Korean MMLU-ProX dataset
BASE_OUTPUT_DIR = "../4_evaluation_results/MMLU_ProX_0shot"  # Output directory
BATCH_SIZE = 8

# Import performance analyzer
try:
    import sys
    sys.path.append('../')
    from performance_analyzer import create_enhanced_summary
except ImportError:
    logger.warning("Performance analyzer not available. Using basic summary.")
    create_enhanced_summary = None

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
    ModelConfig(
        name="Qwen2.5-3B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models2/Qwen2.5-3B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="google_gemma-3-4b-it-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models2/google_gemma-3-4b-it-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models2/Llama-3.2-3B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models2/DeepSeek-R1-Distill-Qwen-1.5B-ToW",
        use_quantization=False
    ),
]

# --- General Configuration ---
MMLU_PROX_EN_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_en.json"
MMLU_PROX_KO_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_Ko.json"
BASE_OUTPUT_DIR = "mmlu_prox_0shot"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def create_0shot_prompt(item, language="en"):
    """
    Creates a 0-shot MMLU-ProX prompt that instructs the model 
    to use Chain-of-Thought and a specific answer format.
    """
    if language == "ko":
        # 1. 전체 지시문
        prompt_parts = ["다음 질문을 읽고, 이제 단계적으로 생각하여 최종 정답을 '#### 정답: [선택지]' 형식으로 제시하시오."]
    else:
        prompt_parts = ["Read the following question, Let's think step by step, and provide the final answer in the format '#### Answer: [Option]'."]
    
    prompt_parts.append("")
    
    # 2. 실제 문제 제시
    question = item.get("question", "")
    options = []
    for i in range(10):
        option_key = f"option_{i}"
        if option_key in item and item[option_key].strip() and item[option_key].strip() != "N/A":
            options.append(f"{chr(65+i)}. {item[option_key]}")
    
    prompt_parts.append(f"Question: {question}")
    prompt_parts.extend(options)
    prompt_parts.append("")
    
    # 3. 모델이 CoT를 시작하도록 유도
    if language == "ko":
        prompt_parts.append("단계적으로 생각해봅시다.")
    else:
        prompt_parts.append("Let's think step by step.")
        
    return "\n".join(prompt_parts)

def extract_answer_first_token(model_output):
    """
    Extract answer from model output using first token approach.
    Supports A-J for 10 options.
    """
    if not model_output:
        return None
        
    cleaned_output = model_output.strip().upper()
    valid_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # First, look for immediate A-J at the start
    if cleaned_output and cleaned_output[0] in valid_answers:
        return cleaned_output[0]
    
    import re
    
    # Priority 1: Structured answer patterns (most reliable)
    structured_patterns = [
        r'####\s*(?:정답|답|ANSWER)\s*:?\s*([A-J])',  # #### Answer: A or #### 정답: A
        r'(?:정답|답|ANSWER)\s*:?\s*([A-J])',        # Answer: A or 정답: A
        r'(?:따라서|그러므로|SO)\s+(?:정답은|답은|THE\s+ANSWER\s+IS)\s+([A-J])',  # So the answer is A
    ]
    
    for pattern in structured_patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[-1]  # Return the last match (final answer)
    
    # Priority 2: Start of text patterns
    start_patterns = [
        r'^\s*([A-J])[\.\)\]\s]',  # A. or A) or A] at start
        r'^\s*\(?([A-J])\)?\s*[\.:;]',  # (A): or A. or A:
        r'^\s*([A-J])\s*$',          # Just A at start of line
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, cleaned_output, re.MULTILINE)
        if match:
            return match.group(1)
    
    # Priority 3: Last resort - find A-J near end of text (avoid random letters in middle)
    # Only look in last 100 characters to avoid picking up random letters
    last_part = cleaned_output[-100:] if len(cleaned_output) > 100 else cleaned_output
    
    # Look for isolated A-J characters near the end
    end_patterns = [
        r'([A-J])(?:\s*[\.:;]?\s*$)',  # A at end with optional punctuation
        r'(?:\s|^)([A-J])(?:\s|$)',    # A surrounded by whitespace
    ]
    
    for pattern in end_patterns:
        matches = re.findall(pattern, last_part)
        if matches:
            return matches[-1]  # Return the last match
    
    # Priority 4: Absolute fallback - scan from end backwards
    # This avoids picking random letters from the beginning/middle of text
    for i in range(len(cleaned_output) - 1, -1, -1):
        if cleaned_output[i] in valid_answers:
            # Check if this letter appears to be part of an answer pattern
            context_start = max(0, i - 20)
            context_end = min(len(cleaned_output), i + 20)
            context = cleaned_output[context_start:context_end]
            
            # Avoid letters that are clearly part of words
            if i > 0 and cleaned_output[i-1].isalnum():
                continue
            if i < len(cleaned_output) - 1 and cleaned_output[i+1].isalnum():
                continue
                
            return cleaned_output[i]
    
    return None

def process_single_with_retry(model, tokenizer, prompt, max_retries=5):
    """
    Process a single prompt with retry logic for answer extraction failures
    Only retries when answer extraction fails (not on genuine model errors)
    """
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    return_dict_in_generate=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            output_tokens = outputs['sequences'][0][input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            # Try to extract answer
            extracted_answer = extract_answer_first_token(generated_text)
            if extracted_answer is not None:
                return generated_text, extracted_answer
            else:
                # Answer extraction failed - try again if we have retries left
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Failed to extract answer, retrying...")
                    # Small delay before retry
                    time.sleep(0.1 + random.random() * 0.1)
                    continue
                else:
                    logger.warning(f"Final attempt failed - could not extract answer after {max_retries} attempts")
                    return generated_text, None
                    
        except Exception as e:
            logger.error(f"Retry {attempt + 1}/{max_retries}: Model inference error: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.2 + random.random() * 0.2)
                continue
            else:
                # Return error info after all retries exhausted
                return f"ERROR after {max_retries} attempts: {str(e)}", None
    
    return f"EXTRACTION_FAILED after {max_retries} attempts", None

def load_jsonl_dataset(filepath):
    """Loads dataset from a JSONL file."""
    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth(item):
    """Returns the ground truth answer letter."""
    answer = item.get("answer", "")
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCDEFGHIJ":
        return answer.upper()
    
    # Fallback to answer_index
    answer_index = item.get("answer_index", -1)
    if isinstance(answer_index, int) and 0 <= answer_index <= 9:
        return chr(65 + answer_index)
    return None

def process_batch(model, tokenizer, batch_prompts, batch_indices):
    """Process a batch of prompts efficiently."""
    try:
        # Tokenize batch
        batch_inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048  # Longer context for complex questions
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        batch_results = []
        for i, (sequence, input_length) in enumerate(zip(outputs.sequences, batch_inputs['input_ids'].shape[1:])):
            # Decode only the generated part
            output_tokens = sequence[input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            # Decode the full sequence (prompt + generation)
            full_text = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            
            extracted_answer = extract_answer_first_token(generated_text)
            
            batch_results.append({
                'index': batch_indices[i],
                'raw_output': generated_text,
                'full_generation': full_text,
                'extracted_answer': extracted_answer
            })
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        # Fallback to individual processing
        individual_results = []
        for prompt, idx in zip(batch_prompts, batch_indices):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=4096).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=0.0,
                    )
                # Decode only the generated part
                output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                
                # Decode the full sequence (prompt + generation)
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                extracted_answer = extract_answer_first_token(generated_text)
                
                individual_results.append({
                    'index': idx,
                    'raw_output': generated_text,
                    'full_generation': full_text,
                    'extracted_answer': extracted_answer
                })
            except Exception as individual_error:
                logger.error(f"Individual processing error for index {idx}: {individual_error}")
                individual_results.append({
                    'index': idx,
                    'raw_output': f"ERROR: {str(individual_error)[:100]}",
                    'full_generation': f"ERROR: {str(individual_error)[:100]}",
                    'extracted_answer': None
                })
        
        return individual_results

# --- Evaluation Function ---
def evaluate_single_model_on_datasets(config: ModelConfig, mmlu_prox_en_data: list, mmlu_prox_ko_data: list, model_specific_output_dir: str):
    """
    Performs 0-shot MMLU-ProX evaluation for a single model on both English and Korean datasets.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_0shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_0shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_0shot.json")

    # Setup Logging
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 0-shot MMLU-ProX Evaluation for Model: {config.name} ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    
    try:
        # Load Model and Tokenizer
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_load_path, 
                    cache_dir=CACHE_DIR,
                    padding_side='left',
                    trust_remote_code=True
                )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model {config.model_id}...")
        quantization_config_bnb = None
        if config.use_quantization:
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

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

        if config.adapter_path:
            logger.info(f"Loading adapter from: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            logger.info("Successfully loaded LoRA adapter.")

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # Gemma 모델에서만 컴파일 비활성화
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")
            
        # Prepare results storage
        all_results = {
            "mmlu_prox_en": {"correct": 0, "total": 0, "details": [], "raw_generations": []},
            "mmlu_prox_ko": {"correct": 0, "total": 0, "details": [], "raw_generations": []}
        }

        # Evaluate MMLU-ProX English
        logger.info("Starting MMLU-ProX English evaluation...")
        pbar_en = tqdm(range(0, len(mmlu_prox_en_data), BATCH_SIZE), desc="Evaluating MMLU-ProX English (errors: 0)")
        for i in pbar_en:
            batch_data = mmlu_prox_en_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_0shot_prompt(item, "en")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth, batch_prompt in zip(batch_results, batch_ground_truths, batch_prompts):
                extracted_answer = result['extracted_answer']
                raw_output = result['raw_output']
                
                # Retry logic for failed extractions
                if not extracted_answer and not raw_output.startswith("ERROR"):
                    logger.warning(f"Batch extraction failed for English item {result['index']}, attempting individual retry...")
                    retry_text, retry_answer = process_single_with_retry(model, tokenizer, batch_prompt)
                    
                    if retry_answer is not None:
                        extracted_answer = retry_answer
                        raw_output = retry_text
                        logger.info(f"English retry successful for item {result['index']}: extracted '{retry_answer}'")
                    else:
                        # Even retry failed
                        if not retry_text.startswith("ERROR"):
                            logger.warning(f"English item {result['index']}: Failed to extract answer after retries")
                            raw_output = f"EXTRACTION_FAILED: {retry_text}"
                        else:
                            logger.error(f"English item {result['index']}: Model error: {retry_text}")
                            raw_output = retry_text
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                # Only count items with valid extracted answers for total
                if extracted_answer:
                    all_results["mmlu_prox_en"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_en"]["correct"] += 1
                
                all_results["mmlu_prox_en"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "raw_output": raw_output
                })
                
                all_results["mmlu_prox_en"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": raw_output,
                    "full_generation": raw_output,
                    "extracted_answer": extracted_answer
                })
            
            # Update progress bar with current error count
            current_en_errors = len(mmlu_prox_en_data[:i+BATCH_SIZE]) - all_results["mmlu_prox_en"]["total"]
            pbar_en.set_description(f"Evaluating MMLU-ProX English (errors: {current_en_errors})")

        logger.info(f"MMLU-ProX English evaluation completed: {all_results['mmlu_prox_en']['correct']}/{all_results['mmlu_prox_en']['total']}")

        # Evaluate MMLU-ProX Korean
        logger.info("Starting MMLU-ProX Korean evaluation...")
        pbar_ko = tqdm(range(0, len(mmlu_prox_ko_data), BATCH_SIZE), desc="Evaluating MMLU-ProX Korean (errors: 0)")
        for i in pbar_ko:
            batch_data = mmlu_prox_ko_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_0shot_prompt(item, "ko")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth, batch_prompt in zip(batch_results, batch_ground_truths, batch_prompts):
                extracted_answer = result['extracted_answer']
                raw_output = result['raw_output']
                
                # Retry logic for failed extractions
                if not extracted_answer and not raw_output.startswith("ERROR"):
                    logger.warning(f"Batch extraction failed for Korean item {result['index']}, attempting individual retry...")
                    retry_text, retry_answer = process_single_with_retry(model, tokenizer, batch_prompt)
                    
                    if retry_answer is not None:
                        extracted_answer = retry_answer
                        raw_output = retry_text
                        logger.info(f"Korean retry successful for item {result['index']}: extracted '{retry_answer}'")
                    else:
                        # Even retry failed
                        if not retry_text.startswith("ERROR"):
                            logger.warning(f"Korean item {result['index']}: Failed to extract answer after retries")
                            raw_output = f"EXTRACTION_FAILED: {retry_text}"
                        else:
                            logger.error(f"Korean item {result['index']}: Model error: {retry_text}")
                            raw_output = retry_text
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                # Only count items with valid extracted answers for total
                if extracted_answer:
                    all_results["mmlu_prox_ko"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_ko"]["correct"] += 1
                
                all_results["mmlu_prox_ko"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "raw_output": raw_output
                })
                
                all_results["mmlu_prox_ko"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": raw_output,
                    "full_generation": raw_output,
                    "extracted_answer": extracted_answer
                })
            
            # Update progress bar with current error count
            current_ko_errors = len(mmlu_prox_ko_data[:i+BATCH_SIZE]) - all_results["mmlu_prox_ko"]["total"]
            pbar_ko.set_description(f"Evaluating MMLU-ProX Korean (errors: {current_ko_errors})")

        logger.info(f"MMLU-ProX Korean evaluation completed: {all_results['mmlu_prox_ko']['correct']}/{all_results['mmlu_prox_ko']['total']}")

        # Calculate strict accuracies (including errors/skips)
        en_strict_accuracy = (all_results["mmlu_prox_en"]["correct"] / len(mmlu_prox_en_data) * 100) if len(mmlu_prox_en_data) > 0 else 0
        ko_strict_accuracy = (all_results["mmlu_prox_ko"]["correct"] / len(mmlu_prox_ko_data) * 100) if len(mmlu_prox_ko_data) > 0 else 0
        
        # Calculate error/skip counts
        en_errors_skipped = len(mmlu_prox_en_data) - all_results["mmlu_prox_en"]["total"]
        ko_errors_skipped = len(mmlu_prox_ko_data) - all_results["mmlu_prox_ko"]["total"]

        logger.info(f"--- Final Results for {config.name} ---")
        logger.info(f"MMLU-ProX English Strict Accuracy: {en_strict_accuracy:.2f}% ({all_results['mmlu_prox_en']['correct']}/{len(mmlu_prox_en_data)}) [Errors/Skipped: {en_errors_skipped}]")
        logger.info(f"MMLU-ProX Korean Strict Accuracy: {ko_strict_accuracy:.2f}% ({all_results['mmlu_prox_ko']['correct']}/{len(mmlu_prox_ko_data)}) [Errors/Skipped: {ko_errors_skipped}]")

        # Save Results
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "0-shot MMLU-ProX",
            "evaluation_date": datetime.now().isoformat(),
            "mmlu_prox_en_results": {
                "accuracy_strict": en_strict_accuracy,
                "correct_predictions": all_results["mmlu_prox_en"]["correct"],
                "total_predictions": all_results["mmlu_prox_en"]["total"],
                "total_items": len(mmlu_prox_en_data),
                "errors_or_skipped": en_errors_skipped,
                "details": all_results["mmlu_prox_en"]["details"]
            },
            "mmlu_prox_ko_results": {
                "accuracy_strict": ko_strict_accuracy,
                "correct_predictions": all_results["mmlu_prox_ko"]["correct"],
                "total_predictions": all_results["mmlu_prox_ko"]["total"],
                "total_items": len(mmlu_prox_ko_data),
                "errors_or_skipped": ko_errors_skipped,
                "details": all_results["mmlu_prox_ko"]["details"]
            }
        }

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # Save raw generations
        raw_generations_summary = {
            "mmlu_prox_en": all_results["mmlu_prox_en"]["raw_generations"],
            "mmlu_prox_ko": all_results["mmlu_prox_ko"]["raw_generations"]
        }
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_summary, f, indent=2, ensure_ascii=False)

        return {
            "model_name": config.name,
            "mmlu_prox_en_accuracy_strict": en_strict_accuracy,
            "mmlu_prox_ko_accuracy_strict": ko_strict_accuracy,
            "mmlu_prox_en_correct": all_results["mmlu_prox_en"]["correct"],
            "mmlu_prox_en_total": all_results["mmlu_prox_en"]["total"],
            "mmlu_prox_en_total_items": len(mmlu_prox_en_data),
            "mmlu_prox_en_errors_skipped": en_errors_skipped,
            "mmlu_prox_ko_correct": all_results["mmlu_prox_ko"]["correct"],
            "mmlu_prox_ko_total": all_results["mmlu_prox_ko"]["total"],
            "mmlu_prox_ko_total_items": len(mmlu_prox_ko_data),
            "mmlu_prox_ko_errors_skipped": ko_errors_skipped
        }

    except Exception as e:
        logger.exception(f"Critical error during evaluation for {config.name}: {e}")
        return {
            "model_name": config.name,
            "mmlu_prox_en_accuracy_strict": 0.0,
            "mmlu_prox_ko_accuracy_strict": 0.0,
            "mmlu_prox_en_correct": 0,
            "mmlu_prox_en_total": 0,
            "mmlu_prox_en_total_items": len(mmlu_prox_en_data) if 'mmlu_prox_en_data' in locals() else 0,
            "mmlu_prox_en_errors_skipped": len(mmlu_prox_en_data) if 'mmlu_prox_en_data' in locals() else 0,
            "mmlu_prox_ko_correct": 0,
            "mmlu_prox_ko_total": 0,
            "mmlu_prox_ko_total_items": len(mmlu_prox_ko_data) if 'mmlu_prox_ko_data' in locals() else 0,
            "mmlu_prox_ko_errors_skipped": len(mmlu_prox_ko_data) if 'mmlu_prox_ko_data' in locals() else 0,
            "error": str(e)
        }
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            root_logger.removeHandler(file_handler)
            file_handler.close()

def main():
    # Load datasets
    mmlu_prox_en_data = load_jsonl_dataset(MMLU_PROX_EN_DATASET_PATH)
    mmlu_prox_ko_data = load_jsonl_dataset(MMLU_PROX_KO_DATASET_PATH)
    
    if not mmlu_prox_en_data or not mmlu_prox_ko_data:
        logger.error("Failed to load datasets.")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Store all model results for summary
    all_model_results = []

    # Evaluate each model
    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        
        model_result = evaluate_single_model_on_datasets(config, mmlu_prox_en_data, mmlu_prox_ko_data, model_specific_output_dir)
        all_model_results.append(model_result)

    # Generate summary
    summary_data = {
        "evaluation_info": {
            "evaluation_type": "0-shot MMLU-ProX",
            "evaluation_date": datetime.now().isoformat(),
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_mmlu_prox_en_items": len(mmlu_prox_en_data),
            "total_mmlu_prox_ko_items": len(mmlu_prox_ko_data),
            "note": "Only strict accuracy reported (includes errors/skips in total count)"
        },
        "model_results": all_model_results,
        "summary_statistics": {
            "best_mmlu_prox_en_model": max(all_model_results, key=lambda x: x.get("mmlu_prox_en_accuracy_strict", 0))["model_name"] if all_model_results else "N/A",
            "best_mmlu_prox_ko_model": max(all_model_results, key=lambda x: x.get("mmlu_prox_ko_accuracy_strict", 0))["model_name"] if all_model_results else "N/A",
            "average_mmlu_prox_en_accuracy_strict": sum(x.get("mmlu_prox_en_accuracy_strict", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0,
            "average_mmlu_prox_ko_accuracy_strict": sum(x.get("mmlu_prox_ko_accuracy_strict", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0
        }
    }

    # Enhanced summary with performance analysis
    if create_enhanced_summary:
        # Prepare model results for analysis
        model_results_for_analysis = []
        en_results_for_analysis = []
        ko_results_for_analysis = []
        
        for result in all_model_results:
            if 'error' not in result:
                # Create combined accuracy metric for analysis
                en_accuracy = result.get('mmlu_prox_en_accuracy_strict', 0)
                ko_accuracy = result.get('mmlu_prox_ko_accuracy_strict', 0)
                combined_accuracy = (en_accuracy + ko_accuracy) / 2
                
                analysis_result = {
                    "model_name": result["model_name"],
                    "accuracy_strict": combined_accuracy,
                    "mmlu_prox_en_accuracy": en_accuracy,
                    "mmlu_prox_ko_accuracy": ko_accuracy,
                    "correct_predictions": result.get('mmlu_prox_en_correct', 0) + result.get('mmlu_prox_ko_correct', 0),
                    "total_items": result.get('mmlu_prox_en_total_items', 0) + result.get('mmlu_prox_ko_total_items', 0)
                }
                model_results_for_analysis.append(analysis_result)
                
                # Separate English and Korean results
                en_analysis_result = {
                    "model_name": result["model_name"],
                    "accuracy_strict": en_accuracy,
                    "correct_predictions": result.get('mmlu_prox_en_correct', 0),
                    "total_items": result.get('mmlu_prox_en_total_items', 0)
                }
                en_results_for_analysis.append(en_analysis_result)
                
                ko_analysis_result = {
                    "model_name": result["model_name"],
                    "accuracy_strict": ko_accuracy,
                    "correct_predictions": result.get('mmlu_prox_ko_correct', 0),
                    "total_items": result.get('mmlu_prox_ko_total_items', 0)
                }
                ko_results_for_analysis.append(ko_analysis_result)
        
        # Combined summary
        enhanced_summary = create_enhanced_summary(
            model_results=model_results_for_analysis,
            evaluation_info=summary_data["evaluation_info"],
            primary_metric="accuracy_strict",
            subject_metric=None  # MMLU_ProX doesn't have subject breakdown
        )
        enhanced_summary["original_detailed_results"] = summary_data
        
        # English-only summary
        en_evaluation_info = summary_data["evaluation_info"].copy()
        en_evaluation_info["dataset_language"] = "English"
        en_evaluation_info["total_items"] = len(mmlu_prox_en_data)
        en_enhanced_summary = create_enhanced_summary(
            model_results=en_results_for_analysis,
            evaluation_info=en_evaluation_info,
            primary_metric="accuracy_strict",
            subject_metric=None
        )
        
        # Korean-only summary
        ko_evaluation_info = summary_data["evaluation_info"].copy()
        ko_evaluation_info["dataset_language"] = "Korean"
        ko_evaluation_info["total_items"] = len(mmlu_prox_ko_data)
        ko_enhanced_summary = create_enhanced_summary(
            model_results=ko_results_for_analysis,
            evaluation_info=ko_evaluation_info,
            primary_metric="accuracy_strict",
            subject_metric=None
        )
        
        # Save all summaries
        summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY.json")
        en_summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY_EN.json")
        ko_summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY_KO.json")
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)
        with open(en_summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_enhanced_summary, f, indent=2, ensure_ascii=False)
        with open(ko_summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_enhanced_summary, f, indent=2, ensure_ascii=False)
            
        # Log key insights
        perf_analysis = enhanced_summary["performance_analysis"]
        logger.info(f"🏆 Best performing model (Combined): {perf_analysis['best_model']}")
        logger.info(f"📊 Average combined accuracy: {perf_analysis['average_score']:.2f}%")
        logger.info(f"📈 Performance gap: {perf_analysis['performance_gap']:.2f}%p")
        
        en_perf_analysis = en_enhanced_summary["performance_analysis"]
        ko_perf_analysis = ko_enhanced_summary["performance_analysis"]
        logger.info(f"🏆 Best performing model (English): {en_perf_analysis['best_model']}")
        logger.info(f"🏆 Best performing model (Korean): {ko_perf_analysis['best_model']}")
        
    else:
        # Fallback to basic summary with separate language files
        summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY.json")
        en_summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY_EN.json")
        ko_summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY_KO.json")
        
        # Create English-only summary
        en_summary_data = {
            "evaluation_info": summary_data["evaluation_info"].copy(),
            "model_results": [{
                "model_name": result["model_name"],
                "mmlu_prox_en_accuracy_strict": result.get("mmlu_prox_en_accuracy_strict", 0),
                "mmlu_prox_en_correct": result.get("mmlu_prox_en_correct", 0),
                "mmlu_prox_en_total": result.get("mmlu_prox_en_total", 0),
                "mmlu_prox_en_total_items": result.get("mmlu_prox_en_total_items", 0),
                "mmlu_prox_en_errors_skipped": result.get("mmlu_prox_en_errors_skipped", 0)
            } for result in all_model_results],
            "summary_statistics": {
                "best_mmlu_prox_en_model": summary_data["summary_statistics"]["best_mmlu_prox_en_model"],
                "average_mmlu_prox_en_accuracy_strict": summary_data["summary_statistics"]["average_mmlu_prox_en_accuracy_strict"]
            }
        }
        en_summary_data["evaluation_info"]["dataset_language"] = "English"
        en_summary_data["evaluation_info"]["total_items"] = len(mmlu_prox_en_data)
        
        # Create Korean-only summary
        ko_summary_data = {
            "evaluation_info": summary_data["evaluation_info"].copy(),
            "model_results": [{
                "model_name": result["model_name"],
                "mmlu_prox_ko_accuracy_strict": result.get("mmlu_prox_ko_accuracy_strict", 0),
                "mmlu_prox_ko_correct": result.get("mmlu_prox_ko_correct", 0),
                "mmlu_prox_ko_total": result.get("mmlu_prox_ko_total", 0),
                "mmlu_prox_ko_total_items": result.get("mmlu_prox_ko_total_items", 0),
                "mmlu_prox_ko_errors_skipped": result.get("mmlu_prox_ko_errors_skipped", 0)
            } for result in all_model_results],
            "summary_statistics": {
                "best_mmlu_prox_ko_model": summary_data["summary_statistics"]["best_mmlu_prox_ko_model"],
                "average_mmlu_prox_ko_accuracy_strict": summary_data["summary_statistics"]["average_mmlu_prox_ko_accuracy_strict"]
            }
        }
        ko_summary_data["evaluation_info"]["dataset_language"] = "Korean"
        ko_summary_data["evaluation_info"]["total_items"] = len(mmlu_prox_ko_data)
        
        # Save all summaries
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        with open(en_summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_summary_data, f, indent=2, ensure_ascii=False)
        with open(ko_summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Summary saved to: {summary_filepath}")
    logger.info("=== FINAL SUMMARY ===")
    for result in all_model_results:
        logger.info(f"{result['model_name']}:")
        logger.info(f"  MMLU-ProX EN Strict: {result.get('mmlu_prox_en_accuracy_strict', 0):.2f}%")
        logger.info(f"  MMLU-ProX KO Strict: {result.get('mmlu_prox_ko_accuracy_strict', 0):.2f}%")

if __name__ == "__main__":
    main()