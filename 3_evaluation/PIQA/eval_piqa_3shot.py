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
PIQA_DATASET_PATH = "../../2_datasets/PIQA/validation.json"  # Path to PIQA dataset
KOPIQA_DATASET_PATH = "../../2_datasets/PIQA/Ko-PIQA_validation.json"  # Path to Ko-PIQA dataset
BASE_OUTPUT_DIR = "../4_evaluation_results/PIQA_3shot"  # Output directory
BATCH_SIZE = 16

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
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/Qwen2.5-3B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="google_gemma-3-4b-it-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/google_gemma-3-4b-it-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.2-3B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/Llama-3.2-3B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-1.5B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models_2/DeepSeek-R1-Distill-Qwen-1.5B-ToW",
        use_quantization=False
    ),
]

# --- General Configuration ---
PIQA_DATASET_PATH = "../../2_datasets/PIQA/piqa.json"
KO_PIQA_DATASET_PATH = "../../2_datasets/PIQA/ko-piqa.json"
BASE_OUTPUT_DIR = "piqa_5shot"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Few-Shot Examples ---
ENGLISH_FEW_SHOT_EXAMPLES = [
    {
        "goal": "How to separate egg whites from egg yolks?",
        "sol1": "Crack the egg and pour the contents from one half of the shell to the other, allowing the white to fall into a bowl below while keeping the yolk in the shell.",
        "sol2": "Crack the egg and pour everything into a bowl, then use a spoon to fish out the yolk.",
        "label": 0,
        "reasoning": "When separating eggs, the goal is to keep the yolk intact while collecting the white. Method A uses the traditional shell-transfer technique which naturally allows the white to flow out while the yolk stays contained in the shell due to its thicker consistency. Method B would work but is messier and risks breaking the yolk when fishing it out with a spoon, potentially contaminating the white."
    },
    {
        "goal": "To make a simple bookmark",
        "sol1": "Cut a piece of cardboard to size, decorate it, and punch a hole at the top to thread a ribbon through.",
        "sol2": "Cut a piece of paper to size, fold it in half, and staple the open edges together.",
        "label": 0,
        "reasoning": "For a bookmark to be functional, it should be durable and have a way to easily locate it. Method A creates a sturdy cardboard bookmark with a ribbon that can hang out of the book, making it easy to find. Method B creates a paper pocket that would be less durable and doesn't provide a way to easily locate the bookmark in a closed book."
    },
    {
        "goal": "How to remove a splinter from your finger?",
        "sol1": "Use sterilized tweezers to gently pull the splinter out in the same direction it entered the skin.",
        "sol2": "Push the splinter deeper into the skin until it comes out the other side.",
        "label": 0,
        "reasoning": "Safe splinter removal requires minimizing tissue damage and infection risk. Method A follows proper first aid by using sterile tools and removing the splinter along its entry path, which minimizes tissue tearing. Method B is dangerous as it would cause unnecessary injury, push bacteria deeper, and potentially break the splinter inside the tissue."
    }
]
KOREAN_FEW_SHOT_EXAMPLES = [
    {
        "goal": "달걀 흰자와 달걀 노른자를 어떻게 분리하나요?",
        "sol1": "달걀을 깨서 껍데기 반쪽에서 다른 반쪽으로 내용물을 부으면서 노른자는 껍데기에 유지하고 흰자는 아래 그릇으로 떨어뜨린다.",
        "sol2": "달걀을 깨서 모든 것을 그릇에 부은 다음 숟가락으로 노른자를 건져낸다.",
        "label": 0,
        "reasoning": "달걀을 분리할 때 목표는 노른자를 터뜨리지 않으면서 흰자를 분리하는 것입니다. 방법 A는 전통적인 껍데기 이동 기법으로, 노른자의 더 진한 농도 때문에 껍데기에 머물면서 흰자만 자연스럽게 흘러내립니다. 방법 B도 가능하지만 숟가락으로 건져낼 때 노른자가 터질 위험이 있어 흰자를 오염시킬 수 있습니다."
    },
    {
        "goal": "간단한 책갈피를 만들려면",
        "sol1": "판지를 크기에 맞게 자르고 장식한 다음 위쪽에 구멍을 뚫어 리본을 끼운다.",
        "sol2": "종이를 크기에 맞게 자르고 반으로 접은 다음 열린 가장자리를 함께 스테이플러로 고정한다.",
        "label": 0,
        "reasoning": "책갈피가 실용적이려면 내구성이 있고 쉽게 찾을 수 있어야 합니다. 방법 A는 튼튼한 판지로 만들고 리본이 책 밖으로 나와 쉽게 찾을 수 있습니다. 방법 B는 종이 주머니를 만드는 것으로 내구성이 떨어지고 닫힌 책에서 책갈피를 쉽게 찾을 수 있는 방법을 제공하지 않습니다."
    },
    {
        "goal": "손가락에서 가시를 어떻게 제거하나요?",
        "sol1": "멸균된 핀셋을 사용하여 가시가 피부에 들어간 같은 방향으로 부드럽게 뽑아낸다.",
        "sol2": "가시를 피부 깊숙이 밀어넣어서 반대편으로 나오게 한다.",
        "label": 0,
        "reasoning": "가시를 제거하는 가장 적절한 방법은 가시가 피부에 들어간 방향과 같은 방향으로 부드럽게 뽑아내는 것입니다. 이 방법은 가시가 피부 내부에서 더 깊게 밀려 들어가지 않도록 하고, 피부를 상처내지 않으면서 가시를 효과적으로 제거할 수 있습니다. 반면, 가시를 피부 깊숙이 밀어넣어 반대편으로 나오게 하는 방법은 가시가 더 깊은 부위로 밀리거나 제거하기 더 어려워질 수 있기 때문에 위험할 수 있습니다. 또한, 이 방법은 추가적인 상처나 감염을 유발할 수 있습니다. 따라서 첫 번째 방법인 멸균된 핀셋을 사용하여 가시가 들어간 방향으로 부드럽게 뽑아내는 방법이 가장 안전하고 효과적입니다."
    }
]

# --- Helper Functions ---
def create_3shot_prompt(item, few_shot_examples, language="en"):
    """
    Creates a 5-shot PIQA prompt with actual reasoning examples.
    """
    if language == "ko":
        prompt_parts = ["다음은 물리적 상식에 대한 다지선다형 질문입니다."]
    else:
        prompt_parts = ["The following are multiple choice questions about physical commonsense."]
    
    prompt_parts.append("")
    
    # Add few-shot examples with actual reasoning
    for example in few_shot_examples[:5]:  # Only use first 5 examples
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        reasoning = example["reasoning"]
        correct_answer = "A" if example["label"] == 0 else "B"
        
        prompt_parts.append(f"Goal: {goal}")
        prompt_parts.append(f"A. {sol1}")
        prompt_parts.append(f"B. {sol2}")
        
        if language == "ko":
            prompt_parts.append(f"응답: 단계적으로 생각해봅시다. {reasoning} #### 따라서 정답: {correct_answer}. #### {correct_answer}")
        else:
            prompt_parts.append(f"Response: Let's think step by step. {reasoning} #### Therefore Answer: {correct_answer}. #### {correct_answer}")
        prompt_parts.append("")
    
    # Add the test question
    goal = item.get("goal", "")
    sol1 = item.get("sol1", "")
    sol2 = item.get("sol2", "")
    
    prompt_parts.append(f"Goal: {goal}")
    prompt_parts.append(f"A. {sol1}")
    prompt_parts.append(f"B. {sol2}")
    
    if language == "ko":
        prompt_parts.append("응답: 단계적으로 생각해봅시다.")
    else:
        prompt_parts.append("Response: Let's think step by step.")
    
    return "\n".join(prompt_parts)

def extract_final_answer(model_output):
    """
    Extract the final answer (A or B) from model output using structured patterns first.
    Supports A-B for 2 options (PIQA format).
    """
    if not model_output:
        return None
        
    cleaned_output = model_output.strip().upper()
    valid_answers = ['A', 'B']
    
    import re
    
    # Priority 1: Structured answer patterns (most reliable)
    structured_patterns = [
        r'####\s*(?:정답|답|ANSWER|THEREFORE\s+ANSWER)\s*:?\s*([AB])',  # #### Answer: A or #### 정답: A
        r'(?:정답|답|ANSWER)\s*:?\s*([AB])',        # Answer: A or 정답: A
        r'(?:따라서|그러므로|SO|THEREFORE)\s+(?:정답은|답은|정답|답|THE\s+ANSWER\s+IS|ANSWER\s+IS)\s*:?\s*([AB])',  # So the answer is A
    ]
    
    for pattern in structured_patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[-1]  # Return the last match (final answer)
    
    # Priority 2: Start of text patterns
    start_patterns = [
        r'^\s*([AB])[\.\)\]\s]',  # A. or A) or A] at start
        r'^\s*\(?([AB])\)?\s*[\.:;]',  # (A): or A. or A:
        r'^\s*([AB])\s*$',          # Just A at start of line
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, cleaned_output, re.MULTILINE)
        if match:
            return match.group(1)
    
    # Priority 3: Last resort - find A-B near end of text (avoid random letters in middle)
    # Only look in last 100 characters to avoid picking up random letters
    last_part = cleaned_output[-100:] if len(cleaned_output) > 100 else cleaned_output
    
    # Look for isolated A-B characters near the end
    end_patterns = [
        r'([AB])(?:\s*[\.:;]?\s*$)',  # A at end with optional punctuation
        r'(?:\s|^)([AB])(?:\s|$)',    # A surrounded by whitespace
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
                max_length=2048
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,  # PIQA typically only needs 1 token (A or B)
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    return_dict_in_generate=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            output_tokens = outputs['sequences'][0][input_length:]
            generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            # Try to extract answer
            extracted_answer = extract_final_answer(generated_text)
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

def load_dataset(filepath):
    """Loads dataset from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth(item):
    """Returns the ground truth answer letter."""
    label = item.get("label", -1)
    if isinstance(label, int) and label in [0, 1]:
        return "A" if label == 0 else "B"
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
            max_length=2048
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,  # Allow reasoning process
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
            
            extracted_answer = extract_final_answer(generated_text)
            
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
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=2048).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,  # Allow reasoning process
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
                
                extracted_answer = extract_final_answer(generated_text)
                
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
def evaluate_single_model_on_datasets(config: ModelConfig, piqa_data: list, ko_piqa_data: list, model_specific_output_dir: str):
    """
    Performs 5-shot PIQA and Ko-PIQA evaluation for a single model.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_5shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot.json")

    # Setup Logging
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 5-shot PIQA/Ko-PIQA Evaluation for Model: {config.name} ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    
    try:
        # Load Model and Tokenizer
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        
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
            "piqa": {"correct": 0, "total": 0, "details": [], "raw_generations": []},
            "ko_piqa": {"correct": 0, "total": 0, "details": [], "raw_generations": []}
        }

        # Evaluate PIQA (English)
        logger.info("Starting PIQA (English) evaluation...")
        pbar_piqa = tqdm(range(0, len(piqa_data), BATCH_SIZE), desc="Evaluating PIQA (English, errors: 0)")
        for i in pbar_piqa:
            batch_data = piqa_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_3shot_prompt(item, ENGLISH_FEW_SHOT_EXAMPLES, "en")
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
                    logger.warning(f"Batch extraction failed for PIQA item {result['index']}, attempting individual retry...")
                    retry_text, retry_answer = process_single_with_retry(model, tokenizer, batch_prompt)
                    
                    if retry_answer is not None:
                        extracted_answer = retry_answer
                        raw_output = retry_text
                        logger.info(f"PIQA retry successful for item {result['index']}: extracted '{retry_answer}'")
                    else:
                        # Even retry failed
                        if not retry_text.startswith("ERROR"):
                            logger.warning(f"PIQA item {result['index']}: Failed to extract answer after retries")
                            raw_output = f"EXTRACTION_FAILED: {retry_text}"
                        else:
                            logger.error(f"PIQA item {result['index']}: Model error: {retry_text}")
                            raw_output = retry_text
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                if extracted_answer:
                    all_results["piqa"]["total"] += 1
                    if is_correct:
                        all_results["piqa"]["correct"] += 1
                
                all_results["piqa"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "raw_output": raw_output
                })
                
                all_results["piqa"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": raw_output,
                    "full_generation": raw_output,
                    "extracted_answer": extracted_answer
                })
            
            # Update progress bar with current error count
            current_piqa_errors = len(piqa_data[:i+BATCH_SIZE]) - all_results["piqa"]["total"]
            pbar_piqa.set_description(f"Evaluating PIQA (English, errors: {current_piqa_errors})")

        logger.info(f"PIQA evaluation completed: {all_results['piqa']['correct']}/{all_results['piqa']['total']}")

        # Evaluate Ko-PIQA (Korean)
        logger.info("Starting Ko-PIQA (Korean) evaluation...")
        pbar_ko_piqa = tqdm(range(0, len(ko_piqa_data), BATCH_SIZE), desc="Evaluating Ko-PIQA (Korean, errors: 0)")
        for i in pbar_ko_piqa:
            batch_data = ko_piqa_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_3shot_prompt(item, KOREAN_FEW_SHOT_EXAMPLES, "ko")
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
                    logger.warning(f"Batch extraction failed for Ko-PIQA item {result['index']}, attempting individual retry...")
                    retry_text, retry_answer = process_single_with_retry(model, tokenizer, batch_prompt)
                    
                    if retry_answer is not None:
                        extracted_answer = retry_answer
                        raw_output = retry_text
                        logger.info(f"Ko-PIQA retry successful for item {result['index']}: extracted '{retry_answer}'")
                    else:
                        # Even retry failed
                        if not retry_text.startswith("ERROR"):
                            logger.warning(f"Ko-PIQA item {result['index']}: Failed to extract answer after retries")
                            raw_output = f"EXTRACTION_FAILED: {retry_text}"
                        else:
                            logger.error(f"Ko-PIQA item {result['index']}: Model error: {retry_text}")
                            raw_output = retry_text
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                if extracted_answer:
                    all_results["ko_piqa"]["total"] += 1
                    if is_correct:
                        all_results["ko_piqa"]["correct"] += 1
                
                all_results["ko_piqa"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "raw_output": raw_output
                })
                
                all_results["ko_piqa"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": raw_output,
                    "full_generation": raw_output,
                    "extracted_answer": extracted_answer
                })
            
            # Update progress bar with current error count
            current_ko_piqa_errors = len(ko_piqa_data[:i+BATCH_SIZE]) - all_results["ko_piqa"]["total"]
            pbar_ko_piqa.set_description(f"Evaluating Ko-PIQA (Korean, errors: {current_ko_piqa_errors})")

        logger.info(f"Ko-PIQA evaluation completed: {all_results['ko_piqa']['correct']}/{all_results['ko_piqa']['total']}")

        # Calculate accuracies - both standard and strict
        # Standard accuracy: based on valid predictions only
        piqa_accuracy_standard = (all_results["piqa"]["correct"] / all_results["piqa"]["total"] * 100) if all_results["piqa"]["total"] > 0 else 0
        ko_piqa_accuracy_standard = (all_results["ko_piqa"]["correct"] / all_results["ko_piqa"]["total"] * 100) if all_results["ko_piqa"]["total"] > 0 else 0
        
        # Strict accuracy: based on total dataset including errors/skips
        piqa_accuracy_strict = (all_results["piqa"]["correct"] / len(piqa_data) * 100) if len(piqa_data) > 0 else 0
        ko_piqa_accuracy_strict = (all_results["ko_piqa"]["correct"] / len(ko_piqa_data) * 100) if len(ko_piqa_data) > 0 else 0
        
        # Calculate error/skip counts
        piqa_errors_skipped = len(piqa_data) - all_results["piqa"]["total"]
        ko_piqa_errors_skipped = len(ko_piqa_data) - all_results["ko_piqa"]["total"]

        logger.info(f"--- Final Results for {config.name} ---")
        logger.info(f"PIQA Standard Accuracy: {piqa_accuracy_standard:.2f}% ({all_results['piqa']['correct']}/{all_results['piqa']['total']})")
        logger.info(f"PIQA Strict Accuracy: {piqa_accuracy_strict:.2f}% ({all_results['piqa']['correct']}/{len(piqa_data)}) [Errors/Skipped: {piqa_errors_skipped}]")
        logger.info(f"Ko-PIQA Standard Accuracy: {ko_piqa_accuracy_standard:.2f}% ({all_results['ko_piqa']['correct']}/{all_results['ko_piqa']['total']})")
        logger.info(f"Ko-PIQA Strict Accuracy: {ko_piqa_accuracy_strict:.2f}% ({all_results['ko_piqa']['correct']}/{len(ko_piqa_data)}) [Errors/Skipped: {ko_piqa_errors_skipped}]")

        # Save Results
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "5-shot PIQA/Ko-PIQA",
            "evaluation_date": datetime.now().isoformat(),
            "piqa_results": {
                "accuracy_standard": piqa_accuracy_standard,
                "accuracy_strict": piqa_accuracy_strict,
                "correct_predictions": all_results["piqa"]["correct"],
                "total_predictions": all_results["piqa"]["total"],
                "total_items": len(piqa_data),
                "errors_or_skipped": piqa_errors_skipped,
                "details": all_results["piqa"]["details"]
            },
            "ko_piqa_results": {
                "accuracy_standard": ko_piqa_accuracy_standard,
                "accuracy_strict": ko_piqa_accuracy_strict,
                "correct_predictions": all_results["ko_piqa"]["correct"],
                "total_predictions": all_results["ko_piqa"]["total"],
                "total_items": len(ko_piqa_data),
                "errors_or_skipped": ko_piqa_errors_skipped,
                "details": all_results["ko_piqa"]["details"]
            }
        }

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # Save raw generations
        raw_generations_summary = {
            "piqa": all_results["piqa"]["raw_generations"],
            "ko_piqa": all_results["ko_piqa"]["raw_generations"]
        }
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_summary, f, indent=2, ensure_ascii=False)

        return {
            "model_name": config.name,
            "piqa_accuracy_standard": piqa_accuracy_standard,
            "piqa_accuracy_strict": piqa_accuracy_strict,
            "ko_piqa_accuracy_standard": ko_piqa_accuracy_standard,
            "ko_piqa_accuracy_strict": ko_piqa_accuracy_strict,
            "piqa_correct": all_results["piqa"]["correct"],
            "piqa_total": all_results["piqa"]["total"],
            "piqa_total_items": len(piqa_data),
            "piqa_errors_skipped": piqa_errors_skipped,
            "ko_piqa_correct": all_results["ko_piqa"]["correct"],
            "ko_piqa_total": all_results["ko_piqa"]["total"],
            "ko_piqa_total_items": len(ko_piqa_data),
            "ko_piqa_errors_skipped": ko_piqa_errors_skipped
        }

    except Exception as e:
        logger.exception(f"Critical error during evaluation for {config.name}: {e}")
        return {
            "model_name": config.name,
            "piqa_accuracy_standard": 0.0,
            "piqa_accuracy_strict": 0.0,
            "ko_piqa_accuracy_standard": 0.0,
            "ko_piqa_accuracy_strict": 0.0,
            "piqa_correct": 0,
            "piqa_total": 0,
            "piqa_total_items": len(piqa_data) if 'piqa_data' in locals() else 0,
            "piqa_errors_skipped": len(piqa_data) if 'piqa_data' in locals() else 0,
            "ko_piqa_correct": 0,
            "ko_piqa_total": 0,
            "ko_piqa_total_items": len(ko_piqa_data) if 'ko_piqa_data' in locals() else 0,
            "ko_piqa_errors_skipped": len(ko_piqa_data) if 'ko_piqa_data' in locals() else 0,
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
    piqa_data = load_dataset(PIQA_DATASET_PATH)
    ko_piqa_data = load_dataset(KO_PIQA_DATASET_PATH)
    
    if not piqa_data or not ko_piqa_data:
        logger.error("Failed to load datasets.")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Store all model results for summary
    all_model_results = []

    # Evaluate each model
    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        
        model_result = evaluate_single_model_on_datasets(config, piqa_data, ko_piqa_data, model_specific_output_dir)
        all_model_results.append(model_result)

    # Generate summary
    summary_data = {
        "evaluation_info": {
            "evaluation_type": "5-shot PIQA/Ko-PIQA",
            "evaluation_date": datetime.now().isoformat(),
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_piqa_items": len(piqa_data),
            "total_ko_piqa_items": len(ko_piqa_data)
        },
        "model_results": all_model_results,
        "summary_statistics": {
            "best_piqa_model_standard": max(all_model_results, key=lambda x: x.get("piqa_accuracy_standard", 0))["model_name"] if all_model_results else "N/A",
            "best_piqa_model_strict": max(all_model_results, key=lambda x: x.get("piqa_accuracy_strict", 0))["model_name"] if all_model_results else "N/A",
            "best_ko_piqa_model_standard": max(all_model_results, key=lambda x: x.get("ko_piqa_accuracy_standard", 0))["model_name"] if all_model_results else "N/A",
            "best_ko_piqa_model_strict": max(all_model_results, key=lambda x: x.get("ko_piqa_accuracy_strict", 0))["model_name"] if all_model_results else "N/A",
            "average_piqa_accuracy_standard": sum(x.get("piqa_accuracy_standard", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0,
            "average_piqa_accuracy_strict": sum(x.get("piqa_accuracy_strict", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0,
            "average_ko_piqa_accuracy_standard": sum(x.get("ko_piqa_accuracy_standard", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0,
            "average_ko_piqa_accuracy_strict": sum(x.get("ko_piqa_accuracy_strict", 0) for x in all_model_results) / len(all_model_results) if all_model_results else 0
        }
    }

    # Save summary
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "piqa_summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Summary saved to: {summary_filepath}")
    logger.info("=== FINAL SUMMARY ===")
    for result in all_model_results:
        logger.info(f"{result['model_name']}:")
        logger.info(f"  PIQA Standard: {result.get('piqa_accuracy_standard', 0):.2f}% | Strict: {result.get('piqa_accuracy_strict', 0):.2f}%")
        logger.info(f"  Ko-PIQA Standard: {result.get('ko_piqa_accuracy_standard', 0):.2f}% | Strict: {result.get('ko_piqa_accuracy_strict', 0):.2f}%")

if __name__ == "__main__":
    main()