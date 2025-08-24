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

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_id: str
    adapter_path: str = None
    use_quantization: bool = True
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
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),

    # TOW Trained Models
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
]

# --- General Configuration ---
MMLU_PROX_EN_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_en.json"
MMLU_PROX_KO_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_Ko.json"
BASE_OUTPUT_DIR = "mmlu_prox_5shot"
BATCH_SIZE = 8
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
        "question": "Which of the following statements about DNA replication is correct?",
        "option_0": "DNA replication occurs in the 3' to 5' direction",
        "option_1": "DNA replication is semiconservative",
        "option_2": "DNA replication only occurs during mitosis", 
        "option_3": "DNA replication produces identical copies",
        "option_4": "DNA replication is conservative",
        "option_5": "DNA replication occurs bidirectionally",
        "option_6": "DNA replication is discontinuous",
        "option_7": "All of the above",
        "option_8": "None of the above",
        "option_9": "DNA replication is continuous",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "What is the primary function of the mitochondria in eukaryotic cells?",
        "option_0": "Protein synthesis",
        "option_1": "ATP production",
        "option_2": "DNA storage", 
        "option_3": "Waste removal",
        "option_4": "Cell division",
        "option_5": "Photosynthesis",
        "option_6": "Lipid synthesis",
        "option_7": "RNA processing",
        "option_8": "Calcium storage",
        "option_9": "Carbohydrate metabolism",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "Which principle of physics explains why objects in motion tend to stay in motion?",
        "option_0": "Newton's second law",
        "option_1": "Newton's first law", 
        "option_2": "Newton's third law",
        "option_3": "Law of conservation of energy",
        "option_4": "Law of conservation of momentum",
        "option_5": "Bernoulli's principle",
        "option_6": "Archimedes' principle",
        "option_7": "Pascal's principle",
        "option_8": "Hooke's law",
        "option_9": "Coulomb's law",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "What is the derivative of f(x) = x³ + 2x² - 5x + 3?",
        "option_0": "3x² + 4x - 5",
        "option_1": "x² + 2x - 5",
        "option_2": "3x² + 4x + 5", 
        "option_3": "x³ + 4x - 5",
        "option_4": "3x + 4",
        "option_5": "6x + 4",
        "option_6": "3x² - 5",
        "option_7": "x² + 4x",
        "option_8": "3x² + 2x - 5",
        "option_9": "Cannot be determined",
        "answer": "A",
        "answer_index": 0
    },
    {
        "question": "Which economic theory suggests that government spending can stimulate economic growth during recessions?",
        "option_0": "Monetarism",
        "option_1": "Keynesian economics",
        "option_2": "Supply-side economics",
        "option_3": "Austrian economics", 
        "option_4": "Classical economics",
        "option_5": "Behavioral economics",
        "option_6": "Neoclassical economics", 
        "option_7": "Chicago school economics",
        "option_8": "Post-Keynesian economics",
        "option_9": "Institutional economics",
        "answer": "B",
        "answer_index": 1
    }
]

KOREAN_FEW_SHOT_EXAMPLES = [
    {
        "question": "DNA 복제에 관한 다음 설명 중 올바른 것은?",
        "option_0": "DNA 복제는 3'에서 5' 방향으로 일어난다",
        "option_1": "DNA 복제는 반보존적이다",
        "option_2": "DNA 복제는 유사분열 동안에만 일어난다",
        "option_3": "DNA 복제는 동일한 사본을 만든다",
        "option_4": "DNA 복제는 보존적이다",
        "option_5": "DNA 복제는 양방향으로 일어난다",
        "option_6": "DNA 복제는 불연속적이다",
        "option_7": "위의 모든 것",
        "option_8": "위의 것 중 없음",
        "option_9": "DNA 복제는 연속적이다",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "진핵세포에서 미토콘드리아의 주요 기능은 무엇인가?",
        "option_0": "단백질 합성",
        "option_1": "ATP 생산",
        "option_2": "DNA 저장",
        "option_3": "노폐물 제거",
        "option_4": "세포 분열",
        "option_5": "광합성",
        "option_6": "지질 합성",
        "option_7": "RNA 처리",
        "option_8": "칼슘 저장",
        "option_9": "탄수화물 대사",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "운동하는 물체가 계속 운동하려는 경향을 설명하는 물리학 원리는?",
        "option_0": "뉴턴의 제2법칙",
        "option_1": "뉴턴의 제1법칙",
        "option_2": "뉴턴의 제3법칙",
        "option_3": "에너지 보존 법칙",
        "option_4": "운동량 보존 법칙",
        "option_5": "베르누이의 원리",
        "option_6": "아르키메데스의 원리",
        "option_7": "파스칼의 원리",
        "option_8": "후크의 법칙",
        "option_9": "쿨롱의 법칙",
        "answer": "B",
        "answer_index": 1
    },
    {
        "question": "f(x) = x³ + 2x² - 5x + 3의 도함수는?",
        "option_0": "3x² + 4x - 5",
        "option_1": "x² + 2x - 5",
        "option_2": "3x² + 4x + 5",
        "option_3": "x³ + 4x - 5",
        "option_4": "3x + 4",
        "option_5": "6x + 4",
        "option_6": "3x² - 5",
        "option_7": "x² + 4x",
        "option_8": "3x² + 2x - 5",
        "option_9": "결정할 수 없음",
        "answer": "A",
        "answer_index": 0
    },
    {
        "question": "경기 침체기에 정부 지출이 경제 성장을 촉진할 수 있다고 주장하는 경제 이론은?",
        "option_0": "통화주의",
        "option_1": "케인즈 경제학",
        "option_2": "공급 경제학",
        "option_3": "오스트리아 경제학",
        "option_4": "고전 경제학",
        "option_5": "행동 경제학",
        "option_6": "신고전 경제학",
        "option_7": "시카고 학파 경제학",
        "option_8": "포스트 케인즈 경제학",
        "option_9": "제도 경제학",
        "answer": "B",
        "answer_index": 1
    }
]

# --- Helper Functions ---
def create_5shot_prompt(item, few_shot_examples, language="en"):
    """
    Creates a 5-shot MMLU-ProX prompt for a given test item.
    """
    if language == "ko":
        prompt_parts = ["다음은 다양한 학문 분야의 전문적이고 어려운 다지선다형 질문입니다."]
    else:
        prompt_parts = ["The following are challenging multiple choice questions from various academic disciplines."]
    
    prompt_parts.append("")
    
    # Add few-shot examples
    for example in few_shot_examples:
        question = example["question"]
        options = []
        for i in range(10):
            option_key = f"option_{i}"
            if option_key in example and example[option_key].strip() and example[option_key].strip() != "N/A":
                options.append(f"{chr(65+i)}. {example[option_key]}")
        
        correct_answer = example["answer"]
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.extend(options)
        if language == "ko":
            prompt_parts.append(f"응답: 단계적으로 생각해봅시다. [사고 과정] 따라서 답은 {correct_answer}입니다.")
        else:
            prompt_parts.append(f"Response: Let's think step by step. [thinking process] So the answer is {correct_answer}.")
        prompt_parts.append("")
    
    # Add the test question
    question = item.get("question", "")
    options = []
    for i in range(10):
        option_key = f"option_{i}"
        if option_key in item and item[option_key].strip() and item[option_key].strip() != "N/A":
            options.append(f"{chr(65+i)}. {item[option_key]}")
    
    prompt_parts.append(f"Question: {question}")
    prompt_parts.extend(options)
    prompt_parts.append("")
    
    if language == "ko":
        prompt_parts.append("선택지에서 문자만을 최종 답변으로 선택해야 합니다.")
        prompt_parts.append("응답: 단계적으로 생각해봅시다.")
    else:
        prompt_parts.append("You should ONLY choose the letters from the options as your final answer.")
        prompt_parts.append("Response: Let's think step by step.")
    
    return "\n".join(prompt_parts)

def extract_answer_first_token(model_output):
    """
    Extract answer from model output using multiple strategies.
    Supports A-J for 10 options.
    Prioritizes patterns that appear later in the text (final answer).
    """
    if not model_output:
        return None
        
    cleaned_output = model_output.strip().upper()
    valid_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    import re
    
    # Strategy 1: Look for "ANSWER:" or "ANSWER IS" patterns (most reliable for final answer)
    final_answer_patterns = [
        r'(?:ANSWER\s*:?\s*|ANSWER\s+IS\s*:?\s*|THE\s+ANSWER\s+IS\s*:?\s*)([A-J])(?:\b|\.|$)',
        r'(?:THEREFORE\s*,?\s*|SO\s*,?\s*|THUS\s*,?\s*)(?:THE\s+ANSWER\s+IS\s*)?([A-J])(?:\b|\.|$)',
        r'(?:FINAL\s+ANSWER\s*:?\s*|CONCLUSION\s*:?\s*)([A-J])(?:\b|\.|$)'
    ]
    
    for pattern in final_answer_patterns:
        matches = list(re.finditer(pattern, cleaned_output))
        if matches:
            # Return the last match (most likely to be final answer)
            return matches[-1].group(1)
    
    # Strategy 2: Look for parentheses pattern, prioritizing later occurrences
    paren_matches = list(re.finditer(r'\(([A-J])\)', cleaned_output))
    if paren_matches:
        return paren_matches[-1].group(1)
    
    # Strategy 3: Look for standalone A-J at word boundaries, prioritizing later ones
    standalone_matches = list(re.finditer(r'\b([A-J])\b', cleaned_output))
    if standalone_matches:
        return standalone_matches[-1].group(1)
    
    # Strategy 4: Look for "option X" or "choice X" patterns
    option_matches = list(re.finditer(r'(?:OPTION|CHOICE)\s*([A-J])', cleaned_output))
    if option_matches:
        return option_matches[-1].group(1)
    
    # Strategy 5: Look for sentence-ending patterns
    sentence_end_matches = list(re.finditer(r'([A-J])(?:\.|!|\?|$)', cleaned_output))
    if sentence_end_matches:
        return sentence_end_matches[-1].group(1)
    
    # Strategy 6: First character if it's A-J (only if very short output)
    if len(cleaned_output) <= 3 and cleaned_output and cleaned_output[0] in valid_answers:
        return cleaned_output[0]
    
    # Strategy 7: Last resort - find the last A-J in the text
    last_answer = None
    for char in cleaned_output:
        if char in valid_answers:
            last_answer = char
    
    return last_answer

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
            max_length=4096  # Longer context for complex questions
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
    Performs 5-shot MMLU-ProX evaluation for a single model on both English and Korean datasets.
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

    logger.info(f"--- Starting 5-shot MMLU-ProX Evaluation for Model: {config.name} ---")
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
                    
                prompt = create_5shot_prompt(item, ENGLISH_FEW_SHOT_EXAMPLES, "en")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth in zip(batch_results, batch_ground_truths):
                is_correct = result['extracted_answer'] == ground_truth if result['extracted_answer'] else False
                
                # Only count items with valid extracted answers for total
                if result['extracted_answer']:
                    all_results["mmlu_prox_en"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_en"]["correct"] += 1
                
                all_results["mmlu_prox_en"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": result['extracted_answer'],
                    "is_correct": is_correct,
                    "raw_output": result['raw_output']
                })
                
                all_results["mmlu_prox_en"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": result['raw_output'],
                    "full_generation": result.get('full_generation', result['raw_output']),
                    "extracted_answer": result['extracted_answer']
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
                    
                prompt = create_5shot_prompt(item, KOREAN_FEW_SHOT_EXAMPLES, "ko")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth in zip(batch_results, batch_ground_truths):
                is_correct = result['extracted_answer'] == ground_truth if result['extracted_answer'] else False
                
                # Only count items with valid extracted answers for total
                if result['extracted_answer']:
                    all_results["mmlu_prox_ko"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_ko"]["correct"] += 1
                
                all_results["mmlu_prox_ko"]["details"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "predicted_answer": result['extracted_answer'],
                    "is_correct": is_correct,
                    "raw_output": result['raw_output']
                })
                
                all_results["mmlu_prox_ko"]["raw_generations"].append({
                    "index": result['index'],
                    "ground_truth": ground_truth,
                    "raw_output": result['raw_output'],
                    "full_generation": result.get('full_generation', result['raw_output']),
                    "extracted_answer": result['extracted_answer']
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
            "evaluation_type": "5-shot MMLU-ProX",
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
            "evaluation_type": "5-shot MMLU-ProX",
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

    # Save summary as SUMMARY.json
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Summary saved to: {summary_filepath}")
    logger.info("=== FINAL SUMMARY ===")
    for result in all_model_results:
        logger.info(f"{result['model_name']}:")
        logger.info(f"  MMLU-ProX EN Strict: {result.get('mmlu_prox_en_accuracy_strict', 0):.2f}%")
        logger.info(f"  MMLU-ProX KO Strict: {result.get('mmlu_prox_ko_accuracy_strict', 0):.2f}%")

if __name__ == "__main__":
    main()