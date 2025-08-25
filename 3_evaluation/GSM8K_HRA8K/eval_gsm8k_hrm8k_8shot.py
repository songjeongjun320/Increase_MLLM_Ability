#!/usr/bin/env python3
"""
GSM8K (HRM8K) Evaluation Script
- Evaluates mathematical reasoning capability on Korean translated GSM8K dataset
- Extracts numerical answers from model outputs
- Saves detailed results per model and creates final summary
"""

import os
import json
import logging
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from dataclasses import dataclass, field
import gc
import sys
from pathlib import Path
from datetime import datetime

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
    # Base Models
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

    # TOW Model
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

# --- Configuration ---
DATASET_PATH = "../../2_datasets/HRM8K_TEXT/GSM8K-test.json"
BASE_OUTPUT_DIR = "gsm8k_hrm8k_8shot_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"
BATCH_SIZE = 32

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- GSM8K 8-shot CoT Examples ---
GSM8K_8SHOT_COT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### So the answer is 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### So the answer is 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### So the answer is 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### So the answer is 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 * 2 = 4 more toys. 5 + 4 = 9. #### So the answer is 9. #### 9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### So the answer is 29. #### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more on wednesday, he had 35 - 2 = 33 golf balls. #### So the answer is 33. #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### So the answer is 8. #### 8"""

# --- GSM8K 8-shot Korean CoT Examples ---
GSM8K_8SHOT_KOR_COT_EXAMPLES = """Question: 숲에 15그루의 나무가 있습니다. 숲 관리인들이 오늘 숲에 나무를 심을 예정입니다. 작업이 끝나면 21그루의 나무가 있을 것입니다. 숲 관리인들이 오늘 몇 그루의 나무를 심었나요?
Answer: Let's think step by step. 원래 15그루의 나무가 있었습니다. 나무를 더 심은 후에는 21그루가 되었습니다. 따라서 21 - 15 = 6그루를 심었습니다. #### 따라서 정답은 6. #### 6

Question: 주차장에 3대의 차가 있고 2대의 차가 더 도착했다면, 주차장에는 몇 대의 차가 있나요?
Answer: Let's think step by step. 원래 3대의 차가 있었습니다. 2대의 차가 더 도착했습니다. 3 + 2 = 5입니다. #### 5

Question: Leah는 32개의 초콜릿을 가지고 있었고 그녀의 여동생은 42개를 가지고 있었습니다. 그들이 35개를 먹었다면, 총 몇 개가 남았나요?
Answer: Let's think step by step. 원래 Leah는 32개의 초콜릿을 가지고 있었습니다. 그녀의 여동생은 42개를 가지고 있었습니다. 따라서 총합은 32 + 42 = 74개였습니다. 35개를 먹은 후에는 74 - 35 = 39개가 남았습니다. #### 따라서 정답은 39. #### 39

Question: Jason은 20개의 막대사탕을 가지고 있었습니다. 그는 Denny에게 몇 개의 막대사탕을 주었습니다. 이제 Jason은 12개의 막대사탕을 가지고 있습니다. Jason이 Denny에게 몇 개의 막대사탕을 주었나요?
Answer: Let's think step by step. Jason은 처음에 20개의 막대사탕을 가지고 있었습니다. Denny에게 몇 개를 준 후에는 12개가 남았습니다. 따라서 Denny에게 20 - 12 = 8개를 주었습니다. #### 따라서 정답은 8. #### 8

Question: Shawn은 5개의 장난감을 가지고 있습니다. 크리스마스에 그는 엄마와 아빠로부터 각각 2개씩의 장난감을 받았습니다. 이제 그는 몇 개의 장난감을 가지고 있나요?
Answer: Let's think step by step. Shawn은 처음에 5개의 장난감을 가지고 있었습니다. 엄마와 아빠로부터 각각 2개씩 받았다면, 그것은 2 * 2 = 4개의 추가 장난감입니다. 5 + 4 = 9입니다. #### 따라서 정답은 9. #### 9

Question: 서버실에 9대의 컴퓨터가 있었습니다. 월요일부터 목요일까지 매일 5대씩 더 설치되었습니다. 이제 서버실에는 몇 대의 컴퓨터가 있나요?
Answer: Let's think step by step. 원래 9대의 컴퓨터가 있었습니다. 4일 동안 매일 5대씩 추가되었습니다. 따라서 5 * 4 = 20대의 컴퓨터가 추가되었습니다. 9 + 20 = 29입니다. #### 따라서 정답은 29. #### 29

Question: Michael은 58개의 골프공을 가지고 있었습니다. 화요일에 그는 23개의 골프공을 잃어버렸습니다. 수요일에는 2개를 더 잃어버렸습니다. 수요일 끝에 그는 몇 개의 골프공을 가지고 있었나요?
Answer: Let's think step by step. Michael은 처음에 58개의 골프공을 가지고 있었습니다. 화요일에 23개를 잃어버린 후, 그는 58 - 23 = 35개를 가지고 있었습니다. 수요일에 2개를 더 잃어버린 후, 그는 35 - 2 = 33개의 골프공을 가지고 있었습니다. #### 따라서 정답은 33. #### 33

Question: Olivia는 23달러를 가지고 있습니다. 그녀는 개당 3달러인 베이글 5개를 샀습니다. 그녀에게 얼마가 남았나요?
Answer: Let's think step by step. Olivia는 23달러를 가지고 있었습니다. 개당 3달러인 베이글 5개는 5 * 3 = 15달러입니다. 따라서 그녀에게는 23 - 15 = 8달러가 남았습니다. #### 따라서 정답은 8. #### 8"""

# --- Helper Functions ---
def load_gsm8k_data(filepath):
    """Load GSM8K dataset from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Data is not a list format")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Not all items in list are dictionaries")
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_gsm8k_prompt(text, is_korean=False):
    """Create GSM8K 8-shot CoT evaluation prompt following standard methodology"""    
    if is_korean:
        # Use Korean 8-shot Chain-of-Thought prompt for Korean questions
        prompt = f"""{GSM8K_8SHOT_KOR_COT_EXAMPLES}

Question: {text}
Answer: Let's think step by step. [Thinking] #### 따라서 답은 [ANSWER]. #### [ANSWER]."""
    else:
        # Use English 8-shot Chain-of-Thought prompt for English questions
        prompt = f"""{GSM8K_8SHOT_COT_EXAMPLES}

Question: {text}
Answer: Let's think step by step. [THINKING] #### The answer is [ANSWER]. #### [ANSWER]."""
    
    return prompt

def extract_numerical_answer(model_output):
    """
    Extract numerical answer from model output
    Prioritizes standard GSM8K CoT format "The answer is [number]"
    Also handles Korean patterns like "답: 18", "정답: 18.0", etc.
    """
    # Clean the output
    cleaned_output = model_output.strip()
    
    # Patterns to match numerical answers - prioritize #### format first
    patterns = [
        r'####\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # New #### format: "#### 18" (highest priority)
        r'답[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # Korean format: "답: 18"
        r'The answer is\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # Standard English GSM8K format: "The answer is 18"
        r'(?:정답|Answer)[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # 정답: 18, Answer: 18
        r'(?:답|정답|Answer)\s*(?:은|는|is)?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # 답은 18, 정답은 18
        r'(?:따라서|그러므로|그래서|결론적으로|최종적으로|Hence|Therefore)\s*(?:답|정답|answer)?[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # 따라서 답: 18
        r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?)(?:\s*(?:입니다|이다|\.|\s*$))',  # 18 달러입니다
        r'(?:총|합계|전체|Total)\s*[:：]?\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)',  # 총: 18
        r'=\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)(?:\s*(?:달러|원|개|명|미터|센티미터|킬로미터|시간|일|dollars?|won|pieces?|meters?|hours?|days?))?(?:\s*$)',  # = 18
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Take the last match (usually the final answer)
            answer_str = matches[-1].replace(',', '').strip()
            try:
                # Try to convert to float
                answer = float(answer_str)
                return answer
            except ValueError:
                continue
    
    # Last resort: find any number in the last line or paragraph
    lines = cleaned_output.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r'([+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)', line)
        if numbers:
            try:
                # Take the last number in the line
                answer_str = numbers[-1].replace(',', '')
                return float(answer_str)
            except ValueError:
                continue
    
    return None

def check_numerical_match(predicted, ground_truth, tolerance=1e-6):
    """
    Check if predicted answer matches ground truth with tolerance
    """
    if predicted is None or ground_truth is None:
        return False
    
    try:
        pred_float = float(predicted)
        gt_float = float(ground_truth)
        return abs(pred_float - gt_float) < tolerance
    except (ValueError, TypeError):
        return False

def evaluate_single_model(config: ModelConfig, gsm8k_data: list, model_output_dir: str):
    """
    Evaluate single model on GSM8K dataset
    """
    os.makedirs(model_output_dir, exist_ok=True)
    results_filepath = os.path.join(model_output_dir, f"results_{config.name}.json")
    log_filepath = os.path.join(model_output_dir, f"eval_{config.name}.log")
    raw_gen_filepath = os.path.join(model_output_dir, f"raw_generations_{config.name}.json")

    # Setup logging for this model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler is not file_handler:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logger.debug(f"Error removing old file handler: {e}")
    
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)

    logger.info(f"--- Starting GSM8K Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Results will be saved to: {results_filepath}")
    logger.info(f"Raw generations will be saved to: {raw_gen_filepath}")
    logger.info(f"Using Device: {DEVICE}, DType: {config.torch_dtype}")
    logger.info(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # Load Model and Tokenizer
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("Tokenizer lacks both pad and eos tokens. Adding new pad token '[PAD]'.")
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

        # Resize model embeddings if needed
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Load LoRA adapter if specified
        if config.adapter_path:
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")
            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")
            
            try:
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            logger.info("No LoRA adapter path specified. Using base model directly.")

        # Configure tokenizer padding
        if tokenizer.pad_token == tokenizer.eos_token and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings'):
            logger.warning("Resizing model embeddings after load due to added PAD token.")
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # Run Evaluation
        correct_predictions_korean = 0
        total_predictions_korean = 0
        errors_or_skipped_korean = 0
        results_details_korean = []
        
        correct_predictions_english = 0
        total_predictions_english = 0
        errors_or_skipped_english = 0
        results_details_english = []

        logger.info("Starting GSM8K inference loop...")
        logger.info(f"Dataset size: {len(gsm8k_data)}")

        # Process each item for both Korean and English versions
        for idx, item in enumerate(tqdm(gsm8k_data, desc=f"Evaluating {config.name} (GSM8K)")):
            ground_truth = item.get("answer", None)
            if ground_truth is None:
                logger.warning(f"Item with no ground truth found at index {idx}. Skipping.")
                errors_or_skipped_korean += 1
                errors_or_skipped_english += 1
                continue

            question = item.get("question", "")
            original = item.get("original", "")
            
            # Check if we have both Korean and English versions
            has_korean = question and original and question != original
            
            # Process Korean version (translated question)
            if has_korean:
                try:
                    korean_prompt = create_gsm8k_prompt(question, is_korean=True)
                    inputs = tokenizer(korean_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            temperature=1.0,
                        )
                    
                    input_lengths = inputs['input_ids'].shape[1]
                    output_only_tokens = outputs[:, input_lengths:]
                    korean_gen_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
                    
                    korean_answer = extract_numerical_answer(korean_gen_text)
                    is_correct_korean = False

                    if korean_answer is not None:
                        total_predictions_korean += 1
                        if check_numerical_match(korean_answer, ground_truth):
                            correct_predictions_korean += 1
                            is_correct_korean = True
                    else:
                        logger.warning(f"Korean - Item {idx}: Failed to extract answer from: '{korean_gen_text[:100]}...'")
                        errors_or_skipped_korean += 1
                        korean_gen_text = f"EXTRACTION_FAILED: {korean_gen_text}"

                    results_details_korean.append({
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_raw_output": korean_gen_text,
                        "extracted_answer": korean_answer,
                        "is_correct": is_correct_korean
                    })

                    raw_generations_list.append({
                        "index": idx,
                        "language": "Korean",
                        "question": question,
                        "original": original,
                        "ground_truth": ground_truth,
                        "raw_output": korean_gen_text,
                        "extracted_answer": korean_answer
                    })

                except Exception as e:
                    logger.error(f"Korean - Item {idx}: Inference error: {e}")
                    errors_or_skipped_korean += 1

            # Process English version (original question)  
            if original:
                try:
                    english_prompt = create_gsm8k_prompt(original, is_korean=False)
                    inputs = tokenizer(english_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            temperature=1.0,
                        )
                    
                    input_lengths = inputs['input_ids'].shape[1]
                    output_only_tokens = outputs[:, input_lengths:]
                    english_gen_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
                    
                    english_answer = extract_numerical_answer(english_gen_text)
                    is_correct_english = False

                    if english_answer is not None:
                        total_predictions_english += 1
                        if check_numerical_match(english_answer, ground_truth):
                            correct_predictions_english += 1
                            is_correct_english = True
                    else:
                        logger.warning(f"English - Item {idx}: Failed to extract answer from: '{english_gen_text[:100]}...'")
                        errors_or_skipped_english += 1
                        english_gen_text = f"EXTRACTION_FAILED: {english_gen_text}"

                    results_details_english.append({
                        "index": idx,
                        "question": original,
                        "ground_truth": ground_truth,
                        "model_raw_output": english_gen_text,
                        "extracted_answer": english_answer,
                        "is_correct": is_correct_english
                    })

                    raw_generations_list.append({
                        "index": idx,
                        "language": "English",
                        "question": question,
                        "original": original,
                        "ground_truth": ground_truth,
                        "raw_output": english_gen_text,
                        "extracted_answer": english_answer
                    })

                except Exception as e:
                    logger.error(f"English - Item {idx}: Inference error: {e}")
                    errors_or_skipped_english += 1

        # Final Results
        logger.info(f"Inference loop finished for {config.name}.")
        
        # Calculate accuracies for Korean
        accuracy_standard_korean = (correct_predictions_korean / total_predictions_korean * 100) if total_predictions_korean > 0 else 0
        accuracy_strict_korean = (correct_predictions_korean / len(gsm8k_data) * 100) if len(gsm8k_data) > 0 else 0

        # Calculate accuracies for English
        accuracy_standard_english = (correct_predictions_english / total_predictions_english * 100) if total_predictions_english > 0 else 0
        accuracy_strict_english = (correct_predictions_english / len(gsm8k_data) * 100) if len(gsm8k_data) > 0 else 0

        logger.info(f"--- GSM8K Results for {config.name} ({config.model_id}) ---")
        logger.info(f"Total Questions: {len(gsm8k_data)}")
        logger.info(f"=== Korean Results ===")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions_korean}")
        logger.info(f"Correct Predictions: {correct_predictions_korean}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped_korean}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard_korean:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_questions): {accuracy_strict_korean:.2f}%")
        logger.info(f"=== English Results ===")
        logger.info(f"Valid Predictions (Answer Extracted): {total_predictions_english}")
        logger.info(f"Correct Predictions: {correct_predictions_english}")
        logger.info(f"Errors or Skipped Items: {errors_or_skipped_english}")
        logger.info(f"Accuracy Standard (correct / valid_predictions): {accuracy_standard_english:.2f}%")
        logger.info(f"Accuracy Strict (correct / total_questions): {accuracy_strict_english:.2f}%")

        # Save Results
        config_dict_serializable = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.__dict__.items()}
        final_summary = {
            "model_config": config_dict_serializable,
            "dataset_path": DATASET_PATH,
            "evaluation_type": "GSM8K (HRM8K Korean and English Separate)",
            "total_questions": len(gsm8k_data),
            "korean_results": {
                "valid_predictions": total_predictions_korean,
                "correct_predictions": correct_predictions_korean,
                "errors_or_skipped": errors_or_skipped_korean,
                "accuracy_standard": accuracy_standard_korean,
                "accuracy_strict": accuracy_strict_korean,
                "details": results_details_korean
            },
            "english_results": {
                "valid_predictions": total_predictions_english,
                "correct_predictions": correct_predictions_english,
                "errors_or_skipped": errors_or_skipped_english,
                "accuracy_standard": accuracy_standard_english,
                "accuracy_strict": accuracy_strict_english,
                "details": results_details_english
            }
        }

        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file {results_filepath}: {e}")

        # Save Raw Generations
        logger.info(f"Saving raw model generations to {raw_gen_filepath}...")
        try:
            with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw generations saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save raw generations file {raw_gen_filepath}: {e}")

        return final_summary

    except Exception as e:
        logger.exception(f"Critical error during evaluation for {config.name}: {e}")
        return None

    finally:
        # Clean up resources
        logger.info(f"Cleaning up resources for {config.name}...")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Resources cleaned up for {config.name}.")
        
        # Remove file handler
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            try:
                root_logger.removeHandler(file_handler)
                file_handler.close()
            except Exception as e:
                logger.debug(f"Error closing/removing file handler: {e}")

def create_final_summary(all_results: list, base_output_dir: str):
    """Create final summary JSON with all model results"""
    final_results_korean = []
    final_results_english = []
    
    for result in all_results:
        if result is not None:
            # Korean results
            korean_summary = {
                "model_name": result["model_config"]["name"],
                "model_id": result["model_config"]["model_id"],
                "adapter_path": result["model_config"].get("adapter_path", None),
                "total_questions": result["total_questions"],
                "correct_predictions": result["korean_results"]["correct_predictions"],
                "valid_predictions": result["korean_results"]["valid_predictions"],
                "errors_or_skipped": result["korean_results"]["errors_or_skipped"],
                "accuracy_standard": result["korean_results"]["accuracy_standard"],
                "accuracy_strict": result["korean_results"]["accuracy_strict"],
                "evaluation_date": result.get("evaluation_date", "N/A")
            }
            final_results_korean.append(korean_summary)
            
            # English results
            english_summary = {
                "model_name": result["model_config"]["name"],
                "model_id": result["model_config"]["model_id"],
                "adapter_path": result["model_config"].get("adapter_path", None),
                "total_questions": result["total_questions"],
                "correct_predictions": result["english_results"]["correct_predictions"],
                "valid_predictions": result["english_results"]["valid_predictions"],
                "errors_or_skipped": result["english_results"]["errors_or_skipped"],
                "accuracy_standard": result["english_results"]["accuracy_standard"],
                "accuracy_strict": result["english_results"]["accuracy_strict"],
                "evaluation_date": result.get("evaluation_date", "N/A")
            }
            final_results_english.append(english_summary)
    
    # Sort by accuracy (strict) descending
    final_results_korean.sort(key=lambda x: x["accuracy_strict"], reverse=True)
    final_results_english.sort(key=lambda x: x["accuracy_strict"], reverse=True)
    
    final_summary = {
        "evaluation_type": "GSM8K (HRM8K Korean and English Separate Evaluation)",
        "dataset_info": {
            "name": "GSM8K-test (Korean translated and English original)",
            "path": DATASET_PATH,
            "total_questions": final_results_korean[0]["total_questions"] if final_results_korean else 0
        },
        "evaluation_summary": {
            "models_evaluated": len(final_results_korean),
            "best_model_korean": final_results_korean[0]["model_name"] if final_results_korean else "N/A",
            "best_accuracy_korean": final_results_korean[0]["accuracy_strict"] if final_results_korean else 0.0,
            "best_model_english": final_results_english[0]["model_name"] if final_results_english else "N/A",
            "best_accuracy_english": final_results_english[0]["accuracy_strict"] if final_results_english else 0.0
        },
        "korean_results": final_results_korean,
        "english_results": final_results_english
    }
    
    final_json_path = os.path.join(base_output_dir, "final_gsm8k_results.json")
    try:
        # Enhanced summary with performance analysis
        if create_enhanced_summary and final_results_korean and final_results_english:
            evaluation_info = {
                "evaluation_type": "GSM8K (HRM8K Korean and English Separate Evaluation)",
                "evaluation_date": datetime.now().isoformat(),
                "dataset_path": DATASET_PATH,
                "total_models_evaluated": len(final_results_korean)
            }
            
            # Analyze Korean results
            korean_enhanced = create_enhanced_summary(
                model_results=final_results_korean,
                evaluation_info=evaluation_info,
                primary_metric="accuracy_strict",
                subject_metric=None  # GSM8K doesn't have subject breakdown
            )
            
            # Analyze English results
            english_enhanced = create_enhanced_summary(
                model_results=final_results_english,
                evaluation_info=evaluation_info,
                primary_metric="accuracy_strict",
                subject_metric=None
            )
            
            # Combine analyses
            enhanced_summary = {
                "evaluation_info": evaluation_info,
                "korean_analysis": korean_enhanced,
                "english_analysis": english_enhanced,
                "original_summary": final_summary,
                "language_comparison": {
                    "korean_avg_score": korean_enhanced["performance_analysis"]["average_score"],
                    "english_avg_score": english_enhanced["performance_analysis"]["average_score"],
                    "korean_best_model": korean_enhanced["performance_analysis"]["best_model"],
                    "english_best_model": english_enhanced["performance_analysis"]["best_model"],
                    "performance_difference": english_enhanced["performance_analysis"]["average_score"] - korean_enhanced["performance_analysis"]["average_score"]
                }
            }
            
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)
                
            # Log key insights
            logger.info(f"🏆 Best Korean model: {korean_enhanced['performance_analysis']['best_model']} ({korean_enhanced['performance_analysis']['average_score']:.2f}%)")
            logger.info(f"🏆 Best English model: {english_enhanced['performance_analysis']['best_model']} ({english_enhanced['performance_analysis']['average_score']:.2f}%)")
            logger.info(f"📈 Language performance gap: {abs(enhanced_summary['language_comparison']['performance_difference']):.2f}%p")
            
        else:
            # Fallback to basic summary
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Summary saved to {final_json_path}")
        
        # Also create separate CSV files for Korean and English results
        korean_csv_path = os.path.join(base_output_dir, "gsm8k_results_korean.csv")
        with open(korean_csv_path, 'w', encoding='utf-8') as f:
            f.write("Model Name,Accuracy Standard (%),Accuracy Strict (%),Correct,Valid,Total\n")
            for result in final_results_korean:
                f.write(f"{result['model_name']},{result['accuracy_standard']:.2f},{result['accuracy_strict']:.2f},{result['correct_predictions']},{result['valid_predictions']},{result['total_questions']}\n")
        logger.info(f"Korean CSV summary saved to {korean_csv_path}")
        
        english_csv_path = os.path.join(base_output_dir, "gsm8k_results_english.csv")
        with open(english_csv_path, 'w', encoding='utf-8') as f:
            f.write("Model Name,Accuracy Standard (%),Accuracy Strict (%),Correct,Valid,Total\n")
            for result in final_results_english:
                f.write(f"{result['model_name']},{result['accuracy_standard']:.2f},{result['accuracy_strict']:.2f},{result['correct_predictions']},{result['valid_predictions']},{result['total_questions']}\n")
        logger.info(f"English CSV summary saved to {english_csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to save final summary: {e}")

def main():
    """Main execution function"""
    logger.info(f"Loading GSM8K data from: {DATASET_PATH}")
    gsm8k_data = load_gsm8k_data(DATASET_PATH)
    if gsm8k_data is None:
        logger.error("Could not load GSM8K data. Exiting.")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")

    all_results = []

    # Evaluate each model
    for config in MODEL_CONFIGS:
        logger.info(f"\n===== Starting Evaluation for Model: {config.name} =====\n")
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_output_dir, exist_ok=True)
        logger.info(f"Output for model {config.name} will be in: {model_output_dir}")

        result = evaluate_single_model(config, gsm8k_data, model_output_dir)
        if result is not None:
            result["evaluation_date"] = str(Path().resolve()).split('/')[-1]  # Simple timestamp
            all_results.append(result)

        logger.info(f"\n===== Finished Evaluation for Model: {config.name} =====")
        print("-" * 80)

    # Create final summary
    logger.info("Creating final summary of all results...")
    create_final_summary(all_results, BASE_OUTPUT_DIR)
    
    logger.info("All GSM8K evaluations complete.")

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    import transformers
    logger.info(f"Transformers library version: {transformers.__version__}")
    logger.info(f"Torch library version: {torch.__version__}")
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Using cache directory: {CACHE_DIR}")

    main()