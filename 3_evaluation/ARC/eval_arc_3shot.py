import os
import json
import logging
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    # Fallback for older tqdm versions
    from contextlib import nullcontext
    logging_redirect_tqdm = nullcontext
from dataclasses import dataclass, field
import gc
import sys
import time
import random

# Import ToW token checker
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from check_tokenizer import check_tow_tokens_for_eval

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high") 

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

    # ModelConfig(
    #     name="llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="qwem-2.5-3b-pt-tow-09_11_2epoch_allenai-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_2epoch_allenai-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-pt-tow-09_11_2epoch_allenai-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_2epoch_allenai-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="olmo-2-0425-1b-tow-09_11_2epoch_allenai-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_allenai-merged",
    #     use_quantization=False
    # ),

    # ModelConfig(
    #     name="llama-3.2-3b-tow-09_11_2epoch_org_initialize-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-tow-09_11_2epoch_org_initialize-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="qwem-2.5-3b-pt-tow-09_11_2epoch_org_initialize-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_2epoch_org_initialize-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-pt-tow-09_11_2epoch_org_initialize-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_2epoch_org_initialize-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="olmo-2-0425-1b-tow-09_11_2epoch_org_initialize-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_org_initialize-merged",
    #     use_quantization=False
    # ),

    # ModelConfig(
    #     name="llama-3.2-3b-tow-09_11_2epoch_fix_tow-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-tow-09_11_2epoch_fix_tow-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="qwem-2.5-3b-tow-09_11_2epoch_fix_tow-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-tow-09_11_2epoch_fix_tow-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="gemma-3-4b-tow-09_11_2epoch_fix_tow-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-tow-09_11_2epoch_fix_tow-merged",
    #     use_quantization=False
    # ),
    # ModelConfig(
    #     name="olmo-2-0425-1b-tow-09_11_2epoch_fix_tow-merged",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_2epoch_fix_tow-merged",
    #     use_quantization=False
    # ),

    # 10 Epochs
    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_10epoch-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt-tow-09_11_10epoch-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt-tow-09_11_10epoch-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="olmo-2-0425-1b-tow-09_11_10epoch-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_allenai-merged",
        use_quantization=False
    ),
]

# --- General Configuration ---
ARC_DATASET_PATH = "../../2_datasets/ARC/ARC.json"
KO_ARC_DATASET_PATH = "../../2_datasets/ARC/Ko-ARC.json"
BASE_OUTPUT_DIR = "arc_3shot_09_11_tokenizer_added"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"
BATCH_SIZE = 16

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ARC_5SHOT_EXAMPLES = [
    {
        "question": "Which of the following is the primary source of energy for most ecosystems on Earth?",
        "options": {
            "A": "Fungi",
            "B": "Herbivores",
            "C": "The Sun",
            "D": "Carnivores"
        },
        "cot_content": """Let's think step by step. The question asks for the primary, or ultimate, source of energy for most ecosystems. Let's analyze the options. A food chain shows how energy is transferred. Option A, Fungi, are decomposers. They get energy from dead organic material, so they are part of the energy cycle, but not the primary source. Option B, Herbivores, are primary consumers. They get their energy by eating plants (producers). So they are not the source. Option D, Carnivores, are secondary or tertiary consumers. They get energy by eating other animals. They are even further down the energy transfer chain. Option C, The Sun. Plants (producers) use sunlight to create their own food through photosynthesis. This chemical energy is the foundation of almost every food chain. Therefore, the sun is the primary source of energy.""",
        "answer": "C"
    },
    {
        "question": "Which physical process describes a liquid turning into a gas?",
        "options": {
            "A": "Melting",
            "B": "Freezing",
            "C": "Condensation",
            "D": "Evaporation"
        },
        "cot_content": """Let's think step by step. The question is about the phase transition from liquid to gas. Option A, Melting, is the process of a solid turning into a liquid. This is incorrect. Option B, Freezing, is the process of a liquid turning into a solid. This is incorrect. Option C, Condensation, is the process of a gas turning into a liquid. This is the reverse of what the question asks. This is incorrect. Option D, Evaporation (or boiling), is the process where a liquid substance becomes a gas. This directly matches the question.""",
        "answer": "D"
    },
    {
        "question": "A student wants to test how the amount of sunlight affects the growth of a bean plant. Which of the following is the independent variable in her experiment?",
        "options": {
            "A": "the height of the plant",
            "B": "the amount of water given to the plant",
            "C": "the amount of sunlight",
            "D": "the type of soil"
        },
        "cot_content": """Let's think step by step. An experiment tests how an independent variable affects a dependent variable. The independent variable is the one factor that the scientist intentionally changes or manipulates. The student wants to see the effect *of* the amount of sunlight. This is the factor she will change. Option A, the height of the plant, is what is being measured to see the effect. This is the dependent variable. Option B, the amount of water, and Option D, the type of soil, should be kept the same for all plants to ensure a fair test. These are controlled variables. Option C, the amount of sunlight, is the one thing the student is purposefully changing to observe its effect on growth. Therefore, it is the independent variable.""",
        "answer": "C"
    },
]

KO_ARC_5SHOT_EXAMPLES = [
    {
        "question": "다음 중 지구상 대부분의 생태계에서 주요 에너지원은 무엇입니까?",
        "options": {
            "A": "균류",
            "B": "초식동물",
            "C": "태양",
            "D": "육식동물"
        },
        "cot_content": """단계별로 생각해봅시다. 이 질문은 대부분의 생태계에서 가장 근원적인 에너지 공급원이 무엇인지 묻고 있습니다. 선택지를 분석해 봅시다. 먹이 사슬은 에너지가 어떻게 전달되는지를 보여줍니다. 선택지 A, 균류는 분해자입니다. 죽은 유기물로부터 에너지를 얻으므로 에너지 순환의 일부이지만 근원적인 에너지원은 아닙니다. 선택지 B, 초식동물은 1차 소비자입니다. 식물(생산자)을 먹음으로써 에너지를 얻으므로 에너지원이 아닙니다. 선택지 D, 육식동물은 2차 또는 3차 소비자입니다. 다른 동물을 먹음으로써 에너지를 얻으며, 에너지 전달 단계에서 더 뒤에 있습니다. 선택지 C, 태양. 식물(생산자)은 광합성을 통해 태양빛을 이용하여 스스로 양분을 만듭니다. 이 화학 에너지가 거의 모든 먹이 사슬의 기초가 됩니다. 따라서 태양이 주요 에너지원입니다.""",
        "answer": "C"
    },
    {
        "question": "액체가 기체로 변하는 물리적 과정은 무엇입니까?",
        "options": {
            "A": "융해",
            "B": "응고",
            "C": "액화",
            "D": "증발"
        },
        "cot_content": """단계별로 생각해봅시다. 이 질문은 액체에서 기체로의 상태 변화에 관한 것입니다. 선택지 A, 융해는 고체가 액체로 변하는 과정입니다. 틀렸습니다. 선택지 B, 응고는 액체가 고체로 변하는 과정입니다. 틀렸습니다. 선택지 C, 액화는 기체가 액체로 변하는 과정입니다. 질문과 반대되는 과정입니다. 틀렸습니다. 선택지 D, 증발(또는 끓음)은 액체 물질이 기체로 변하는 과정입니다. 이는 질문과 정확히 일치합니다.""",
        "answer": "D"
    },
    {
        "question": "한 학생이 햇빛의 양이 강낭콩 식물의 성장에 미치는 영향을 시험하고 싶어 합니다. 이 실험에서 독립 변인은 무엇입니까?",
        "options": {
            "A": "식물의 키",
            "B": "식물에게 주는 물의 양",
            "C": "햇빛의 양",
            "D": "토양의 종류"
        },
        "cot_content": """단계별로 생각해봅시다. 실험은 독립 변인이 종속 변인에 미치는 영향을 시험합니다. 독립 변인은 과학자가 의도적으로 변화시키거나 조작하는 하나의 요인입니다. 학생은 햇빛의 양이 미치는 '영향'을 보고 싶어 하므로, 햇빛의 양이 바로 학생이 변화시킬 요인입니다. 선택지 A, 식물의 키는 햇빛의 영향을 확인하기 위해 측정되는 것입니다. 이것은 종속 변인입니다. 선택지 B, 물의 양과 선택지 D, 토양의 종류는 공정한 실험을 위해 모든 식물에게 동일하게 유지되어야 합니다. 이것들은 통제 변인입니다. 선택지 C, 햇빛의 양은 학생이 성장에 미치는 영향을 관찰하기 위해 의도적으로 변화시키는 유일한 것입니다. 따라서 이것이 독립 변인입니다.""",
        "answer": "C"
    },
]

# --- Helper Functions for 3-shot ARC Evaluation ---
def create_3shot_prompt(item, examples, dataset_type="arc"):
    """
    (최종 개선 버전)
    딕셔셔너리 리스트 형태의 고품질 3-shot 예제를 사용하여
    ARC / Ko-ARC 평가 프롬프트를 동적으로 생성합니다.
    """
    if dataset_type == "arc":
        prompt_parts = ["The following are multiple choice questions about science and reasoning. You MUST choose one of the option A~D.\n"]
        response_header = "Response:"
        cot_trigger = "Let's think step by step."
        final_answer_prefix = "Therefore Answer:"
        
    else:  # ko-arc
        prompt_parts = ["다음은 과학과 추론에 관한 객관식 문제들입니다. A부터 D까지의 보기중 무조건 하나의 답만 선택하세요.\n"]
        response_header = "응답:"
        cot_trigger = "단계적으로 생각해봅시다."
        final_answer_prefix = "따라서 정답:"

    # 1. 3개의 예제를 동적으로 생성합니다.
    for example in examples:
        # 예제 딕셔너리에서 각 부분을 가져옵니다.
        question = example["question"]
        options_dict = example["options"]
        cot_content = example["cot_content"] # 실제 추론 과정을 사용합니다.
        answer = example["answer"]

        # 질문과 선택지를 프롬프트에 추가합니다.
        prompt_parts.append(question)
        for key, value in sorted(options_dict.items()):
            prompt_parts.append(f"{key}. {value}")
        
        # 실제 추론 과정과 최종 답변 형식을 포함한 완전한 응답 블록을 만듭니다.
        full_response_block = f"{response_header} {cot_content} #### {final_answer_prefix} {{{answer}}}. #### {{{answer}}}."
        prompt_parts.append(full_response_block)
        prompt_parts.append("") # 예제 사이에 빈 줄 추가

    # 2. 모델이 풀어야 할 실제 문제를 추가합니다.
    test_question = item.get("question", "")
    prompt_parts.append(test_question)
    # 실제 데이터셋('item')의 선택지 형식에 맞춰 처리합니다.
    for key in ['A', 'B', 'C', 'D']:
        if key in item:
            prompt_parts.append(f"{key}. {item[key]}")
    prompt_parts.append("")

    # 3. 모델의 추론을 유도하는 깔끔한 시작 신호로 프롬프트를 마무리합니다.
    prompt_parts.append(f"{response_header} {cot_trigger}")
    
    return "\n".join(prompt_parts)


def process_single_with_retry(model, tokenizer, prompt, max_retries=0):
    """
    Process a single prompt with retry logic for answer extraction failures
    Only retries when answer extraction fails (not on genuine model errors)
    """
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(DEVICE)
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            
            input_lengths = inputs['input_ids'].shape[1]
            output_only_tokens = outputs[:, input_lengths:]
            generated_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
            
            # Try to extract answer
            extracted_answer = extract_answer_robust(generated_text)
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

def extract_answer_robust(model_output: str) -> str:
    """
    Extract the final answer (A, B, C, D) from model output using STRICT validation.
    Returns None if no clear structured answer is found.
    STRICT MODE: Only accepts {} format - unified across all evaluation scripts.
    """
    if not model_output:
        return None

    cleaned_output = model_output.strip().upper()

    import re

    # STRICT: Only accept {} format for consistency across all evaluation scripts
    box_pattern = r'\{([A-D])\}'
    box_matches = re.findall(box_pattern, cleaned_output)
    if box_matches:
        return box_matches[-1]  # Return the last match (final answer)

    # No fallback patterns - forces models to use {} format only
    return None

def load_arc_data(filepath):
    """Loads ARC data from a JSON file."""
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
    answer = item.get("answer", "")
    if answer in ['A', 'B', 'C', 'D']:
        return answer
    return None

def select_examples(data, test_item, num_examples=5):
    """
    Select examples for few-shot prompting, excluding the test item.
    """
    examples = []
    test_id = test_item.get("id", "")
    
    for item in data:
        if item.get("id", "") != test_id and len(examples) < num_examples:
            examples.append(item)
    
    return examples

def save_failure_cases(failure_cases, model_name, output_dir):
    """
    Save failure cases to a separate JSON file for analysis.
    """
    failure_filepath = os.path.join(output_dir, f"failure_cases_{model_name}_3shot.json")
    
    with open(failure_filepath, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(failure_cases)} failure cases to {failure_filepath}")

# --- Single Model Evaluation Function with 3-shot Prompting ---
def evaluate_single_model(config: ModelConfig, arc_data: list, ko_arc_data: list, model_specific_output_dir: str):
    """
    Performs 3-shot ARC evaluation for a single model on both datasets.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_3shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_3shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_3shot.json")

    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove previous file handlers to avoid duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 3-shot Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # --- Load Model and Tokenizer ---
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR, padding_side='left')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # Gemma 모델에서만 컴파일 비활성화
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")
            
        # --- Evaluate on both datasets ---
        all_results = {}
        all_failure_cases = {}  # Store failure cases for both datasets
        
        datasets = [
            ("ARC", arc_data, "arc"),
            ("Ko-ARC", ko_arc_data, "ko-arc")
        ]
        
        for dataset_name, dataset, dataset_type in datasets:
            logger.info(f"Starting evaluation on {dataset_name} dataset...")
            
            if dataset_type == "arc":
                examples_to_use = ARC_5SHOT_EXAMPLES
            else:  # "ko-arc"
                examples_to_use = KO_ARC_5SHOT_EXAMPLES

            correct_predictions = 0
            total_predictions = 0
            errors_or_skipped = 0
            results_details = []
            failure_cases = []  # Store failure cases for this dataset
            
            # Batch processing loop with tqdm logging redirect
            num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
            
            with logging_redirect_tqdm():
                pbar = tqdm(range(num_batches), 
                           desc=f"Evaluating {config.name} on {dataset_name} (3-shot, errors: 0)",
                           ncols=100,  # 고정 너비
                           unit="batch",
                           leave=True,
                           dynamic_ncols=False,
                           file=sys.stdout,
                           position=0)
                
                for i in pbar:
                    batch_start = i * BATCH_SIZE
                    batch_end = batch_start + BATCH_SIZE
                    batch = dataset[batch_start:batch_end]
                    
                    prompts = []
                    ground_truths = []
                    valid_items_in_batch = []

                    for item in batch:
                        ground_truth = get_ground_truth(item)
                        if ground_truth is None:
                            errors_or_skipped += 1
                            # Log skipped item if needed
                            continue

                        prompt = create_3shot_prompt(item, examples_to_use, dataset_type)
                        prompts.append(prompt)
                        ground_truths.append(ground_truth)
                        valid_items_in_batch.append(item)

                    if not prompts:
                        continue

                    try:
                        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(DEVICE)
                        
                        with torch.inference_mode():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=False,
                            )
                        
                        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        
                        input_lengths = inputs['input_ids'].shape[1]
                        # The generated text includes the prompt, so we need to remove it.
                        # Be careful with batch_decode, it might handle prompts differently.
                        # A safer way is to decode only the generated part.
                        output_only_tokens = outputs[:, input_lengths:]
                        decoded_outputs = tokenizer.batch_decode(output_only_tokens, skip_special_tokens=True)


                        for j, (item, ground_truth, gen_text) in enumerate(zip(valid_items_in_batch, ground_truths, decoded_outputs)):
                            generated_text_log = gen_text.strip()
                            model_answer_log = extract_answer_robust(generated_text_log)
                            is_correct_log = False

                            if model_answer_log:
                                total_predictions += 1
                                if model_answer_log == ground_truth:
                                    correct_predictions += 1
                                    is_correct_log = True
                                else:
                                    # This is a wrong answer - add to failure cases
                                    failure_cases.append({
                                        "index": batch_start + j,
                                        "id": item.get("id", ""),
                                        "dataset": dataset_name,
                                        "question": item.get("question", ""),
                                        "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                        "ground_truth": ground_truth,
                                        "predicted_answer": model_answer_log,
                                        "raw_output": generated_text_log,
                                        "failure_type": "incorrect_answer"
                                    })
                            else:
                                # Batch extraction failed, try individual retry for this item
                                logger.warning(f"Batch extraction failed for item {batch_start + j}, attempting individual retry...")
                                prompt = create_3shot_prompt(item, examples_to_use, dataset_type)
                                retry_text, retry_answer = process_single_with_retry(model, tokenizer, prompt)
                            
                                if retry_answer is not None:
                                    generated_text_log = retry_text
                                    model_answer_log = retry_answer
                                    total_predictions += 1
                                    if model_answer_log == ground_truth:
                                        correct_predictions += 1
                                        is_correct_log = True
                                    else:
                                        # This is a wrong answer after retry - add to failure cases
                                        failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("id", ""),
                                            "dataset": dataset_name,
                                            "question": item.get("question", ""),
                                            "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                            "ground_truth": ground_truth,
                                            "predicted_answer": model_answer_log,
                                            "raw_output": generated_text_log,
                                            "failure_type": "incorrect_answer_after_retry"
                                        })
                                    logger.info(f"Retry successful for item {batch_start + j}: extracted '{retry_answer}'")
                                else:
                                    # Even retry failed - add to failure cases
                                    if not retry_text.startswith("ERROR"):
                                        logger.warning(f"Item {batch_start + j}: Failed to extract answer after retries")
                                        errors_or_skipped += 1
                                        generated_text_log = f"EXTRACTION_FAILED: {retry_text}"
                                        failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("id", ""),
                                            "dataset": dataset_name,
                                            "question": item.get("question", ""),
                                            "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                            "ground_truth": ground_truth,
                                            "predicted_answer": -1,
                                            "raw_output": generated_text_log,
                                            "failure_type": "extraction_failed"
                                        })
                                    else:
                                        # This was a model error, not extraction failure  
                                        logger.error(f"Item {batch_start + j}: Model error: {retry_text}")
                                        errors_or_skipped += 1
                                        generated_text_log = retry_text
                                        failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("id", ""),
                                            "dataset": dataset_name,
                                            "question": item.get("question", ""),
                                            "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                            "ground_truth": ground_truth,
                                            "predicted_answer": -1,
                                            "raw_output": generated_text_log,
                                            "failure_type": "model_error"
                                        })

                            current_item_index = batch_start + j # or find a better way to get original index
                            results_details.append({
                                "index": current_item_index, 
                                "id": item.get("id", ""),
                                "ground_truth": ground_truth, 
                                "model_raw_output": generated_text_log,
                                "predicted_answer": model_answer_log, 
                                "is_correct": is_correct_log
                            })
                            
                            raw_generations_list.append({
                                "dataset": dataset_name,
                                "index": current_item_index, 
                                "id": item.get("id", ""),
                                "ground_truth": ground_truth,
                                "raw_output": generated_text_log, 
                                "extracted_answer": model_answer_log
                            })

                    except Exception as e:
                        logger.error(f"Batch {i}: Inference error: {e}", exc_info=False)
                        # Add all items in this batch to failure cases
                        for j, (item, ground_truth) in enumerate(zip(valid_items_in_batch, ground_truths)):
                            failure_cases.append({
                                "index": batch_start + j,
                                "id": item.get("id", ""),
                                "dataset": dataset_name,
                                "question": item.get("question", ""),
                                "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                "ground_truth": ground_truth,
                                "predicted_answer": -1,
                                "raw_output": f"BATCH_ERROR: {str(e)}",
                                "failure_type": "batch_inference_error"
                            })
                        errors_or_skipped += len(prompts)
                
                    # Update progress bar with current error count
                    pbar.set_description(f"Evaluating {config.name} on {dataset_name} (3-shot, errors: {errors_or_skipped})")

            
            # Calculate accuracy
            accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            accuracy_strict = (correct_predictions / len(dataset) * 100) if len(dataset) > 0 else 0

            logger.info(f"--- 3-shot {dataset_name} Results for {config.name} ---")
            logger.info(f"Test Items: {len(dataset)}")
            logger.info(f"Valid Predictions: {total_predictions}")
            logger.info(f"Correct Predictions: {correct_predictions}")
            logger.info(f"Failure Cases: {len(failure_cases)}")
            logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
            logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")
            
            all_results[dataset_name] = {
                "test_items": len(dataset),
                "valid_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "failure_cases_count": len(failure_cases),
                "accuracy_standard": accuracy_standard,
                "accuracy_strict": accuracy_strict,
                "details": results_details
            }
            
            # Store failure cases for this dataset
            all_failure_cases[dataset_name] = failure_cases

        # --- Save Results ---
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "3-shot ARC Challenge",
            "datasets": all_results
        }
        
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)
        
        # Save failure cases for both datasets combined
        all_failure_cases_combined = []
        for dataset_name, cases in all_failure_cases.items():
            all_failure_cases_combined.extend(cases)
        
        save_failure_cases(all_failure_cases_combined, config.name, model_specific_output_dir)

        return all_results

    except Exception as e:
        logger.exception(f"A critical error occurred during evaluation for {config.name}: {e}")
        return None
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
    arc_data = load_arc_data(ARC_DATASET_PATH)
    ko_arc_data = load_arc_data(KO_ARC_DATASET_PATH)
    
    if not arc_data or not ko_arc_data:
        logger.error("Failed to load one or both datasets")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Store summary results for all models
    summary_results = {}

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        
        results = evaluate_single_model(config, arc_data, ko_arc_data, model_specific_output_dir)
        
        if results:
            summary_results[config.name] = {
                "model_id": config.model_id,
                "adapter_path": config.adapter_path,
                "ARC_accuracy_standard": results["ARC"]["accuracy_standard"],
                "ARC_accuracy_strict": results["ARC"]["accuracy_strict"],
                "ARC_failure_cases": results["ARC"]["failure_cases_count"],
                "Ko-ARC_accuracy_standard": results["Ko-ARC"]["accuracy_standard"],
                "Ko-ARC_accuracy_strict": results["Ko-ARC"]["accuracy_strict"],
                "Ko-ARC_failure_cases": results["Ko-ARC"]["failure_cases_count"]
            }

    # Save summary results
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary results saved to: {summary_filepath}")
    
    # Print summary table
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    print(f"{'Model Name':<30} {'ARC Acc (%)':<15} {'ARC Fails':<12} {'Ko-ARC Acc (%)':<17} {'Ko-ARC Fails':<15}")
    print("-"*100)
    
    for model_name, results in summary_results.items():
        arc_acc = results["ARC_accuracy_standard"]
        ko_arc_acc = results["Ko-ARC_accuracy_standard"]
        arc_fails = results["ARC_failure_cases"]
        ko_arc_fails = results["Ko-ARC_failure_cases"]
        print(f"{model_name:<30} {arc_acc:<15.2f} {arc_fails:<12} {ko_arc_acc:<17.2f} {ko_arc_fails:<15}")
    
    print("="*100)

if __name__ == "__main__":
    main()