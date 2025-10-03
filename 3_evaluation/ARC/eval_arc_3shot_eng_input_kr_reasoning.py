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
    ModelConfig(
        name="llama-3.2-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
        use_quantization=False
    ),
]

# --- General Configuration ---
ARC_DATASET_PATH = "../../2_datasets/ARC/ARC.json"
BASE_OUTPUT_DIR = "arc_3shot_09_11_eng_input_kr_reasoning"
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

# English questions and options for few-shot examples
ARC_5SHOT_EXAMPLES = [
    {
        "question": "Which of the following is the primary source of energy for most ecosystems on Earth?",
        "options": {
            "A": "Fungi",
            "B": "Herbivores",
            "C": "The Sun",
            "D": "Carnivores"
        },
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
        "answer": "C"
    },
]

# Korean reasoning (cot_content) for few-shot examples
KO_ARC_5SHOT_EXAMPLES = [
    {
        "cot_content": """단계별로 생각해봅시다. 이 질문은 대부분의 생태계에서 가장 근원적인 에너지 공급원이 무엇인지 묻고 있습니다. 선택지를 분석해 봅시다. 먹이 사슬은 에너지가 어떻게 전달되는지를 보여줍니다. 선택지 A, 균류는 분해자입니다. 죽은 유기물로부터 에너지를 얻으므로 에너지 순환의 일부이지만 근원적인 에너지원은 아닙니다. 선택지 B, 초식동물은 1차 소비자입니다. 식물(생산자)을 먹음으로써 에너지를 얻으므로 에너지원이 아닙니다. 선택지 D, 육식동물은 2차 또는 3차 소비자입니다. 다른 동물을 먹음으로써 에너지를 얻으며, 에너지 전달 단계에서 더 뒤에 있습니다. 선택지 C, 태양. 식물(생산자)은 광합성을 통해 태양빛을 이용하여 스스로 양분을 만듭니다. 이 화학 에너지가 거의 모든 먹이 사슬의 기초가 됩니다. 따라서 태양이 주요 에너지원입니다.""",
        "answer": "C"
    },
    {
        "cot_content": """단계별로 생각해봅시다. 이 질문은 액체에서 기체로의 상태 변화에 관한 것입니다. 선택지 A, 융해는 고체가 액체로 변하는 과정입니다. 틀렸습니다. 선택지 B, 응고는 액체가 고체로 변하는 과정입니다. 틀렸습니다. 선택지 C, 액화는 기체가 액체로 변하는 과정입니다. 질문과 반대되는 과정입니다. 틀렸습니다. 선택지 D, 증발(또는 끓음)은 액체 물질이 기체로 변하는 과정입니다. 이는 질문과 정확히 일치합니다.""",
        "answer": "D"
    },
    {
        "cot_content": """단계별로 생각해봅시다. 실험은 독립 변인이 종속 변인에 미치는 영향을 시험합니다. 독립 변인은 과학자가 의도적으로 변화시키거나 조작하는 하나의 요인입니다. 학생은 햇빛의 양이 미치는 '영향'을 보고 싶어 하므로, 햇빛의 양이 바로 학생이 변화시킬 요인입니다. 선택지 A, 식물의 키는 햇빛의 영향을 확인하기 위해 측정되는 것입니다. 이것은 종속 변인입니다. 선택지 B, 물의 양과 선택지 D, 토양의 종류는 공정한 실험을 위해 모든 식물에게 동일하게 유지되어야 합니다. 이것들은 통제 변인입니다. 선택지 C, 햇빛의 양은 학생이 성장에 미치는 영향을 관찰하기 위해 의도적으로 변화시키는 유일한 것입니다. 따라서 이것이 독립 변인입니다.""",
        "answer": "C"
    },
]

# --- Helper Functions for 3-shot ARC Evaluation ---
def create_3shot_prompt(item, add_bos_token=False, bos_token=""):
    """
    Creates few-shot prompt with English questions/options but Korean reasoning.
    - Uses ARC_5SHOT_EXAMPLES for English questions and options
    - Uses KO_ARC_5SHOT_EXAMPLES for Korean reasoning (cot_content)
    - Always uses "arc" dataset type settings
    - Uses Korean cot_trigger: "단계적으로 생각해봅시다."
    """
    # Always use ARC (English) format for headers
    prompt_parts = ["The following are multiple choice questions about science and reasoning. You MUST choose one of the option A~D.\n"]
    response_header = "Response:"
    cot_trigger = "단계적으로 생각해봅시다."  # Korean trigger
    final_answer_prefix = "Therefore Answer:"

    # 1. Create 3 few-shot examples combining English questions with Korean reasoning
    for eng_example, ko_example in zip(ARC_5SHOT_EXAMPLES, KO_ARC_5SHOT_EXAMPLES):
        # Use English question and options from ARC_5SHOT_EXAMPLES
        question = eng_example["question"]
        options_dict = eng_example["options"]

        # Use Korean reasoning from KO_ARC_5SHOT_EXAMPLES
        cot_content = ko_example["cot_content"]
        answer = ko_example["answer"]  # Use Korean example's answer

        # Add question and options to prompt
        prompt_parts.append(question)
        for key, value in sorted(options_dict.items()):
            prompt_parts.append(f"{key}. {value}")

        # Create complete response block with Korean reasoning
        full_response_block = f"{response_header} {cot_content} #### {final_answer_prefix} {{{answer}}}. #### {{{answer}}}."
        prompt_parts.append(full_response_block)
        prompt_parts.append("")  # Empty line between examples

    # 2. Add the actual test question (from ARC dataset - English)
    test_question = item.get("question", "")
    prompt_parts.append(test_question)
    for key in ['A', 'B', 'C', 'D']:
        if key in item:
            prompt_parts.append(f"{key}. {item[key]}")
    prompt_parts.append("")

    # 3. End with Korean reasoning trigger
    prompt_parts.append(f"{response_header} {cot_trigger}")

    final_prompt = "\n".join(prompt_parts)

    # Add BOS token if needed (for OLMo models)
    if add_bos_token and bos_token:
        logger.warning("OLMo BOS 토큰 추가 임시 비활성화 (디버깅용)")
        pass

    return final_prompt


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

def save_failure_cases(failure_cases, model_name, output_dir):
    """
    Save failure cases to a separate JSON file for analysis.
    """
    failure_filepath = os.path.join(output_dir, f"failure_cases_{model_name}_3shot.json")

    with open(failure_filepath, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(failure_cases)} failure cases to {failure_filepath}")

# --- Single Model Evaluation Function with 3-shot Prompting ---
def evaluate_single_model(config: ModelConfig, arc_data: list, model_specific_output_dir: str):
    """
    Performs 3-shot ARC evaluation for a single model.
    Uses English input (ARC dataset) with Korean reasoning.
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

        # OLMo specific tokenizer settings
        if "olmo" in config.name.lower():
            logger.info("OLMo 모델 감지: under-trained tokens 문제 해결을 위한 토크나이저 설정")

            if tokenizer.pad_token is None:
                if tokenizer.unk_token:
                    tokenizer.pad_token = tokenizer.unk_token
                    logger.info(f"OLMo PAD 토큰: UNK 토큰 사용 ({tokenizer.unk_token})")
                else:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"OLMo PAD 토큰: EOS 토큰 사용 ({tokenizer.eos_token})")

            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
                logger.info(f"OLMo BOS 토큰: EOS 토큰 사용 ({tokenizer.eos_token})")

            tokenizer.padding_side = 'left'
            logger.info("OLMo 토크나이저: left padding 설정")

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
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")

            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")

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

                if "olmo" in config.name.lower():
                    current_vocab_size = model.get_input_embeddings().weight.shape[0]
                    logger.info(f"OLMo LoRA: 현재 임베딩 크기 {current_vocab_size}, 타겟 크기 {target_vocab_size}")
                    logger.warning("OLMo LoRA: 임베딩 리사이즈 생략 (모델 무결성 보호)")
                else:
                    if target_vocab_size:
                        current_vocab_size = model.get_input_embeddings().weight.shape[0]
                        if current_vocab_size != target_vocab_size:
                            logger.info(f"Resizing model from {current_vocab_size} to {target_vocab_size} for LoRA compatibility")
                            model.resize_token_embeddings(target_vocab_size)
                    else:
                        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                            model.resize_token_embeddings(len(tokenizer))

            except Exception as e:
                logger.warning(f"Could not determine LoRA vocab size: {e}. Using tokenizer length.")
                if "olmo" in config.name.lower():
                    logger.warning("OLMo 모델: 예외 상황에서도 임베딩 리사이즈 생략")
                else:
                    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                        model.resize_token_embeddings(len(tokenizer))

            try:
                model = PeftModel.from_pretrained(model, absolute_adapter_path)
                logger.info("Successfully loaded LoRA adapter.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter from {absolute_adapter_path}: {e}")
                raise e
        else:
            model_embed_size = model.get_input_embeddings().weight.shape[0]
            tokenizer_vocab_size = len(tokenizer)

            if "olmo" in config.name.lower():
                logger.info(f"OLMo 모델 임베딩 크기: {model_embed_size}")
                logger.info(f"OLMo 토크나이저 vocab 크기: {tokenizer_vocab_size}")

                if model_embed_size != tokenizer_vocab_size:
                    logger.error(f"❌ OLMo 크기 불일치 발견! 모델: {model_embed_size}, 토크나이저: {tokenizer_vocab_size}")
                    logger.info("🔧 OLMo 토큰 임베딩 크기 조정 중...")
                    model.resize_token_embeddings(len(tokenizer))
                    logger.info("✅ OLMo 토큰 임베딩 크기 조정 완료")
                else:
                    logger.info("✅ OLMo 모델과 토크나이저 크기 일치")
            else:
                if model_embed_size != tokenizer_vocab_size:
                    logger.info(f"Resizing model token embeddings from {model_embed_size} to {tokenizer_vocab_size}")
                    model.resize_token_embeddings(len(tokenizer))
            logger.info("No LoRA adapter path specified. Using the base model directly.")

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # Gemma model specific settings
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")

        # OLMo model specific settings
        if "olmo" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("OLMo 모델 감지: torch compilation 비활성화")

        # --- Evaluate on ARC dataset (English input) ---
        logger.info(f"Starting evaluation on ARC dataset with Korean reasoning...")

        is_olmo_model = "olmo" in config.name.lower()
        add_bos_for_olmo = False
        if is_olmo_model:
            logger.info("OLMo 모델 감지: BOS 토큰 추가 비활성화")

        correct_predictions = 0
        total_predictions = 0
        errors_or_skipped = 0
        results_details = []
        failure_cases = []

        # Batch processing loop
        num_batches = (len(arc_data) + BATCH_SIZE - 1) // BATCH_SIZE

        with logging_redirect_tqdm():
            pbar = tqdm(range(num_batches),
                       desc=f"Evaluating {config.name} on ARC (3-shot, errors: 0)",
                       ncols=100,
                       unit="batch",
                       leave=True,
                       dynamic_ncols=False,
                       file=sys.stdout,
                       position=0)

            for i in pbar:
                batch_start = i * BATCH_SIZE
                batch_end = batch_start + BATCH_SIZE
                batch = arc_data[batch_start:batch_end]

                prompts = []
                ground_truths = []
                valid_items_in_batch = []

                for item in batch:
                    ground_truth = get_ground_truth(item)
                    if ground_truth is None:
                        errors_or_skipped += 1
                        continue

                    prompt = create_3shot_prompt(item,
                                                add_bos_token=add_bos_for_olmo,
                                                bos_token=tokenizer.bos_token if add_bos_for_olmo else "")
                    prompts.append(prompt)
                    ground_truths.append(ground_truth)
                    valid_items_in_batch.append(item)

                if not prompts:
                    continue

                try:
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(DEVICE)

                    with torch.inference_mode():
                        if "olmo" in config.name.lower():
                            bad_words = ["setattr", "ForcedSuppressWarnings", "RI", "kommsetattr", "despre", "empire", "FLICT", "PrivateKey", "TestCase"]
                            bad_words_ids = []
                            for word in bad_words:
                                try:
                                    word_ids = tokenizer.encode(word, add_special_tokens=False)
                                    if len(word_ids) > 0:
                                        bad_words_ids.append(word_ids)
                                except:
                                    continue

                            generation_kwargs = {
                                "max_new_tokens": 512,
                                "do_sample": True,
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "repetition_penalty": 1.1,
                                "pad_token_id": tokenizer.pad_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                                "use_cache": True,
                            }

                            if bad_words_ids:
                                generation_kwargs["bad_words_ids"] = bad_words_ids
                        else:
                            generation_kwargs = {
                                "max_new_tokens": 512,
                                "pad_token_id": tokenizer.pad_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                                "do_sample": False,
                            }

                        outputs = model.generate(
                            **inputs,
                            **generation_kwargs
                        )

                    input_lengths = inputs['input_ids'].shape[1]
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
                                failure_cases.append({
                                    "index": batch_start + j,
                                    "id": item.get("id", ""),
                                    "dataset": "ARC",
                                    "question": item.get("question", ""),
                                    "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                    "ground_truth": ground_truth,
                                    "predicted_answer": model_answer_log,
                                    "raw_output": generated_text_log,
                                    "failure_type": "incorrect_answer"
                                })
                        else:
                            if j == 0:
                                logger.warning(f"Batch item {batch_start + j}: Failed to extract answer")
                            errors_or_skipped += 1
                            generated_text_log = f"BATCH_EXTRACTION_FAILED: {gen_text.strip()}"
                            failure_cases.append({
                                "index": batch_start + j,
                                "id": item.get("id", ""),
                                "dataset": "ARC",
                                "question": item.get("question", ""),
                                "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                                "ground_truth": ground_truth,
                                "predicted_answer": -1,
                                "raw_output": generated_text_log,
                                "failure_type": "batch_extraction_failed"
                            })
                            model_answer_log = None
                            is_correct_log = False

                        current_item_index = batch_start + j
                        results_details.append({
                            "index": current_item_index,
                            "id": item.get("id", ""),
                            "ground_truth": ground_truth,
                            "model_raw_output": generated_text_log,
                            "predicted_answer": model_answer_log,
                            "is_correct": is_correct_log
                        })

                        raw_generations_list.append({
                            "dataset": "ARC",
                            "index": current_item_index,
                            "id": item.get("id", ""),
                            "ground_truth": ground_truth,
                            "raw_output": generated_text_log,
                            "extracted_answer": model_answer_log
                        })

                except Exception as e:
                    logger.error(f"Batch {i}: Inference error: {e}", exc_info=False)
                    for j, (item, ground_truth) in enumerate(zip(valid_items_in_batch, ground_truths)):
                        failure_cases.append({
                            "index": batch_start + j,
                            "id": item.get("id", ""),
                            "dataset": "ARC",
                            "question": item.get("question", ""),
                            "options": {k: v for k, v in item.items() if k in ['A', 'B', 'C', 'D']},
                            "ground_truth": ground_truth,
                            "predicted_answer": -1,
                            "raw_output": f"BATCH_ERROR: {str(e)}",
                            "failure_type": "batch_inference_error"
                        })
                    errors_or_skipped += len(prompts)

                pbar.set_description(f"Evaluating {config.name} on ARC (3-shot, errors: {errors_or_skipped})")


        # Calculate accuracy
        accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        accuracy_strict = (correct_predictions / len(arc_data) * 100) if len(arc_data) > 0 else 0

        logger.info(f"--- 3-shot ARC (English input, Korean reasoning) Results for {config.name} ---")
        logger.info(f"Test Items: {len(arc_data)}")
        logger.info(f"Valid Predictions: {total_predictions}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Failure Cases: {len(failure_cases)}")
        logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
        logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")

        all_results = {
            "test_items": len(arc_data),
            "valid_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "failure_cases_count": len(failure_cases),
            "accuracy_standard": accuracy_standard,
            "accuracy_strict": accuracy_strict,
            "details": results_details
        }

        # --- Save Results ---
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "3-shot ARC Challenge (English input, Korean reasoning)",
            "results": all_results
        }

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)

        save_failure_cases(failure_cases, config.name, model_specific_output_dir)

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
    # Load ARC dataset only (English)
    arc_data = load_arc_data(ARC_DATASET_PATH)

    if not arc_data:
        logger.error("Failed to load ARC dataset")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Store summary results for all models
    summary_results = {}

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)

        results = evaluate_single_model(config, arc_data, model_specific_output_dir)

        if results:
            summary_results[config.name] = {
                "model_id": config.model_id,
                "adapter_path": config.adapter_path,
                "ARC_accuracy_standard": results["accuracy_standard"],
                "ARC_accuracy_strict": results["accuracy_strict"],
                "ARC_failure_cases": results["failure_cases_count"]
            }

    # Save summary results
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary results saved to: {summary_filepath}")

    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - English Input with Korean Reasoning")
    print("="*80)
    print(f"{'Model Name':<50} {'ARC Acc (%)':<15} {'ARC Fails':<15}")
    print("-"*80)

    for model_name, results in summary_results.items():
        arc_acc = results["ARC_accuracy_standard"]
        arc_fails = results["ARC_failure_cases"]
        print(f"{model_name:<50} {arc_acc:<15.2f} {arc_fails:<15}")

    print("="*80)

if __name__ == "__main__":
    main()
