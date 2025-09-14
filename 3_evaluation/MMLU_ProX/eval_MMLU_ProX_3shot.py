import os
import json
import logging
import torch
import warnings 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import gc
import sys
from datetime import datetime
import time
import random

torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # "medium"도 OK

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    # ModelConfig(
    #     name="llama-3.2-3b-pt",
    #     model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
    #     use_quantization=False
    # ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-09_11_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),
    ModelConfig(
        name="llama-3.2-3b-pt-tow-org-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-org-merged",
        use_quantization=False
    ),

    ModelConfig(
        name="qwem-2.5-3b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="qwem-2.5-3b-pt-tow-09_11_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/qwem-2.5-3b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),

    ModelConfig(
        name="gemma-3-4b-pt",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt",
        use_quantization=False
    ),
    ModelConfig(
        name="gemma-3-4b-pt-tow-09_11_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/gemma-3-4b-pt-tow-09_11_allenai-merged",
        use_quantization=False
    ),

    ModelConfig(
        name="olmo-2-0425-1b",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/olmo-2-0425-1b",
        use_quantization=False
    ),
    ModelConfig(
        name="olmo-2-0425-1b-tow-09_11_allenai-merged",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/olmo-2-0425-1b-tow-09_11_allenai-merged",
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
]


# --- General Configuration ---
MMLU_PROX_EN_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_en.jsonl"
MMLU_PROX_KO_DATASET_PATH = "../../2_datasets/MMLU_ProX/MMLU_ProX_ko.jsonl"
BASE_OUTPUT_DIR = "mmlu_prox_3shot"
BATCH_SIZE = 16
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

ENGLISH_FEW_SHOT_EXAMPLES = [
    {
        "question": "In a race, Alice finished before Bob. Charlie finished after David. Bob finished before David. Who finished last?",
        "options": {
            "A": "Alice",
            "B": "Bob",
            "C": "Charlie",
            "D": "David",
            "E": "Eva",
            "F": "Fabian",
            "G": "Giorgio",
            "H": "Hailey",
            "I": "Isla",
            "J": "Cannot be determined"
        },
        "cot_content": """Response: Let's think step by step. Let's break down the statements. Let '>' mean 'finished before'. Statement 1: Alice > Bob. Statement 3: Bob > David. Combining these two, we get the order: Alice > Bob > David. Statement 2: Charlie finished after David, which means David > Charlie. Combining all information, the complete order is: Alice > Bob > David > Charlie. The person who finished last is the one at the very end of this chain. That person is Charlie.""", 
        "answer": "C"
    },
    {
        "question": "A net force of 50 Newtons is applied to a 10 kg object. What is the acceleration of the object?",
        "options": {
            "A": "0.2 m/s²",
            "B": "5 m/s²",
            "C": "40 m/s²",
            "D": "500 m/s²",
            "E": "50 m/s²",
            "F": "10 m/s²",
            "G": "300 m/s²",
            "H": "3 m/s²",
            "I": "38 m/s²",
            "J": "900 m/s²"
        },
        "cot_content": """Response: Let's think step by step. The question asks for acceleration given a net force and a mass. The relevant physical principle is Newton's Second Law of Motion. The formula is Force = mass × acceleration (F = ma). We need to rearrange the formula to solve for acceleration: acceleration = Force / mass (a = F/m). The given values are Force (F) = 50 N and mass (m) = 10 kg. Substitute the values into the rearranged formula: a = 50 N / 10 kg. The calculation gives a = 5 m/s². This matches option B.""",
        "answer": "B"
    },
    {
        "question": "From which country did the United States purchase the Louisiana Territory in 1803?",
        "options": {
            "A": "Spain",
            "B": "Great Britain",
            "C": "Mexico",
            "D": "Russia",
            "E": "China",
            "F": "The Netherlands",
            "G": "Korea",
            "H": "Qatar",
            "I": "France",
            "J": "Japan"
        },
        "cot_content": """Response: Let's think step by step. The question is about the Louisiana Purchase in 1803. I need to recall the historical context of that period in North America. The major European powers with territory were Spain, Great Britain, and France. At that time, the leader of France was Napoleon Bonaparte. He was engaged in wars in Europe and needed funds. The territory, known as Louisiana, was difficult for France to control and defend from afar. Therefore, Napoleon decided to sell the vast territory to the young United States to finance his military campaigns. This event is known as the Louisiana Purchase. This historical fact confirms the purchase was made from France.""",
        "answer": "I"
    },
#     {
#         "question": "What are the primary products of photosynthesis?",
#         "options": {
#             "A": "Carbon dioxide and water",
#             "B": "Glucose and water",
#             "C": "Oxygen and carbon dioxide",
#             "D": "Glucose and oxygen",
#             "E": "Sunlight and water"
#         },
#         "cot_content": """Response: Let's think step by step.
# Photosynthesis is the process plants use to convert light energy into chemical energy.
# First, let's identify the inputs (reactants). Plants take in carbon dioxide (CO₂), water (H₂O), and sunlight.
# The process then converts these inputs into outputs (products).
# One main product is a sugar called glucose (C₆H₁₂O₆), which the plant uses as food/energy.
# The other main product is oxygen (O₂), which is released into the atmosphere as a byproduct.
# Therefore, the primary products are glucose and oxygen. This corresponds to option D.""",
#         "answer": "D"
#     },
#     {
#         "question": "Who is the author of the famous line, \"To be, or not to be: that is the question\"?",
#         "options": {
#             "A": "Christopher Marlowe",
#             "B": "John Milton",
#             "C": "William Shakespeare",
#             "D": "Charles Dickens",
#             "E": "Jane Austen"
#         },
#         "cot_content": """Response: Let's think step by step.
# This is one of the most famous quotes in English literature.
# I need to identify which play and author it comes from.
# The line is a soliloquy from the play Hamlet.
# The author of Hamlet is William Shakespeare, the famous English playwright.
# The other authors are known for different works: John Milton for Paradise Lost, Charles Dickens for novels like A Tale of Two Cities, etc. The style and origin firmly point to Shakespeare.""",
#         "answer": "C"
#     }
]

KOREAN_FEW_SHOT_EXAMPLES = [
    {
        "question": "경주에서 앨리스는 밥보다 먼저 들어왔다. 찰리는 데이비드보다 늦게 들어왔다. 밥은 데이비드보다 먼저 들어왔다. 누가 가장 꼴찌로 들어왔는가?",
        "options": {
            "A": "앨리스",
            "B": "밥",
            "C": "찰리",
            "D": "데이비드",
            "E": "이바",
            "F": "파비안",
            "G": "조지",
            "H": "헤일리",
            "I": "아일라",
            "J": "결정할 수 없음"
        },
        "cot_content": """응답: 단계별로 생각해봅시다. 주어진 문장들을 분석해 보겠습니다. '>'를 '먼저 들어왔다'는 의미로 사용하겠습니다. 문장 1: 앨리스 > 밥. 문장 3: 밥 > 데이비드. 이 두 문장을 조합하면 순서는 앨리스 > 밥 > 데이비드 입니다. 문장 2: 찰리는 데이비드보다 늦게 들어왔다, 즉 데이비드 > 찰리 입니다. 모든 정보를 종합하면, 전체 순서는 앨리스 > 밥 > 데이비드 > 찰리 입니다. 가장 꼴찌로 들어온 사람은 이 순서의 맨 마지막에 있는 사람입니다. 그 사람은 찰리입니다.""",        "answer": "C"
    },
    {
        "question": "10kg의 물체에 50 뉴턴(N)의 알짜힘이 가해졌다. 이 물체의 가속도는 얼마인가?",
        "options": {
            "A": "0.2 m/s²",
            "B": "5 m/s²",
            "C": "40 m/s²",
            "D": "500 m/s²",
            "E": "50 m/s²",
            "F": "10 m/s²",
            "G": "300 m/s²",
            "H": "3 m/s²",
            "I": "38 m/s²",
            "J": "900 m/s²"
        },
        "cot_content": """응답: 단계별로 생각해봅시다. 이 질문은 알짜힘과 질량이 주어졌을 때 가속도를 구하는 문제입니다. 관련된 물리 법칙은 뉴턴의 운동 제2법칙입니다. 공식은 힘 = 질량 × 가속도 (F = ma) 입니다. 가속도를 구하기 위해 공식을 변형해야 합니다: 가속도 = 힘 / 질량 (a = F/m). 주어진 값은 힘 (F) = 50 N 이고, 질량 (m) = 10 kg 입니다. 변형된 공식에 값을 대입합니다: a = 50 N / 10 kg. 계산 결과 a = 5 m/s² 입니다. 이는 선택지 B와 일치합니다.""", 
        "answer": "B"
    },
    {
        "question": "1803년 미국은 어느 나라로부터 루이지애나 영토를 매입했는가?",
        "options": {
            "A": "스페인",
            "B": "영국",
            "C": "멕시코",
            "D": "러시아",
            "E": "중국",
            "F": "네덜란드",
            "G": "대한민국",
            "H": "카타르",
            "I": "프랑스",
            "J": "일본"
        },
        "cot_content": """응답: 단계별로 생각해봅시다. 이 질문은 1803년의 '루이지애나 매입'에 관한 것입니다. 당시 북미 대륙의 역사적 상황을 떠올려야 합니다. 영토를 가진 주요 유럽 국가는 스페인, 영국, 프랑스였습니다. 그 시기 프랑스의 지도자는 나폴레옹 보나파르트였습니다. 그는 유럽에서 전쟁을 치르고 있었고 자금이 필요했습니다. 루이지애나로 알려진 영토는 프랑스가 멀리서 통제하고 방어하기 어려웠습니다. 따라서 나폴레옹은 그의 군사 작전 자금을 마련하기 위해 광대한 영토를 신생 국가인 미국에 팔기로 결정했습니다. 이 사건이 바로 루이지애나 매입입니다. 이 역사적 사실은 해당 영토를 프랑스로부터 매입했음을 확인시켜 줍니다.""", 
        "answer": "I"
    },
#     {
#         "question": "광합성의 주된 생성물은 무엇인가?",
#         "options": {
#             "A": "이산화탄소와 물",
#             "B": "포도당과 물",
#             "C": "산소와 이산화탄소",
#             "D": "포도당과 산소",
#             "E": "햇빛과 물"
#         },
#         "cot_content": """응답: 단계별로 생각해봅시다.
# 광합성은 식물이 빛 에너지를 화학 에너지로 전환하는 과정입니다.
# 먼저, 투입물(반응물)이 무엇인지 확인합니다. 식물은 이산화탄소(CO₂), 물(H₂O), 그리고 햇빛을 흡수합니다.
# 이 과정은 투입물을 산출물(생성물)로 변환합니다.
# 주요 생성물 중 하나는 식물이 식량/에너지로 사용하는 포도당(C₆H₁₂O₆)이라는 당입니다.
# 다른 주요 생성물은 부산물로서 대기 중으로 방출되는 산소(O₂)입니다.
# 따라서, 주된 생성물은 포도당과 산소입니다. 이는 선택지 D에 해당합니다.""",
#         "answer": "D"
#     },
#     {
#         "question": "\"죽느냐 사느냐, 그것이 문제로다\"라는 유명한 대사를 쓴 작가는 누구인가?",
#         "options": {
#             "A": "크리스토퍼 말로",
#             "B": "존 밀턴",
#             "C": "윌리엄 셰익스피어",
#             "D": "찰스 디킨스",
#             "E": "제인 오스틴"
#         },
#         "cot_content": """응답: 단계별로 생각해봅시다.
# 이것은 영문학에서 가장 유명한 인용구 중 하나입니다.
# 어떤 희곡과 작가로부터 나왔는지 식별해야 합니다.
# 이 대사는 희곡 «햄릿»에 나오는 독백입니다.
# «햄릿»의 저자는 영국의 유명한 극작가인 윌리엄 셰익스피어입니다.
# 다른 작가들은 다른 작품으로 유명합니다: 존 밀턴은 «실낙원», 찰스 디킨스는 «두 도시 이야기»와 같은 소설로 유명합니다. 문체와 출처를 볼 때 셰익스피어가 확실합니다.""",
#         "answer": "C"
#     }
]

def create_3shot_prompt(item, few_shot_examples, language="en"):
    """
    Creates a 5-shot MMLU-ProX prompt for a given test item.
    (Corrected Version)
    """
    if language == "ko":
        prompt_parts = ["다음은 다양한 학문 분야의 전문적이고 어려운 다지선다형 질문입니다. A부터 J까지의 보기중 무조건 하나의 답만 선택하세요.\n"]
    else:
        prompt_parts = ["The following are challenging multiple choice questions from various academic disciplines. You MUST choose one of the option A~J.\n"]
    
    # Add few-shot examples
    for example in few_shot_examples:
        # 1. 질문, CoT 내용, 정답을 딕셔너리에서 직접 가져옵니다.
        question = example["question"]
        correct_answer = example["answer"]
        cot_reasoning = example["cot_content"] # 실제 추론 내용을 가져옵니다.

        prompt_parts.append(f"Question: {question}")
        
        # 2. 예제의 옵션 처리 방식을 수정합니다. (options 딕셔너리 순회)
        #    sorted()를 사용하여 A, B, C 순서를 보장합니다.
        options = []
        for key, value in sorted(example["options"].items()):
            options.append(f"{key}. {value}")
        prompt_parts.extend(options)
        prompt_parts.append(cot_reasoning)

        if language == "ko":
            prompt_parts.append(f"#### 따라서 정답은 {correct_answer} 입니다.")
            prompt_parts.append(f"#### 정답: {correct_answer}")
        else:
            prompt_parts.append(f"#### So the answer is {correct_answer}.")
            prompt_parts.append(f"#### Answer: {correct_answer}.")
        prompt_parts.append("")
    
    # Add the test question
    question = item.get("question", "")
    options = []
    for i in range(10):
        option_key = f"option_{i}"
        # MMLU-ProX 데이터셋의 실제 'item'은 이 형식을 따르므로 이 로직은 유지합니다.
        if option_key in item and item[option_key] and str(item[option_key]).strip() and str(item[option_key]).strip() != "N/A":
            options.append(f"{chr(65+i)}. {item[option_key]}")
    
    prompt_parts.append(f"Question: {question}")
    prompt_parts.extend(options)
    prompt_parts.append("")
    
    if language == "ko":
        prompt_parts.append("응답: 단계별로 생각해봅시다.")
    else:
        prompt_parts.append("Response: Let's think step by step.")
    
    return "\n".join(prompt_parts)


import re

def extract_answer_first_token(model_output):
    """
    요청사항에 따라 단순화되고 개선된 답변 추출 함수입니다.
    두 가지 우선순위에 따라 답변을 찾습니다.

    1. (최우선) 모델의 출력이 'A.', 'B)', 'C' 등 정답 선택지로 시작하는지 확인합니다.
    2. 위 조건에 맞지 않을 경우, '#### Answer:' 와 같은 명확한 구조적 패턴을 찾습니다.
    """
    if not model_output:
        return None

    if "Question:" in model_output:
        model_output = model_output.split("Question:")[0]
    if "질문:" in model_output:
        model_output = model_output.split("질문:")[0]

    # 분석할 텍스트를 정리합니다.
    cleaned_output = model_output.strip().upper()

    structured_patterns = [
        r'####\s*(?:정답|답|ANSWER|THEREFORE\s+ANSWER)\s*:?\s*\{?([A-J])\}?',  # #### Answer: A or #### 정답: A or {A}
        r'\{([A-J])\}',  # {A} box format matching prompt style
        r'(?:정답|답|ANSWER)\s*:?\s*\{?([A-J])\}?',        # Answer: A or 정답: A or {A}
        r'(?:따라서|그러므로|SO|THEREFORE)\s+(?:정답은|답은|정답|답|THE\s+ANSWER\s+IS|ANSWER\s+IS)\s*:?\s*\{?([A-J])\}?',  # So the answer is A or {A}
    ]

    for pattern in structured_patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[0]  # Return the first match (avoid repetitions/hallucinations)
    
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
    # 위 두 조건으로도 찾지 못하면 추출 실패로 간주
    return None


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
        
        with torch.inference_mode():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                return_dict_in_generate=False,  # 그대로 False (속도 유리)
                output_scores=False,
                num_beams=1,
            )
        
        batch_results = []
        input_len = batch_inputs['input_ids'].shape[1]

        gen_only = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        
        for i, generated_text in enumerate(decoded_texts):
            generated_text = generated_text.strip()
            extracted_answer = extract_answer_first_token(generated_text)

            # (선택) full_generation 저장이 꼭 필요 없다면 None으로 두어 I/O 줄이기
            # full_text = full_decoded[i].strip() if save_full else None

            batch_results.append({
                'index': batch_indices[i],
                'raw_output': generated_text,
                'full_generation': generated_text,  # 필요 없으면 None로 바꾸세요
                'extracted_answer': extracted_answer
            })
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        # Fallback to individual processing
        individual_results = []
        for prompt, idx in zip(batch_prompts, batch_indices):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1500).to(DEVICE)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=0.0,
                    )
                # Decode only the generated part
                input_len = inputs['input_ids'].shape[1]
                gen_only = outputs[:, input_len:]          # outputs가 Tensor이므로 이렇게 자름
                generated_text = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()                
                
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
    failures_filepath = os.path.join(model_specific_output_dir, f"failures_{config.name}_5shot.json")
    all_outputs_filepath = os.path.join(model_specific_output_dir, f"all_outputs_{config.name}_5shot.json")  # 새로 추가

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
            device_map="auto",
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
            
        all_results = {
            "mmlu_prox_en": {
                "correct": 0, 
                "total": 0, 
                "details": [], 
                "raw_generations": [], 
                "failures": [],
                "all_outputs": []  # 새로 추가: 모든 output 저장
            },
            "mmlu_prox_ko": {
                "correct": 0, 
                "total": 0, 
                "details": [], 
                "raw_generations": [], 
                "failures": [],
                "all_outputs": []  # 새로 추가: 모든 output 저장
            }
        }

        # Evaluate MMLU-ProX English
        logger.info("Starting MMLU-ProX English evaluation...")
        pbar_en = tqdm(range(0, len(mmlu_prox_en_data), BATCH_SIZE), desc="Evaluating MMLU-ProX English (errors: 0)")
        for i in pbar_en:
            batch_data = mmlu_prox_en_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            batch_items = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_3shot_prompt(item, ENGLISH_FEW_SHOT_EXAMPLES, "en")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
                batch_items.append(item)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth, batch_prompt, item in zip(batch_results, batch_ground_truths, batch_prompts, batch_items):
                extracted_answer = result['extracted_answer']
                raw_output = result['raw_output']
                
                # No retry logic - just move on if extraction fails for speed
                if not extracted_answer and not raw_output.startswith("ERROR"):
                    logger.warning(f"English item {result['index']}: Failed to extract answer - skipping retry for speed")
                    raw_output = f"EXTRACTION_FAILED: {raw_output}"
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                # 모든 케이스를 all_outputs에 저장 (새로 추가)
                all_output_case = {
                    "index": result['index'],
                    "question": item.get("question", ""),
                    "subject": item.get("subject", ""),
                    "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                    "prompt": batch_prompt,  # 전체 프롬프트도 저장
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "raw_output": raw_output,
                    "is_correct": is_correct,
                    "status": "correct" if is_correct else ("extraction_failed" if not extracted_answer else "incorrect_answer")
                }
                all_results["mmlu_prox_en"]["all_outputs"].append(all_output_case)
                
                # Only count items with valid extracted answers for total
                if extracted_answer:
                    all_results["mmlu_prox_en"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_en"]["correct"] += 1
                    else:
                        # 틀린 케이스를 failures에 저장
                        failure_case = {
                            "index": result['index'],
                            "question": item.get("question", ""),
                            "subject": item.get("subject", ""),
                            "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                            "ground_truth": ground_truth,
                            "predicted_answer": extracted_answer,
                            "raw_output": raw_output,
                            "error_type": "incorrect_answer"
                        }
                        all_results["mmlu_prox_en"]["failures"].append(failure_case)
                else:
                    # 추출 실패 케이스도 failures에 저장
                    failure_case = {
                        "index": result['index'],
                        "question": item.get("question", ""),
                        "subject": item.get("subject", ""),
                        "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                        "ground_truth": ground_truth,
                        "predicted_answer": None,
                        "raw_output": raw_output,
                        "error_type": "extraction_failed"
                    }
                    all_results["mmlu_prox_en"]["failures"].append(failure_case)
                
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
            
            current_en_errors = len(mmlu_prox_en_data[:i+BATCH_SIZE]) - all_results["mmlu_prox_en"]["total"]
            pbar_en.set_description(f"Evaluating MMLU-ProX English (errors: {current_en_errors})")

        logger.info(f"MMLU-ProX English evaluation completed: {all_results['mmlu_prox_en']['correct']}/{all_results['mmlu_prox_en']['total']}")

        # Save English results immediately after English evaluation
        en_strict_accuracy = (all_results["mmlu_prox_en"]["correct"] / len(mmlu_prox_en_data) * 100) if len(mmlu_prox_en_data) > 0 else 0
        en_errors_skipped = len(mmlu_prox_en_data) - all_results["mmlu_prox_en"]["total"]
        
        # Save English results
        en_results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot_en.json")
        en_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "5-shot MMLU-ProX English",
            "evaluation_date": datetime.now().isoformat(),
            "language": "en",
            "results": {
                "accuracy_strict": en_strict_accuracy,
                "correct_predictions": all_results["mmlu_prox_en"]["correct"],
                "total_predictions": all_results["mmlu_prox_en"]["total"],
                "total_items": len(mmlu_prox_en_data),
                "errors_or_skipped": en_errors_skipped,
                "details": all_results["mmlu_prox_en"]["details"]
            }
        }
        
        with open(en_results_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_summary, f, indent=2, ensure_ascii=False)
        
        # Save English raw generations
        en_raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot_en.json")
        en_raw_generations = {
            "model_name": config.name,
            "language": "en", 
            "evaluation_date": datetime.now().isoformat(),
            "raw_generations": all_results["mmlu_prox_en"]["raw_generations"]
        }
        with open(en_raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_raw_generations, f, indent=2, ensure_ascii=False)
        
        # Save English failures
        en_failures_filepath = os.path.join(model_specific_output_dir, f"failures_{config.name}_5shot_en.json")
        en_failures_summary = {
            "model_name": config.name,
            "language": "en",
            "evaluation_date": datetime.now().isoformat(),
            "total_failures": len(all_results["mmlu_prox_en"]["failures"]),
            "incorrect_answers": len([f for f in all_results["mmlu_prox_en"]["failures"] if f["error_type"] == "incorrect_answer"]),
            "extraction_failures": len([f for f in all_results["mmlu_prox_en"]["failures"] if f["error_type"] == "extraction_failed"]),
            "failure_cases": all_results["mmlu_prox_en"]["failures"]
        }
        
        with open(en_failures_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_failures_summary, f, indent=2, ensure_ascii=False)
        
        # Save English all outputs (새로 추가)
        en_all_outputs_filepath = os.path.join(model_specific_output_dir, f"all_outputs_{config.name}_5shot_en.json")
        en_all_outputs_summary = {
            "model_name": config.name,
            "language": "en",
            "evaluation_date": datetime.now().isoformat(),
            "total_items": len(mmlu_prox_en_data),
            "total_outputs": len(all_results["mmlu_prox_en"]["all_outputs"]),
            "correct_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "correct"]),
            "incorrect_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "incorrect_answer"]),
            "extraction_failed_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "extraction_failed"]),
            "all_outputs": all_results["mmlu_prox_en"]["all_outputs"]
        }
        
        with open(en_all_outputs_filepath, 'w', encoding='utf-8') as f:
            json.dump(en_all_outputs_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"English results saved: {en_results_filepath}")
        logger.info(f"English raw generations saved: {en_raw_gen_filepath}")
        logger.info(f"English failures saved: {en_failures_filepath}")
        logger.info(f"English all outputs saved: {en_all_outputs_filepath}")  # 새로 추가

        # Evaluate MMLU-ProX Korean (동일한 패턴으로 수정)
        logger.info("Starting MMLU-ProX Korean evaluation...")
        pbar_ko = tqdm(range(0, len(mmlu_prox_ko_data), BATCH_SIZE), desc="Evaluating MMLU-ProX Korean (errors: 0)")
        for i in pbar_ko:
            batch_data = mmlu_prox_ko_data[i:i+BATCH_SIZE]
            batch_prompts = []
            batch_indices = []
            batch_ground_truths = []
            batch_items = []
            
            for j, item in enumerate(batch_data):
                ground_truth = get_ground_truth(item)
                if ground_truth is None:
                    continue
                    
                prompt = create_3shot_prompt(item, KOREAN_FEW_SHOT_EXAMPLES, "ko")
                batch_prompts.append(prompt)
                batch_indices.append(i + j)
                batch_ground_truths.append(ground_truth)
                batch_items.append(item)
            
            if not batch_prompts:
                continue
                
            batch_results = process_batch(model, tokenizer, batch_prompts, batch_indices)
            
            for result, ground_truth, batch_prompt, item in zip(batch_results, batch_ground_truths, batch_prompts, batch_items):
                extracted_answer = result['extracted_answer']
                raw_output = result['raw_output']
                
                # No retry logic - just move on if extraction fails for speed
                if not extracted_answer and not raw_output.startswith("ERROR"):
                    logger.warning(f"Korean item {result['index']}: Failed to extract answer - skipping retry for speed")
                    raw_output = f"EXTRACTION_FAILED: {raw_output}"
                
                is_correct = extracted_answer == ground_truth if extracted_answer else False
                
                # 모든 케이스를 all_outputs에 저장 (새로 추가)
                all_output_case = {
                    "index": result['index'],
                    "question": item.get("question", ""),
                    "subject": item.get("subject", ""),
                    "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                    "prompt": batch_prompt,  # 전체 프롬프트도 저장
                    "ground_truth": ground_truth,
                    "predicted_answer": extracted_answer,
                    "raw_output": raw_output,
                    "is_correct": is_correct,
                    "status": "correct" if is_correct else ("extraction_failed" if not extracted_answer else "incorrect_answer")
                }
                all_results["mmlu_prox_ko"]["all_outputs"].append(all_output_case)
                
                # Only count items with valid extracted answers for total
                if extracted_answer:
                    all_results["mmlu_prox_ko"]["total"] += 1
                    if is_correct:
                        all_results["mmlu_prox_ko"]["correct"] += 1
                    else:
                        # 틀린 케이스를 failures에 저장
                        failure_case = {
                            "index": result['index'],
                            "question": item.get("question", ""),
                            "subject": item.get("subject", ""),
                            "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                            "ground_truth": ground_truth,
                            "predicted_answer": extracted_answer,
                            "raw_output": raw_output,
                            "error_type": "incorrect_answer"
                        }
                        all_results["mmlu_prox_ko"]["failures"].append(failure_case)
                else:
                    # 추출 실패 케이스도 failures에 저장
                    failure_case = {
                        "index": result['index'],
                        "question": item.get("question", ""),
                        "subject": item.get("subject", ""),
                        "options": {f"option_{i}": item.get(f"option_{i}", "") for i in range(10) if f"option_{i}" in item and item[f"option_{i}"] and str(item[f"option_{i}"]).strip() != "N/A"},
                        "ground_truth": ground_truth,
                        "predicted_answer": None,
                        "raw_output": raw_output,
                        "error_type": "extraction_failed"
                    }
                    all_results["mmlu_prox_ko"]["failures"].append(failure_case)
                
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
            
            current_ko_errors = len(mmlu_prox_ko_data[:i+BATCH_SIZE]) - all_results["mmlu_prox_ko"]["total"]
            pbar_ko.set_description(f"Evaluating MMLU-ProX Korean (errors: {current_ko_errors})")

        logger.info(f"MMLU-ProX Korean evaluation completed: {all_results['mmlu_prox_ko']['correct']}/{all_results['mmlu_prox_ko']['total']}")

        # Save Korean results immediately after Korean evaluation
        ko_strict_accuracy = (all_results["mmlu_prox_ko"]["correct"] / len(mmlu_prox_ko_data) * 100) if len(mmlu_prox_ko_data) > 0 else 0
        ko_errors_skipped = len(mmlu_prox_ko_data) - all_results["mmlu_prox_ko"]["total"]
        
        # Save Korean results
        ko_results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot_ko.json")
        ko_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "5-shot MMLU-ProX Korean",
            "evaluation_date": datetime.now().isoformat(),
            "language": "ko",
            "results": {
                "accuracy_strict": ko_strict_accuracy,
                "correct_predictions": all_results["mmlu_prox_ko"]["correct"],
                "total_predictions": all_results["mmlu_prox_ko"]["total"],
                "total_items": len(mmlu_prox_ko_data),
                "errors_or_skipped": ko_errors_skipped,
                "details": all_results["mmlu_prox_ko"]["details"]
            }
        }
        
        with open(ko_results_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_summary, f, indent=2, ensure_ascii=False)
        
        # Save Korean raw generations
        ko_raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot_ko.json")
        ko_raw_generations = {
            "model_name": config.name,
            "language": "ko",
            "evaluation_date": datetime.now().isoformat(), 
            "raw_generations": all_results["mmlu_prox_ko"]["raw_generations"]
        }
        with open(ko_raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_raw_generations, f, indent=2, ensure_ascii=False)
        
        # Save Korean failures
        ko_failures_filepath = os.path.join(model_specific_output_dir, f"failures_{config.name}_5shot_ko.json")
        ko_failures_summary = {
            "model_name": config.name,
            "language": "ko",
            "evaluation_date": datetime.now().isoformat(),
            "total_failures": len(all_results["mmlu_prox_ko"]["failures"]),
            "incorrect_answers": len([f for f in all_results["mmlu_prox_ko"]["failures"] if f["error_type"] == "incorrect_answer"]),
            "extraction_failures": len([f for f in all_results["mmlu_prox_ko"]["failures"] if f["error_type"] == "extraction_failed"]),
            "failure_cases": all_results["mmlu_prox_ko"]["failures"]
        }
        
        with open(ko_failures_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_failures_summary, f, indent=2, ensure_ascii=False)
        
        # Save Korean all outputs (새로 추가)
        ko_all_outputs_filepath = os.path.join(model_specific_output_dir, f"all_outputs_{config.name}_5shot_ko.json")
        ko_all_outputs_summary = {
            "model_name": config.name,
            "language": "ko",
            "evaluation_date": datetime.now().isoformat(),
            "total_items": len(mmlu_prox_ko_data),
            "total_outputs": len(all_results["mmlu_prox_ko"]["all_outputs"]),
            "correct_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "correct"]),
            "incorrect_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "incorrect_answer"]),
            "extraction_failed_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "extraction_failed"]),
            "all_outputs": all_results["mmlu_prox_ko"]["all_outputs"]
        }
        
        with open(ko_all_outputs_filepath, 'w', encoding='utf-8') as f:
            json.dump(ko_all_outputs_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Korean results saved: {ko_results_filepath}")
        logger.info(f"Korean raw generations saved: {ko_raw_gen_filepath}")
        logger.info(f"Korean failures saved: {ko_failures_filepath}")
        logger.info(f"Korean all outputs saved: {ko_all_outputs_filepath}")  # 새로 추가

        # Calculate strict accuracies (including errors/skips) for final summary
        en_strict_accuracy = (all_results["mmlu_prox_en"]["correct"] / len(mmlu_prox_en_data) * 100) if len(mmlu_prox_en_data) > 0 else 0
        
        # Calculate error/skip counts
        en_errors_skipped = len(mmlu_prox_en_data) - all_results["mmlu_prox_en"]["total"]

        logger.info(f"--- Final Results for {config.name} ---")
        logger.info(f"MMLU-ProX English Strict Accuracy: {en_strict_accuracy:.2f}% ({all_results['mmlu_prox_en']['correct']}/{len(mmlu_prox_en_data)}) [Errors/Skipped: {en_errors_skipped}]")
        logger.info(f"MMLU-ProX Korean Strict Accuracy: {ko_strict_accuracy:.2f}% ({all_results['mmlu_prox_ko']['correct']}/{len(mmlu_prox_ko_data)}) [Errors/Skipped: {ko_errors_skipped}]")

        # Save combined results for compatibility
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
        
        # Save combined raw generations for compatibility
        raw_generations_summary = {
            "mmlu_prox_en": all_results["mmlu_prox_en"]["raw_generations"],
            "mmlu_prox_ko": all_results["mmlu_prox_ko"]["raw_generations"]
        }
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_summary, f, indent=2, ensure_ascii=False)
        
        # Save combined failures for compatibility
        failures_summary = {
            "model_name": config.name,
            "evaluation_date": datetime.now().isoformat(),
            "mmlu_prox_en": {
                "total_failures": len(all_results["mmlu_prox_en"]["failures"]),
                "incorrect_answers": len([f for f in all_results["mmlu_prox_en"]["failures"] if f["error_type"] == "incorrect_answer"]),
                "extraction_failures": len([f for f in all_results["mmlu_prox_en"]["failures"] if f["error_type"] == "extraction_failed"]),
                "failure_cases": all_results["mmlu_prox_en"]["failures"]
            },
            "mmlu_prox_ko": {
                "total_failures": len(all_results["mmlu_prox_ko"]["failures"]),
                "incorrect_answers": len([f for f in all_results["mmlu_prox_ko"]["failures"] if f["error_type"] == "incorrect_answer"]),
                "extraction_failures": len([f for f in all_results["mmlu_prox_ko"]["failures"] if f["error_type"] == "extraction_failed"]),
                "failure_cases": all_results["mmlu_prox_ko"]["failures"]
            }
        }
        
        with open(failures_filepath, 'w', encoding='utf-8') as f:
            json.dump(failures_summary, f, indent=2, ensure_ascii=False)
        
        # Save combined all outputs (새로 추가)
        all_outputs_summary = {
            "model_name": config.name,
            "evaluation_date": datetime.now().isoformat(),
            "mmlu_prox_en": {
                "total_items": len(mmlu_prox_en_data),
                "total_outputs": len(all_results["mmlu_prox_en"]["all_outputs"]),
                "correct_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "correct"]),
                "incorrect_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "incorrect_answer"]),
                "extraction_failed_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "extraction_failed"]),
                "no_ground_truth_count": len([o for o in all_results["mmlu_prox_en"]["all_outputs"] if o["status"] == "no_ground_truth"]),
                "all_outputs": all_results["mmlu_prox_en"]["all_outputs"]
            },
            "mmlu_prox_ko": {
                "total_items": len(mmlu_prox_ko_data),
                "total_outputs": len(all_results["mmlu_prox_ko"]["all_outputs"]),
                "correct_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "correct"]),
                "incorrect_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "incorrect_answer"]),
                "extraction_failed_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "extraction_failed"]),
                "no_ground_truth_count": len([o for o in all_results["mmlu_prox_ko"]["all_outputs"] if o["status"] == "no_ground_truth"]),
                "all_outputs": all_results["mmlu_prox_ko"]["all_outputs"]
            }
        }
        
        with open(all_outputs_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_outputs_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Combined results saved: {results_filepath}")
        logger.info(f"Combined raw generations saved: {raw_gen_filepath}")
        logger.info(f"Combined failures saved: {failures_filepath}")
        logger.info(f"Combined all outputs saved: {all_outputs_filepath}")  # 새로 추가

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
        
        # 에러 발생시에도 기본 JSON 파일들을 저장
        error_results = {
            "mmlu_prox_en": {
                "correct": 0, 
                "total": 0, 
                "details": [], 
                "raw_generations": [], 
                "failures": [],
                "all_outputs": []
            },
            "mmlu_prox_ko": {
                "correct": 0, 
                "total": 0, 
                "details": [], 
                "raw_generations": [], 
                "failures": [],
                "all_outputs": []
            }
        }
        
        # 에러 정보를 포함한 JSON 파일들 저장
        try:
            error_summary_en = {
                "model_config": {k: str(v) for k, v in config.__dict__.items()},
                "evaluation_type": "5-shot MMLU-ProX English",
                "evaluation_date": datetime.now().isoformat(),
                "language": "en",
                "error": str(e),
                "results": {
                    "accuracy_strict": 0.0,
                    "correct_predictions": 0,
                    "total_predictions": 0,
                    "total_items": len(mmlu_prox_en_data) if 'mmlu_prox_en_data' in locals() else 0,
                    "errors_or_skipped": len(mmlu_prox_en_data) if 'mmlu_prox_en_data' in locals() else 0,
                    "details": []
                }
            }
            
            error_summary_ko = {
                "model_config": {k: str(v) for k, v in config.__dict__.items()},
                "evaluation_type": "5-shot MMLU-ProX Korean", 
                "evaluation_date": datetime.now().isoformat(),
                "language": "ko",
                "error": str(e),
                "results": {
                    "accuracy_strict": 0.0,
                    "correct_predictions": 0,
                    "total_predictions": 0,
                    "total_items": len(mmlu_prox_ko_data) if 'mmlu_prox_ko_data' in locals() else 0,
                    "errors_or_skipped": len(mmlu_prox_ko_data) if 'mmlu_prox_ko_data' in locals() else 0,
                    "details": []
                }
            }
            
            # 에러 결과 저장
            en_results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot_en.json")
            ko_results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot_ko.json")
            
            with open(en_results_filepath, 'w', encoding='utf-8') as f:
                json.dump(error_summary_en, f, indent=2, ensure_ascii=False)
            with open(ko_results_filepath, 'w', encoding='utf-8') as f:
                json.dump(error_summary_ko, f, indent=2, ensure_ascii=False)
                
            # 빈 raw generations와 failures 파일도 생성
            empty_raw_gen = {"model_name": config.name, "error": str(e), "raw_generations": []}
            empty_failures = {"model_name": config.name, "error": str(e), "failure_cases": []}
            empty_all_outputs = {"model_name": config.name, "error": str(e), "all_outputs": []}
            
            for lang in ["en", "ko"]:
                with open(os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot_{lang}.json"), 'w', encoding='utf-8') as f:
                    json.dump({**empty_raw_gen, "language": lang}, f, indent=2, ensure_ascii=False)
                with open(os.path.join(model_specific_output_dir, f"failures_{config.name}_5shot_{lang}.json"), 'w', encoding='utf-8') as f:
                    json.dump({**empty_failures, "language": lang}, f, indent=2, ensure_ascii=False)
                with open(os.path.join(model_specific_output_dir, f"all_outputs_{config.name}_5shot_{lang}.json"), 'w', encoding='utf-8') as f:
                    json.dump({**empty_all_outputs, "language": lang}, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Error result files saved for {config.name}")
            
        except Exception as save_error:
            logger.error(f"Failed to save error results for {config.name}: {save_error}")
        
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
        # Clean up model and tokenizer
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up logging handler
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
            root_logger.removeHandler(file_handler)
            file_handler.close()
        
        logger.info(f"--- Cleanup completed for model: {config.name} ---")


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
        logger.info(f"Starting evaluation for model: {config.name}")
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        
        try:
            model_result = evaluate_single_model_on_datasets(config, mmlu_prox_en_data, mmlu_prox_ko_data, model_specific_output_dir)
            all_model_results.append(model_result)
            
            # 성공/실패 상태 로깅
            if "error" in model_result:
                logger.error(f"Model {config.name} evaluation failed: {model_result['error']}")
            else:
                logger.info(f"Model {config.name} evaluation completed successfully")
                logger.info(f"  EN Accuracy: {model_result.get('mmlu_prox_en_accuracy_strict', 0):.2f}%")
                logger.info(f"  KO Accuracy: {model_result.get('mmlu_prox_ko_accuracy_strict', 0):.2f}%")
                
        except Exception as e:
            logger.exception(f"Unexpected error evaluating model {config.name}: {e}")
            # 최소한의 에러 결과라도 저장
            error_result = {
                "model_name": config.name,
                "mmlu_prox_en_accuracy_strict": 0.0,
                "mmlu_prox_ko_accuracy_strict": 0.0,
                "mmlu_prox_en_correct": 0,
                "mmlu_prox_en_total": 0,
                "mmlu_prox_en_total_items": len(mmlu_prox_en_data),
                "mmlu_prox_en_errors_skipped": len(mmlu_prox_en_data),
                "mmlu_prox_ko_correct": 0,
                "mmlu_prox_ko_total": 0,
                "mmlu_prox_ko_total_items": len(mmlu_prox_ko_data),
                "mmlu_prox_ko_errors_skipped": len(mmlu_prox_ko_data),
                "error": str(e)
            }
            all_model_results.append(error_result)

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

    # Enhanced summary with performance analysis
    if create_enhanced_summary:
        # Prepare model results for analysis
        model_results_for_analysis = []
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
        
        enhanced_summary = create_enhanced_summary(
            model_results=model_results_for_analysis,
            evaluation_info=summary_data["evaluation_info"],
            primary_metric="accuracy_strict",
            subject_metric=None  # MMLU_ProX doesn't have subject breakdown
        )
        
        # Merge with original summary data
        enhanced_summary["original_detailed_results"] = summary_data
        
        summary_filepath = os.path.join(BASE_OUTPUT_DIR, "SUMMARY.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)
            
        # Log key insights
        perf_analysis = enhanced_summary["performance_analysis"]
        logger.info(f"🏆 Best performing model: {perf_analysis['best_model']}")
        logger.info(f"📊 Average combined accuracy: {perf_analysis['average_score']:.2f}%")
        logger.info(f"📈 Performance gap: {perf_analysis['performance_gap']:.2f}%p")
        
    else:
        # Fallback to basic summary
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