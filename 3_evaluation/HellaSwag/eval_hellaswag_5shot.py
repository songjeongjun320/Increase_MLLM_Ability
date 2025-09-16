#!/usr/bin/env python3
"""
HellaSwag 5-shot evaluation script
Evaluates models on English and Korean HellaSwag test datasets
"""

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

# --- General Configuration ---
HELLASWAG_DATASET_PATH = "../../2_datasets/HellaSwag/hellaswag_validation.json"
KO_HELLASWAG_DATASET_PATH = "../../2_datasets/HellaSwag/ko_hellaswag_validation.json"
BASE_OUTPUT_DIR = "hellaswag_5shot_results_tokenizer_added"
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

# --- 5-shot Examples for English HellaSwag ---
HELLASWAG_5SHOT_EXAMPLES = [
    {
        "ctx": "A woman is outside with a small group of dogs in the snow. She",
        "endings": [
            "is pulling a sled through the snow containing a small dog.",
            "uses the leash to pull the dogs closer to her.",
            "gets ready to release the dogs by pulling on the leash.",
            "touches the dogs' heads and necks."
        ],
        "answer": 0,
        "cot_content": "Let's think step by step. The context describes a woman outside with dogs in the snow. Looking at the options: A) pulling a sled through the snow with a small dog - this fits the snowy outdoor setting and dogs context perfectly. B) using leash to pull dogs closer - possible but doesn't relate strongly to the snowy setting. C) getting ready to release dogs - doesn't connect well with the snow context. D) touching dogs' heads and necks - too generic and doesn't use the snow setting. The most logical continuation that uses all context elements is option A."
    },
    {
        "ctx": "A camera pans over a snowy mountain and shows a person skiing down a slope. The person",
        "endings": [
            "continues skiing down the mountain, going around several turns.",
            "is shown skiing down the mountain in slow motion.",
            "stops skiing and starts to walk down the mountain.",
            "jumps over several obstacles while skiing."
        ],
        "answer": 0,
        "cot_content": "Let's think step by step. The scene shows someone skiing down a snowy slope. Looking at the most natural continuation: A) continuing to ski down with turns - this is the most natural progression of skiing down a slope. B) skiing in slow motion - this describes a filming technique, not what the person is doing. C) stopping to walk - this would be unusual behavior while actively skiing down. D) jumping obstacles - this introduces new elements not established in the context. Option A provides the most logical and natural continuation of the skiing activity."
    },
    {
        "ctx": "A man is standing in a kitchen next to a blender. He",
        "endings": [
            "pours ingredients into the blender and turns it on.",
            "starts washing dishes in the sink nearby.",
            "opens the refrigerator to get something out.",
            "begins chopping vegetables on a cutting board."
        ],
        "answer": 0,
        "cot_content": "Let's think step by step. A man is positioned next to a blender in the kitchen. The most logical action would relate to using the blender. A) pouring ingredients and turning it on - this directly uses the blender that was specifically mentioned. B) washing dishes - doesn't relate to the blender at all. C) opening refrigerator - possible but ignores the blender context. D) chopping vegetables - also ignores the blender. The context specifically mentions the blender, so the most coherent continuation involves actually using it, which is option A."
    },
    {
        "ctx": "The credits of the movie are shown on the screen. Then we",
        "endings": [
            "see a person working in their office.",
            "see the opening scene of the movie.",
            "are shown the movie theater from outside.",
            "see people leaving the movie theater."
        ],
        "answer": 3,
        "cot_content": "Let's think step by step. Movie credits are typically shown at the end of a film. After credits finish, what would logically happen? A) seeing someone in an office - this doesn't connect to the movie ending context. B) seeing opening scene - this would be going backwards, which doesn't make sense after credits. C) showing theater from outside - possible but less directly connected to the end-of-movie moment. D) seeing people leaving the theater - this is the most logical sequence as people typically leave after credits finish. Option D provides the most natural temporal progression from credits ending."
    },
    {
        "ctx": "A young child is sitting at a table with an adult. The adult",
        "endings": [
            "is helping the child with their homework.",
            "is reading a newspaper while ignoring the child.",
            "gets up and leaves the room suddenly.",
            "is watching television across the room."
        ],
        "answer": 0,
        "cot_content": "Let's think step by step. A child and adult are sitting together at a table. This setting suggests interaction and engagement. A) helping with homework - this fits perfectly with adults and children at a table together, showing positive interaction. B) reading newspaper while ignoring child - this contradicts the 'together at table' setup. C) getting up and leaving - this breaks the established scene of being together. D) watching TV across the room - this contradicts them being 'at a table' together. The context of sitting together at a table most naturally leads to collaborative activity like homework help, making option A the most logical."
    }
]

# --- 5-shot Examples for Korean HellaSwag ---
KO_HELLASWAG_5SHOT_EXAMPLES = [
    {
        "ctx": "한 여성이 눈 속에서 작은 개들과 함께 밖에 있다. 그녀는",
        "endings": [
            "작은 개가 들어있는 썰매를 눈 속에서 끌고 있다.",
            "목줄을 사용해서 개들을 자신에게 더 가까이 끌어당긴다.",
            "목줄을 당겨서 개들을 놓아줄 준비를 한다.",
            "개들의 머리와 목을 만진다."
        ],
        "answer": 0,
        "cot_content": "단계별로 생각해봅시다. 맥락은 한 여성이 눈 속에서 개들과 함께 <ToW>After establishing the context of a woman in snow with dogs, \"밖에\" (outside) is the logical next word to complete the setting description, emphasizing the outdoor snowy environment which is crucial for understanding the scenario.</ToW> 밖에 있다는 것입니다. 선택지를 보면: A) 작은 개가 들어있는 썰매를 눈 속에서 끄는 것 - 이는 눈이 있는 야외 <ToW>After describing option A as sledding in snow and noting it relates to the snowy outdoor context, \"환경과\" (environment and) is the necessary next word to continue explaining how this option fits with the established setting before mentioning the dogs.</ToW> 환경과 개들이라는 맥락에 완벽하게 맞습니다. B) 목줄로 개들을 가까이 끌어당기기 - 가능하지만 눈이라는 환경과 강하게 연관되지 <ToW>After stating that option B is possible but doesn't strongly relate to the snow environment, \"않습니다\" (is not) is the logical conclusion word to complete the negative assessment of this option's fit with the snowy context.</ToW> 않습니다. C) 개들을 놓아줄 준비하기 - 눈 맥락과 잘 연결되지 않습니다. D) 개들의 머리와 목 만지기 - 너무 <ToW>After introducing option D about touching dogs' heads and necks, \"일반적이고\" (general and) is the logical next word to criticize this option as being too generic and not specific to the snowy environment context.</ToW> 일반적이고 눈 환경을 활용하지 못합니다. 모든 맥락 요소를 활용하는 가장 논리적인 연속은 선택지 <ToW>After analyzing all options and concluding that the most logical continuation uses all contextual elements, \"A입니다\" (is A) is the necessary final word to identify option A as the correct answer.</ToW> A입니다."    },
    {
        "ctx": "카메라가 눈 덮인 산을 비추며 한 사람이 슬로프를 스키로 내려오는 모습을 보여준다. 그 사람은",
        "endings": [
            "계속해서 산을 따라 스키를 타며 여러 번 커브를 돈다.",
            "슬로우 모션으로 산을 따라 스키를 타는 모습이 보여진다.",
            "스키를 멈추고 산을 걸어서 내려가기 시작한다.",
            "스키를 타면서 여러 장애물을 뛰어넘는다."
        ],
        "answer": 0,
        "cot_content": "단계별로 생각해봅시다. 장면은 누군가가 눈 덮인 슬로프를 <ToW>After describing the scene of someone on a snow-covered slope, \"스키로\" (by skiing) is the logical next word to specify the method of movement down the slope, establishing the skiing activity before describing the downward motion.</ToW> 스키로 내려오는 것을 보여줍니다. 가장 자연스러운 연속을 보면: A) 커브를 돌며 계속 스키를 타고 내려가기 - 이는 슬로프를 스키로 내려가는 가장 <ToW>After describing option A as continuing to ski down while turning curves and noting it relates to skiing down slopes, \"자연스러운\" (natural) is the necessary next word to evaluate this option as the most natural progression of the skiing activity.</ToW> 자연스러운 진행입니다. B) 슬로우 모션으로 스키 타기 - 이는 촬영 기법을 설명하는 것이지, 그 사람이 하는 <ToW>After explaining that option B describes a filming technique rather than what the person is doing, \"행동이\" (action) is the logical next word to contrast the filming technique with the actual physical action being performed.</ToW> 행동이 아닙니다. C) 스키를 멈추고 걷기 - 적극적으로 스키를 타고 있는 중에 이런 행동은 이상합니다. D) 장애물 뛰어넘기 - 맥락에서 설정되지 않은 새로운 요소를 도입합니다. 선택지 A가 스키 활동의 가장 논리적이고 <ToW>After stating that option A provides the most logical progression of skiing activity, \"자연스러운\" (natural) is the necessary next word to complete the final assessment, emphasizing both the logical and natural qualities of this continuation.</ToW> 자연스러운 연속을 제공합니다."    },
    {
        "ctx": "한 남자가 블렌더 옆 부엌에 서 있다. 그는",
        "endings": [
            "블렌더에 재료를 넣고 켠다.",
            "근처 싱크대에서 설거지를 시작한다.",
            "냉장고를 열어 뭔가를 꺼낸다.",
            "도마 위에서 야채를 썰기 시작한다."
        ],
        "answer": 0,
        "cot_content": "단계별로 생각해봅시다. 한 남자가 부엌에서 블렌더 옆에 위치해 있습니다. 가장 논리적인 행동은 <ToW>After establishing the scene of a man positioned next to a blender in the kitchen, \"블렌더\" (blender) is the logical next word to specify that the most logical action would be related to the blender that was specifically mentioned in the context.</ToW> 블렌더 사용과 관련될 것입니다. A) 재료를 넣고 블렌더를 켜기 - 이는 특별히 언급된 블렌더를 <ToW>After describing option A as putting ingredients in and turning on the blender, \"직접\" (directly) is the necessary next word to emphasize that this option involves direct use of the specifically mentioned blender.</ToW> 직접 사용하는 것입니다. B) 설거지하기 - 블렌더와 전혀 관련이 없습니다. C) 냉장고 열기 - 가능하지만 블렌더 맥락을 무시합니다. D) 야채 썰기 - 역시 블렌더를 <ToW>After explaining that vegetable cutting is another option but noting it also relates to the blender context, \"무시합니다\" (ignores) is the logical next word to criticize this option for disregarding the blender that was specifically mentioned.</ToW> 무시합니다. 맥락에서 블렌더를 특별히 언급했으므로, 가장 일관된 연속은 실제로 그것을 <ToW>After noting that the blender was specifically mentioned in the context and stating that the most consistent continuation is actually to do something with it, \"사용하는\" (use) is the necessary next word to complete the logical conclusion about using the blender.</ToW> 사용하는 것이며, 이는 선택지 A입니다."    },
    {
        "ctx": "영화의 크레딧이 화면에 나타난다. 그러면 우리는",
        "endings": [
            "사무실에서 일하고 있는 사람을 본다.",
            "영화의 오프닝 장면을 본다.",
            "영화관을 밖에서 본다.",
            "영화관을 나가는 사람들을 본다."
        ],
        "answer": 3,
        "cot_content": "단계별로 생각해봅시다. 영화 크레딧은 보통 영화 마지막에 나옵니다. 크레딧이 끝난 후에 논리적으로 무엇이 <ToW>After establishing that movie credits usually appear at the end of films and asking what logically happens after credits end, \"일어날까요\" (will happen) is the necessary next word to complete the rhetorical question about the logical sequence of events.</ToW> 일어날까요? A) 사무실에서 일하는 사람 보기 - 이는 영화 끝 맥락과 <ToW>After describing option A about seeing people working in an office and noting it relates to the movie ending context, \"연결되지\" (is not connected) is the logical next word to criticize this option for lacking connection to the movie theater context.</ToW> 연결되지 않습니다. B) 오프닝 장면 보기 - 크레딧 후에 거꾸로 <ToW>After explaining option B about seeing opening scenes and noting it involves going backwards after credits, \"가는\" (going) is the necessary next word to complete the criticism that going backwards in the sequence doesn't make sense.</ToW> 가는 것은 말이 안 됩니다. C) 영화관을 밖에서 보기 - 가능하지만 영화 끝 순간과 직접적으로 <ToW>After describing option C as possible but noting it doesn't directly relate to the movie ending moment, \"연결되지\" (is not connected) is the logical next word to explain why this option lacks direct connection to the immediate post-credits moment.</ToW> 연결되지 않습니다. D) 영화관을 나가는 사람들 보기 - 이는 크레딧이 끝난 후 사람들이 보통 <ToW>After describing option D about seeing people leaving the theater and explaining this happens after credits end because people usually do something, \"나가기\" (leave) is the necessary next word to complete the explanation of typical audience behavior.</ToW> 나가기 때문에 가장 논리적인 순서입니다. 선택지 D가 크레딧 <ToW>After concluding that option D provides the most logical sequence, \"종료로부터\" (from the ending of) is the necessary next word to specify that option D provides the most natural temporal progression from the credits ending.</ToW> 종료로부터 가장 자연스러운 시간적 진행을 제공합니다."    },
    {
        "ctx": "어린 아이가 어른과 함께 테이블에 앉아 있다. 그 어른은",
        "endings": [
            "아이의 숙제를 도와주고 있다.",
            "아이를 무시하면서 신문을 읽고 있다.",
            "갑자기 일어나서 방을 나간다.",
            "방 건너편에서 텔레비전을 보고 있다."
        ],
        "answer": 0,
        "cot_content": "단계별로 생각해봅시다. 아이와 어른이 테이블에 함께 앉아 있습니다. 이 상황은 상호작용과 참여를 시사합니다. A) 숙제 도와주기 - 이는 어른과 <ToW>After describing option A as helping with homework and noting it involves an adult, \"아이가\" (child) is the logical next word to complete the pairing that matches perfectly with the established context of an adult and child together at a table.</ToW> 아이가 테이블에 함께 있는 상황과 완벽하게 맞으며, 긍정적인 상호작용을 보여줍니다. B) 아이를 <ToW>After starting to describe option B about doing something to the child, \"무시하며\" (ignoring while) is the necessary next word to complete the description of this negative behavior that contrasts with the collaborative setting.</ToW> 무시하며 신문 읽기 - 이는 '테이블에 함께' 있다는 설정과 <ToW>After explaining that ignoring the child while reading newspaper relates to being 'together at the table', \"모순됩니다\" (contradicts) is the logical next word to criticize how this behavior contradicts the togetherness implied by the setting.</ToW> 모순됩니다. C) 일어나서 나가기 - 이는 함께 있다는 기존 <ToW>After describing option C as getting up and leaving, and noting it affects the existing state of being together, \"장면을\" (scene) is the necessary next word to specify what gets disrupted by this action.</ToW> 장면을 깨뜨립니다. D) 방 건너편에서 TV 보기 - 이는 '테이블에' 함께 있다는 것과 <ToW>After explaining that watching TV from across the room relates to being 'together at the table', \"모순됩니다\" (contradicts) is the logical next word to point out how this action contradicts the table setting.</ToW> 모순됩니다. 테이블에 함께 앉아 있다는 맥락은 가장 자연스럽게 숙제 도움 같은 <ToW>After explaining that the context of sitting together at a table most naturally leads to activities like homework help, \"협력적\" (collaborative) is the necessary next word to specify the type of activity that best fits this setting.</ToW> 협력적 활동으로 이어지므로, 선택지 A가 가장 논리적입니다."    }
]

# --- Helper Functions for 5-shot HellaSwag Evaluation ---
def create_5shot_prompt(item, examples, dataset_type="hellaswag"):
    """Create 5-shot prompt for HellaSwag evaluation"""
    if dataset_type == "hellaswag":
        prompt_parts = ["The following are reading comprehension questions. Choose the most logical continuation from the given options.\n"]
        response_header = "#### Answer:"
        cot_trigger = "Let's think step by step."
        final_answer_prefix = "#### Therefore, the answer is:"

    else:  # ko-hellaswag
        prompt_parts = ["다음은 독해 문제들입니다. 주어진 선택지 중 가장 논리적인 연속을 선택하세요.\n"]
        response_header = "#### 정답:"
        cot_trigger = "단계별로 생각해봅시다."
        final_answer_prefix = "#### 따라서 정답은:"

    # Add 5 examples
    for i, example in enumerate(examples):
        ctx = example["ctx"]
        endings = example["endings"]
        cot_content = example["cot_content"]
        answer = example["answer"]

        # Question and options
        prompt_parts.append(f"Context: {ctx}")
        for j, ending in enumerate(endings):
            option_letter = chr(65 + j)  # A, B, C, D
            prompt_parts.append(f"{option_letter}. {ending}")

        # Complete response with reasoning and answer
        answer_letter = chr(65 + answer)  # Convert 0,1,2,3 to A,B,C,D
        full_response = f"{response_header} {cot_content} {final_answer_prefix} {{{answer_letter}}}. #### {{{answer_letter}}}."
        prompt_parts.append(full_response)
        prompt_parts.append("") # Empty line between examples

    # Add the test item
    test_ctx = item.get("ctx", "")
    test_endings = item.get("endings", [])

    prompt_parts.append(f"Context: {test_ctx}")
    for j, ending in enumerate(test_endings):
        option_letter = chr(65 + j)  # A, B, C, D
        prompt_parts.append(f"{option_letter}. {ending}")
    prompt_parts.append("")

    # Start reasoning
    prompt_parts.append(f"{response_header} {cot_trigger}")

    return "\n".join(prompt_parts)

def process_single_with_retry(model, tokenizer, prompt, max_retries=0):
    """Process a single prompt with retry logic for answer extraction failures"""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)

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
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Failed to extract answer, retrying...")
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
                return f"ERROR after {max_retries} attempts: {str(e)}", None

    return f"EXTRACTION_FAILED after {max_retries} attempts", None

def extract_answer_robust(model_output: str) -> int:
    """Extract the final answer (A, B, C, D) from model output for HellaSwag and convert to 0,1,2,3
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
        # Convert A,B,C,D to 0,1,2,3 for HellaSwag
        return ord(box_matches[-1]) - ord('A')  # Use last match (final answer)

    # No fallback patterns - forces models to use {} format only
    return None

def load_hellaswag_data(filepath):
    """Loads HellaSwag data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Sampling 10000 -> 1000
        data = data[:1000]
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth(item):
    """Returns the ground truth answer index."""
    # HellaSwag uses 'label' field for the correct answer index
    answer = item.get("label", -1)

    # Handle string labels (convert to int)
    if isinstance(answer, str):
        if answer in ["0", "1", "2", "3"]:
            return int(answer)
        elif answer == "":
            return -1  # Test dataset with empty labels

    # Handle integer labels
    if answer in [0, 1, 2, 3]:
        return answer

    return None

def save_failure_cases(failure_cases, model_name, output_dir):
    """Save failure cases to a separate JSON file for analysis."""
    failure_filepath = os.path.join(output_dir, f"failure_cases_{model_name}_5shot.json")

    with open(failure_filepath, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(failure_cases)} failure cases to {failure_filepath}")

# --- Single Model Evaluation Function with 5-shot Prompting ---
def evaluate_single_model(config: ModelConfig, hellaswag_data: list, ko_hellaswag_data: list, model_specific_output_dir: str):
    """Performs 5-shot HellaSwag evaluation for a single model on both datasets."""
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_5shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot.json")

    # Setup Logging for this specific model
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()

    # Remove previous file handlers to avoid duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 5-shot HellaSwag Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # Load Model and Tokenizer
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
            # Handle LoRA adapter loading (similar to ARC code)
            absolute_adapter_path = os.path.abspath(config.adapter_path)
            logger.info(f"LoRA adapter specified. Loading adapter from: {absolute_adapter_path}")

            if not os.path.isdir(absolute_adapter_path):
                logger.error(f"Adapter path does not exist or is not a directory: {absolute_adapter_path}")
                raise FileNotFoundError(f"Adapter path not found: {absolute_adapter_path}")

            # Resize token embeddings if needed
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
            if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
                model.resize_token_embeddings(len(tokenizer))
            logger.info("No LoRA adapter path specified. Using the base model directly.")

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # Disable compilation for Gemma models
        if "gemma" in config.name.lower():
            torch._dynamo.config.disable = True
            logger.info("Disabled torch compilation for Gemma model")

        # Evaluate on both datasets
        all_results = {}
        all_failure_cases = {}

        datasets = [
            ("HellaSwag", hellaswag_data, "hellaswag"),
            ("Ko-HellaSwag", ko_hellaswag_data, "ko-hellaswag")
        ]

        for dataset_name, dataset, dataset_type in datasets:
            logger.info(f"Starting evaluation on {dataset_name} dataset...")

            if dataset_type == "hellaswag":
                examples_to_use = HELLASWAG_5SHOT_EXAMPLES
            else:  # "ko-hellaswag"
                examples_to_use = KO_HELLASWAG_5SHOT_EXAMPLES

            correct_predictions = 0
            total_predictions = 0
            errors_or_skipped = 0
            results_details = []
            failure_cases = []
            all_ground_truths = []  # Track all ground truths for the dataset

            # Batch processing loop
            num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

            with logging_redirect_tqdm():
                pbar = tqdm(range(num_batches),
                           desc=f"Evaluating {config.name} on {dataset_name} (5-shot, errors: 0)",
                           ncols=100,
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
                            continue

                        prompt = create_5shot_prompt(item, examples_to_use, dataset_type)
                        prompts.append(prompt)
                        ground_truths.append(ground_truth)
                        all_ground_truths.append(ground_truth)  # Track all ground truths
                        valid_items_in_batch.append(item)

                    if not prompts:
                        continue

                    try:
                        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)

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
                        decoded_outputs = tokenizer.batch_decode(output_only_tokens, skip_special_tokens=True)

                        for j, (item, ground_truth, gen_text) in enumerate(zip(valid_items_in_batch, ground_truths, decoded_outputs)):
                            generated_text_log = gen_text.strip()
                            model_answer_log = extract_answer_robust(generated_text_log)
                            is_correct_log = False

                            if model_answer_log is not None:
                                total_predictions += 1
                                # Only evaluate correctness if we have valid ground truth
                                if ground_truth != -1:
                                    if model_answer_log == ground_truth:
                                        correct_predictions += 1
                                        is_correct_log = True
                                    else:
                                        # Wrong answer - add to failure cases
                                        failure_cases.append({
                                        "index": batch_start + j,
                                        "id": item.get("ind", ""),
                                        "dataset": dataset_name,
                                        "context": item.get("ctx", ""),
                                        "endings": item.get("endings", []),
                                        "ground_truth": ground_truth,
                                        "predicted_answer": model_answer_log,
                                        "raw_output": generated_text_log,
                                        "failure_type": "incorrect_answer"
                                        })
                                # For test datasets with no ground truth, just log the prediction without failure
                                else:
                                    pass  # No ground truth available, cannot determine correctness
                            else:
                                # Batch extraction failed, try individual retry
                                logger.warning(f"Batch extraction failed for item {batch_start + j}, attempting individual retry...")
                                prompt = create_5shot_prompt(item, examples_to_use, dataset_type)
                                retry_text, retry_answer = process_single_with_retry(model, tokenizer, prompt)

                                if retry_answer is not None:
                                    generated_text_log = retry_text
                                    model_answer_log = retry_answer
                                    total_predictions += 1
                                    # Only evaluate correctness if we have valid ground truth
                                    if ground_truth != -1:
                                        if model_answer_log == ground_truth:
                                            correct_predictions += 1
                                            is_correct_log = True
                                        else:
                                            failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("ind", ""),
                                            "dataset": dataset_name,
                                            "context": item.get("ctx", ""),
                                            "endings": item.get("endings", []),
                                            "ground_truth": ground_truth,
                                            "predicted_answer": model_answer_log,
                                            "raw_output": generated_text_log,
                                            "failure_type": "incorrect_answer_after_retry"
                                            })
                                    logger.info(f"Retry successful for item {batch_start + j}: extracted '{retry_answer}'")
                                else:
                                    # Even retry failed
                                    if not retry_text.startswith("ERROR"):
                                        logger.warning(f"Item {batch_start + j}: Failed to extract answer after retries")
                                        errors_or_skipped += 1
                                        generated_text_log = f"EXTRACTION_FAILED: {retry_text}"
                                        failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("ind", ""),
                                            "dataset": dataset_name,
                                            "context": item.get("ctx", ""),
                                            "endings": item.get("endings", []),
                                            "ground_truth": ground_truth,
                                            "predicted_answer": None,
                                            "raw_output": generated_text_log,
                                            "failure_type": "extraction_failed"
                                        })
                                    else:
                                        logger.error(f"Item {batch_start + j}: Model error: {retry_text}")
                                        errors_or_skipped += 1
                                        generated_text_log = retry_text
                                        failure_cases.append({
                                            "index": batch_start + j,
                                            "id": item.get("ind", ""),
                                            "dataset": dataset_name,
                                            "context": item.get("ctx", ""),
                                            "endings": item.get("endings", []),
                                            "ground_truth": ground_truth,
                                            "predicted_answer": None,
                                            "raw_output": generated_text_log,
                                            "failure_type": "model_error"
                                        })

                            current_item_index = batch_start + j
                            results_details.append({
                                "index": current_item_index,
                                "id": item.get("ind", ""),
                                "ground_truth": ground_truth,
                                "model_raw_output": generated_text_log,
                                "predicted_answer": model_answer_log,
                                "is_correct": is_correct_log
                            })

                            raw_generations_list.append({
                                "dataset": dataset_name,
                                "index": current_item_index,
                                "id": item.get("ind", ""),
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
                                "id": item.get("ind", ""),
                                "dataset": dataset_name,
                                "context": item.get("ctx", ""),
                                "endings": item.get("endings", []),
                                "ground_truth": ground_truth,
                                "predicted_answer": None,
                                "raw_output": f"BATCH_ERROR: {str(e)}",
                                "failure_type": "batch_inference_error"
                            })
                        errors_or_skipped += len(prompts)

                    # Update progress bar
                    pbar.set_description(f"Evaluating {config.name} on {dataset_name} (5-shot, errors: {errors_or_skipped})")

            # Calculate accuracy (only for datasets with ground truth)
            has_ground_truth = any(gt != -1 for gt in all_ground_truths if len(all_ground_truths) > 0)
            valid_ground_truth_count = sum(1 for gt in all_ground_truths if gt != -1) if len(all_ground_truths) > 0 else 0

            if has_ground_truth:
                accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                accuracy_strict = (correct_predictions / valid_ground_truth_count * 100) if valid_ground_truth_count > 0 else 0
            else:
                accuracy_standard = None  # No ground truth available
                accuracy_strict = None

            logger.info(f"--- 5-shot {dataset_name} Results for {config.name} ---")
            logger.info(f"Test Items: {len(dataset)}")
            logger.info(f"Valid Predictions: {total_predictions}")

            if has_ground_truth:
                logger.info(f"Correct Predictions: {correct_predictions}")
                logger.info(f"Failure Cases: {len(failure_cases)}")
                logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
                logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")
            else:
                logger.info(f"Ground Truth: Not available (test dataset)")
                logger.info(f"Inference Success Rate: {(total_predictions / len(dataset) * 100):.2f}%")
                logger.info(f"Note: Accuracy cannot be calculated without ground truth labels")

            all_results[dataset_name] = {
                "test_items": len(dataset),
                "valid_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "failure_cases_count": len(failure_cases),
                "accuracy_standard": accuracy_standard,
                "accuracy_strict": accuracy_strict,
                "has_ground_truth": has_ground_truth,
                "inference_success_rate": (total_predictions / len(dataset) * 100) if len(dataset) > 0 else 0,
                "details": results_details
            }

            all_failure_cases[dataset_name] = failure_cases

        # Save Results
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "5-shot HellaSwag Challenge",
            "datasets": all_results
        }

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)

        # Save failure cases
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
    hellaswag_data = load_hellaswag_data(HELLASWAG_DATASET_PATH)
    ko_hellaswag_data = load_hellaswag_data(KO_HELLASWAG_DATASET_PATH)

    if not hellaswag_data or not ko_hellaswag_data:
        logger.error("Failed to load one or both datasets")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Store summary results for all models
    summary_results = {}

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)

        results = evaluate_single_model(config, hellaswag_data, ko_hellaswag_data, model_specific_output_dir)

        if results:
            summary_results[config.name] = {
                "model_id": config.model_id,
                "adapter_path": config.adapter_path,
                "HellaSwag_accuracy_standard": results["HellaSwag"]["accuracy_standard"],
                "HellaSwag_accuracy_strict": results["HellaSwag"]["accuracy_strict"],
                "HellaSwag_failure_cases": results["HellaSwag"]["failure_cases_count"],
                "HellaSwag_inference_success_rate": results["HellaSwag"]["inference_success_rate"],
                "Ko-HellaSwag_accuracy_standard": results["Ko-HellaSwag"]["accuracy_standard"],
                "Ko-HellaSwag_accuracy_strict": results["Ko-HellaSwag"]["accuracy_strict"],
                "Ko-HellaSwag_failure_cases": results["Ko-HellaSwag"]["failure_cases_count"],
                "Ko-HellaSwag_inference_success_rate": results["Ko-HellaSwag"]["inference_success_rate"]
            }

    # Save summary results
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary results saved to: {summary_filepath}")

    # Print summary table
    print("\n" + "="*130)
    print("HELLASWAG 5-SHOT EVALUATION SUMMARY")
    print("="*130)
    print(f"{'Model Name':<35} {'HSwag Success (%)':<17} {'HSwag Acc (%)':<15} {'Ko-HSwag Success (%)':<19} {'Ko-HSwag Acc (%)':<17}")
    print("-"*130)

    for model_name, results in summary_results.items():
        hswag_success = results["HellaSwag_inference_success_rate"]
        hswag_acc = results["HellaSwag_accuracy_standard"]
        ko_hswag_success = results["Ko-HellaSwag_inference_success_rate"]
        ko_hswag_acc = results["Ko-HellaSwag_accuracy_standard"]

        hswag_acc_str = f"{hswag_acc:.2f}" if hswag_acc is not None else "N/A"
        ko_hswag_acc_str = f"{ko_hswag_acc:.2f}" if ko_hswag_acc is not None else "N/A"

        print(f"{model_name:<35} {hswag_success:<17.2f} {hswag_acc_str:<15} {ko_hswag_success:<19.2f} {ko_hswag_acc_str:<17}")

    print("-"*130)
    print("Note: Accuracy shows 'N/A' for test datasets without ground truth labels")
    print("Success rate shows percentage of successful model inferences")
    print("="*130)

if __name__ == "__main__":
    main()