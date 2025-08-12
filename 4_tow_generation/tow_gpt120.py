#!/usr/bin/env python3
"""
tow_gpt120.py

Generate 'Thought of Words' (ToW) dataset using the GPT-OSS-120B model.
This script reads sentences from klue_all.json, generates an English ToW
for each, and saves the result to a new JSON file.

[REQUIREMENTS]
- 80GB+ GPU (NVIDIA H100/A100)
- Sufficient system RAM (128GB+) and disk space
- The 'openai/gpt-oss-120b' model must be downloaded beforehand.
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 설정 (Configuration) ---
# 이전에 다운로드한 120B 모델의 경로를 지정하세요.
MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/1_models/gpt_oss/gpt-oss-120b"
INPUT_JSON_PATH = "./klue_all.json"  # KLUE 데이터셋 파일 경로
OUTPUT_JSON_PATH = "./klue_all_with_tow_120b.json" # 결과가 저장될 파일 경로
MODEL_ID = "openai/gpt-oss-120b"

def generate_tow_dataset():
    """
    Loads the GPT-OSS-120B model and generates ToW for each sentence in the input JSON.
    """
    # 모델 경로 확인
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        print(f"[ERROR] 모델을 찾을 수 없습니다: {MODEL_PATH}")
        print(f"[INFO] 먼저 'download_gpt_oss_120b.py' 스크립트를 실행하여 모델을 다운로드하세요.")
        return

    # GPU 가용성 확인
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 70 * (1024**3):
        print("[ERROR] 이 스크립트를 실행하려면 80GB 이상의 VRAM을 가진 CUDA GPU가 필요합니다.")
        return

    print(f"[INFO] '{MODEL_ID}' 모델을 로드합니다. 몇 분 정도 소요될 수 있습니다...")

    # 토크나이저 및 모델 로드
    # 메모리 사용량을 최적화하여 대형 모델을 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",          # 자동으로 GPU에 모델 레이어를 분배
        low_cpu_mem_usage=True,     # CPU 메모리 사용량 최소화
        trust_remote_code=True
    )

    # EOS 토큰이 PAD 토큰으로 사용되도록 설정 (생성 시 경고 방지)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    print(f"[INFO] 총 {len(data)}개의 문장에 대해 ToW 생성을 시작합니다...")

    # tqdm을 사용하여 진행 상황 표시
    for item in tqdm(data, desc="Generating ToW"):
        sentence = item['sentence']

        # 모델에게 역할을 부여하는 프롬프트 구성
        prompt = (
            "You are an expert AI that understands context and nuance. "
            "Generate an English 'Thought-of-Words' (ToW) for the following Korean sentence. "
            "The ToW should explain the context, intent, or implicit meaning in a concise English phrase, enclosed in <ToW> tags. "
            "Do not repeat the original sentence.\n\n"
            f"Korean Sentence: \"{sentence}\"\n\n"
            "Generated ToW:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # ToW 생성
        with torch.no_grad():
            # 프롬프트를 제외한 새로운 토큰만 생성하도록 설정
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=60,  # 생성할 최대 토큰 수
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95
            )

        # 입력 부분을 제외하고 새로 생성된 텍스트만 디코딩
        generated_ids = output_sequences[0, inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # 생성된 텍스트가 <ToW> 태그를 포함하도록 보정
        if not generated_text.startswith("<ToW>"):
             generated_text = f"<ToW>{generated_text}"
        if not generated_text.endswith("</ToW>"):
             generated_text = f"{generated_text}</ToW>"

        # 결과 저장
        new_item = {
            'id': item['id'],
            'sentence': sentence,
            'tow_sentence': f"{sentence} {generated_text}"
        }
        results.append(new_item)

    # 최종 결과를 JSON 파일로 저장
    print(f"\n[SUCCESS] ToW 생성이 완료되었습니다. '{OUTPUT_JSON_PATH}' 파일에 결과를 저장합니다.")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_tow_dataset()