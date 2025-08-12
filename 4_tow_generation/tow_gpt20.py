#!/usr/bin/env python3
"""
tow_gpt20.py

Generate 'Thought of Words' (ToW) dataset using the GPT-OSS-20B model.
This script reads sentences from klue_all.json, generates an English ToW
for each, and saves the result to a new JSON file.

[REQUIREMENTS]
- 24GB+ GPU (NVIDIA RTX 3090/4090)
- The 'openai/gpt-oss-20b' model must be downloaded beforehand.
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- 설정 (Configuration) ---
# 20B 모델 경로로 수정
MODEL_PATH = "/path/to/your/models/gpt_oss/gpt-oss-20b"
INPUT_JSON_PATH = "./klue_all.json"
# 결과 파일명 수정
OUTPUT_JSON_PATH = "./klue_all_with_tow_20b.json"
# 모델 ID 수정
MODEL_ID = "openai/gpt-oss-20b"

def generate_tow_dataset():
    """
    Loads the GPT-OSS-20B model and generates ToW for each sentence in the input JSON.
    """
    # 모델 경로 확인
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        print(f"[ERROR] 모델을 찾을 수 없습니다: {MODEL_PATH}")
        print(f"[INFO] 먼저 'download_gpt_oss_20b.py' 스크립트를 실행하여 모델을 다운로드하세요.")
        return

    # GPU 가용성 확인
    if not torch.cuda.is_available():
        print("[ERROR] 이 스크립트를 실행하려면 CUDA GPU가 필요합니다.")
        return

    print(f"[INFO] '{MODEL_ID}' 모델을 로드합니다. 잠시 기다려주세요...")

    # 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    print(f"[INFO] 총 {len(data)}개의 문장에 대해 ToW 생성을 시작합니다...")

    for item in tqdm(data, desc="Generating ToW"):
        sentence = item['sentence']
        prompt = (
            "You are an expert AI that understands context and nuance. "
            "Generate an English 'Thought-of-Words' (ToW) for the following Korean sentence. "
            "The ToW should explain the context, intent, or implicit meaning in a concise English phrase, enclosed in <ToW> tags. "
            "Do not repeat the original sentence.\n\n"
            f"Korean Sentence: \"{sentence}\"\n\n"
            "Generated ToW:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95
            )
        
        generated_ids = output_sequences[0, inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not generated_text.startswith("<ToW>"):
             generated_text = f"<ToW>{generated_text}"
        if not generated_text.endswith("</ToW>"):
             generated_text = f"{generated_text}</ToW>"

        new_item = {
            'id': item['id'],
            'sentence': sentence,
            'tow_sentence': f"{sentence} {generated_text}"
        }
        results.append(new_item)

    print(f"\n[SUCCESS] ToW 생성이 완료되었습니다. '{OUTPUT_JSON_PATH}' 파일에 결과를 저장합니다.")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_tow_dataset()