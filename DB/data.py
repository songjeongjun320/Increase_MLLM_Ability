from datasets import load_dataset
import json
import os

# 디렉토리 생성 (없으면)
output_dir = "MMLU"
os.makedirs(output_dir, exist_ok=True)

# 1. OpenAI MMMLU KO_KR 데이터셋 로드 및 저장
ds_ko_kr = load_dataset("openai/MMMLU", "KO_KR")

# 데이터셋을 .json 파일로 저장
with open(os.path.join(output_dir, "MMLU_KO_KR.json"), "w", encoding="utf-8") as f:
    json.dump(ds_ko_kr.to_dict(), f, ensure_ascii=False, indent=4)

# 2. CAIS MMLU 데이터셋 로드 및 저장
ds_origin = load_dataset("cais/mmlu", "all")

# 데이터셋을 .json 파일로 저장
with open(os.path.join(output_dir, "MMLU_Origin.json"), "w", encoding="utf-8") as f:
    json.dump(ds_origin.to_dict(), f, ensure_ascii=False, indent=4)

print("파일이 저장되었습니다.")
