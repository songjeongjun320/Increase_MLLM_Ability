import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# --- 설정 (본인 환경에 맞게 수정하세요) ---
# 1. 원본 베이스 모델 경로 (처음 학습 시작할 때 사용했던 모델)
base_model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/qwem-2.5-3b-pt"

# 2. 2400 스텝까지 학습된 LoRA 어댑터가 있는 폴더 경로
adapter_path = "./tow_trained_models/qwem"

# 3. 병합된 모델을 저장할 새 폴더 이름
output_merged_model_path = "./merged_models/qwem-merged-2400"
# -----------------------------------------

print(f"1. 기본 모델 로딩: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"2. 토크나이저 로딩 및 저장: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 병합된 모델 폴더에 토크나이저 파일을 미리 저장합니다.
tokenizer.save_pretrained(output_merged_model_path)

print(f"3. LoRA 어댑터 로딩: {adapter_path}")
# PeftModel을 사용해 기본 모델 위에 어댑터를 로드합니다.
model = PeftModel.from_pretrained(base_model, adapter_path)

print("4. 모델 병합 및 언로딩 중...")
# merge_and_unload()를 호출하여 어댑터 가중치를 기본 모델에 합칩니다.
model = model.merge_and_unload()
print("   병합 완료.")

print(f"5. 병합된 전체 모델 저장: {output_merged_model_path}")
model.save_pretrained(output_merged_model_path)

print("\n모든 작업이 완료되었습니다!")
print(f"병합된 모델이 '{output_merged_model_path}' 경로에 저장되었습니다.")
print("이제 이 경로를 --model_name_or_path 로 사용하여 학습을 재개할 수 있습니다.")