import os
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # PyTorch 임포트 추가

# --- 설정 ---
repo_id = "meta-llama/Llama-3.1-8B-Instruct"
download_path = "/scratch/jsong132/Increase_MLLM_Ability/Technical_Llama3.1_8B_Instruct" # 새 모델을 위한 경로

print(f"모델 및 토크나이저 파일을 다음 경로에 다운로드합니다: {download_path}")
os.makedirs(download_path, exist_ok=True)

# 다운로드할 파일 목록 (meta-llama/Llama-3.1-8B-Instruct 기준)
files_to_download = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]

# --- 파일 다운로드 ---
for filename in files_to_download:
    print(f"{filename} 다운로드 중...")
    try:
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=download_path,
            # token=HUGGING_FACE_TOKEN, # 토큰을 직접 전달할 경우
            # token=True, # huggingface-cli login으로 인증된 토큰 사용 시
        )
        print(f"{filename} 다운로드 완료: {downloaded_file_path}")
    except Exception as e:
        print(f"{filename} 다운로드 실패: {e}")
        print("Hugging Face Hub에 로그인되어 있고 모델 접근 권한이 있는지 확인하세요.")
        print("예: 터미널에서 'huggingface-cli login' 실행")
        # 실패 시 중단하거나 계속 진행할 수 있습니다. 여기서는 일단 계속 진행합니다.
        pass

print("\n모든 지정된 파일 다운로드 시도 완료.")

# --- 로컬 경로에서 모델 및 토크나이저 로드 ---
print(f"\n'{download_path}' 경로에서 토크나이저 로드 중...")
try:
    # 로컬 경로에서 직접 로드할 때는 해당 경로를 `from_pretrained`에 전달합니다.
    tokenizer = AutoTokenizer.from_pretrained(
        download_path,
    )
    print("토크나이저 로드 성공.")
except Exception as e:
    print(f"토크나이저 로드 실패: {e}")
    exit()

print(f"\n'{download_path}' 경로에서 모델 로드 중...")
try:
    # 로컬 경로에서 직접 로드
    # 8B 모델은 메모리를 많이 사용하므로, dtype과 device_map을 적절히 설정하는 것이 좋습니다.
    model = AutoModelForCausalLM.from_pretrained(
        download_path,
        torch_dtype=torch.bfloat16,  # 또는 torch.float16 (GPU 지원 여부 확인)
        device_map="auto",           # 사용 가능한 장치에 자동으로 모델 분산 (GPU, CPU)
        # token=HUGGING_FACE_TOKEN # 로컬 로드 시에도 필요할 수 있음
        # token=True
    )
    print("모델 로드 성공.")
    model.eval() # 추론 모드로 설정
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("RAM/VRAM이 충분한지, 다운로드된 파일들이 올바른지 확인하세요.")
    print("Llama-3.1-8B 모델은 bfloat16으로도 최소 16GB 이상의 VRAM이 필요할 수 있습니다.")
    exit()

# --- 간단한 테스트 (선택 사항) ---
print("\n텍스트 생성 테스트 중...")
try:
    prompt = "The capital of South Korea is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # 모델과 같은 장치로 이동

    # 생성 파라미터 설정
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
except Exception as e:
    print(f"텍스트 생성 중 오류 발생: {e}")

print("\n스크립트 실행 완료.")