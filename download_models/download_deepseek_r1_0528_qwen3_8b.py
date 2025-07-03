import os
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # PyTorch 임포트 추가

# --- 설정 ---
repo_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
download_path = "/scratch/jsong132/Increase_MLLM_Ability/DeepSeek_R1_0528_Qwen3_8B" # DeepSeek 모델을 위한 경로

print(f"모델 및 토크나이저 파일을 다음 경로에 다운로드합니다: {download_path}")
os.makedirs(download_path, exist_ok=True)

# 다운로드할 파일 목록 (deepseek-ai/DeepSeek-R1-Distill-Llama-70B 기준)
files_to_download = [
    # 설정 파일들
    "config.json",
    "model.safetensors.index.json",
    
    # 모델 파일들 (17개 파일)
    "model-00001-of-000002.safetensors",
    "model-00002-of-000002.safetensors",
    
    # 토크나이저 파일들
    "tokenizer.json",
    "tokenizer_config.json",
]

# 총 파일 크기 계산 (대략적)
total_size_gb = (8.95 + 8.69 + 1.58 + 8.69 + 8.42 + 8.69 + 8.42 + 8.69 + 8.42 + 
                8.69 + 8.42 + 8.69 + 8.42 + 8.69 + 8.42 + 8.69 + 10.5)
print(f"예상 다운로드 크기: 약 {total_size_gb:.1f} GB")
print("⚠️  이 모델은 매우 큽니다. 충분한 저장 공간과 안정적인 인터넷 연결이 필요합니다.")

# 사용자 확인
user_confirm = input("다운로드를 계속하시겠습니까? (y/n): ").strip().lower()
if user_confirm not in ['y', 'yes', '예']:
    print("다운로드가 취소되었습니다.")
    exit()

# --- 파일 다운로드 ---
print(f"\n총 {len(files_to_download)}개 파일 다운로드 시작...")
downloaded_files = []
failed_files = []

for i, filename in enumerate(files_to_download, 1):
    print(f"\n[{i}/{len(files_to_download)}] {filename} 다운로드 중...")
    try:
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=download_path,
            resume_download=True,  # 중단된 다운로드 재개
            # token=HUGGING_FACE_TOKEN, # 토큰을 직접 전달할 경우
            # token=True, # huggingface-cli login으로 인증된 토큰 사용 시
        )
        print(f"✅ {filename} 다운로드 완료: {downloaded_file_path}")
        downloaded_files.append(filename)
    except Exception as e:
        print(f"❌ {filename} 다운로드 실패: {e}")
        failed_files.append(filename)
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("⚠️  모델 접근 권한이 필요할 수 있습니다.")
            print("1. Hugging Face Hub에 로그인: 'huggingface-cli login'")
            print("2. DeepSeek 모델 페이지에서 접근 권한 요청")
        elif "token" in str(e).lower():
            print("⚠️  인증 토큰이 필요합니다. 'huggingface-cli login' 실행")

print(f"\n다운로드 완료: {len(downloaded_files)}개 파일")
if failed_files:
    print(f"다운로드 실패: {len(failed_files)}개 파일")
    print("실패한 파일들:", failed_files)

# --- 다운로드 검증 ---
print("\n다운로드된 파일 확인 중...")
for filename in files_to_download:
    file_path = os.path.join(download_path, filename)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024**3)  # GB 단위
        print(f"✅ {filename}: {file_size:.2f} GB")
    else:
        print(f"❌ {filename}: 파일 없음")

# --- 로컬 경로에서 모델 및 토크나이저 로드 (선택사항) ---
load_test = input("\n다운로드 완료 후 모델 로드 테스트를 하시겠습니까? (y/n): ").strip().lower()

if load_test in ['y', 'yes', '예']:
    print(f"\n'{download_path}' 경로에서 토크나이저 로드 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            download_path,
            trust_remote_code=True,
        )
        print("✅ 토크나이저 로드 성공.")
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        exit()

    print(f"\n'{download_path}' 경로에서 모델 로드 중...")
    print("⚠️  70B 모델은 매우 큰 메모리가 필요합니다. (최소 140GB+ VRAM 또는 적절한 양자화)")
    
    try:
        # 70B 모델은 메모리를 매우 많이 사용하므로, 양자화 또는 적절한 device_map 설정 필요
        model = AutoModelForCausalLM.from_pretrained(
            download_path,
            torch_dtype=torch.bfloat16,    # 메모리 절약을 위한 bfloat16
            device_map="auto",             # 사용 가능한 장치에 자동으로 모델 분산
            trust_remote_code=True,
            low_cpu_mem_usage=True,        # CPU 메모리 사용량 최적화
            # offload_folder="./offload",  # 필요시 디스크로 오프로드
        )
        print("✅ 모델 로드 성공.")
        model.eval() # 추론 모드로 설정
        
        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("메모리 부족일 가능성이 높습니다. 다음을 시도해보세요:")
        print("1. 더 많은 GPU 메모리 확보")
        print("2. 양자화 사용 (8-bit 또는 4-bit)")
        print("3. CPU 오프로딩 활용")
        print("4. 모델을 여러 GPU에 분산")
        exit()

    # --- 간단한 테스트 ---
    print("\n텍스트 생성 테스트 중...")
    try:
        prompt = "The capital of South Korea is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 입력을 모델과 같은 장치로 이동
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 생성 파라미터 설정
        with torch.no_grad():  # 메모리 절약
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("✅ 텍스트 생성 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 텍스트 생성 중 오류 발생: {e}")
        if "out of memory" in str(e).lower():
            print("GPU 메모리 부족입니다. 더 작은 배치 크기나 양자화를 사용해보세요.")

print("\n스크립트 실행 완료.")
print(f"모델 파일 위치: {download_path}")
print("이제 이 경로를 사용하여 로컬에서 모델을 로드할 수 있습니다.")