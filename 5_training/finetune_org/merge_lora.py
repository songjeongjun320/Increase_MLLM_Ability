import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil
import glob
from safetensors import safe_open

# --- 설정 (본인 환경에 맞게 수정하세요) ---
base_model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/gemma-3-4b-pt"
adapter_path = "./tow_trained_models/gemma-3-4b-tow-09_11_2epoch_fix_tow"
output_merged_model_path = "./merged_models/gemma-3-4b-tow-09_11_2epoch_fix_tow-merged"
# -----------------------------------------

def get_adapter_vocab_size(adapter_path):
    """어댑터에서 실제 vocab size 추출"""
    try:
        # safetensors 파일 우선 검색
        safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
        pytorch_files = glob.glob(os.path.join(adapter_path, "*.bin"))
        
        if safetensor_files:
            with safe_open(safetensor_files[0], framework="pt") as f:
                for key in f.keys():
                    if any(target in key for target in ['embed_tokens.weight', 'lm_head.weight']):
                        return f.get_tensor(key).shape[0]
        
        elif pytorch_files:
            checkpoint = torch.load(pytorch_files[0], map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            for key, tensor in state_dict.items():
                if any(target in key for target in ['embed_tokens.weight', 'lm_head.weight']):
                    return tensor.shape[0]
            del checkpoint
            
    except Exception as e:
        print(f"Error extracting vocab size: {e}")
    return None

print(f"1. 기본 모델 로딩: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"2. 토크나이저 로딩: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 현재 모델과 어댑터의 vocab size 확인
base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
adapter_vocab_size = get_adapter_vocab_size(adapter_path)

print(f"베이스 모델 vocab size: {base_vocab_size}")
print(f"어댑터 vocab size: {adapter_vocab_size}")

# vocab size가 다르면 베이스 모델을 어댑터에 맞춰 조정
if adapter_vocab_size and adapter_vocab_size != base_vocab_size:
    print(f"3. vocab size 불일치 해결: {base_vocab_size} -> {adapter_vocab_size}")
    base_model.resize_token_embeddings(adapter_vocab_size)
    print("   베이스 모델 embedding 크기 조정 완료")
else:
    print("3. vocab size 일치함 - 조정 불필요")

print(f"4. LoRA 어댑터 로딩: {adapter_path}")
try:
    # PeftModel을 사용해 기본 모델 위에 어댑터를 로드합니다.
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("   어댑터 로딩 성공")
except RuntimeError as e:
    if "size mismatch" in str(e):
        print(f"   여전히 size mismatch 발생: {e}")
        print("   대안 방법으로 어댑터를 수동 로드합니다...")
        
        # 수동으로 어댑터 로드 (embedding layer 제외)
        model = load_adapter_manually(base_model, adapter_path)
    else:
        raise e

def load_adapter_manually(base_model, adapter_path):
    """수동으로 어댑터를 로드 (embedding 관련 문제 우회)"""
    from peft import LoraConfig, get_peft_model
    import json
    
    # adapter config 로드
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        peft_config = json.load(f)
    
    # LoraConfig 생성
    lora_config = LoraConfig(**peft_config)
    
    # 빈 PEFT 모델 생성
    model = get_peft_model(base_model, lora_config)
    
    # state dict 로드
    safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
    pytorch_files = glob.glob(os.path.join(adapter_path, "*.bin"))
    
    if safetensor_files:
        with safe_open(safetensor_files[0], framework="pt") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
    elif pytorch_files:
        checkpoint = torch.load(pytorch_files[0], map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        raise FileNotFoundError("No adapter weights found")
    
    # embedding layer는 제외하고 로드
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(skip in key for skip in ['embed_tokens.weight', 'lm_head.weight']):
            filtered_state_dict[key] = value
    
    # 필터링된 state dict 로드
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"   수동 로드 완료: {len(filtered_state_dict)} 파라미터 로드됨")
    if missing_keys:
        print(f"   누락된 키 (예상됨): {len(missing_keys)}개")
    
    return model

print("5. 모델 병합 및 언로딩 중...")
# merge_and_unload()를 호출하여 어댑터 가중치를 기본 모델에 합칩니다.
try:
    model = model.merge_and_unload()
    print("   병합 완료")
except Exception as e:
    print(f"   병합 중 오류 발생: {e}")
    print("   가능한 원인: LoRA 어댑터에 embedding layer가 포함되어 있음")
    raise e

# 출력 폴더 생성
os.makedirs(output_merged_model_path, exist_ok=True)

print(f"6. 토크나이저 저장: {output_merged_model_path}")
tokenizer.save_pretrained(output_merged_model_path)

print(f"7. 병합된 전체 모델 저장: {output_merged_model_path}")
model.save_pretrained(output_merged_model_path)

print("\n모든 작업이 완료되었습니다!")
print(f"병합된 모델이 '{output_merged_model_path}' 경로에 저장되었습니다.")
print("이제 이 경로를 --model_name_or_path 로 사용하여 학습을 재개할 수 있습니다.")

# 최종 확인
final_model = AutoModelForCausalLM.from_pretrained(output_merged_model_path)
final_tokenizer = AutoTokenizer.from_pretrained(output_merged_model_path)
print(f"\n최종 확인:")
print(f"병합된 모델 vocab size: {final_model.get_input_embeddings().weight.shape[0]}")
print(f"병합된 토크나이저 vocab size: {len(final_tokenizer)}")