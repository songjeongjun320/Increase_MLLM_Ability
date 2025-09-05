#!/usr/bin/env python3
"""
Hugging Face 모델 다운로드 스크립트
지정된 폴더에 직접 저장합니다.
"""

import os
import torch
import json
from typing import Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

def download_and_save_model(model_name: str, save_dir: str, model_type: str) -> Dict[str, Any]:
    """
    모델을 다운로드하고 지정된 디렉토리에 직접 저장
    """
    print(f"📥 {model_name} 다운로드 중...")
    
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델과 토크나이저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 지정된 디렉토리에 직접 저장
        print(f"💾 {save_dir}에 저장 중...")
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        # config.json에 model_type 추가 (오류 방지)
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config["model_type"] = model_type
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"🔧 config.json에 model_type='{model_type}' 추가")
        
        print(f"✅ {model_name} → {save_dir} 저장 완료!")
        
        return {
            "tokenizer": tokenizer, 
            "model": model, 
            "model_name": model_name,
            "local_path": save_dir,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ {model_name} 저장 실패: {e}")
        return {"success": False, "error": str(e)}

def check_gpu_memory():
    """
    GPU 메모리 상태 확인
    """
    if torch.cuda.is_available():
        print(f"🖥️  GPU 사용 가능: {torch.cuda.get_device_name()}")
        print(f"📊 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU를 사용합니다.")

def main():
    """
    메인 함수 - 모든 모델을 지정된 폴더에 직접 다운로드
    """
    print("🚀 Hugging Face 모델 다운로드를 시작합니다...\n")
    
    # GPU 메모리 확인
    check_gpu_memory()
    print()
    
    # 기본 저장 위치
    base_models_dir = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models"
    os.makedirs(base_models_dir, exist_ok=True)
    
    # 다운로드할 모델 정보
    models_to_download = [
        # {
        #     "model_name": "meta-llama/Llama-3.2-3B",
        #     "folder_name": "llama-3.2-3b-pt",
        #     "model_type": "llama",
        #     "display_name": "🦙 Llama 3.2 3B"
        # },
        # {
        #     "model_name": "Qwen/Qwen2.5-3B", 
        #     "folder_name": "qwem-2.5-3b-pt",
        #     "model_type": "qwen2",
        #     "display_name": "🤖 Qwen 2.5 3B"
        # },
        {
            "model_name": "google/gemma-3-4b-pt",
            "folder_name": "gemma-3-4b-pt", 
            "model_type": "gemma3",
            "display_name": "💎 Gemma 3 4B"
        }
    ]
    
    print(f"📁 모델 저장 위치: {base_models_dir}")
    print("📂 각 모델이 저장될 폴더:")
    for model_info in models_to_download:
        folder_path = os.path.join(base_models_dir, model_info["folder_name"])
        print(f"   └── {folder_path}/")
        print(f"       ├── config.json")
        print(f"       ├── tokenizer.json") 
        print(f"       ├── tokenizer_config.json")
        print(f"       ├── model files...")
        print(f"       └── ...")
    print()
    
    # 모델 다운로드 및 저장
    downloaded_models = []
    
    for i, model_info in enumerate(models_to_download, 1):
        print(f"\n{'='*60}")
        print(f"📥 [{i}/{len(models_to_download)}] {model_info['display_name']} 처리 중...")
        print(f"{'='*60}")
        
        model_name = model_info["model_name"]
        folder_name = model_info["folder_name"]
        model_type = model_info["model_type"]
        save_path = os.path.join(base_models_dir, folder_name)
        
        print(f"🔗 Hugging Face: {model_name}")
        print(f"📁 저장 경로: {save_path}")
        
        # 모델 다운로드 및 저장
        result = download_and_save_model(model_name, save_path, model_type)
        
        if result.get("success"):
            downloaded_models.append(result)
            
            # 저장된 파일 확인
            if os.path.exists(save_path):
                file_count = len([f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))])
                folder_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(save_path)
                                for filename in filenames) / (1024**3)
                print(f"📊 저장된 파일: {file_count}개, 크기: {folder_size:.1f}GB")
        else:
            print(f"❌ {model_info['display_name']} 저장 실패")
        
        # 메모리 정리
        if 'tokenizer' in result:
            del result['tokenizer']
        if 'model' in result:
            del result['model']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print("📋 다운로드 결과 요약")
    print(f"{'='*60}")
    
    if downloaded_models:
        print(f"🎉 총 {len(downloaded_models)}개 모델이 성공적으로 저장되었습니다!\n")
        
        for i, model_info in enumerate(downloaded_models, 1):
            folder_name = os.path.basename(model_info['local_path'])
            print(f"{i}. ✅ {folder_name}")
            print(f"   📂 {model_info['local_path']}")
            
            # 주요 파일 확인
            key_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            for key_file in key_files:
                file_path = os.path.join(model_info['local_path'], key_file)
                status = "✅" if os.path.exists(file_path) else "❌"
                print(f"   {status} {key_file}")
            print()
        
        print(f"📁 모든 모델이 {base_models_dir} 하위의 지정된 폴더에 저장되었습니다.")
        
    else:
        print("❌ 성공적으로 저장된 모델이 없습니다.")
    
    return downloaded_models

def test_model_loading():
    """
    저장된 모델들이 제대로 로드되는지 테스트
    """
    print("\n🧪 저장된 모델 로딩 테스트...")
    
    base_models_dir = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models"
    model_folders = ["llama-3.2-3b-pt", "qwem-2.5-3b-pt", "gemma-3-4b-pt"]
    
    for folder in model_folders:
        model_path = os.path.join(base_models_dir, folder)
        if os.path.exists(model_path):
            try:
                print(f"🔄 {folder} 로딩 테스트 중...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"   ✅ 토크나이저 로딩 성공")
                
                # 메모리 절약을 위해 모델은 로딩하지 않고 config만 확인
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f"   ✅ config.json 확인 (model_type: {config.get('model_type', 'None')})")
                else:
                    print(f"   ❌ config.json 없음")
                    
            except Exception as e:
                print(f"   ❌ {folder} 로딩 실패: {e}")
        else:
            print(f"❌ {folder} 폴더를 찾을 수 없습니다: {model_path}")

if __name__ == "__main__":
    # 필요한 라이브러리 설치 안내
    print("📦 필요한 라이브러리:")
    print("pip install transformers torch accelerate")
    print("pip install sentencepiece protobuf")
    print("pip install Pillow")
    print("-" * 60)
    
    # 모델 다운로드 실행
    downloaded_models = main()
    
    # 로딩 테스트 옵션
    if downloaded_models:
        test_loading = input("\n🧪 저장된 모델 로딩 테스트를 수행하시겠습니까? (y/n): ").lower().strip()
        if test_loading == 'y':
            test_model_loading()
    
    print(f"\n{'='*60}")
    print("🎉 모든 작업이 완료되었습니다!")
    print("📁 지정된 폴더에 모델들이 직접 저장되었습니다.")
    print(f"{'='*60}")