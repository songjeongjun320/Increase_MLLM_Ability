#!/usr/bin/env python3
"""
기존 merged_models 폴더의 토크나이저에 ToW 토큰을 추가하는 스크립트
"""

import os
from transformers import AutoTokenizer

def update_tokenizer_with_tow(model_folder_path):
    """
    기존 모델 폴더의 토크나이저에 ToW 토큰을 추가하고 저장
    """
    print(f"Processing: {model_folder_path}")
    
    # 1. 기존 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        print(f"  Original vocab size: {len(tokenizer)}")
        print(f"  Original additional_special_tokens: {tokenizer.additional_special_tokens}")
    except Exception as e:
        print(f"  ❌ Error loading tokenizer: {e}")
        return False
    
    # 2. ToW 토큰 존재 확인
    existing_special_tokens = tokenizer.additional_special_tokens or []
    tow_start_exists = '<ToW>' in existing_special_tokens
    tow_end_exists = '</ToW>' in existing_special_tokens
    
    if tow_start_exists and tow_end_exists:
        print(f"  ✅ ToW tokens already exist - skipping")
        return True
    
    # 3. ToW 토큰 추가
    tokens_to_add = []
    if not tow_start_exists:
        tokens_to_add.append('<ToW>')
    if not tow_end_exists:
        tokens_to_add.append('</ToW>')
    
    if tokens_to_add:
        all_special_tokens = existing_special_tokens + tokens_to_add
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
        print(f"  ➕ Added tokens: {tokens_to_add}")
        print(f"  📊 New vocab size: {len(tokenizer)}")
    
    # 4. 백업 생성 (선택사항)
    backup_dir = os.path.join(model_folder_path, "tokenizer_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        # 기존 토크나이저 파일들 백업
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", 
            "special_tokens_map.json", "added_tokens.json"
        ]
        for file in tokenizer_files:
            src = os.path.join(model_folder_path, file)
            dst = os.path.join(backup_dir, file)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
        print(f"  💾 Backup created at: {backup_dir}")
    
    # 5. 수정된 토크나이저 저장
    try:
        tokenizer.save_pretrained(model_folder_path)
        print(f"  ✅ Tokenizer updated successfully")
    except Exception as e:
        print(f"  ❌ Error saving tokenizer: {e}")
        return False
    
    # 6. 검증
    try:
        test_tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        tow_start_id = test_tokenizer.convert_tokens_to_ids('<ToW>')
        tow_end_id = test_tokenizer.convert_tokens_to_ids('</ToW>')
        
        if tow_start_id != test_tokenizer.unk_token_id and tow_end_id != test_tokenizer.unk_token_id:
            print(f"  ✅ Validation successful - ToW IDs: {tow_start_id}, {tow_end_id}")
            
            # 토큰화 테스트
            test_text = "Question: <ToW> thinking </ToW> Answer"
            tokens = test_tokenizer.tokenize(test_text)
            if '<ToW>' in str(tokens) and '</ToW>' in str(tokens):
                print(f"  ✅ Tokenization test passed")
            else:
                print(f"  ⚠️ Tokenization test: {tokens}")
        else:
            print(f"  ❌ Validation failed - ToW tokens not properly saved")
            return False
            
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False
    
    return True

def main():
    # 모델 폴더들 경로 설정
    base_path = "/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models"
    
    print("Starting tokenizer update process...")
    print(f"Scanning all folders in: {base_path}")
    print("=" * 60)
    
    # base_path 밑의 모든 폴더 자동 검색 (tokenizer 파일 존재 여부 무관하게 모든 폴더 처리)
    model_folders = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            folder_path = os.path.join(base_path, item)
            if os.path.isdir(folder_path):
                model_folders.append(item)
                print(f"Found folder: {item}")
    else:
        print(f"❌ Base path not found: {base_path}")
        return
    
    if not model_folders:
        print("❌ No folders found")
        return
    
    print(f"Total model folders found: {len(model_folders)}")
    print("-" * 60)
    
    success_count = 0
    for folder in model_folders:
        folder_path = os.path.join(base_path, folder)
        
        if update_tokenizer_with_tow(folder_path):
            success_count += 1
        print("-" * 60)
    
    print(f"Process completed: {success_count}/{len(model_folders)} models updated successfully")

if __name__ == "__main__":
    main()