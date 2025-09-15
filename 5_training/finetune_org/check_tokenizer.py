#!/usr/bin/env python3
"""
ToW Token Checker - 모델 경로에서 ToW 토큰 존재 여부를 확인하는 스크립트
"""

import os
import json
from transformers import AutoTokenizer
from pathlib import Path

def check_tow_tokens(model_path):
    """
    주어진 모델 경로에서 ToW 토큰의 존재 여부를 상세히 확인
    """
    print(f"Checking ToW tokens in: {model_path}")
    print("=" * 80)
    
    # 1. 경로 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ Error: Path does not exist: {model_path}")
        return
    
    # 2. 토크나이저 파일들 확인
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model", 
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    print("📁 Tokenizer files check:")
    for file in tokenizer_files:
        file_path = os.path.join(model_path, file)
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {exists} {file}")
    
    print()
    
    try:
        # 3. 토크나이저 로딩
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 4. 기본 정보
        print(f"📊 Basic tokenizer info:")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        print(f"  Tokenizer class: {tokenizer.__class__.__name__}")
        print()
        
        # 5. 특수 토큰 확인
        print("🔍 Special tokens:")
        special_tokens = {
            'bos_token': tokenizer.bos_token,
            'eos_token': tokenizer.eos_token,
            'unk_token': tokenizer.unk_token,
            'pad_token': tokenizer.pad_token,
        }
        
        for name, token in special_tokens.items():
            print(f"  {name}: {token} (ID: {tokenizer.convert_tokens_to_ids(token) if token else 'None'})")
        
        # 6. Additional special tokens 확인
        additional_tokens = tokenizer.additional_special_tokens
        print(f"  additional_special_tokens: {additional_tokens}")
        print()
        
        # 7. ToW 토큰 존재 확인
        print("🎯 ToW Token Analysis:")
        
        # Vocab에서 ToW 토큰 찾기
        vocab = tokenizer.get_vocab()
        tow_start_in_vocab = '<ToW>' in vocab
        tow_end_in_vocab = '</ToW>' in vocab
        
        print(f"  <ToW> in vocab: {tow_start_in_vocab}")
        print(f"  </ToW> in vocab: {tow_end_in_vocab}")
        
        if tow_start_in_vocab:
            tow_start_id = vocab['<ToW>']
            print(f"    <ToW> ID: {tow_start_id}")
        
        if tow_end_in_vocab:
            tow_end_id = vocab['</ToW>']
            print(f"    </ToW> ID: {tow_end_id}")
        
        # Additional special tokens에서 확인
        tow_start_in_additional = '<ToW>' in (additional_tokens or [])
        tow_end_in_additional = '</ToW>' in (additional_tokens or [])
        
        print(f"  <ToW> in additional_special_tokens: {tow_start_in_additional}")
        print(f"  </ToW> in additional_special_tokens: {tow_end_in_additional}")
        
        # 8. 토큰화 테스트
        print("\n🧪 Tokenization test:")
        test_text = "Question: What is 2+2? <ToW> Let me think </ToW> Answer: 4"
        
        try:
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.encode(test_text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            
            print(f"  Input: {test_text}")
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {token_ids}")
            print(f"  Decoded: {decoded}")
            
            # ToW 토큰이 올바르게 토큰화되는지 확인
            tow_start_found = any('<ToW>' in str(token) for token in tokens)
            tow_end_found = any('</ToW>' in str(token) for token in tokens)
            
            print(f"  <ToW> found in tokens: {tow_start_found}")
            print(f"  </ToW> found in tokens: {tow_end_found}")
            
        except Exception as e:
            print(f"  ❌ Tokenization test failed: {e}")
        
        # 9. 개별 토큰 테스트
        print("\n🔬 Individual token tests:")
        for token in ['<ToW>', '</ToW>']:
            try:
                direct_tokens = tokenizer.tokenize(token)
                direct_ids = tokenizer.encode(token, add_special_tokens=False)
                convert_id = tokenizer.convert_tokens_to_ids(token)
                
                print(f"  {token}:")
                print(f"    tokenize(): {direct_tokens}")
                print(f"    encode(): {direct_ids}")
                print(f"    convert_tokens_to_ids(): {convert_id}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 10. 요약
        print(f"\n📋 Summary:")
        tow_exists = tow_start_in_vocab and tow_end_in_vocab
        print(f"  ToW tokens exist: {'✅ YES' if tow_exists else '❌ NO'}")
        
        if tow_exists:
            print(f"  Status: ToW tokens are properly configured")
        else:
            print(f"  Status: ToW tokens need to be added")
            
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")

if __name__ == "__main__":
    # 모델 경로 설정
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-tow-09_11_2epoch_fix_tow-merged"
    
    check_tow_tokens(model_path)