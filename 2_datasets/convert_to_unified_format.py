#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 파일들을 통일된 양식으로 변환하는 스크립트

현재 세 파일의 구조:
1. kornli_texts.json: {"metadata": {...}, "texts": [...]}
2. ko_strategyqa_texts.json: {"metadata": {...}, "texts": [...]}
3. kobest_texts.json: {"metadata": {...}, "texts": [...]}

목표 양식:
[
    {
        "sentence": "텍스트 내용",
        "id": "파일번호_인덱스"
    },
    ...
]
"""

import json
import os
import sys
from typing import List, Dict

# Windows에서 UTF-8 출력을 위한 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def convert_file_to_unified_format(file_path: str, file_number: int) -> List[Dict[str, str]]:
    """
    JSON 파일을 통일된 양식으로 변환
    
    Args:
        file_path: 변환할 JSON 파일 경로
        file_number: 파일 식별 번호 (1: kornli, 2: ko_strategyqa, 3: kobest)
    
    Returns:
        통일된 양식의 데이터 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # texts 배열에서 데이터 추출
    texts = data.get('texts', [])
    
    # 통일된 양식으로 변환
    unified_data = []
    for idx, text in enumerate(texts):
        unified_data.append({
            "sentence": text.strip(),  # 앞뒤 공백 제거
            "id": f"{file_number}_{idx}"
        })
    
    return unified_data

def main():
    """메인 함수"""
    # 파일 경로 설정
    base_dir = r"C:\Users\songj\OneDrive\Desktop\Increase_MLLM_Ability\2_datasets"
    
    files_to_convert = [
        {
            "path": os.path.join(base_dir, "kakaobrain-kornli", "kornli_texts.json"),
            "number": 1,
            "name": "kornli"
        },
        {
            "path": os.path.join(base_dir, "Ko-StrategyQA", "ko_strategyqa_texts.json"),
            "number": 2,
            "name": "ko_strategyqa"
        },
        {
            "path": os.path.join(base_dir, "kobest", "kobest_texts.json"),
            "number": 3,
            "name": "kobest"
        }
    ]
    
    all_converted_data = []
    
    print("JSON files converting to unified format...")
    
    for file_info in files_to_convert:
        file_path = file_info["path"]
        file_number = file_info["number"]
        file_name = file_info["name"]
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"Processing {file_name} file...")
        
        try:
            converted_data = convert_file_to_unified_format(file_path, file_number)
            all_converted_data.extend(converted_data)
            print(f"[OK] {file_name}: {len(converted_data):,} items converted")
            
        except Exception as e:
            print(f"Error converting {file_name}: {e}")
            continue
    
    # 통합된 데이터를 새 파일로 저장
    output_path = os.path.join(base_dir, "unified_texts.json")
    
    print(f"\nSaving unified data to: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Conversion completed!")
        print(f"Total {len(all_converted_data):,} items converted to unified format.")
        
        # 각 파일별 통계 출력
        print("\nFile statistics:")
        for file_info in files_to_convert:
            file_number = file_info["number"]
            file_name = file_info["name"]
            count = len([item for item in all_converted_data if item["id"].startswith(f"{file_number}_")])
            print(f"  - {file_name}: {count:,} items")
            
        # 샘플 데이터 출력
        print("\nSample converted data (first 3 items):")
        for i, item in enumerate(all_converted_data[:3]):
            print(f"  {i+1}. ID: {item['id']}")
            print(f"     Text: {item['sentence'][:100]}{'...' if len(item['sentence']) > 100 else ''}")
            print()
            
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()