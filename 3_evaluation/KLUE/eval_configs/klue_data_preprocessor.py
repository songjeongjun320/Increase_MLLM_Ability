#!/usr/bin/env python3
"""
KLUE 데이터셋 전처리 유틸리티
특히 DP (Dependency Parsing)와 DST (Dialogue State Tracking)의 복잡한 구조를 처리
"""

import json
from typing import List, Dict, Any, Union
import re

def preprocess_dp_target(head_list: List[int]) -> str:
    """
    DP(Dependency Parsing)의 head 리스트를 공백으로 구분된 문자열로 변환
    
    Args:
        head_list: head 인덱스 리스트 (예: [2, 3, 14, 5, 14, 7, 10, 10, 10, 11, 12, 14, 14, 0])
    
    Returns:
        공백으로 구분된 문자열 (예: "2 3 14 5 14 7 10 10 10 11 12 14 14 0")
    """
    return " ".join(map(str, head_list))

def extract_last_user_state(dialogue: List[Dict[str, Any]]) -> List[str]:
    """
    DST(Dialogue State Tracking)에서 마지막 사용자 턴의 상태를 추출
    
    Args:
        dialogue: 대화 리스트
    
    Returns:
        마지막 사용자 턴의 상태 리스트
    """
    user_turns = [turn for turn in dialogue if turn.get('role') == 'user']
    if user_turns:
        return user_turns[-1].get('state', [])
    return []

def format_dst_state(state_list: List[str]) -> str:
    """
    DST 상태 리스트를 문자열로 포맷팅
    
    Args:
        state_list: 상태 리스트 (예: ['관광-종류-박물관', '관광-지역-서울 중앙'])
    
    Returns:
        포맷팅된 문자열
    """
    return ", ".join(state_list) if state_list else "상태 없음"

def extract_mrc_answer(answers: Union[List[Dict], Dict]) -> str:
    """
    MRC(Machine Reading Comprehension)에서 첫 번째 답변 텍스트 추출
    
    Args:
        answers: 답변 객체 (리스트 또는 딕셔너리)
    
    Returns:
        첫 번째 답변의 텍스트
    """
    if isinstance(answers, list) and answers:
        return answers[0].get('text', '답할 수 없음')
    elif isinstance(answers, dict):
        return answers.get('text', '답할 수 없음')
    return '답할 수 없음'

def validate_sts_score(score_text: str) -> float:
    """
    STS(Semantic Textual Similarity) 점수 검증 및 변환
    
    Args:
        score_text: 점수 텍스트
    
    Returns:
        유효한 점수 (0-5 범위)
    """
    try:
        # 정규식으로 점수 추출
        match = re.search(r'([0-5](?:\.[0-9]+)?)', score_text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(5.0, score))  # 0-5 범위로 제한
    except:
        pass
    return 0.0  # 기본값

def preprocess_klue_dataset(dataset_name: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    KLUE 데이터셋별 전처리 수행
    
    Args:
        dataset_name: 데이터셋 이름 ('dp', 'dst', 'mrc', 'sts' 등)
        data_point: 개별 데이터 포인트
    
    Returns:
        전처리된 데이터 포인트
    """
    processed_data = data_point.copy()
    
    if dataset_name == 'dp':
        # DP: head 리스트를 문자열로 변환
        if 'head' in processed_data and isinstance(processed_data['head'], list):
            processed_data['head_str'] = preprocess_dp_target(processed_data['head'])
    
    elif dataset_name == 'dst' or dataset_name == 'wos':
        # DST: 마지막 사용자 상태 추출
        if 'dialogue' in processed_data:
            last_state = extract_last_user_state(processed_data['dialogue'])
            processed_data['last_user_state'] = format_dst_state(last_state)
    
    elif dataset_name == 'mrc':
        # MRC: 답변 텍스트 추출
        if 'answers' in processed_data:
            processed_data['answer_text'] = extract_mrc_answer(processed_data['answers'])
    
    elif dataset_name == 'sts':
        # STS: 라벨 정규화
        if 'labels' in processed_data and 'label' in processed_data['labels']:
            # 이미 0-5 범위로 정규화된 라벨 사용
            processed_data['normalized_label'] = processed_data['labels']['label']
    
    return processed_data

def create_evaluation_script():
    """
    평가 스크립트 생성 (예제)
    """
    script = """
# KLUE 평가 실행 예제
python -m lm_eval \\
    --model hf \\
    --model_args pretrained=your_model_path \\
    --tasks klue_tc,klue_sts,klue_nli,klue_re,klue_dp,klue_mrc,klue_dst \\
    --num_fewshot 3 \\
    --batch_size auto \\
    --output_path ./klue_results.json \\
    --verbosity INFO

# 개별 태스크 실행
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_tc --output_path ./tc_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_sts --output_path ./sts_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_nli --output_path ./nli_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_re --output_path ./re_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_dp --output_path ./dp_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_mrc --output_path ./mrc_results.json
python -m lm_eval --model hf --model_args pretrained=your_model_path --tasks klue_dst --output_path ./dst_results.json
"""
    return script.strip()

if __name__ == "__main__":
    # 예제 사용법
    sample_dp_data = {
        'sentence': '해당 그림을 보면 디즈니 공주들이 브리트니 스피어스의 앨범이나 뮤직비디오, 화보 속 모습을 똑같이 재연했다.',
        'head': [2, 3, 14, 5, 14, 7, 10, 10, 10, 11, 12, 14, 14, 0]
    }
    
    processed = preprocess_klue_dataset('dp', sample_dp_data)
    print("DP 전처리 결과:")
    print(f"원본 head: {processed['head']}")
    print(f"변환된 head_str: {processed['head_str']}")
    
    # 평가 스크립트 출력
    print("\n평가 스크립트:")
    print(create_evaluation_script())