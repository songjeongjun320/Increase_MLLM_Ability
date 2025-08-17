# MMLU & KMMLU 5-Shot Evaluation 방법론

이 문서는 업데이트된 MMLU (Massive Multitask Language Understanding)와 KMMLU (Korean MMLU) 평가 시스템의 표준 5-shot evaluation 방법론을 설명합니다.

## 개요

기존 평가 시스템이 0-shot evaluation을 사용했던 것을 표준 **5-shot prompting** 방식으로 업그레이드하여 더 정확하고 신뢰성 있는 평가를 제공합니다.

## 표준 MMLU 5-Shot Evaluation 방법론

### 1. 기본 원칙

**5-shot prompting**은 MMLU 벤치마크의 표준 평가 방법으로, 다음과 같은 특징을 갖습니다:

- **Development Set 활용**: 각 과목별로 5개의 예제를 few-shot examples로 사용
- **First Token Generation**: 모델이 생성하는 첫 번째 토큰만을 답변으로 사용
- **표준 Prompt Template**: 일관된 형식으로 모든 모델에서 동일한 조건 제공
- **Subject-Specific Examples**: 같은 과목의 예제만 few-shot으로 사용

### 2. Prompt Template

#### 영어 MMLU (표준 형식)
```
The following are multiple choice questions (with answers) about [Subject Name].

[Example 1 Question]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
Answer: [Correct Letter]

[Example 2 Question]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
Answer: [Correct Letter]

[... 5 examples total ...]

[Test Question]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
Answer:
```

#### 한국어 KMMLU 형식
```
다음은 [과목명]에 관한 객관식 문제(정답 포함)입니다.

[예제 문제 1]
A. [선택지 1]
B. [선택지 2]
C. [선택지 3]
D. [선택지 4]
정답: [정답 알파벳]

[예제 문제 2]
A. [선택지 1]
B. [선택지 2]
C. [선택지 3]
D. [선택지 4]
정답: [정답 알파벳]

[... 총 5개 예제 ...]

[테스트 문제]
A. [선택지 1]
B. [선택지 2]
C. [선택지 3]
D. [선택지 4]
정답:
```

### 3. 데이터 분할 전략

#### Development/Test Split
- **Development Set**: 각 과목별로 처음 5개 문제를 few-shot 예제로 사용
- **Test Set**: 나머지 문제들을 실제 평가용으로 사용
- **일관성 보장**: 모든 모델에서 동일한 few-shot 예제 사용

#### 과목별 처리
- 각 과목마다 독립적으로 5개의 development examples 생성
- 과목이 5개 미만의 문제를 가진 경우 경고 출력 및 적절한 처리
- 과목별 결과를 종합하여 전체 성능 측정

### 4. 모델 생성 및 답변 추출

#### Generation Settings
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=1,  # 첫 번째 토큰만 생성
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,  # 결정적 출력
)
```

#### Answer Extraction
- **First Token Only**: 생성된 첫 번째 토큰에서 A, B, C, D 추출
- **정확한 매칭**: 생성된 토큰이 정확히 A, B, C, D 중 하나여야 함
- **실패 처리**: 유효한 답변을 추출할 수 없는 경우 오류로 처리

## 구현된 기능

### 1. 핵심 함수들

#### MMLU (영어)
- `prepare_mmlu_data_with_dev_split()`: 데이터 분할
- `create_5shot_prompt()`: 5-shot prompt 생성
- `extract_answer_first_token()`: 첫 토큰 기반 답변 추출
- `parse_choices_from_question()`: 선택지 파싱

#### KMMLU (한국어)
- `prepare_kmmlu_data_with_dev_split()`: 한국어 데이터 분할
- `create_5shot_korean_prompt()`: 한국어 5-shot prompt 생성
- `extract_korean_answer_first_token()`: 한국어 첫 토큰 답변 추출
- `parse_korean_choices_from_question()`: 한국어 선택지 파싱

### 2. 데이터 처리

#### 입력 데이터 형식
```json
{
  "question": "문제 텍스트",
  "answer": 1,  // 1-4 또는 0-3 (자동 처리)
  "subject": "과목명",
  "original": "영어 원문 (한국어 데이터의 경우)"
}
```

#### 출력 결과 형식
```json
{
  "model_config": {...},
  "evaluation_type": "5-shot MMLU",
  "total_original_items": 470,
  "dev_examples_per_subject": 5,
  "test_items": 455,
  "valid_predictions": 440,
  "correct_predictions": 380,
  "accuracy": 86.36,
  "subjects_with_dev_examples": ["abstract_algebra", "college_mathematics", "high_school_mathematics"],
  "details": [...]
}
```

### 3. 성능 개선사항

#### 토큰 효율성
- **0-shot**: `max_new_tokens=15` → **5-shot**: `max_new_tokens=1`
- 불필요한 텍스트 생성 방지
- 더 빠르고 일관된 평가

#### 정확성 향상
- 표준 benchmark 방법론 준수
- 일관된 few-shot 예제 사용
- 개선된 답변 추출 로직

#### 로깅 및 모니터링
- 더 자세한 진행 상황 로깅
- Dev/test split 정보 출력
- 과목별 통계 제공

## 사용 방법

### 1. MMLU 평가 실행
```bash
cd 3_evaluation
python eval_mmlu.py
```

### 2. KMMLU 평가 실행
```bash
cd 3_evaluation
python eval_kmmlu.py
```

### 3. 결과 확인
평가 결과는 다음 디렉토리에 저장됩니다:
- **MMLU**: `evaluation_results_mmlu_5shot_tow_model/`
- **KMMLU**: `evaluation_results_kmmlu_5shot_tow_model/`

각 모델별로 별도 하위 디렉토리 생성:
- `results_[model_name].json`: 상세 결과
- `eval_[model_name].log`: 평가 로그
- `raw_generations_[model_name].json`: 원본 생성 텍스트

## 성능 기대치

### MMLU 벤치마크 참고 성능
- **GPT-3**: 43.9% (5-shot)
- **GPT-4**: 86.4% (5-shot)
- **Llama-2-70B**: ~70% (5-shot)

### 5-shot vs 0-shot 성능 차이
일반적으로 5-shot evaluation이 0-shot보다 **5-15% 높은** 성능을 보입니다.

## 개발 히스토리

### 기존 문제점
1. **0-shot evaluation**: 표준 benchmark와 다른 방법 사용
2. **비효율적 토큰 생성**: 불필요하게 많은 토큰 생성
3. **복잡한 답변 추출**: 정규식 기반의 복잡한 로직
4. **일관성 부족**: 모델별로 다른 조건에서 평가

### 개선사항
1. **표준 5-shot prompting** 구현
2. **First token generation** 적용
3. **일관된 prompt template** 사용
4. **개선된 데이터 분할** 전략
5. **더 나은 로깅 및 모니터링**

## 주의사항

### 1. 데이터 호환성
- 현재 수학 관련 3개 과목만 지원 (abstract_algebra, college_mathematics, high_school_mathematics)
- 전체 57개 MMLU 과목 지원을 위해서는 완전한 MMLU 데이터셋 필요

### 2. 모델 호환성
- **Transformers 라이브러리** 호환 모델만 지원
- **PEFT/LoRA 어댑터** 지원
- **4-bit quantization** 지원

### 3. 성능 고려사항
- **메모리 사용량**: 5-shot prompting은 더 긴 입력 시퀀스 필요
- **추론 시간**: prompt가 길어지므로 약간의 시간 증가
- **정확성**: 표준 방법론을 따르므로 더 신뢰할 수 있는 결과

## 향후 개선 계획

1. **전체 MMLU 데이터셋** 지원
2. **Logit-based scoring** 옵션 추가
3. **Chain-of-Thought (CoT)** prompting 지원
4. **다국어 MMLU** 확장 지원
5. **자동 hyperparameter tuning**

## 참고문헌

1. Hendrycks, D., et al. (2020). "Measuring Massive Multitask Language Understanding." ICLR 2021.
2. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.
3. EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
4. Hugging Face MMLU Dataset: https://huggingface.co/datasets/cais/mmlu

---

이 evaluation 시스템은 표준 MMLU benchmark 방법론을 정확히 구현하여 신뢰할 수 있고 비교 가능한 평가 결과를 제공합니다.