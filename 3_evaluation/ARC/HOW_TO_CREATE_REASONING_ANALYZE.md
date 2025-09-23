# How to Create Reasoning Analysis - Step by Step Guide

이 문서는 ARC 데이터셋을 활용하여 모델의 언어별 추론 능력을 분석하는 전체 과정을 설명합니다.

## 📁 프로젝트 구조

```
3_evaluation/ARC/
├── basemodel_results/          # 영어+영어추론, 한국어+한국어추론 결과
│   ├── results_gemma-3-4b-pt_3shot.json
│   ├── results_llama-3.2-3b-pt_3shot.json
│   └── results_qwem-2.5-3b-pt_3shot.json
├── basemodel_eng_reasoning/    # 한국어+영어추론 결과
│   ├── results_gemma-3-4b-pt_3shot.json
│   ├── results_llama-3.2-3b-pt_3shot.json
│   └── results_qwem-2.5-3b-pt_3shot.json
├── step1_results/              # 1단계 분석 결과
│   ├── eng_correct-kr_incorrect_gemma-3-4b-pt.json
│   ├── eng_correct-kr_incorrect_llama-3.2-3b-pt.json
│   └── eng_correct-kr_incorrect_qwem-2.5-3b-pt.json
├── step2_results/              # 2단계 분석 결과
│   ├── kr_input_eng_reasoning_correct_gemma-3-4b-pt.json
│   ├── kr_input_eng_reasoning_correct_llama-3.2-3b-pt.json
│   ├── kr_input_eng_reasoning_correct_qwem-2.5-3b-pt.json
│   ├── kr_input_eng_reasoning_incorrect_gemma-3-4b-pt.json
│   ├── kr_input_eng_reasoning_incorrect_llama-3.2-3b-pt.json
│   └── kr_input_eng_reasoning_incorrect_qwem-2.5-3b-pt.json
├── step3_results/              # 3단계 분석 결과
│   ├── kr_input_unsolvable_gemma-3-4b-pt.json
│   ├── kr_input_unsolvable_llama-3.2-3b-pt.json
│   └── kr_input_unsolvable_qwem-2.5-3b-pt.json
├── step4_results/              # 4단계 분석 결과 (최종)
│   ├── comprehensive_reasoning_analysis_gemma-3-4b-pt.json
│   ├── comprehensive_reasoning_analysis_llama-3.2-3b-pt.json
│   └── comprehensive_reasoning_analysis_qwem-2.5-3b-pt.json
├── 1_find_eng_correct_kr_incorrect.py     # 1단계 스크립트
├── 2_extract_eng_reasoning_results.py     # 2단계 스크립트
├── 3_find_common_unsolvable_ids.py        # 3단계 스크립트
├── 4_create_comprehensive_analysis.py     # 4단계 스크립트
└── HOW_TO_CREATE_REASONING_ANALYZE.md     # 이 가이드 문서
```

## 🔄 분석 과정 (4단계)

### 1단계: 영어 정답 & 한국어 오답 ID 추출
**파일**: `find_eng_correct_kr_incorrect.py`

**목적**: ARC(영어)에서는 정답이지만 Ko-ARC(한국어)에서는 오답인 문제 ID들을 찾기

**입력 데이터**:
- `basemodel_results/results_{model}_3shot.json` 파일들
- 각 파일의 `datasets.ARC.details`와 `datasets.Ko-ARC.details` 섹션

**처리 로직**:
```python
# ARC와 Ko-ARC 결과를 비교
for item_id in arc_results:
    if item_id in ko_arc_results:
        arc_correct = arc_results[item_id]      # 영어+영어추론 결과
        ko_arc_correct = ko_arc_results[item_id] # 한국어+한국어추론 결과

        if arc_correct and not ko_arc_correct:  # 영어O, 한국어X
            eng_correct_kr_incorrect.append(item_id)
```

**출력 파일**:
- `eng_correct-kr_incorrect_{model}.json` (모델별 3개 파일)
- 각 파일에는 영어로는 맞췄지만 한국어로는 틀린 ID 리스트

**결과 예시**:
- gemma-3-4b-pt: 229개 항목
- llama-3.2-3b-pt: 362개 항목
- qwem-2.5-3b-pt: 386개 항목

---

### 2단계: 한국어 입력 + 영어 추론 결과 분석
**파일**: `extract_eng_reasoning_results.py`

**목적**: 한국어 질문에 영어로 추론한 결과를 정답/오답으로 분류

**입력 데이터**:
- `basemodel_eng_reasoning/results_{model}_3shot.json` 파일들
- 각 파일의 `datasets.Ko-ARC.details` 섹션

**처리 로직**:
```python
# Ko-ARC 데이터에서 is_correct 기준으로 분류
for item in ko_arc_data:
    if item['is_correct']:
        correct_ids.append(item['id'])      # 한국어+영어추론 성공
    else:
        incorrect_ids.append(item['id'])    # 한국어+영어추론 실패
```

**출력 파일**:
- `kr_input_eng_reasoning_correct_{model}.json` (한국어+영어추론 성공)
- `kr_input_eng_reasoning_incorrect_{model}.json` (한국어+영어추론 실패)

**결과 예시**:
- gemma-3-4b-pt: 696개 성공, 471개 실패
- llama-3.2-3b-pt: 501개 성공, 666개 실패
- qwem-2.5-3b-pt: 743개 성공, 424개 실패

---

### 3단계: 추론 언어와 무관하게 해결 불가능한 문제 찾기
**파일**: `find_common_unsolvable_ids.py`

**목적**: 영어로는 해결 가능하지만, 한국어 입력일 때는 추론 언어와 관계없이 해결 불가능한 문제들 식별

**입력 데이터**:
- 1단계 결과: `eng_correct-kr_incorrect_{model}.json`
- 2단계 결과: `kr_input_eng_reasoning_incorrect_{model}.json`

**처리 로직**:
```python
# 두 집합의 교집합 구하기
eng_correct_kr_incorrect_ids = set(파일1의_ids)
kr_input_eng_reasoning_incorrect_ids = set(파일2의_ids)

common_ids = eng_correct_kr_incorrect_ids.intersection(kr_input_eng_reasoning_incorrect_ids)
```

**분석 의미**:
- 영어 입력 + 영어 추론 = ✅ 정답
- 한국어 입력 + 한국어 추론 = ❌ 오답
- 한국어 입력 + 영어 추론 = ❌ 오답
- **결론**: 한국어 입력 자체가 문제를 해결하는 데 장벽이 되는 경우

**출력 파일**:
- `kr_input_unsolvable_{model}.json`

**결과 예시**:
- gemma-3-4b-pt: 114개 항목
- llama-3.2-3b-pt: 202개 항목
- qwem-2.5-3b-pt: 181개 항목

---

### 4단계: 종합 추론 과정 비교 분석
**파일**: `create_comprehensive_analysis.py`

**목적**: 같은 문제에 대한 3가지 추론 방식의 전체 과정과 결과를 비교

**입력 데이터**:
- 3단계 결과: `kr_input_unsolvable_{model}.json` (분석 대상 ID 리스트)
- `basemodel_results/results_{model}_3shot.json` (영어+영어, 한국어+한국어 추론)
- `basemodel_eng_reasoning/results_{model}_3shot.json` (한국어+영어 추론)

**처리 로직**:
```python
for item_id in unsolvable_ids:
    # 각 추론 방식에서 해당 ID의 상세 정보 수집
    eng_eng_item = arc_details.get(item_id)           # 영어+영어
    kr_kr_item = ko_arc_details.get(item_id)          # 한국어+한국어
    kr_eng_item = kr_eng_details.get(item_id)         # 한국어+영어

    # 3가지 방식의 raw_output, predicted_answer, is_correct 정보 통합
```

**출력 파일**:
- `comprehensive_reasoning_analysis_{model}.json`

**파일 구조**:
```json
{
  "model_name": "모델명",
  "total_count": "분석된 문제 수",
  "items": [
    {
      "id": "문제ID",
      "ground_truth": "정답",
      "reasoning_comparisons": {
        "eng_eng_reasoning": {
          "predicted_answer": "예측답안",
          "is_correct": true,
          "model_raw_output": "전체 추론 과정..."
        },
        "kr_kr_reasoning": {
          "predicted_answer": "예측답안",
          "is_correct": false,
          "model_raw_output": "전체 추론 과정..."
        },
        "kr_eng_reasoning": {
          "predicted_answer": "예측답안",
          "is_correct": false,
          "model_raw_output": "전체 추론 과정..."
        }
      }
    }
  ]
}
```

## 🎯 최종 분석 가능한 인사이트

### 1. 언어별 성능 차이
- 영어 vs 한국어 입력 시 성능 차이 정량화
- 모델별 언어 편향성 분석

### 2. 추론 언어의 영향
- 한국어 질문 + 영어 추론 vs 한국어 질문 + 한국어 추론
- 추론 언어 변경의 효과성 측정

### 3. 언어 장벽 문제 식별
- 입력 언어가 문제 이해에 미치는 영향
- 번역/언어 이해 단계에서의 정보 손실

### 4. 모델별 특성 비교
- 각 모델의 다국어 처리 능력 차이
- 추론 과정에서의 언어별 패턴 분석

## 🔧 다른 데이터셋 적용 시 수정 사항

### 1. 파일 경로 변경
각 파이썬 스크립트에서 다음 경로들을 새 데이터셋에 맞게 수정:
```python
basemodel_results_dir = Path("새로운/데이터셋/경로/basemodel_results")
basemodel_eng_reasoning_dir = Path("새로운/데이터셋/경로/basemodel_eng_reasoning")
```

### 2. 데이터셋 이름 변경
데이터 구조에서 `"ARC"`, `"Ko-ARC"` 키를 새로운 데이터셋의 키로 변경:
```python
arc_data = data.get('datasets', {}).get('새로운_영어_데이터셋명', {}).get('details', [])
ko_arc_data = data.get('datasets', {}).get('새로운_한국어_데이터셋명', {}).get('details', [])
```

### 3. 모델 이름 패턴 수정
파일명에서 모델명을 추출하는 로직 조정:
```python
model_name = json_file.stem.replace('results_', '').replace('_3shot', '')
# 새로운 파일명 패턴에 맞게 수정
```

### 4. 출력 파일명 변경
분석 결과 파일명을 새 데이터셋에 맞게 수정:
```python
output_file = f"새로운데이터셋_eng_correct-kr_incorrect_{model}.json"
```

## 📝 실행 순서

1. **데이터 준비**: basemodel_results와 basemodel_eng_reasoning 폴더에 결과 파일 배치
2. **1단계 실행**: `python find_eng_correct_kr_incorrect.py`
3. **2단계 실행**: `python extract_eng_reasoning_results.py`
4. **3단계 실행**: `python find_common_unsolvable_ids.py`
5. **4단계 실행**: `python create_comprehensive_analysis.py`
6. **결과 확인**: 생성된 JSON 파일들을 통해 분석 수행

각 단계는 이전 단계의 출력 파일을 입력으로 사용하므로 순차적으로 실행해야 합니다.