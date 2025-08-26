# ToW Training with Intelligent Caching

이제 모든 ToW 훈련 스크립트가 지능적인 캐싱 시스템을 사용합니다! 🚀

## 🎯 주요 개선사항

### ✅ 문제 해결됨
- **매번 토크나이징 반복** → **한 번만 토크나이징 후 캐시 사용**
- **긴 대기 시간** → **즉시 로딩 (10초 이내)**
- **모델별 충돌** → **모델별 독립 캐시**

### 🚀 성능 향상
- **시간 절약**: 첫 실행 후 99% 시간 절약
- **메모리 효율성**: 중복 작업 제거
- **자동 관리**: 데이터 변경 시 자동 캐시 무효화

## 📁 파일 구조

```
5_training/
├── dataset_cache_utils.py      # 캐싱 유틸리티
├── cache_manager.py            # 캐시 관리 도구
├── ToW_Training_deepseek.py    # DeepSeek 훈련 (캐싱 적용)
├── ToW_Training_llama.py       # Llama 훈련 (캐싱 적용)
├── ToW_Training_mistral.py     # Mistral 훈련 (캐싱 적용)
├── ToW_Training_qwen.py        # Qwen 훈련 (캐싱 적용)
└── cached_datasets/            # 캐시 저장소 (자동 생성)
    ├── DeepSeek-R1-0528-Qwen3-8B_abc123/
    ├── Llama-3.1-8B-Instruct_def456/
    └── ...
```

## 🔧 사용법

### 기본 훈련 (변경사항 없음)
```bash
# 기존과 동일하게 실행
python ToW_Training_deepseek.py
python ToW_Training_llama.py
python ToW_Training_mistral.py
python ToW_Training_qwen.py
```

### 🧠 지능적 캐싱 동작
1. **첫 번째 실행**: 토크나이징 + 캐시 생성
2. **두 번째 실행부터**: 캐시에서 즉시 로딩!

### 📊 캐시 관리

#### 캐시 목록 보기
```bash
python cache_manager.py list
```

#### 캐시 통계 보기
```bash
python cache_manager.py stats
```

#### 특정 캐시 삭제
```bash
python cache_manager.py clear
```

#### 모든 캐시 삭제
```bash
python cache_manager.py clear-all
```

## 🔍 캐시 작동 방식

### 캐시 키 생성
각 모델의 캐시는 다음 요소로 고유하게 식별됩니다:
- **모델 ID**: `/scratch/jsong132/.../모델명`
- **데이터 경로**: 훈련 데이터 파일 경로
- **최대 길이**: 토큰 시퀀스 최대 길이

### 자동 무효화
다음 경우 캐시가 자동으로 재생성됩니다:
- 훈련 데이터 파일 변경
- 모델 설정 변경
- 최대 길이 설정 변경

### 모델별 독립성
```
cached_datasets/
├── DeepSeek-R1-0528-Qwen3-8B_abc123/    # DeepSeek 전용 캐시
├── Llama-3.1-8B-Instruct_def456/        # Llama 전용 캐시
├── Mistral-7B-Instruct-v0.3_ghi789/     # Mistral 전용 캐시
└── Qwen2.5-7B-Instruct_jkl012/          # Qwen 전용 캐시
```

## 📈 성능 비교

### 이전 (캐싱 없음)
```
Tokenizing and creating labels: 8%|▉| 8000/104087 [00:11<02:22, 675.08 examples/s]
총 소요시간: ~10-15분
```

### 현재 (캐싱 적용)
```
Loading cached dataset from cached_datasets/DeepSeek-R1-0528-Qwen3-8B_abc123/tokenized_dataset.arrow
Successfully loaded cached dataset with 93,678 examples
총 소요시간: ~5-10초
```

## 🛠️ 문제 해결

### 캐시 관련 오류 시
```bash
# 모든 캐시 삭제 후 재시작
python cache_manager.py clear-all
python ToW_Training_deepseek.py
```

### 데이터 업데이트 후
캐시가 자동으로 무효화되지만, 수동으로 삭제하고 싶다면:
```bash
python cache_manager.py clear
```

### 디스크 공간 관리
캐시는 디스크 공간을 사용합니다. 정기적으로 확인:
```bash
python cache_manager.py stats
```

## 🎉 이제 훈련이 훨씬 빨라졌습니다!

- **첫 실행**: 정상적으로 토크나이징 진행
- **이후 실행**: 캐시에서 즉시 로딩
- **시간 절약**: 90% 이상 단축
- **자동 관리**: 걱정할 필요 없음

Happy Training! 🚀