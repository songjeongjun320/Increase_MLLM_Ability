# KLUE 벤치마크 평가 실행 가이드

이 디렉토리는 완전히 개선된 KLUE (Korean Language Understanding Evaluation) 벤치마크 평가를 위한 파일들을 포함합니다.

## 📁 파일 구조

```
klue_evaluation/
├── tc.yaml              # Topic Classification 설정
├── sts.yaml             # Semantic Textual Similarity 설정  
├── nli.yaml             # Natural Language Inference 설정
├── re.yaml              # Relation Extraction 설정 (새로 완성)
├── dp.yaml              # Dependency Parsing 설정 (수정됨)
├── mrc.yaml             # Machine Reading Comprehension 설정 (수정됨)
├── dst.yaml             # Dialogue State Tracking 설정 (수정됨)
├── model_configs.yaml   # 모델 설정 파일
├── run_klue_evaluation.py           # 🚀 메인 실행 스크립트
├── klue_data_preprocessor.py        # 데이터 전처리 유틸리티
├── validate_klue_config.py          # 설정 검증 스크립트
└── README.md           # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install lm-eval transformers datasets torch pyyaml

# accelerate 설정 (GPU 사용 시)
accelerate config
```

### 2. 모델 경로 설정
`model_configs.yaml` 파일을 수정해서 실제 모델 경로를 입력하세요:

```yaml
models:
  - name: "your-model-name"
    path: "/path/to/your/model"
    adapter: ""  # LoRA 어댑터 경로 (선택사항)
```

### 3. 전체 평가 실행
```bash
# 모든 모델에 대해 모든 KLUE 태스크 평가
python run_klue_evaluation.py

# 결과 저장 디렉토리 지정
python run_klue_evaluation.py --results_dir ./my_results
```

### 4. 결과 확인
평가 완료 후 `klue_evaluation_results/` 디렉토리에서 결과를 확인할 수 있습니다:
- `모델명_태스크명.json`: 개별 결과 파일
- `klue_evaluation_summary_YYYYMMDD_HHMMSS.json`: 전체 결과 요약

## 🔧 고급 사용법

### 개별 태스크만 실행
```bash
# 단일 태스크만 평가하고 싶은 경우
python -m lm_eval \
    --model hf \
    --model_args pretrained=/path/to/your/model \
    --tasks tc \
    --num_fewshot 3 \
    --batch_size auto \
    --output_path ./tc_results.json
```

### 설정 검증
```bash
# 평가 전에 설정이 올바른지 확인
python validate_klue_config.py
```

### 모델 설정 템플릿 생성
```bash
# 새로운 model_configs.yaml 템플릿 생성
python run_klue_evaluation.py --create_template
```

## 📊 평가 태스크 상세

| 태스크 | 설명 | Few-shot | 메트릭 | 예상 시간 |
|--------|------|----------|--------|-----------|
| **TC** | 주제 분류 (뉴스 제목 → 7개 카테고리) | 3 | Accuracy | ~10분 |
| **STS** | 의미 유사성 (문장 쌍 → 0-5점) | 3 | Pearson r | ~15분 |
| **NLI** | 자연어 추론 (전제-가설 → 함의/모순/중립) | 3 | Accuracy | ~15분 |
| **RE** | 관계 추출 (문장+개체 → 30개 관계) | 2 | macro F1 | ~20분 |
| **DP** | 구문 분석 (문장 → head 인덱스) | 1 | Exact Match | ~30분 |
| **MRC** | 기계 독해 (지문+질문 → 답변) | 2 | EM, F1 | ~25분 |
| **DST** | 대화 상태 추적 (대화 → 슬롯-값) | 1 | Exact Match | ~20분 |

**총 예상 시간: 모델당 ~2-3시간** (GPU 성능에 따라 차이)

## ⚡ 성능 최적화

### GPU 메모리 최적화
```bash
# 배치 크기 자동 조정
--batch_size auto

# 수동 배치 크기 설정 (메모리 부족 시)
--batch_size 4
```

### 병렬 처리
```bash
# accelerate로 멀티 GPU 사용
accelerate launch -m lm_eval --model hf --model_args pretrained=/path/to/model --tasks tc,sts,nli
```

## 📋 체크리스트

평가 실행 전에 다음을 확인하세요:

- [ ] 모든 YAML 설정 파일이 같은 디렉토리에 있음
- [ ] `model_configs.yaml`에 올바른 모델 경로 설정
- [ ] 충분한 디스크 공간 (결과 파일용)
- [ ] GPU 메모리 충분함 (최소 8GB 권장)
- [ ] 인터넷 연결 (KLUE 데이터셋 다운로드용)

## 🐛 문제 해결

### 일반적인 오류

**1. 모델 로딩 실패**
```
❌ 모델 경로 없음: /path/to/model
```
→ `model_configs.yaml`에서 올바른 경로 확인

**2. KLUE 데이터셋 로딩 실패**
```
❌ datasets.exceptions.DatasetNotFoundError
```
→ 인터넷 연결 확인, `datasets` 라이브러리 최신 버전 설치

**3. GPU 메모리 부족**
```
torch.cuda.OutOfMemoryError
```
→ `--batch_size 1` 또는 더 작은 값 사용

**4. 특정 태스크 실패**
- **DP (Dependency Parsing)**: 가장 복잡한 태스크, 실패 가능성 높음
- **DST (Dialogue State Tracking)**: 복잡한 대화 구조로 인한 파싱 오류 가능

### 로그 확인
```bash
# 자세한 로그 출력
python run_klue_evaluation.py --verbosity DEBUG
```

## 📈 결과 해석

### 태스크별 성능 기준
- **TC**: 85%+ (우수), 80%+ (보통)
- **STS**: 0.85+ (우수), 0.80+ (보통)  
- **NLI**: 80%+ (우수), 75%+ (보통)
- **RE**: 70%+ (우수), 65%+ (보통