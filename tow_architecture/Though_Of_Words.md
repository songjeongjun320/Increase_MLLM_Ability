# Thoughts of Words (TOW) 논문 요약

## 1. TOW(Thoughts of Words) 예시 및 역할

**TOW (Thoughts of Words)**는 언어 모델이 예측하는 각 단어에 대해 그 예측이 어떻게 이루어졌는지에 대한 **정교한 설명**을 추가하는 데이터 증강 기법입니다. 이 기법은 기본적으로, 각 단어에 대해 **"다음 단어가 어떻게 예측되는지에 대한 이유"**를 모델에 주입하여 모델이 보다 논리적이고 정확한 예측을 하도록 학습시킵니다.

### 예시
- **일반적인 문장**:
  - "Bruce Lee met with the Kung-fu movie director at noon regarding his martial arts education."
  
- **TOW가 적용된 문장**:
  - "Bruce Lee met with the Kung-fu movie director at noon regarding his martial arts education; the lunch lasted <ToW>soft-consistency: A temporal value with likely ranges to be between a few minutes to 2-3 hours</ToW>."

이 예시에서, "<ToW>" 태그로 표시된 부분은 모델이 'lunch lasted'라는 구절과 어떻게 연관될 수 있는지를 설명하는 **생각**(thought)입니다. 이 **생각**은 단순히 "lunch lasted 2 hours"를 예측하는 것이 아니라, "lunch가 2시간 동안 지속될 것"이라는 시간적 맥락과 함께 **추론적 설명**을 제공하는 것입니다.

### Option 2 원칙: ToW 내부는 항상 영어
- 모든 소스 언어(한국어/중국어/일본어/영어 등)에서 생성하더라도, `<ToW>...</ToW>` 안의 **생각 텍스트는 반드시 영어**로 강제됩니다.
- 형식 강제 규칙:
  - 토큰 형식은 반드시 `<ToW>English reasoning...</ToW>`를 따릅니다.
  - ToW 내부 텍스트는 비-ASCII 문자를 제거하고, 공백을 정규화하며, 길이가 과도하면 200자 이내로 절단합니다.
  - 내부 텍스트가 비어 있거나 영어가 아닐 경우 기본 문구로 대체합니다(예: "Contextual reasoning applied.").

## 2. 데이터베이스 및 TOW 토큰 작용

**TOW 데이터베이스**는 모델의 **사전 훈련 데이터**에 대한 확장입니다. TOW는 문장에서 예측되는 단어들에 대해 **정확한 이유를 설명하는 텍스트**를 생성하고, 이를 데이터베이스의 각 문맥에 주입합니다. TOW는 **토큰화된 문장**의 각 단어에 대해, 그 단어가 예측되는 이유에 대한 **세부 설명을 토큰**으로 추가합니다.

### TOW 토큰화 및 데이터베이스 구성
- 각 단어의 예측 이유를 설명하는 **TOW 토큰**은 **원본 문장의 문맥**을 기반으로 생성됩니다.
- **TOW 토큰**은 크게 4가지 범주로 나뉩니다:
  1. **Trivial (사소한 단어)**: 너무 자주 나타나거나 특별한 의미를 가지지 않는 단어.
  2. **Exact Match (정확히 일치하는 예측)**: 모델이 예측한 단어가 실제 다음 단어와 일치하는 경우.
  3. **Soft Consistent (부분적으로 일치)**: 모델이 예측한 단어가 정확하게 일치하지 않지만 문맥상 논리적으로 유사한 경우.
  4. **Unpredictable (예측 불가능)**: 모델이 예측할 수 없는 단어로, 이를 통해 모델이 잘못된 예측을 방지할 수 있습니다.

### TOW 토큰 작용
- **언어 모델 훈련에의 기여**:
  - TOW는 모델에게 각 예측이 문맥 내에서 **어떻게 합리적으로 도출되었는지**에 대해 훈련시킵니다. 예를 들어, 모델이 'Bruce Lee'라는 주제에서 'Kung-fu movie director'라는 단어를 예측했을 때, 'lunch'라는 단어가 올 가능성에 대한 **시간적 맥락과 설명**을 제공하여, 모델이 더 합리적이고 정확한 예측을 할 수 있도록 돕습니다.
  
- **생각의 토큰화 및 효과**:
  - 각 **생각 토큰**은 **문맥적 관련성**을 강조하며, 다음 단어를 예측하는 데 있어서 단순히 빈번히 등장하는 단어들이 아니라 **문맥에 맞는 단어를 선택하도록** 합니다. TOW는 모델이 보다 **정교하고 체계적인 추론**을 할 수 있도록 도와줍니다.

### TOW의 작용 예시
1. **생각의 생성**:
   - "Bruce Lee met with the Kung-fu movie director at noon regarding his martial arts education; the lunch lasted <ToW>soft-consistency: likely a temporal value with likely ranges to be between a few minutes to 2-3 hours</ToW>."
   - 이 부분에서 TOW는 'lunch'가 2시간 지속된다고 예측하는 이유에 대해 설명합니다.
   
2. **생각의 분류**:
   - 생성된 생각을 **"exact match"** 또는 **"soft consistent"**로 분류하여, 모델이 예측할 때 이 정보들을 참고할 수 있도록 합니다.

### TOW 토큰의 역할
- **언어 모델 훈련에의 기여**:
  - TOW는 모델에게 각 예측이 문맥 내에서 **어떻게 합리적으로 도출되었는지**에 대해 훈련시킵니다. 예를 들어, 모델이 'Bruce Lee'라는 주제에서 'Kung-fu movie director'라는 단어를 예측했을 때, 'lunch'라는 단어가 올 가능성에 대한 **시간적 맥락과 설명**을 제공하여, 모델이 더 합리적이고 정확한 예측을 할 수 있도록 돕습니다.
  
- **생각의 토큰화 및 효과**:
  - 각 **생각 토큰**은 **문맥적 관련성**을 강조하며, 다음 단어를 예측하는 데 있어서 단순히 빈번히 등장하는 단어들이 아니라 **문맥에 맞는 단어를 선택하도록** 합니다. TOW는 모델이 보다 **정교하고 체계적인 추론**을 할 수 있도록 도와줍니다.

## 3. 실험 결과 및 효과

TOW는 다양한 **추론 데이터셋**에서 모델 성능을 크게 향상시킵니다. 예를 들어, **GSM8K**, **CSQA**, **StrategyQA** 등에서 모델의 **정확도**가 7%에서 9%까지 향상되었습니다. 또한, **hallucination**(허위 정보 생성) 문제를 **10%까지 감소**시킬 수 있었습니다.

- TOW는 **정확한 추론을 강화**하고, 모델이 **잘못된 정보**를 생성하지 않도록 도와주는 중요한 역할을 합니다.

## 4. 구현 노트 (Option 2 적용 사항)
- 데이터 증강 파이프라인(`tow_architecture/data_augmentation/pipeline.py`): ToW 생성 시 영어 강제 및 포맷 위생 처리 적용.
- 교차언어 TOW(`tow_architecture/core/cross_lingual_tow.py`): 모든 ToW를 영어로 생성하고 최종 위생 처리.
- 사고 토큰 프로세서(`tow_architecture/core/thought_processor.py`): ToW 포맷터에서 영어 강제 및 포맷 위생 처리.
- 유틸(`tow_architecture/utils/text_utils.py`): `enforce_english_text`, `sanitize_tow_token`, `validate_tow_token_format` 추가.
- 데이터셋 생성 스크립트(`DB/option2_tow_generator.py`): 최종 저장 전에 ToW 토큰을 재검증/정규화.

