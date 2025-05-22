import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from konlpy.tag import Mecab # 또는 Okt, Kkma 등
from bert_score import score as bert_scorer # score 함수 이름 충돌 방지
import torch
import nltk

# NLTK 'punkt' 리소스 다운로드 (최초 실행 시 필요할 수 있음)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 설정 ---
ENG_JSON_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/C4/c4_200_eng.json"
KR_JSON_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/C4/c4_200_kr.json"
JSON_KEY = "c4_first_200_sentences"

# 한국어 형태소 분석기 초기화
# Mecab을 사용할 경우:
try:
    korean_tokenizer = Mecab()
    print("Mecab 사용 중")
except Exception as e:
    print(f"Mecab 로드 실패: {e}. Okt로 대체합니다.")
    from konlpy.tag import Okt
    korean_tokenizer = Okt()
    print("Okt 사용 중")

# --- 함수 정의 ---

def load_sentences_from_json(filepath, key):
    """JSON 파일에서 문장 리스트를 로드합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(key, [])
    except FileNotFoundError:
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"오류: {filepath} 파일이 올바른 JSON 형식이 아닙니다.")
        return []
    except Exception as e:
        print(f"파일 로드 중 오류 발생 ({filepath}): {e}")
        return []

def tokenize_korean(text):
    """한국어 텍스트를 토큰화합니다 (형태소 분석)."""
    if isinstance(korean_tokenizer, Mecab):
        return korean_tokenizer.morphs(text)
    elif 'Okt' in str(type(korean_tokenizer)):
        return korean_tokenizer.morphs(text)
    else:
        return text.split()


def calculate_bleu(reference_tokens, candidate_tokens):
    """BLEU 점수를 계산합니다."""
    chencherry = SmoothingFunction()
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=chencherry.method1, weights=(0.25, 0.25, 0.25, 0.25))


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    eng_sentences = load_sentences_from_json(ENG_JSON_PATH, JSON_KEY)
    kr_sentences = load_sentences_from_json(KR_JSON_PATH, JSON_KEY)

    if not eng_sentences or not kr_sentences:
        print("문장 데이터를 로드하지 못했습니다. 프로그램을 종료합니다.")
        exit()

    if len(eng_sentences) != len(kr_sentences):
        print("경고: 영어 문장과 한국어 문장의 수가 다릅니다. 짧은 쪽 길이에 맞춰 비교합니다.")
        min_len = min(len(eng_sentences), len(kr_sentences))
        eng_sentences = eng_sentences[:min_len]
        kr_sentences = kr_sentences[:min_len]

    num_sentences = len(eng_sentences) # 비교할 문장 쌍의 수
    print(f"총 {num_sentences}개의 문장 쌍을 비교합니다.\n")

    print("BERTScore 계산 중... (시간이 다소 소요될 수 있습니다)")
    P, R, F1_bert = bert_scorer(eng_sentences, kr_sentences, lang="multi", verbose=False, model_type="bert-base-multilingual-cased")

    individual_results = []
    total_bleu_score = 0.0
    total_bert_f1_score = 0.0

    for i in range(num_sentences):
        eng_sent = eng_sentences[i]
        kr_sent = kr_sentences[i]

        eng_tokens = word_tokenize(eng_sent.lower())
        kr_tokens = tokenize_korean(kr_sent)

        bleu_score = calculate_bleu(kr_tokens, eng_tokens)
        bert_f1_score = F1_bert[i].item()

        total_bleu_score += bleu_score
        total_bert_f1_score += bert_f1_score

        individual_results.append({
            "index": i + 1,
            "english_sentence": eng_sent,
            "korean_sentence": kr_sent,
            "bleu_score": round(bleu_score, 4),
            "bert_f1_score": round(bert_f1_score, 4)
        })

        if (i + 1) % 10 == 0 or i == num_sentences - 1:
            print(f"\n--- 비교 결과 {i+1} ---")
            print(f"English: {eng_sent}")
            print(f"Korean: {kr_sent}")
            print(f"BLEU Score (KR as ref, EN as hyp): {bleu_score:.4f}")
            print(f"BERTScore F1: {bert_f1_score:.4f}")

    # 평균 계산
    avg_bleu_score = total_bleu_score / num_sentences if num_sentences > 0 else 0.0
    avg_bert_f1_score = total_bert_f1_score / num_sentences if num_sentences > 0 else 0.0

    print(f"\n--- 전체 평균 점수 ---")
    print(f"평균 BLEU Score: {avg_bleu_score:.4f}")
    print(f"평균 BERTScore F1: {avg_bert_f1_score:.4f}")

    # 전체 결과를 JSON 파일로 저장 (개별 결과와 평균 점수 포함)
    output_filename = "comparison_results_with_average.json" # 파일 이름 변경
    
    final_output_data = {
        "average_scores": {
            "average_bleu_score": round(avg_bleu_score, 4),
            "average_bert_f1_score": round(avg_bert_f1_score, 4)
        },
        "individual_results": individual_results
    }

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(final_output_data, outfile, ensure_ascii=False, indent=4)
    print(f"\n전체 비교 결과 (평균 포함)가 {output_filename} 에 저장되었습니다.")