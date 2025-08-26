#!/bin/bash

# --- 설정 부분 ---
export TRANSFORMERS_VERBOSITY=debug

MODEL_NAMES=(
    "Qwen2.5-7B-Instruct"
    "Mistral-8B-Instruct-2410"
    "Llama-3.1-8B-Instruct"
    "DeepSeek-R1-0528-Qwen3-8B"
    "Qwen2.5-7B-Instruct-ToW"
    "Mistral-8B-Instruct-2410-ToW"
    "Llama-3.1-8B-Instruct-ToW"
    "DeepSeek-R1-0528-Qwen3-8B-ToW"
)
MODEL_PATHS=(
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B"
)
ADAPTER_PATHS=(
    "" "" "" ""
    "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-7B-Instruct-ToW"
    "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Mistral-8B-Instruct-2410-ToW"
    "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.1-8B-Instruct-ToW"
    "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-0528-Qwen3-8B-ToW"
)

# Hugging Face 데이터셋 사용으로 변경
TASKS=("gsm8k_en_hf" "gsm8k_ko_hf")
RESULTS_DIR="./evaluation_results_gsm8k_hf"
mkdir -p $RESULTS_DIR
NUM_FEWSHOT=8

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_TASKS=${#TASKS[@]}
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))
CURRENT_RUN=0

# --- 실행 부분 ---

for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    MODEL_PATH=${MODEL_PATHS[$i]}
    ADAPTER_PATH=${ADAPTER_PATHS[$i]}

    MODEL_ARGS="pretrained=$MODEL_PATH"
    if [ -n "$ADAPTER_PATH" ]; then
        MODEL_ARGS+=",peft=$ADAPTER_PATH,tokenizer=$ADAPTER_PATH"
    fi

    for TASK in "${TASKS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "===================================================="
        echo "평가 시작 (진행: $CURRENT_RUN / $TOTAL_RUNS): 모델 [$NAME] on 태스크 [$TASK]"
        echo "===================================================="

        OUTPUT_FILE="$RESULTS_DIR/${NAME}_${TASK}.json"

        accelerate launch -m lm_eval \
            --model hf \
            --model_args $MODEL_ARGS \
            --tasks $TASK \
            --include_path ./eval_configs \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size auto \
            --output_path $OUTPUT_FILE \
            --log_samples \
            --verbosity INFO \
            --write_out \
            --show_config
        
        echo "평가 완료 (진행: $CURRENT_RUN / $TOTAL_RUNS): 결과가 $OUTPUT_FILE 에 저장되었습니다."
        echo ""
    done
done

echo "모든 GSM8K 평가가 완료되었습니다."

# 결과 요약 생성
echo "===================================================="
echo "결과 요약을 생성하는 중..."
echo "===================================================="

SUMMARY_FILE="$RESULTS_DIR/gsm8k_summary_hf.json"

# JSON 시작
echo "{" > $SUMMARY_FILE
echo '  "evaluation_summary": {' >> $SUMMARY_FILE
echo '    "total_models": '$NUM_MODELS',' >> $SUMMARY_FILE
echo '    "total_tasks": '$NUM_TASKS',' >> $SUMMARY_FILE
echo '    "fewshot": '$NUM_FEWSHOT',' >> $SUMMARY_FILE
echo '    "dataset_source": "Hugging Face (gsm8k + kuotient/gsm8k-ko)",' >> $SUMMARY_FILE
echo '    "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",' >> $SUMMARY_FILE
echo '    "results": {' >> $SUMMARY_FILE

model_count=0
for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    
    if [ $model_count -gt 0 ]; then
        echo ',' >> $SUMMARY_FILE
    fi
    echo '      "'$NAME'": {' >> $SUMMARY_FILE
    
    task_count=0
    for TASK in "${TASKS[@]}"; do
        RESULT_FILE="$RESULTS_DIR/${NAME}_${TASK}.json"
        
        if [ $task_count -gt 0 ]; then
            echo ',' >> $SUMMARY_FILE
        fi
        
        # JSON 파일에서 정확도 추출 (여러 메트릭 확인)
        if [ -f "$RESULT_FILE" ]; then
            ACCURACY=$(python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    if 'results' in data and '$TASK' in data['results']:
        task_results = data['results']['$TASK']
        # 우선순위: exact_match > acc > accuracy > 첫 번째 숫자 메트릭
        if 'exact_match' in task_results:
            print(f'{task_results[\"exact_match\"]:.4f}')
        elif 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.4f}')
        elif 'accuracy' in task_results:
            print(f'{task_results[\"accuracy\"]:.4f}')
        else:
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and key not in ['alias', 'stderr']:
                    print(f'{value:.4f}')
                    break
            else:
                print('null')
    else:
        print('null')
except Exception as e:
    print('null')
            ")
        else
            ACCURACY="null"
        fi
        
        echo -n '        "'$TASK'": '$ACCURACY >> $SUMMARY_FILE
        task_count=$((task_count + 1))
    done
    
    echo '' >> $SUMMARY_FILE
    echo '      }' >> $SUMMARY_FILE
    model_count=$((model_count + 1))
done

echo '    }' >> $SUMMARY_FILE
echo '  }' >> $SUMMARY_FILE
echo '}' >> $SUMMARY_FILE

echo "결과 요약이 $SUMMARY_FILE 에 저장되었습니다."

# 간단한 텍스트 요약도 생성
TEXT_SUMMARY="$RESULTS_DIR/gsm8k_summary_hf.txt"
echo "GSM8K 평가 결과 요약 (Hugging Face 데이터셋)" > $TEXT_SUMMARY
echo "생성 시간: $(date)" >> $TEXT_SUMMARY
echo "Few-shot: $NUM_FEWSHOT" >> $TEXT_SUMMARY
echo "데이터셋: gsm8k (영어) + kuotient/gsm8k-ko (한국어)" >> $TEXT_SUMMARY
echo "===========================================" >> $TEXT_SUMMARY
echo "" >> $TEXT_SUMMARY

printf "%-35s" "모델명" >> $TEXT_SUMMARY
for TASK in "${TASKS[@]}"; do
    if [[ $TASK == *"en"* ]]; then
        printf "%-15s" "영어(GSM8K)" >> $TEXT_SUMMARY
    else
        printf "%-15s" "한국어(GSM8K)" >> $TEXT_SUMMARY
    fi
done
echo "" >> $TEXT_SUMMARY

for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    printf "%-35s" "$NAME" >> $TEXT_SUMMARY
    
    for TASK in "${TASKS[@]}"; do
        RESULT_FILE="$RESULTS_DIR/${NAME}_${TASK}.json"
        if [ -f "$RESULT_FILE" ]; then
            ACCURACY=$(python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    if 'results' in data and '$TASK' in data['results']:
        task_results = data['results']['$TASK']
        if 'exact_match' in task_results:
            print(f'{task_results[\"exact_match\"]:.3f}')
        elif 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.3f}')
        elif 'accuracy' in task_results:
            print(f'{task_results[\"accuracy\"]:.3f}')
        else:
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and key not in ['alias', 'stderr']:
                    print(f'{value:.3f}')
                    break
            else:
                print('N/A')
    else:
        print('N/A')
except Exception as e:
    print('ERR')
            ")
        else
            ACCURACY="N/A"
        fi
        printf "%-15s" "$ACCURACY" >> $TEXT_SUMMARY
    done
    echo "" >> $TEXT_SUMMARY
done

echo ""
echo "텍스트 요약이 $TEXT_SUMMARY 에 저장되었습니다."
echo ""
echo "==== 사용된 설정 정보 ===="
echo "영어 데이터셋: gsm8k (Hugging Face 공식)"
echo "한국어 데이터셋: kuotient/gsm8k-ko"
echo "Few-shot 예제: $NUM_FEWSHOT 개"
echo "총 모델 수: $NUM_MODELS"
echo "총 태스크 수: $NUM_TASKS"