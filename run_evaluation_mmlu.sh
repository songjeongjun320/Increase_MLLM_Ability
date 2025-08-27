#!/bin/bash

export TRANSFORMERS_VERBOSITY=debug

MODEL_NAMES=(
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "google_gemma-3-4b-it"
    "Qwen2.5-3B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "DeepSeek-R1-Distill-Qwen-1.5B-ToW"
    # "google_gemma-3-4b-it-ToW"
    # "Qwen2.5-3B-Instruct-ToW"
    # "Llama-3.2-3B-Instruct-ToW"
)

MODEL_PATHS=(
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
)

ADAPTER_PATHS=(
    ""
    ""
    ""
    ""
    # "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-Distill-Qwen-1.5B-ToW"
    # "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/google_gemma-3-4b-it-ToW"
    # "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-3B-Instruct-ToW"
    # "/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.2-3B-Instruct-ToW"
)

# 기본 태스크 설정
TASKS=("mmlu" "KO_KR")
RESULTS_DIR="./evaluation_results_mmlu"
mkdir -p $RESULTS_DIR
NUM_FEWSHOT=5

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_TASKS=${#TASKS[@]}
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))
CURRENT_RUN=0

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

        # --log_samples 비활성화: 모든 샘플을 저장하지 않음
        accelerate launch -m lm_eval \
            --model hf \
            --model_args $MODEL_ARGS \
            --tasks $TASK \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size auto \
            --output_path $OUTPUT_FILE \
            --verbosity INFO
        
        echo "평가 완료 (진행: $CURRENT_RUN / $TOTAL_RUNS): 결과가 $OUTPUT_FILE 에 저장되었습니다."
        echo ""

        # 오류 케이스만 별도로 저장하는 로직 추가
        if [ -f "$OUTPUT_FILE" ]; then
            ACCURACY=$(python3 -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    # 태스크 키 찾기 (정확한 매치 또는 유사한 키)
    task_key = None
    if 'results' in data:
        if '$TASK' in data['results']:
            task_key = '$TASK'
        else:
            # 유사한 키 찾기
            for key in data['results'].keys():
                if '$TASK' in key or key in '$TASK':
                    task_key = key
                    break
    
    if task_key and 'results' in data:
        task_results = data['results'][task_key]
        if 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.4f}')
        elif 'accuracy' in task_results:
            print(f'{task_results[\"accuracy\"]:.4f}')
        else:
            # 첫 번째 숫자 값 찾기
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and key not in ['alias', 'num_fewshot']:
                    print(f'{value:.4f}')
                    break
            else:
                print('null')
    else:
        print('null')
except Exception as e:
    print(f'Error: {e}')
    print('null')
            ")

            if [ "$ACCURACY" == "null" ]; then
                echo "오류 케이스 발견: 정확도 데이터 없음"
                # 오류 케이스만 별도 파일에 저장
                ERROR_FILE="$RESULTS_DIR/error_samples.json"
                echo "{" >> $ERROR_FILE
                echo '  "model": "'$NAME'",' >> $ERROR_FILE
                echo '  "task": "'$TASK'",' >> $ERROR_FILE
                echo '  "accuracy": "null"' >> $ERROR_FILE
                echo "}" >> $ERROR_FILE
            fi
        fi
    done
done

echo "모든 MMLU 평가가 완료되었습니다."

# 결과 요약 생성
echo "===================================================="
echo "결과 요약을 생성하는 중..."
echo "===================================================="

SUMMARY_FILE="$RESULTS_DIR/mmlu_summary.json"

echo "{" > $SUMMARY_FILE
echo '  "evaluation_summary": {' >> $SUMMARY_FILE
echo '    "total_models": '$NUM_MODELS',' >> $SUMMARY_FILE
echo '    "total_tasks": '$NUM_TASKS',' >> $SUMMARY_FILE
echo '    "fewshot": '$NUM_FEWSHOT',' >> $SUMMARY_FILE
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
        
        if [ -f "$RESULT_FILE" ]; then
            ACCURACY=$(python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print('N/A')
        exit()
    
    task_key = None
    if '$TASK' in data['results']:
        task_key = '$TASK'
    
    if task_key:
        task_results = data['results'][task_key]
        if 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.3f}')
        elif 'accuracy' in task_results:
            print(f'{task_results[\"accuracy\"]:.3f}')
        else:
            print('N/A')
except Exception as e:
    print('N/A')
            ")
            echo -n '        "'$TASK'": '$ACCURACY >> $SUMMARY_FILE
        fi
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

# 최종 요약 출력
echo "===================================================="
echo "MMLU 평가 최종 요약:"
echo "- 평가된 모델 수: $NUM_MODELS"
echo "- 사용된 태스크: ${TASKS[*]}" 
echo "- Few-shot 설정: $NUM_FEWSHOT"
echo "- 생성된 파일:"
echo "  JSON 요약: $SUMMARY_FILE"
echo "  오류 케이스 파일: $RESULTS_DIR/error_samples.json"
echo "===================================================="
