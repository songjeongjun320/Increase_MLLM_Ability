#!/bin/bash

export TRANSFORMERS_VERBOSITY=debug

# 평가할 모델들의 정보를 배열로 정의
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
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct"  # 수정됨
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

# AI2-ARC와 Ko-ARC 데이터셋 사용
TASKS=("arc_challenge" "arc_ko")  # 커스텀 Ko-ARC 태스크 사용

# 커스텀 태스크 설정 경로
CUSTOM_TASK_PATH="./eval_configs"

RESULTS_DIR="./evaluation_results_arc"
mkdir -p $RESULTS_DIR
NUM_FEWSHOT=5  # 5-shot으로 설정

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_TASKS=${#TASKS[@]}
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))
CURRENT_RUN=0

# --- 실행 부분 ---
echo "ARC 평가 시작 (AI2-ARC + Ko-ARC)"
echo "데이터셋: allenai/ai2_arc + 로컬 Ko-ARC"
echo "커스텀 태스크 경로: $CUSTOM_TASK_PATH"
echo "모델 개수: $NUM_MODELS"
echo "태스크 개수: $NUM_TASKS" 
echo "총 실행 횟수: $TOTAL_RUNS"
echo "Few-shot: $NUM_FEWSHOT"
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    MODEL_PATH=${MODEL_PATHS[$i]}
    ADAPTER_PATH=${ADAPTER_PATHS[$i]}

    echo "모델 정보 확인:"
    echo "  이름: $NAME"
    echo "  경로: $MODEL_PATH"
    echo "  어댑터: $ADAPTER_PATH"

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

        # 태스크별로 다른 실행 방식 사용
        if [[ $TASK == "arc_ko" ]]; then
            echo "Ko-ARC 커스텀 태스크 실행 (로컬 Ko-ARC 데이터셋 사용)"
            # 커스텀 Ko-ARC 태스크 실행 시 include_path 옵션 추가
            accelerate launch -m lm_eval \
                --model hf \
                --model_args $MODEL_ARGS \
                --tasks $TASK \
                --include_path $CUSTOM_TASK_PATH \
                --num_fewshot $NUM_FEWSHOT \
                --batch_size auto \
                --output_path $OUTPUT_FILE \
                --verbosity INFO
        else
            echo "AI2-ARC 기본 태스크 실행 (allenai/ai2_arc 사용)"
            # AI2-ARC는 기본 태스크
            accelerate launch -m lm_eval \
                --model hf \
                --model_args $MODEL_ARGS \
                --tasks $TASK \
                --num_fewshot $NUM_FEWSHOT \
                --batch_size auto \
                --output_path $OUTPUT_FILE \
                --verbosity INFO
        fi
        
        echo "평가 완료 (진행: $CURRENT_RUN / $TOTAL_RUNS): 결과가 $OUTPUT_FILE 에 저장되었습니다."
        echo ""
    done
done

echo "모든 ARC 평가가 완료되었습니다."

# 결과 파일들 확인
echo "===================================================="
echo "생성된 결과 파일들:"
ls -la $RESULTS_DIR/*.json 2>/dev/null || echo "생성된 JSON 파일이 없습니다."
echo "===================================================="

# 결과 요약 생성
echo "결과 요약을 생성하는 중..."

SUMMARY_FILE="$RESULTS_DIR/arc_summary.json"

# JSON 시작
echo "{" > $SUMMARY_FILE
echo '  "evaluation_summary": {' >> $SUMMARY_FILE
echo '    "total_models": '$NUM_MODELS',' >> $SUMMARY_FILE
echo '    "total_tasks": '$NUM_TASKS',' >> $SUMMARY_FILE
echo '    "fewshot": '$NUM_FEWSHOT',' >> $SUMMARY_FILE
echo '    "datasets": "allenai/ai2_arc + 로컬 Ko-ARC",' >> $SUMMARY_FILE
echo '    "custom_task_path": "'$CUSTOM_TASK_PATH'",' >> $SUMMARY_FILE
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

# 간단한 텍스트 요약도 생성
TEXT_SUMMARY="$RESULTS_DIR/arc_summary.txt"
echo "ARC 평가 결과 요약 (AI2-ARC + Ko-ARC)" > $TEXT_SUMMARY
echo "생성 시간: $(date)" >> $TEXT_SUMMARY
echo "Few-shot: $NUM_FEWSHOT" >> $TEXT_SUMMARY
echo "데이터셋: allenai/ai2_arc + 로컬 Ko-ARC" >> $TEXT_SUMMARY
echo "커스텀 태스크 경로: $CUSTOM_TASK_PATH" >> $TEXT_SUMMARY
echo "===========================================" >> $TEXT_SUMMARY
echo "" >> $TEXT_SUMMARY

printf "%-30s" "모델명" >> $TEXT_SUMMARY
for TASK in "${TASKS[@]}"; do
    if [[ $TASK == "arc_challenge" ]]; then
        printf "%-15s" "ARC-C(영어)" >> $TEXT_SUMMARY
    elif [[ $TASK == "arc_easy" ]]; then
        printf "%-15s" "ARC-E(영어)" >> $TEXT_SUMMARY
    elif [[ $TASK == "arc_ko" ]]; then
        printf "%-15s" "Ko-ARC(한국어)" >> $TEXT_SUMMARY
    else
        printf "%-15s" "${TASK}" >> $TEXT_SUMMARY
    fi
done
echo "" >> $TEXT_SUMMARY

for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    printf "%-30s" "$NAME" >> $TEXT_SUMMARY
    
    for TASK in "${TASKS[@]}"; do
        RESULT_FILE="$RESULTS_DIR/${NAME}_${TASK}.json"
        if [ -f "$RESULT_FILE" ]; then
            ACCURACY=$(python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print('N/A')
        exit()
    
    # 태스크 키 찾기 (동일한 로직)
    task_key = None
    task_results = None
    
    if '$TASK' in data['results']:
        task_results = data['results']['$TASK']
    else:
        for key in data['results'].keys():
            if '$TASK' in key or key in '$TASK':
                task_results = data['results'][key]
                break
    
    if task_results and isinstance(task_results, dict):
        arc_metrics = ['acc', 'accuracy', 'acc_norm', 'exact_match', 'score', 'correct']
        
        for metric in arc_metrics:
            if metric in task_results:
                print(f'{task_results[metric]:.3f}')
                exit()
        
        exclude_keys = {
            'alias', 'num_fewshot', 'stderr', 'stderr_norm', 'alias_norm',
            'num_examples', 'version', 'task_hash'
        }
        for metric_key, value in task_results.items():
            if (isinstance(value, (int, float)) and 
                metric_key not in exclude_keys and
                not metric_key.endswith('_stderr') and
                not metric_key.startswith('_')):
                print(f'{value:.3f}')
                exit()
        
        print('N/A')
    else:
        print('N/A')
except Exception as e:
    print('N/A')
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

# 최종 요약 출력
echo "===================================================="
echo "ARC 평가 최종 요약:"
echo "- 평가된 모델 수: $NUM_MODELS"
echo "- 사용된 태스크: ${TASKS[*]}" 
echo "- Few-shot 설정: $NUM_FEWSHOT"
echo "- 데이터셋: allenai/ai2_arc (영어) + 로컬 Ko-ARC (한국어)"
echo "- 커스텀 태스크 경로: $CUSTOM_TASK_PATH"
echo "- 생성된 파일:"
echo "  JSON 요약: $SUMMARY_FILE"
echo "  텍스트 요약: $TEXT_SUMMARY"
echo "===================================================="