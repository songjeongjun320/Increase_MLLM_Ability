#!/bin/bash

# --- 설정 부분 ---
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

# 기본 태스크 사용 (커스텀 태스크 문제 해결을 위해)
# MMLU-ProX가 실제로 지원되는 태스크인지 확인 필요
TASKS=("mmlu")  # 일단 기본 MMLU로 시작, MMLU-ProX가 확인되면 변경
RESULTS_DIR="./evaluation_results_mmlu_prox"
mkdir -p $RESULTS_DIR
NUM_FEWSHOT=5

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_TASKS=${#TASKS[@]}
TOTAL_RUNS=$((NUM_MODELS * NUM_TASKS))
CURRENT_RUN=0

echo "MMLU-ProX 평가 시작"
echo "모델 수: $NUM_MODELS"
echo "태스크 수: $NUM_TASKS" 
echo "총 실행 횟수: $TOTAL_RUNS"
echo "Few-shot: $NUM_FEWSHOT"
echo ""

# 사용 가능한 태스크 확인
echo "사용 가능한 MMLU 관련 태스크 확인 중..."
lm-eval --tasks list | grep -i mmlu || echo "MMLU 태스크 목록을 가져올 수 없습니다."
echo ""

# --- 실행 부분 ---

for i in "${!MODEL_NAMES[@]}"; do
    NAME=${MODEL_NAMES[$i]}
    MODEL_PATH=${MODEL_PATHS[$i]}
    ADAPTER_PATH=${ADAPTER_PATHS[$i]}

    echo "모델 정보:"
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

        # 기본 태스크 사용 (--include_path 제거)
        accelerate launch -m lm_eval \
            --model hf \
            --model_args $MODEL_ARGS \
            --tasks $TASK \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size auto \
            --output_path $OUTPUT_FILE \
            --log_samples \
            --verbosity INFO
        
        echo "평가 완료 (진행: $CURRENT_RUN / $TOTAL_RUNS): 결과가 $OUTPUT_FILE 에 저장되었습니다."
        echo ""
    done
done

echo "모든 MMLU-ProX 평가가 완료되었습니다."

# 생성된 파일들 확인
echo "===================================================="
echo "생성된 결과 파일들:"
ls -la $RESULTS_DIR/*.json
echo "===================================================="

# 결과 요약 생성
echo "결과 요약을 생성하는 중..."

SUMMARY_FILE="$RESULTS_DIR/mmlu_prox_summary.json"

# JSON 시작
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
        
        echo "처리 중: $RESULT_FILE"
        
        # JSON 파일에서 정확도 추출 (강화된 디버깅 버전)
        if [ -f "$RESULT_FILE" ]; then
            echo "파일 존재함. JSON 구조 분석 중..."
            
            # 디버깅을 위한 JSON 구조 출력
            python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    print('=== JSON 구조 분석 ===')
    print('Top-level keys:', list(data.keys()))
    if 'results' in data:
        print('Results keys:', list(data['results'].keys()))
        for key in data['results'].keys():
            result_data = data['results'][key]
            if isinstance(result_data, dict):
                print(f'Task {key} metrics: {list(result_data.keys())}')
                # 샘플 값들도 출력
                for metric_key, metric_value in result_data.items():
                    if isinstance(metric_value, (int, float)):
                        print(f'  {metric_key}: {metric_value}')
            else:
                print(f'Task {key} type: {type(result_data)}')
    else:
        print('No results key found!')
    print('====================')
except Exception as e:
    print(f'Error reading file: {e}')
            "
            
            ACCURACY=$(python3 -c "
import json
import sys

try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print('Error: No results key found', file=sys.stderr)
        print('null')
        exit()
    
    # 태스크 키를 찾기 위한 포괄적인 검색
    task_key = None
    task_results = None
    
    print(f'Looking for task: $TASK', file=sys.stderr)
    print(f'Available tasks: {list(data[\"results\"].keys())}', file=sys.stderr)
    
    # 1. 정확한 매치
    if '$TASK' in data['results']:
        task_key = '$TASK'
        task_results = data['results']['$TASK']
        print(f'Found exact match: {task_key}', file=sys.stderr)
    else:
        # 2. 부분 매치 (다양한 패턴 시도)
        search_patterns = [
            '$TASK',
            '$TASK'.lower(),
            '$TASK'.upper(),
            '$TASK'_custom',
            '$TASK'_en_custom',
            '$TASK'_ko_custom'
        ]
        
        for pattern in search_patterns:
            for key in data['results'].keys():
                if (pattern in key or key in pattern or 
                    pattern.lower() in key.lower() or 
                    key.lower() in pattern.lower()):
                    task_key = key
                    task_results = data['results'][key]
                    print(f'Found pattern match: {pattern} -> {task_key}', file=sys.stderr)
                    break
            if task_key:
                break
        
        # 3. 첫 번째 결과 키 사용 (마지막 수단)
        if task_key is None and data['results']:
            task_key = list(data['results'].keys())[0]
            task_results = data['results'][task_key]
            print(f'Using first available key: {task_key}', file=sys.stderr)
    
    if task_results and isinstance(task_results, dict):
        print(f'Available metrics for {task_key}: {list(task_results.keys())}', file=sys.stderr)
        
        # 정확도 값 찾기 (우선순위 순서)
        accuracy_keys = ['acc', 'accuracy', 'acc_norm', 'exact_match', 'f1', 'score']
        
        found_accuracy = False
        for acc_key in accuracy_keys:
            if acc_key in task_results:
                print(f'{task_results[acc_key]:.4f}')
                print(f'Using metric: {acc_key} = {task_results[acc_key]}', file=sys.stderr)
                found_accuracy = True
                break
        
        if not found_accuracy:
            # 첫 번째 숫자 값 찾기 (제외할 키들을 더 포괄적으로)
            exclude_keys = {'alias', 'num_fewshot', 'stderr', 'stderr_norm', 'alias_norm'}
            for metric_key, value in task_results.items():
                if (isinstance(value, (int, float)) and 
                    metric_key not in exclude_keys and
                    not metric_key.endswith('_stderr')):
                    print(f'{value:.4f}')
                    print(f'Using fallback metric: {metric_key} = {value}', file=sys.stderr)
                    found_accuracy = True
                    break
            
            if not found_accuracy:
                print('Error: No valid numeric metrics found', file=sys.stderr)
                print('null')
    else:
        print('Error: No valid task results found or invalid format', file=sys.stderr)
        print('null')
        
except Exception as e:
    print(f'Exception occurred: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    print('null')
            ")
        else
            echo "파일이 존재하지 않음: $RESULT_FILE"
            ACCURACY="null"
        fi
        
        echo "추출된 정확도: $ACCURACY"
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
TEXT_SUMMARY="$RESULTS_DIR/mmlu_prox_summary.txt"
echo "MMLU-ProX 평가 결과 요약" > $TEXT_SUMMARY
echo "생성 시간: $(date)" >> $TEXT_SUMMARY
echo "Few-shot: $NUM_FEWSHOT" >> $TEXT_SUMMARY
echo "사용된 태스크: ${TASKS[*]}" >> $TEXT_SUMMARY
echo "===========================================" >> $TEXT_SUMMARY
echo "" >> $TEXT_SUMMARY

printf "%-30s" "모델명" >> $TEXT_SUMMARY
for TASK in "${TASKS[@]}"; do
    printf "%-20s" "${TASK}" >> $TEXT_SUMMARY
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
        # 패턴 매칭
        for key in data['results'].keys():
            if ('$TASK' in key.lower() or key.lower() in '$TASK'.lower() or
                '$TASK'.lower() in key.lower()):
                task_results = data['results'][key]
                break
        
        if task_results is None and data['results']:
            task_results = data['results'][list(data['results'].keys())[0]]
    
    if task_results and isinstance(task_results, dict):
        # 정확도 값 찾기
        accuracy_keys = ['acc', 'accuracy', 'acc_norm', 'exact_match', 'f1', 'score']
        
        for acc_key in accuracy_keys:
            if acc_key in task_results:
                print(f'{task_results[acc_key]:.3f}')
                exit()
        
        # 첫 번째 숫자 값 찾기
        exclude_keys = {'alias', 'num_fewshot', 'stderr', 'stderr_norm', 'alias_norm'}
        for metric_key, value in task_results.items():
            if (isinstance(value, (int, float)) and 
                metric_key not in exclude_keys and
                not metric_key.endswith('_stderr')):
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
        printf "%-20s" "$ACCURACY" >> $TEXT_SUMMARY
    done
    echo "" >> $TEXT_SUMMARY
done

echo ""
echo "텍스트 요약이 $TEXT_SUMMARY 에 저장되었습니다."

# 최종 요약 출력
echo "===================================================="
echo "MMLU-ProX 평가 최종 요약:"
echo "- 평가된 모델 수: $NUM_MODELS"
echo "- 사용된 태스크: ${TASKS[*]}" 
echo "- Few-shot 설정: $NUM_FEWSHOT"
echo "- 생성된 파일:"
echo "  JSON 요약: $SUMMARY_FILE"
echo "  텍스트 요약: $TEXT_SUMMARY"
echo "===================================================="

echo ""
echo "참고: MMLU-ProX 커스텀 태스크가 존재하지 않을 경우 기본 MMLU로 실행되었습니다."
echo "실제 MMLU-ProX 태스크를 사용하려면 해당 태스크가 올바르게 정의되어 있는지 확인하세요."