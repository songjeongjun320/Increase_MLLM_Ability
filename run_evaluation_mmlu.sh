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
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct"
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it"
    # "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct"
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

TASKS=("mmlu_ko_custom" "mmlu_en_custom")
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

        accelerate launch -m lm_eval \
            --model hf \
            --model_args $MODEL_ARGS \
            --tasks $TASK \
            --include_path ./eval_configs \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size auto \
            --output_path $OUTPUT_FILE \
            --log_samples \
            --verbosity INFO
        
        echo "평가 완료 (진행: $CURRENT_RUN / $TOTAL_RUNS): 결과가 $OUTPUT_FILE 에 저장되었습니다."
        echo ""
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
    if 'results' in data and '$TASK' in data['results']:
        task_results = data['results']['$TASK']
        if 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.4f}')
        else:
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and key != 'alias':
                    print(f'{value:.4f}')
                    break
            else:
                print('null')
    else:
        print('null')
except:
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

# 텍스트 요약 생성
TEXT_SUMMARY="$RESULTS_DIR/mmlu_summary.txt"
echo "MMLU 평가 결과 요약" > $TEXT_SUMMARY
echo "생성 시간: $(date)" >> $TEXT_SUMMARY
echo "Few-shot: $NUM_FEWSHOT" >> $TEXT_SUMMARY
echo "===========================================" >> $TEXT_SUMMARY
echo "" >> $TEXT_SUMMARY

printf "%-30s" "모델명" >> $TEXT_SUMMARY
for TASK in "${TASKS[@]}"; do
    printf "%-15s" "${TASK##*_}" >> $TEXT_SUMMARY
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
    if 'results' in data and '$TASK' in data['results']:
        task_results = data['results']['$TASK']
        if 'acc' in task_results:
            print(f'{task_results[\"acc\"]:.3f}')
        else:
            for key, value in task_results.items():
                if isinstance(value, (int, float)) and key != 'alias':
                    print(f'{value:.3f}')
                    break
            else:
                print('N/A')
    else:
        print('N/A')
except:
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