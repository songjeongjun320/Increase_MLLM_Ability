Trainer 기반 코드를 finetune.py 스타일로 변환하는 단계별 지침을 제시하겠습니다.
1. 핵심 아키텍처 변경
A. Trainer → Accelerate + 수동 루프
제거해야 할 것들:

from transformers import Trainer, TrainingArguments
EnhancedToWTrainer 클래스
trainer.train() 호출
모든 TrainerCallback들

추가해야 할 것들:

from accelerate import Accelerator, InitProcessGroupKwargs
수동 훈련 루프 구현
직접적인 옵티마이저/스케줄러 관리

B. 데이터 처리 방식 변경
기존 (Trainer):
pythondata_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
변경 후 (finetune.py 스타일):
pythondef encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    # prompt-completion 형식의 정확한 마스킹 구현
    # labels에서 prompt 부분을 -100으로 마스킹
2. 데이터 전처리 로직 재구성
A. 현재 문제점 파악
기존 CustomDataCollatorForLanguageModeling은 단순히 input_ids를 labels로 복사하므로 prompt 부분도 학습에 포함됩니다.
B. finetune.py 스타일로 변경

데이터 읽기 함수 구현:

pythondef read_from_jsonl(raw_file):
    outputs = []
    with open(raw_file) as f:
        for line in f:
            data = json.loads(line)
            outputs.append({"prompt": data['prompt'], "completion": data['completion']})
    return outputs

인코딩 함수 적용:

pythonencode_function = partial(
    encode_with_prompt_completion_format,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    add_bos=False,
)
train_dataset = train_dataset.map(encode_function, batched=False)
3. 특수 토큰 처리 통일
A. 토큰 명칭 통일
기존: <ToW>, </ToW>
finetune.py: <hCoT>, </hCoT>
결정: 어떤 토큰을 사용할지 명확히 하고 일관되게 적용
B. 임베딩 초기화 로직 적용
finetune.py의 DeepSpeed 호환 초기화 로직을 그대로 사용:
pythonwith deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
    init_embeddings = embeddings.weight.data[tokenizer.encode('---', add_special_tokens=False)[0], :]
    embeddings.weight.data[len(tokenizer)-1, :] = init_embeddings
    embeddings.weight.data[len(tokenizer)-2, :] = init_embeddings
4. 손실 계산 방식 변경
A. Trainer의 자동 손실 → 수동 손실 계산
제거: 모든 compute_loss override
추가: finetune.py의 조건부 손실 계산 로직
B. Sum reduction 지원 추가
pythonif reduce_loss == "mean":
    loss = outputs.loss
else:
    # sum reduction 구현
    logits = outputs.logits
    labels = batch["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # ... 나머지 구현
5. 훈련 루프 완전 재구성
A. 제거할 섹션들

trainer = EnhancedToWTrainer(...) 초기화
trainer.train() 호출
모든 callback 설정

B. 추가할 핵심 구조
python# 1. Accelerator 초기화
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

# 2. 옵티마이저/스케줄러 설정
optimizer = torch.optim.AdamW(...)
lr_scheduler = get_scheduler(...)

# 3. accelerator.prepare() 호출
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(...)

# 4. 수동 훈련 루프
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = compute_loss_function(outputs, batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
6. 체크포인트 및 저장 시스템 변경
A. Trainer 저장 → Accelerate 저장
제거:

trainer.save_model()
TrainingArguments의 체크포인트 설정

추가:

accelerator.save_state(output_dir)
save_with_accelerate() 함수 사용

B. 체크포인트 복원 로직
pythonlast_checkpoint_path = get_last_checkpoint_path(args)
if last_checkpoint_path:
    accelerator.load_state(last_checkpoint_path)
7. 로깅 및 메트릭 시스템 재구성
A. TrainerCallback → 직접 로깅
제거: 모든 callback 클래스들
추가: 조건부 로깅 로직
pythonif completed_steps % logging_steps == 0:
    avg_loss = accelerator.gather(total_loss).mean().item()
    logger.info(f"Step: {completed_steps}, Loss: {avg_loss}")
B. 메트릭 수집 단순화
복잡한 ToW 메트릭 대신 기본적인 손실과 학습률 추적으로 단순화
8. 분산 훈련 설정 변경
A. 수동 분산 설정 제거
제거:
pythonif "LOCAL_RANK" in os.environ:
    torch.distributed.init_process_group(backend="nccl")
변경: Accelerate가 자동으로 처리하도록 위임
B. DeepSpeed 통합 방식 변경

JSON 설정 파일을 통한 DeepSpeed 설정
accelerator.prepare()를 통한 자동 초기화

9. 인자 처리 시스템 통합
A. 커스텀 Config → FlatArguments
기존 ToWTrainingConfig를 FlatArguments 스타일로 통합
B. ArgumentParserPlus 사용
YAML 및 command line 인자를 모두 지원하는 파서 사용
10. 실행 플로우 재구성
A. main() 함수 완전 재작성
pythondef main(args: FlatArguments):
    # 1. Accelerator 초기화
    # 2. 모델/토크나이저 로딩
    # 3. 데이터셋 처리
    # 4. 옵티마이저 설정
    # 5. 수동 훈련 루프
    # 6. 모델 저장
B. 스크립트 진입점 변경
pythonif __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()
    main(args)
