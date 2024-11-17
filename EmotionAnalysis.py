import os
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import torch
import numpy as np

# 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"  # 사용할 GPU 번호 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 결과 저장 경로
OUTPUT_DIR = '/home/ysw1029'  # 원격 서버의 디렉토리로 수정
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")

# 최적의 하이퍼파라미터
best_hyperparams = {
    'learning_rate': 2.8741541327119172e-05,  # 최적 학습률
    'per_device_train_batch_size': 16,  # 최적 배치 크기
    'num_train_epochs': 2  # 최적 에포크 수
}

# NSMC 데이터셋 로드 및 토큰화
dataset = load_dataset('nsmc')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


def tokenize_function(examples):
    return tokenizer(examples['document'], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 정확도와 F1 스코어 메트릭 로드
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


# F1 스코어와 정확도 계산을 위한 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # 소프트맥스 적용
    predictions = np.argmax(probabilities, axis=-1)  # 확률값에서 가장 큰 인덱스를 예측값으로
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "probabilities": probabilities.tolist(),  # 확률값을 출력을 위함
    }


# 모델 초기화 함수
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)


# 학습 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_hyperparams['learning_rate'],  # 최적 학습률
    per_device_train_batch_size=best_hyperparams['per_device_train_batch_size'],  # 최적 배치 크기
    per_device_eval_batch_size=16,  # 검증 시 배치 크기
    num_train_epochs=best_hyperparams['num_train_epochs'],  # 최적 에포크 수
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",#코사인 어닐링
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    fp16=True,
)

# Trainer 생성
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics
)

# 모델 학습
trainer.train()

# 최종 모델 저장
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# 평가 결과 저장
eval_results_path = os.path.join(OUTPUT_DIR, "eval_results_final.json")
with open(eval_results_path, "w") as f:
    json.dump(trainer.evaluate(), f)

print(f"Model saved to {final_model_dir}")
