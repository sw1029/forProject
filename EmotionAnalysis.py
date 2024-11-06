'''import os
import pickle
import optuna
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"  # 사용할 GPU 번호 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 결과 저장 경로
OUTPUT_DIR = '/home/ysw1029'  # 원격 서버의 디렉토리로 수정

# 경로 설정
study_path = os.path.join(OUTPUT_DIR, "optuna_study.pkl")
checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoint-last")

# GPU 사용 여부 및 장치 이름 확인
import torch

cuda_available = torch.cuda.is_available()
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "No GPU available"
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Device Name: {cuda_device_name}")

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
    predictions = logits.argmax(axis=-1)  # 가장 높은 값의 인덱스를 예측값으로 사용
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


# 모델 초기화 함수
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)


# Optuna 튜닝을 위한 objective 함수 정의
def objective(trial):
    # 학습률, 배치 크기, epoch 수를 최적화할 하이퍼파라미터로 지정
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 4, 16, step=4)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
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

    # 체크포인트가 있는 경우 불러오기
    if os.path.exists(checkpoint_dir):
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("No checkpoint found, starting from scratch.")
        trainer.train()

    # 검증 손실 반환 (최적화 기준)
    eval_result = trainer.evaluate()

    # 매 trial 종료 후 Optuna 스터디 저장
    print(f"Trial {trial.number} finished. Saving Optuna study...")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    return eval_result['eval_loss']


# Optuna 스터디 생성 및 로드
if os.path.exists(study_path):
    print("Loading previous Optuna study...")
    with open(study_path, "rb") as f:
        study = pickle.load(f)
else:
    print("No previous study found. Creating a new one...")
    study = optuna.create_study(direction="minimize")

# Optuna 튜닝 진행
study.optimize(objective, n_trials=10)

# 최종 모델 저장
model_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

# 평가 결과 저장
eval_results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
with open(eval_results_path, "w") as f:
    json.dump(trainer.evaluate(), f)

# Optuna 튜닝 결과 저장 (최종)
hyperparams_path = os.path.join(OUTPUT_DIR, "best_hyperparameters.txt")
with open(hyperparams_path, "w") as f:
    f.write(f"Best trial value: {study.best_trial.value}\n")
    f.write("Best hyperparameters:\n")
    for key, value in study.best_trial.params.items():
        f.write(f"  {key}: {value}\n")
'''

import os
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

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
    predictions = logits.argmax(axis=-1)  # 가장 높은 값의 인덱스를 예측값으로 사용
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
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
    lr_scheduler_type="cosine",
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
