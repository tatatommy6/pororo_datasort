from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

# 1. JSON 로딩 (파인튜닝용 대화 데이터)
with open("pororo_chat.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 메시지를 하나의 텍스트로 변환
def format_chat(example):
    messages = example["messages"]
    dialogue = ""
    for msg in messages:
        dialogue += f"<|{msg['role']}|>\n{msg['content']}\n"
    return {"text": dialogue}

dataset = Dataset.from_list(data).map(format_chat)

# 3. 모델과 토크나이저 로딩
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # 예시용, 다른 모델도 가능
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 4. 토큰화
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
tokenized.set_format("torch")

# 5. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./pororo_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,  # GPU가 있다면 True
)

# 6. Trainer 구성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()

# 7. 모델 저장
trainer.save_model("./pororo_model")