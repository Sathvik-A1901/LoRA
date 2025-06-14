import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# ==== User Configurations ====
MODEL_PATH = "C:/Users/sathv/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"  # Change this to your local Llama 2 directory
DATA_PATH = "llama_training/ollama_training.jsonl"
OUTPUT_DIR = "lora_llama2_finetuned"

# ==== Load Dataset ====
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ==== Load Model and Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token

# ==== Preprocessing Function ====
def preprocess(example):
    prompt = example["prompt"]
    completion = example["completion"]
    text = prompt + "\n" + completion
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v[0] for k, v in tokenized.items()}

processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ==== LoRA Configuration ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==== Training Arguments ====
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator
)

trainer.train()

# ==== Save Model ====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}") 