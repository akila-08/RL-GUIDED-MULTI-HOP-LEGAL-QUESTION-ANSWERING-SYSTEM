import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

with open("data/decompose_dataset.json", "r") as f:
    data = json.load(f)

def clean_question(q):
    q = q.strip()
    if not q.endswith("?"):
        q += "?"
    return q

def format_example(example):
    input_text = "decompose: " + example["complex_question"]
    sub_qs = example["sub_questions"]
    lines = []
    for i, sq in enumerate(sub_qs):
        q = clean_question(sq["question"])
        if i == len(sub_qs) - 1:
            lines.append(f"<apply> {q}")
        else:
            lines.append(f"<rule> {q}")
    target_text = "\n".join(lines)
    return {"input_text": input_text, "target_text": target_text}

formatted_data = [format_example(x) for x in data]
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1)

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 🔧 Add custom tokens so the model knows <rule> and <apply> as single units
special_tokens = {"additional_special_tokens": ["<rule>", "<apply>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

def tokenize(example):
    model_inputs = tokenizer(
        example["input_text"],
        max_length=256,
        truncation=True,
        # 🔧 Remove padding="max_length" — let DataCollator handle padding
    )

    labels = tokenizer(
        text_target=example["target_text"],
        max_length=128,
        truncation=True,
        # 🔧 Remove padding here too
    )

    model_inputs["labels"] = labels["input_ids"]
    # 🔧 Do NOT manually replace pad tokens with -100 here.
    # DataCollatorForSeq2Seq with label_pad_token_id=-100 handles this correctly.
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize,
    batched=False,
    remove_columns=dataset["train"].column_names
)

# 🔧 Set label_pad_token_id so collator masks padding in labels properly
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=4,      # 🔧 increased; collator-based padding is more efficient
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    num_train_epochs=10,
    learning_rate=5e-5,                 # 🔧 lower LR — 1e-4 was too aggressive for flan-t5
    weight_decay=0.01,
    max_grad_norm=1.0,

    fp16=False,                         # 🔧 disable — RTX 2050 FP16 causes NaN with seq2seq

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,

    save_total_limit=2,
    load_best_model_at_end=True,

    label_smoothing_factor=0.0,         # 🔧 disable until training is stable

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./decomp_model")
tokenizer.save_pretrained("./decomp_model")