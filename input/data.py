from summarization.precompiled import *

from datasets import load_dataset
squad = load_dataset("squad_v2")

def preprocess_function(examples):
    inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    targets = [ans["text"][0] if ans["text"] else "no_answer" for ans in examples["answers"]]
    return {"input_text": inputs, "target_text": targets}

tokenized_squad = squad.map(preprocess_function, batched=True)

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_squad = tokenized_squad.map(tokenize_function, batched=True)

tokenized_squad.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_squad['train']
val_dataset = tokenized_squad['validation']

# train_dataset = train_dataset.train_test_split(test_size=0.1)["test"]
# val_dataset = val_dataset.train_test_split(test_size=0.3)["test"]
