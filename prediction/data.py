from precompiled import *

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

dataset = load_dataset("openwebtext", streaming=True, trust_remote_code=True)
text_column = "text"

size = 3000
samples = list(islice(dataset['train'], size))   # much faster than using self-written iterations...
raw_datasets = Dataset.from_dict({text_column: [sample[text_column] for sample in samples]})

tmp = raw_datasets.train_test_split(test_size=0.2)  # very small. since it often runs OOM...
train_dataset = tmp['train']
val_dataset, test_dataset = tmp['test'].train_test_split(test_size=0.15).values()

def preprocess_function(data):
    inputs = tokenizer(data[text_column], max_length=None, truncation=True, padding=True)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

original_test_dataset = test_dataset

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=test_dataset.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds   # GPT2: (bs, seq_length, vocab_size)
    # print(type(preds))
    # print(preds.shape)
    preds = np.argmax(preds, axis=-1)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

def print_rouge_score(trainer: Trainer, test_dataset: Dataset):
    # Note: this is a hack to make the predict() method temporarily work.
    #  Please see the source code of:
    #   Huggingface transformers, [[Train#prediction_loop]]
    trainer.args.predict_with_generate = True
    trainer.args.prediction_loss_only = False
    trainer.compute_metrics = compute_metrics

    res = trainer.predict(test_dataset)
    print(res.metrics)

def print_rouge_score_2(model: nn.Module, test_dataset: Dataset):
    training_args = TrainingArguments(
        output_dir="/tmp",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # eval_accumulation_steps=20,  # (slow) it flushes data into cpu once per 20 steps to prevent GPU OOM: https://stackoverflow.com/a/75793317
        eval_strategy="steps",
        save_strategy='steps',
        fp16=torch.cuda.is_available(),
        prediction_loss_only=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.select(range(1)),
        eval_dataset=val_dataset.select(range(1)),
        compute_metrics=compute_metrics,
    )

    # res = trainer.predict(test_dataset.select(range(10)))
    res = trainer.predict(test_dataset)
    print(res.metrics)

    del trainer
    release()

import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


# Perplexity: Copied & Pasted from the source:
#   https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
#   Unfortunately we cannot directly use that since it doesn't accept model as an argument.
#   It supports batches, so it's very fast.
class Perplexity:
    @staticmethod
    def compute(input_texts, model, tokenizer, batch_size: int = 16,
                add_start_token: bool = True, device=None, max_length=None):
        model = model.to(device)

        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                    tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def print_perplexity(model, tokenizer, test_dataset: Dataset):
    res = Perplexity.compute(test_dataset[text_column], model, tokenizer,
                             # device="cpu",
                             device=device,
                             batch_size=32, max_length=model.config.n_positions)
    # print(res)
    print(f"Perplexity (mean): {res['mean_perplexity']:.2f}")


