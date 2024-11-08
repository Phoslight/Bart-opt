from precompiled import *

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

df = pd.read_csv(INPUT_PATH + "bbc.csv")

text_column = "article"
summary_column = "summary"
raw_datasets = Dataset.from_pandas(df[[text_column, summary_column]])

max_source_length = 512
max_target_length = 128

def preprocess_function(data):
    model_inputs = tokenizer(data[text_column], max_length=max_source_length, padding="max_length", truncation=True)
    labels = tokenizer(text_target=data[summary_column], max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    # num_proc=4,
    remove_columns=raw_datasets.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on train dataset",
)

train_tmp_split = raw_datasets.train_test_split(test_size=0.2)
val_test_split = train_tmp_split['test'].train_test_split(test_size=0.5)
train_dataset, val_dataset, test_dataset = train_tmp_split['train'], val_test_split['train'], val_test_split['test']

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print(type(preds))
    # print(preds.shape)

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

def print_rouge_score_2(model: BartForConditionalGeneration, test_dataset: Dataset):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        save_strategy='steps',
        fp16=torch.cuda.is_available(),
        prediction_loss_only=False,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    res = trainer.predict(test_dataset)
    print(res.metrics)
