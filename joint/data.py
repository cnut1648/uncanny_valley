import os
from collections import namedtuple
from typing import Dict

import numpy as np
import pandas as pd
import wandb
import yaml
from sklearn.preprocessing import LabelEncoder
from torch import nn
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    IntervalStrategy, AutoTokenizer, EarlyStoppingCallback
from transformers.integrations import WandbCallback
from torch.utils.data import DataLoader
from transformers.data import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TDATA_PATH = "../datasets/tdata"

def createDataset(config):
    df = pd.read_hdf("../datasets/ATU.h5", key="default")
    df = df[df["motif"] != "NOT FOUND"]
    atu = df.loc[df.groupby("atu")["atu"].filter(lambda g: len(g) >= config["data"]["atu_filter_no"]).index]
    # atu = df
    atu["labels"] = LabelEncoder().fit_transform(atu["motif"])
    atu = atu[["text", "labels"]]

    dataset = Dataset.from_pandas(atu)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["arch"])

    def tokenize(instance):
        return tokenizer(instance["text"],
                         max_length=128,
                         truncation=True,
                         padding=True
                         )

    dataset = dataset. \
        shuffle(seed=42). \
        map(tokenize, batched=True)

    dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'labels']
    )

    dataset = dataset.train_test_split(test_size=0.1)
    dataset.save_to_disk(TDATA_PATH)
    return dataset



def compute_metrics(pred: namedtuple):
    # y_pred (N, d), y_true (N,)
    y_pred, y_true = pred
    y_pred = y_pred.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



if __name__ == "__main__":
    with open('kl_auxiliary.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    useCachedDataset = False
    if useCachedDataset:
        dataset = load_from_disk(TDATA_PATH)
        dataset.set_format(
            type="torch",
            columns=['input_ids', 'attention_mask', 'labels']
        )
    else:
        dataset = createDataset(config)

    train_ds, test_ds = dataset["train"], dataset['test']
    print(f'train: {len(train_ds)}, test: {len(test_ds)}, columns: {train_ds.features}')

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "./results/checkpoint-875"
    # )
    #
    # dataloader = DataLoader(
    #     test_ds.select(range(12)),
    #     batch_size=4,
    #     drop_last=True
    # )
    #
    # for batch in tqdm(dataloader):
    #     # print(type(batch["input_ids"]))
    #     output = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    #     logits = output.logits.detach().numpy()
    #     y_pred = logits.argmax(-1)
    #     acc = accuracy_score(batch["labels"].detach().numpy(), y_pred)
    #     print(acc)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["arch"], num_labels=config["model"]["num_labels"],
    )

    trainConfig = config["train"]
    train_args = TrainingArguments(
        # model pred/ckpt
        output_dir='./results',
        # tensorboard logs
        logging_dir="./logs",
        num_train_epochs=trainConfig["epoch"],
        per_device_train_batch_size=trainConfig["train_batch_size"],
        per_device_eval_batch_size=trainConfig["eval_batch_size"],
        # x (logging / eval /save) every acc * x_steps
        gradient_accumulation_steps=trainConfig["acc_batch"],
        evaluation_strategy=IntervalStrategy.EPOCH,
        # AdamW
        learning_rate=1e-5,
        warmup_steps=500,
        # apply to all layers but bias / LayerNorm
        weight_decay=0.02,
        # if True, ignore param save_strategy / save_steps / save_total_limit
        load_best_model_at_end=True,
        # report_to=["none"],
        report_to=["wandb"],
        seed=42,
        logging_strategy=IntervalStrategy.STEPS,
    )

    wandb.init(
        project="folklore",
        name="roberta-filtered",
        tags=["roberta-large", "ATU filtered"],
        group="fine-tune"
    )
    trainer = Trainer(
        model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.evaluate()

    # best model
    y_pred_tuple = trainer.predict(test_ds)
    y_pred, y_true, metrics = y_pred_tuple
    acc = accuracy_score(y_true, y_pred.argmax(-1))
    print(acc)

