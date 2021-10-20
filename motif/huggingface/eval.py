import os, sys
from collections import namedtuple
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import wandb
import yaml
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader

from datasets import Dataset, load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
from typing import List
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    IntervalStrategy, AutoTokenizer, EarlyStoppingCallback, AutoConfig, logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import glob
from logging import getLogger
import warnings

def compute_metrics(pred: namedtuple):
    # logits (before-softmax) (N, d), y_true (N,)
    logits, y_true = pred
    y_prob = softmax(logits, axis=-1)
    y_pred = y_prob.argmax(-1)
    # one-vs-rest macro AUC
    macro_auc = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovo")
    weight_auc = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovo")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_auc': macro_auc,
        'weight_auc': weight_auc,
    }

TDATA_PATH = "/home/jiashu/uncanny_valley/datasets/motif/allenai/longformer-base-4096/filter-0_test-0.2"
ckpt_path = "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/99"

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    dataset = load_from_disk(TDATA_PATH)
    dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'label']
    )
    train_ds, test_ds = dataset["train"], dataset['test']

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path
    )
    trainConfig = cfg.train
    output_dir = os.path.join(trainConfig["output_dir"])
    train_args = TrainingArguments(
        # module pred/ckpt
        output_dir=output_dir,
        # tensorboard logs
        logging_dir="./logs",
        num_train_epochs=trainConfig["epoch"],
        per_device_train_batch_size=trainConfig["train_batch_size"],
        per_device_eval_batch_size=trainConfig["eval_batch_size"],
        # x (logging / eval /save) every acc * x_steps
        gradient_accumulation_steps=trainConfig["acc_batch"],
        evaluation_strategy=IntervalStrategy.EPOCH,
        label_smoothing_factor=trainConfig["label_smooth"],
        # AdamW
        learning_rate=trainConfig["lr"],
        warmup_steps=trainConfig["warmup"],
        # apply to all layers but bias / LayerNorm
        weight_decay=trainConfig["wd"],
        save_total_limit=2,
        # if True, ignore param save_strategy / save_steps / save_total_limit
        load_best_model_at_end=True,
        # report_to=["none"],
        report_to=["wandb"],
        seed=cfg.seed,
        logging_strategy=IntervalStrategy.STEPS,
        metric_for_best_model=trainConfig["metric"]
    )
    trainer = Trainer(
        model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=trainConfig["early_stopping_patience"]),
        ],
        compute_metrics=compute_metrics,
    )

    y_pred_tuple = trainer.predict(test_ds)
    logits, y_true, metrics = y_pred_tuple
    y_pred = logits.argmax(-1)
    with open("LF.pl", "wb") as f:
        import pickle
        pickle.dump([y_pred, y_true], f)
    print(metrics)
    acc = accuracy_score(y_true, y_pred)
    print(acc)


if __name__ == "__main__":
    main()