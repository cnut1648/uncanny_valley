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

def load_motif_mapping(path="datasets/label-mapping.txt") -> dict:
    ret = {}
    with open(path) as f:
        for line in f:
            motif, label = line.split("->")
            ret[int(label)] = motif
    return ret

label_mapping = load_motif_mapping("/home/jiashu/uncanny_valley/datasets/label-mapping.txt")

log = getLogger(__name__)
# ignore warning but keep only error
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def createDataset(config):
    """
    build dataset from the h5 file
    also filter out rare *individual ATU*
    """
    df = pd.read_hdf(config.data.h5_file, key=config.data.h5_key)
    atu = df.loc[df.groupby("atu")["atu"].filter(lambda g: len(g) >= config["datamodules"]["atu_filter_no"]).index]
    atu = atu[["text", "atu", "desc", "label"]]

    dataset = Dataset.from_pandas(atu)
    tokenizer = AutoTokenizer.from_pretrained(config["module"]["arch"])

    def tokenize(instance):
        return tokenizer(
            instance["text"],
            max_length=config["module"]["seq_len"],
            truncation="longest_first",
            padding="max_length")

    dataset = dataset. \
        shuffle(seed=config.seed). \
        map(tokenize, batched=True)

    # split by cls (stratified)
    sub_ds = {
        "train": [],
        "test": []
    }
    for cls in np.unique(dataset["label"]):
        cls_ds = dataset. \
            filter(lambda d: d['label'] == int(cls))
        cls_ds = cls_ds.train_test_split(test_size=config.data.test_ratio, seed=config.seed)
        sub_ds["train"].append(cls_ds["train"])
        sub_ds["test"].append(cls_ds["test"])
    
    dataset = DatasetDict({
        split: concatenate_datasets(ds)
        for split, ds in sub_ds.items()
    })
    dataset.save_to_disk(config.data.cached_dir)
    return dataset

def getDataset(config):
    TDATA_PATH = config.data.cached_dir
    try:
        dataset = load_from_disk(TDATA_PATH)
    except:
        # not exist, create one
        dataset = createDataset(config)
    dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'label']
    )
    train_ds, test_ds = dataset["train"], dataset['test']
    log.info(f'train: {len(train_ds)}, test: {len(test_ds)}, columns: {train_ds.features}')

    return train_ds, test_ds


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

def init_wandb(cfg: DictConfig):
    config = {
       f"cfg/{k}": v
       for k, v in OmegaConf.to_container(cfg, resolve=True).items()
    }
    

    wandb.init(
        project=cfg.logger.project,
        config=config,
        tags=cfg.logger.tags,
        group=cfg.logger.group,
    )

    # upload code
    code = wandb.Artifact("project-source", type="code")
    for path in glob.glob(os.path.join(cfg.work_dir, "*.py")):
        code.add_file(path)
    wandb.log_artifact(code)


def plot_heat_map(y_true, y_pred, num_labels):
    cm = confusion_matrix(y_true, y_pred)
    columns: List[str] = [
        label_mapping[i]
        for i in range(num_labels)
    ]
    df = pd.DataFrame(cm, index=columns, columns=columns)
    df.to_csv("eval_cm.csv")

    fig = sns.heatmap(df, annot=True, fmt='d', cmap="Reds")

    wandb.log({
        "confusion_matrix_plot": wandb.Image(fig),
        "confusion_matrix": wandb.Table(dataframe=df)
    })




@hydra.main(config_path="conf", config_name="config")
def fine_tune(cfg: DictConfig) -> float:
    """fine tune bert module"""
    init_wandb(cfg)
    train_ds, test_ds = getDataset(cfg)

    config = AutoConfig.from_pretrained(
        cfg.model.arch, num_labels=cfg.model.num_labels
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.arch, config=config
    )
    id = wandb.run.name.rsplit("-", 1)[1]
    trainConfig = cfg.train
    output_dir = os.path.join(trainConfig["output_dir"], id)
    print("module output dir = ", output_dir)
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
        # save_total_limit=2,
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

    print("logs in dir", os.getcwd())
    print("gpu count = ", trainer.args.n_gpu, "is_fp16 =", trainer.args.fp16)

    trainer.train()
    trainer.evaluate()

    # best module
    trainer.model.save_pretrained(os.path.join(output_dir, "best"))
    y_pred_tuple = trainer.predict(test_ds)
    logits, y_true, metrics = y_pred_tuple
    y_pred = logits.argmax(-1)
    
    plot_heat_map(y_true, y_pred, cfg.model.num_labels)

    acc = accuracy_score(y_true, y_pred)
    print(acc)
    wandb.finish()
    return acc

if __name__ == "__main__":
   fine_tune()