"""
generate embedding from fine tuned model
"""
import os
from collections import namedtuple
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import glob


def getDataset(config):
    """
    build dataset from the h5 file
    """
    atu = pd.read_hdf("/home/jiashu/uncanny_valley/datasets/folklore_noATU.h5", key=config.data.h5_key)
    atu = atu[["text", "folklore",]]

    dataset = Dataset.from_pandas(atu)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["arch"])

    def tokenize(instance):
        return tokenizer(
            instance["text"],
            max_length=config["model"]["seq_len"],
            truncation=True,
            padding=True)

    dataset = dataset. \
        shuffle(seed=config.seed). \
        map(tokenize, batched=True)

    dataset.set_format(
        type="numpy",
        columns=['input_ids', 'attention_mask', "text", "folklore"]
    )

    return dataset


# filter 10
# tokenizer len = 512
# roberta
# https://wandb.ai/cnut1648/motif/runs/3ffta2u6/overview?workspace=user-cnut1648
# BEST_CKPT = "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/72"
# name = "roberta-72"

# LF
# https://wandb.ai/cnut1648/motif/runs/2ll4vfnb/overview?workspace=user-cnut1648
# 0.8768472906403941
# BEST_CKPT = "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/79"
# name = "LF-79"

# filter 0
# tokenizer len = 512
# roberta
# https://wandb.ai/cnut1648/motif/runs/3ffta2u6/overview?workspace=user-cnut1648
# BEST_CKPT = "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/91"
# name = "roberta-91"
BEST_CKPT = "/home/jiashu/uncanny_valley/motif/huggingface/selected_ckpt/99"
name = "LF-99"

# BEST_CKPT = "roberta-large"
# name = "roberta-large"
# BEST_CKPT = "allenai/longformer-base-4096"
# name = "LF-base-4096"
# BEST_CKPT = "sentence-transformers/paraphrase-distilroberta-base-v2"
# name = "SBERT-v2"

# ALSO GENERATE FOR TEXT NOT IN TRAIN [eg. no motif found]

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    print("generating emb for ", name)
    if name.startswith("LF"):
        # must on ruby
        assert "allenai/longformer" in cfg.model.arch
        bsz = 64
    elif name.startswith("roberta"):
        assert "roberta" in cfg.model.arch
        bsz = 256
    elif name.startswith("SBERT"):
        assert "sentence" in cfg.model.arch
        bsz = 256

    dataset = getDataset(cfg)
    data_loader = DataLoader(dataset, batch_size=bsz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(BEST_CKPT)
    # if roberta-large, same as model itself
    # if seq cls model, remove head
    model = model.base_model
    model.eval()
    model.zero_grad()
    model.to(device)
    new_df = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            attention_mask = batch["attention_mask"].to(device)
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=attention_mask
            )
            # (N, max_len, hidden)
            token_emb = output["last_hidden_state"]
            if name.startswith("SBERT"):
                # (N, max_len) -> (N, max_len, hidden)
                input_mask_expanded = attention_mask.unsqueeze(-1) \
                    .expand(token_emb.size()).float()
                # mask out invalid token in max_len 
                # summary by mean of scores
                # (N, hidden)
                emb = torch.sum(token_emb * input_mask_expanded, 1) 
                emb /= torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                # roberta & LF seq clf use [CLS] (<s>) as summary
                # not use pooler
                # (N, hidden)
                emb = token_emb[:, 0, :]
            emb = emb.cpu().numpy()
            # pack all instance in batch
            for sampleid in range(len(batch["text"])):
                new_df.append({
                    "folklore": batch["folklore"][sampleid],
                    "text": batch["text"][sampleid],
                    name: emb[sampleid]
                })
    
    df = pd.DataFrame(new_df)
    atu = pd.read_hdf("/home/jiashu/uncanny_valley/datasets/folklore_noATU.h5", key=cfg.data.h5_key)
    merged = pd.merge(df, atu, how='inner',
            on=["folklore", "text"])
    assert atu.shape[0] == merged.shape[0], "row should be the same"
    print(merged.columns)
    merged.to_hdf("/home/jiashu/uncanny_valley/datasets/folklore_noATU.h5", key=cfg.data.h5_key)

if __name__ == "__main__":
    main()