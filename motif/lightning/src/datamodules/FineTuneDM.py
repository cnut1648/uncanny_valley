import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import torch
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TDataset
from transformers import AutoTokenizer


class FineTuneDM(LightningDataModule):
    """
    Fine tune LM
    """

    def __init__(
            self,
            data_dir: str, train_val_test_split: str,
            # dataloading
            batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
            # tokenization
            arch: str = "", seq_len: int = 0, cache_dir: str = ".",
            **kwargs
    ):
        super().__init__()

        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            arch, use_fast=True
        )
        self.cache_dir = cache_dir
        self.data_dir = Path(data_dir)
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.datasets = {}

    @property
    def num_classes(self) -> int:
        return 7

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        split_index_file = self.data_dir / f"{self.train_val_test_split}.pkl"
        # split ATU dataset into train/val/test
        if not split_index_file.exists():
            atu: pd.DataFrame = pd.read_hdf(
                self.data_dir / "ATU.h5", key="default"
            )
            train, val, test = map(int, self.train_val_test_split.split("_"))
            assert train + val + test == 100

            train_index, val_index = train_test_split(
                atu.index, shuffle=True,
                test_size=(val + test) / 100, stratify=atu['label'])

            if test > 0:
                val_atu = atu[atu.index.isin(val_index)]
                val_index, test_index = train_test_split(
                    val_index, shuffle=True,
                    test_size=test / (test + val), stratify=val_atu['label'])

            with open(split_index_file, "wb") as f:
                split = {
                    "train": train_index.tolist(),
                    "val": val_index.tolist(),
                    "test": test_index.tolist() if test > 0 else None,
                }
                pickle.dump(split, f)

    def setup(self, stage: Optional[str] = None):
        def tokenize(instance):
            return self.tokenizer(
                instance["text"],
                max_length=self.seq_len,
                truncation=True,
                padding=True)

        if stage == "fit":
            split_index_file = self.data_dir / f"{self.train_val_test_split}.pkl"
            with open(split_index_file, "rb") as f:
                split_index = pickle.load(f)
            atu: pd.DataFrame = pd.read_hdf(
                self.data_dir / "ATU.h5", key="default"
            )
            for split, index in split_index.items():
                # ignore (test, None)
                if not (split == "test" and index is None):
                    split_atu = atu[atu.index.isin(index)][["text", "atu", "desc", "label"]]

                    dataset = Dataset.from_pandas(split_atu)
                    dataset = dataset.map(
                        tokenize,
                        batched=True)
                    # import sys; sys.exit(1)
                    # dataset = dataset.map(tokenize, batched=True)
                    dataset.save_to_disk(f"{self.cache_dir}/{split}")
                    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

                    self.datasets[split] = DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        shuffle=(split == "train"),
                    )

    def train_dataloader(self):
        return self.datasets['train']

    def val_dataloader(self):
        return self.datasets['val']

    def test_dataloader(self):
        return self.datasets['test']
