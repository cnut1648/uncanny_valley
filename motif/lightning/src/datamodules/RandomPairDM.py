import pickle
from collections import defaultdict
from pathlib import Path
import random
from typing import Optional, Tuple, Sized, Iterable, List, Dict

import pandas as pd
import torch
import datasets
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TDataset
from torch.utils.data.sampler import Sampler
from transformers import AutoTokenizer


class PairDataset(TDataset):
    def __init__(self,
                 data_source: Dataset,
                 num_positives: int, num_negatives: int,
                 shuffle: bool = False):
        """
        data_source: already ed Dataset.Dataset
        thus ele = torch tensor
        """
        super().__init__()
        self.num_negatives = num_negatives
        self.num_positives = num_positives
        self.data_source = data_source
        # track which is already done
        # goal is go thru all
        # if shuffle: random.shuffle(self.idxs)

    def __getitem__(self, idx):
        """
        Returns:
            idx anchor, #pos positive (same class), #neg negative (diff class)
        """
        label = self.data_source[idx]['label']
        positive_samples = self.data_source.filter(
            lambda x: x['label'] == label
        )
        random_pos = random.sample(range(len(positive_samples)), self.num_positives)

        negative_samples = self.data_source.filter(
            lambda x: x['label'] != label
        )
        random_neg = random.sample(range(len(negative_samples)), self.num_negatives)

        return self.data_source[[idx]], positive_samples[random_pos], negative_samples[random_neg]

    def __len__(self):
        return len(self.data_source)

    @staticmethod
    def collate_fn(batches: List[Tuple[Dict, Dict, Dict]]) -> Dict:
        """
        let K = 1 + #pos + #neg
        Returns:
            dict of keys
                label: (N, K)
                attention_mask: (N, K, L)
                input_ids: (N, K, L)
        """
        ret = defaultdict(list)
        # 3 dict
        for anchor, positives, negatives in batches:
            for k in anchor:
                value = torch.cat([anchor[k], positives[k], negatives[k]], dim=0).unsqueeze(0)
                ret[k].append(value)
        ret = {
            k: torch.cat(v, dim=0)
            for k, v in ret.items()
        }
        return ret


class RandomPairDM(LightningDataModule):
    """
    supervised contrastive learning datamodule
    generated batch of (N, num_neg+num_pos+1),
        where 1st one the anchor, then positive (same label), rest negative (diff label)

    """

    def __init__(
            self,
            data_dir: str, train_val_test_split: str,
            num_positives: int = 1, num_negatives: int = 1,
            # dataloading
            batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
            # tokenization
            arch: str = "", seq_len: int = 0, cache_dir: str = ".",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            arch, use_fast=True
        )
        self.cache_dir = cache_dir
        self.num_negatives = num_negatives
        self.num_positives = num_positives
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
            # truncation="longest_first",
            # padding="max_length")

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
                    dataset = PairDataset(
                        dataset,
                        num_negatives=self.num_negatives, num_positives=self.num_positives)

                    self.datasets[split] = DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        collate_fn=PairDataset.collate_fn,
                        shuffle=(split == "train"),
                    )
                    # self.datasets[split] = DataLoader(
                    #     dataset,
                    #     sampler=PairSampler(dataset,
                    #                         shuffle=(split == "train"),
                    #                         num_negatives=self.num_negatives,
                    #                         num_positives=self.num_positives),
                    #     batch_size=self.batch_size,
                    #     num_workers=self.num_workers,
                    #     pin_memory=self.pin_memory,
                    # )

    def train_dataloader(self):
        return self.datasets['train']

    def val_dataloader(self):
        return self.datasets['val']

    def test_dataloader(self):
        return self.datasets['test']
