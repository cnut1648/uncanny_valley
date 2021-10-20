from typing import Any, List, Optional, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from transformers import AutoModel, AutoConfig, get_scheduler

from src.utils.utils import get_logger

log = get_logger(__name__)


class ContrastiveModule(LightningModule):
    def __init__(
        self,
        arch: str,
        num_positives: int, num_negatives: int,
        temperature: float,
        optcfg: DictConfig,
        schcfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.num_negatives = num_negatives
        self.num_positives = num_positives
        self.schcfg = schcfg
        self.optcfg = optcfg
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(arch)
        self.transformer = AutoModel.from_config(config)
            # TODO
            # custom pooler? like SimCSE
            # config, add_pooling_layer=True
        # )
        pool_size = self.transformer.config.hidden_size
        self.projection = nn.Linear(pool_size, pool_size)

        # loss function
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.transformer(input_ids, attention_mask=attention_mask)

    def step(self, batch: Any):
        """K = 1 + #pos + #neg"""
        N, K, L = batch['input_ids'].shape
        pool_size = self.transformer.config.hidden_size
        assert K == 1 + self.num_negatives + self.num_positives
        input_ids = batch['input_ids'].view(N * K, L)
        attention_mask = batch['attention_mask'].view(N * K, L)
        pooled_embedding = self.forward(input_ids, attention_mask)
        pooled_embedding = pooled_embedding['pooler_output'].view(N, K, pool_size)
        # (N, ?, h)
        anchor, pos, neg = (
            pooled_embedding[:, 0:1],
            pooled_embedding[:, 1:self.num_positives+1],
            pooled_embedding[:, self.num_positives+1:]
        )
        # (N, K-1)
        cos_sim = torch.cat([
            self.cos_sim(anchor, pos) / self.temperature,
            self.cos_sim(anchor, neg) / self.temperature,
        ], dim=1)
        loss = self.criterion(
            cos_sim, torch.zeros(N).type_as(cos_sim).long()
        )
        return loss, cos_sim

    def step_end(self, output: tuple, phase: str):
        loss, cos_sim = output
        self.log(f"{phase}/step/loss", loss, logger=True,
                 # on_step if train; on_epoch if not train
                 on_step=phase == "train", on_epoch=phase != "train")

    def agg_epoch(self, outputs: EPOCH_OUTPUT, phase: str):
        loss = torch.stack([l for l, _ in outputs]).mean()
        self.log(f"{phase}/epoch/loss", loss)

        # (\sum N, K-1)
        cos_sim = torch.cat([
            cs for _, cs in outputs
        ], dim=0)

        # want 0th col cos sim higher than other
        diff = cos_sim[:, 0].mean() - cos_sim[:, 1:].mean()
        self.log(f"{phase}/epoch/cos_sim", diff, logger=True, prog_bar=True,
                 # on_step if train; on_epoch if not train
                 on_step=False, on_epoch=True)

    def training_step(self, batch: Dict, batch_idx: int):
        return self.step(batch)

    def training_step_end(self, outputs: tuple) -> tuple:
        self.epoch_end(outputs, "train")
        return outputs

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.step(batch)

    def validation_step_end(self, outputs: tuple) -> tuple:
        self.epoch_end(outputs, "valid")
        return outputs

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "valid")

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            tokenizer = self.trainer.datamodule.tokenizer
            self.transformer.resize_token_embeddings(len(tokenizer))

            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            effective_batch_size = (self.trainer.datamodule.batch_size *
                                    max(1, self.trainer.num_gpus) * self.trainer.accumulate_grad_batches)
            self.total_steps = int(
                (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.optcfg.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        optimizer = instantiate(
            self.optcfg, params=optimizer_parameters,
            _convert_="partial"
        )

        ret_dict = {'optimizer': optimizer}
        if self.schcfg:
            if self.schcfg.lr_scheduler == "linear_with_warmup":
                if self.schcfg.warmup_updates > 1.0:
                    warmup_steps = int(self.schcfg.warmup_updates)
                else:
                    warmup_steps = int(self.total_steps *
                                       self.schcfg.warmup_updates)
                log.info(
                    f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

                scheduler = get_scheduler(
                    "linear", optimizer=optimizer,
                    num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)

                scheduler = {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                ret_dict["lr_scheduler"] = scheduler
                # TODO : change
                ret_dict["monitor"] = "valid/epoch/loss"

        return ret_dict

