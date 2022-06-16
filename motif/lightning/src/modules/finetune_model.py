from typing import Any, List, Optional, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn.modules.container import ModuleDict
from transformers import get_scheduler, AutoModelForSequenceClassification
from torchmetrics import (
    MetricCollection, Accuracy,
    Recall, Precision, F1, AUROC
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from src.utils.utils import get_logger

log = get_logger(__name__)


class FineTuneModule(LightningModule):
    def __init__(
            self,
            arch: str,
            optcfg: DictConfig,
            arch_ckpt: Optional[str] = None,
            schcfg: Optional[DictConfig] = None,
            **kwargs,
    ):
        super().__init__()

        self.schcfg = schcfg
        self.optcfg = optcfg
        self.save_hyperparameters()

        if arch_ckpt:
            arch = arch_ckpt
        self.transformer = AutoModelForSequenceClassification.from_pretrained(arch, num_labels=7)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # metrics
        mc = MetricCollection({
            "accuracy": Accuracy(threshold=0.0),
            "recall": Recall(threshold=0.0, num_classes=7, average='macro'),
            "precision": Precision(threshold=0.0, num_classes=7, average='macro'),
            "f1": F1(threshold=0.0, num_classes=7, average='macro'),
            "macro_auc": AUROC(num_classes=7, average='macro'),
            # "weighted_auc": AUROC(num_classes=7, average='weighted')
        })
        self.metrics: ModuleDict[str, MetricCollection] = ModuleDict({
            f"{phase}_metric": mc.clone()
            for phase in ["train", "valid", "test"]
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.transformer(input_ids, attention_mask=attention_mask)
        # (N, K)
        return output['logits']
    
    def step(self, batch: Any, batch_idx: int):
        if self.do_augumentation:
            text = batch.pop('text')
            batch.update(
                self.tokenizer(text, padding="max_length", truncation=True, return_tensors='pt')
            )
            batch = self.transfer_batch_to_device(batch, device=self.device, dataloader_idx=batch_idx)
        # (N, K)
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['label'])
        prob = logits.softmax(dim=1)
        return loss, prob.detach()

    def step_end(self, output: tuple, phase: str):
        loss = output['loss']
        self.log(f"{phase}/step/loss", loss, logger=True,
                 # on_step if train; on_epoch if not train
                 on_step=phase == "train", on_epoch=phase != "train")
        metrics = self.metrics[f"{phase}_metric"]
        metrics(output['prob'], output['label'])

    def agg_epoch(self, outputs: EPOCH_OUTPUT, phase: str):
        loss = torch.stack([o['loss'] for o in outputs]).mean()
        self.log(f"{phase}/epoch/loss", loss)
        # (N, K)
        probs = torch.cat([o['prob'] for o in outputs])
        # (N, )
        labels = torch.cat([o['label'] for o in outputs])
        metrics = self.metrics[f"{phase}_metric"]

        y_true = labels.detach().cpu().numpy()
        y_prob = probs.detach().cpu().numpy()
        macro_auc = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovo")
        weight_auc = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovo")
        self.log(f"{phase}/epoch/scikit-macro-auc", macro_auc, logger=True, prog_bar=True,
                     # on_step if train; on_epoch if not train
                     on_step=False, on_epoch=True)
        self.log(f"{phase}/epoch/scikit-weight-auc", weight_auc, logger=True, prog_bar=True,
                     # on_step if train; on_epoch if not train
                     on_step=False, on_epoch=True)

        for metric_name, metric in metrics.items():
            self.log(f"{phase}/epoch/{metric_name}", metric.compute(), logger=True, prog_bar=True,
                     # on_step if train; on_epoch if not train
                     on_step=False, on_epoch=True)

    def training_step(self, batch: Dict, batch_idx: int):
        loss, prob = self.step(batch, batch_idx)
        return {"loss": loss, "prob": prob, 'label': batch['label']}

    def training_step_end(self, outputs: tuple) -> tuple:
        self.step_end(outputs, "train")
        return outputs

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        loss, prob = self.step(batch, batch_idx)
        return {"loss": loss, "prob": prob, 'label': batch['label']}

    def validation_step_end(self, outputs: tuple) -> tuple:
        self.step_end(outputs, "valid")
        return outputs

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.agg_epoch(outputs, "valid")

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.tokenizer = self.trainer.datamodule.tokenizer
            self.transformer.resize_token_embeddings(len(self.tokenizer))

            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            effective_batch_size = (self.trainer.datamodule.batch_size *
                                    max(1, self.trainer.num_gpus) * self.trainer.accumulate_grad_batches)
            self.total_steps = int(
                (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))
            
            self.do_augumentation = self.trainer.datamodule.aug_list is not None

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
                # return [optimizer], [scheduler]
                ret_dict["monitor"] = "valid/epoch/loss"
            else:
                raise NotImplementedError

        return ret_dict
