from typing import List, Optional

import hydra
import os
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)

def get_pl_logger(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
                while True:
                    try:
                        # sometimes fail for unknown reason
                        print(logger.experiment)
                        break
                    except BaseException:
                        pass

                if "wandb" in lg_conf["_target_"]:
                    id = "offline"
                    if not cfg.debug:
                        # will upload this run to cloud
                        log.info(f"wandb url in {logger.experiment.url}")
                        # get id from x-y-id
                        id = logger.experiment.name.rsplit('-', 1)[1]
                        cfg.callback.model_checkpoint.dirpath = os.path.join(
                            cfg.callback.model_checkpoint.dirpath, id
                        )
                    # if debug, not saving checkpoint at all
                    # since del in `touch`
    return loggers

def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning module
    log.info(f"Instantiating model <{config.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.module, optcfg=config.module.optim,
        schcfg=getattr(config.module, "scheduler", None),
        _recursive_=False
     )

    # Init lightning loggers
     logger: List[LightningLoggerBase] = get_pl_logger(config)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the module
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate module on test set, using the best module achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
