import os
from typing import List, Tuple
from pathlib import Path
import hydra
import pyrootutils
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from src import utils
from src.utils.utils import replace_cfg_targets

log = utils.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.ckpt_path
    
    replace_cfg_targets(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    torch.save(cfg, os.path.join(cfg.paths.output_dir, "cfg.pt"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, input_dims=datamodule.input_dims, output_dir=cfg.paths.output_dir)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    
    if cfg.training:
        log.info("Starting evaluating training set!")
        datamodule.setup()
        trainer.test(model=model, dataloaders=datamodule.train_dataloader(), ckpt_path=cfg.ckpt_path)
    
    if cfg.validate:
        log.info("Starting validation!")
        datamodule.setup()
        trainer.test(model=model, dataloaders=datamodule.val_dataloader(), ckpt_path=cfg.ckpt_path)

    if cfg.test:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    
    evaluate(cfg)

if __name__ == "__main__":
    main()
