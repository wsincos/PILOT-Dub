import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import logging
import re
from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightning.datamodule import VoiceCraftDubDataModule
from src.lightning.module import VoiceCraftDubLightningModule
from src.utils.logging import setup_logging


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if cb_conf is None:
            continue
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if lg_conf is None:
            continue
        if "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def _select_resume_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    logger = logging.getLogger(__name__)
    ckpt_files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and os.path.isfile(os.path.join(ckpt_dir, f))
    ]
    if not ckpt_files:
        return None

    valid_ckpt_files = []
    for path in ckpt_files:
        size = os.path.getsize(path)
        if size <= 0:
            logger.warning("Skipping empty checkpoint in resume dir: %s", path)
            continue
        valid_ckpt_files.append(path)
    if not valid_ckpt_files:
        return None

    def _extract_step(path: str) -> Optional[int]:
        match = re.search(r"step[=_](\d+)", os.path.basename(path))
        return int(match.group(1)) if match else None

    def _sort_key(path: str):
        step = _extract_step(path)
        mtime = os.path.getmtime(path)
        return (step if step is not None else -1, mtime)

    return max(valid_ckpt_files, key=_sort_key)


@hydra.main(
    config_path="../configs",
    config_name="v19_strong_real_time_acoustic_interface_formal",
    version_base=None,
)
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_dir = HydraConfig.get().runtime.output_dir
    log_file = os.path.join(run_dir, "train.log")

    setup_logging(level=logging.INFO, log_file=log_file, rank_zero_only=True)
    logger = logging.getLogger(__name__)

    logger.info("===== Training Entry =====")
    logger.info("Run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info("Instantiating DataModule...")
    datamodule: L.LightningDataModule = VoiceCraftDubDataModule(cfg)

    logger.info("Instantiating LightningModule...")
    model: L.LightningModule = VoiceCraftDubLightningModule(cfg)

    logger.info("Instantiating Callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating Loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    logger.info("Instantiating Trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    logger.info("Starting training!")
    if cfg.get("ckpt_dir", None) is not None:
        ckpt_path = _select_resume_checkpoint(cfg.ckpt_dir)
    else:
        ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
