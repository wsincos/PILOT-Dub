# src/utils/logging.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def is_rank_zero() -> bool:
    """
    Best-effort rank0 detection for DDP / SLURM.
    If env vars are not present, assume single-process (rank0).
    """
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v) == 0
        except ValueError:
            # Unexpected value -> don't block logging
            return True
    return True


class RankZeroFilter(logging.Filter):
    """Allow logs only from rank0 process."""
    def filter(self, record: logging.LogRecord) -> bool:
        return is_rank_zero()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank_zero_only: bool = True,
    fmt: str = "[%(asctime)s][%(levelname)s][%(name)s][%(filename)s:%(lineno)d] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Initialize logging ONCE at program entry.
    This configures the ROOT logger with console/file handlers.

    After calling this, anywhere in the code you can do:
        logger = logging.getLogger(__name__)
        logger.info("...")

    Args:
        level: log level for root logger (INFO/DEBUG/...)
        log_file: optional path to a log file
        rank_zero_only: if True, only rank0 prints/writes logs
        fmt/datefmt: logging formatter options
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    # (Hydra/Lightning/interactive envs sometimes pre-configure logging)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    rank_filter = RankZeroFilter() if rank_zero_only else None

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    if rank_filter is not None:
        ch.addFilter(rank_filter)
    root.addHandler(ch)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        if rank_filter is not None:
            fh.addFilter(rank_filter)
        root.addHandler(fh)

    # Make warnings go through logging (optional but nice)

    # lightning
    for lib in ["lightning", "pytorch_lightning", "lightning.fabric", "torch.distributed"]:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(level) # 强制设为 INFO，保证内容能流向 FileHandler
        lib_logger.propagate = True
        # 清理它们自己的 Handler，强制使用 Root Handler
        for h in list(lib_logger.handlers):
            lib_logger.removeHandler(h)

    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """
    Lightweight accessor. Does NOT add handlers.
    Assumes setup_logging() has been called once.
    """
    return logging.getLogger(name)
