"""Reproducible train/validation loaders using the supplied patient folds."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.metadata import split_train_validation
from breast_cnn.data.validation import validate_sample_frame


def seed_everything(seed: int) -> None:
    """Seed local RNGs; this does not promise determinism across hardware."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """PyTorch seeds each worker; derive Python/NumPy seeds from that seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_data_loaders(
    samples: pd.DataFrame,
    project_root: Path | str,
    *,
    validation_fold: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    path_aliases: Mapping[str, str] | None = None,
    horizontal_flip_probability: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    """Validate the index, seed RNGs and return training/validation loaders.

    Test is excluded before fold selection. Each loader has its own generator,
    so iterating validation does not change training's shuffle/worker seeds.
    Neither loader drops the last batch. New calls reset main-process RNGs.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    validate_sample_frame(samples)
    training, validation = split_train_validation(samples, validation_fold)
    if training.empty or validation.empty:
        raise ValueError("Training and validation must both contain samples")
    train_dataset = BreastDCEDataset(
        training,
        project_root,
        path_aliases=path_aliases,
        training=True,
        horizontal_flip_probability=horizontal_flip_probability,
    )
    validation_dataset = BreastDCEDataset(validation, project_root, path_aliases=path_aliases)
    seed_everything(seed)
    # Persistent workers avoid re-spawning processes every epoch (costly on Windows).
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            **common,
        ),
        DataLoader(
            validation_dataset,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            **common,
        ),
    )
