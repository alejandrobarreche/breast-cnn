"""Reproducible train/validation loaders using the supplied patient folds."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.metadata import split_train_validation
from breast_cnn.data.transforms import RandomAffine
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


class PatientBatchSampler(Sampler[list[int]]):
    """Batches made of whole patients: every slice of a patient lands in the same batch.

    Patients are shuffled each epoch with the loader's generator, so resuming a run
    restores the patient order together with the other random states.
    """

    def __init__(
        self, patient_ids: Sequence[str], patients_per_batch: int, generator: torch.Generator
    ) -> None:
        if patients_per_batch < 1:
            raise ValueError("patients_per_batch must be positive")
        self.groups = list(pd.Series(range(len(patient_ids))).groupby(list(patient_ids)).agg(list))
        self.patients_per_batch = patients_per_batch
        self.generator = generator

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for start in range(0, len(order), self.patients_per_batch):
            yield [
                i for g in order[start : start + self.patients_per_batch] for i in self.groups[g]
            ]

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.patients_per_batch)


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
    affine: RandomAffine | None = None,
    patients_per_batch: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Validate the index, seed RNGs and return training/validation loaders.

    Test is excluded before fold selection. Each loader has its own generator,
    so iterating validation does not change training's shuffle/worker seeds.
    Neither loader drops the last batch. New calls reset main-process RNGs.
    With `patients_per_batch`, training batches hold whole patients instead of
    `batch_size` shuffled slices; validation is unchanged.
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
        affine=affine,
    )
    validation_dataset = BreastDCEDataset(validation, project_root, path_aliases=path_aliases)
    seed_everything(seed)
    # Persistent workers avoid re-spawning processes every epoch (costly on Windows).
    common = {
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_generator = torch.Generator().manual_seed(seed)
    if patients_per_batch is None:
        batching = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
    else:
        batching = {
            "batch_sampler": PatientBatchSampler(
                train_dataset.samples["patient_id"].tolist(), patients_per_batch, train_generator
            )
        }
    return (
        DataLoader(train_dataset, generator=train_generator, **batching, **common),
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            **common,
        ),
    )
