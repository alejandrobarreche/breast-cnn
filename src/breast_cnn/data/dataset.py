"""PyTorch access to explicitly selected rows of the source index."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from breast_cnn.data.metadata import resolve_image_path
from breast_cnn.data.transforms import (
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    load_dce_triplet,
)
from breast_cnn.data.validation import PHASE_PATH_COLUMNS, validate_sample_frame


class BreastDCEDataset(Dataset):
    """Read filtered rows, retaining identities and a scalar float32 target.

    Default collation yields image [B,3,256,256] and label [B], including B=1.
    Augmentation is opt-in and allowed only for an explicitly training dataset.
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        project_root: Path | str,
        *,
        path_aliases: Mapping[str, str] | None = None,
        training: bool = False,
        horizontal_flip_probability: float = 0.0,
        affine: RandomAffine | None = None,
    ) -> None:
        validate_sample_frame(samples)
        if training and not samples["split"].eq("train").all():
            raise ValueError("Training dataset may contain only public train samples")
        affine = affine or RandomAffine()
        if not training and (horizontal_flip_probability != 0 or not affine.is_identity):
            raise ValueError("Augmentation is forbidden for validation/test/inference")
        self.samples = samples.copy(deep=True).reset_index(drop=True)
        self.project_root = Path(project_root).expanduser().resolve()
        self.path_aliases = dict(path_aliases or {})
        self.transform = Compose((affine, RandomHorizontalFlip(horizontal_flip_probability)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        try:
            paths = [
                resolve_image_path(row[column], self.project_root, self.path_aliases)
                for column in PHASE_PATH_COLUMNS.values()
            ]
            image = self.transform(load_dce_triplet(*paths))
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"Sample {row['sample_id']}: {error}") from error
        return {
            "image": image,
            "label": torch.tensor(float(row["pCR"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "slice_index": int(row["slice_index"]),
        }
