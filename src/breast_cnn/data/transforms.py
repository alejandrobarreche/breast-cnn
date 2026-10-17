"""Shared deterministic DCE preprocessing and optional joint geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PHASES = ("PRE", "EARLY", "LATE")
IMAGE_SIZE = (256, 256)


def load_dce_triplet(pre: Path, early: Path, late: Path) -> torch.Tensor:
    """Decode PRE/EARLY/LATE to float32 [3,256,256] in [0,1].

    No resize, mode conversion or phase-wise normalization is performed.
    This is also the deterministic entry point for future inference.
    """
    paths = (pre, early, late)
    if len({Path(path).resolve() for path in paths}) != 3:
        raise ValueError("PRE, EARLY and LATE must reference three distinct files")
    channels = []
    for phase, path in zip(PHASES, paths, strict=True):
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != IMAGE_SIZE:
                    raise ValueError(
                        f"Expected 8-bit grayscale PNG {IMAGE_SIZE}, found "
                        f"{image.format}, mode={image.mode}, size={image.size}"
                    )
                image.load()
                array = np.array(image)
                if array.dtype != np.uint8 or array.shape != IMAGE_SIZE:
                    raise ValueError(
                        f"Expected uint8 {IMAGE_SIZE}, found {array.dtype} {array.shape}"
                    )
                channels.append(array)
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid {phase} image {path}: {error}") from error
    return torch.from_numpy(np.stack(channels)).to(torch.float32).div_(255.0)


@dataclass(frozen=True)
class RandomHorizontalFlip:
    """Flip the width axis once for all temporal channels; never resample phases."""

    probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Horizontal flip probability must be in [0,1]")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.probability > 0 and torch.rand(()).item() < self.probability:
            return image.flip(-1)
        return image
