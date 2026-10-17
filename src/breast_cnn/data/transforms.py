"""Shared deterministic DCE preprocessing and optional joint geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


@dataclass(frozen=True)
class RandomAffine:
    """Random rotation, translation and isotropic scaling with one grid for all phases.

    A single sampling grid keeps PRE, EARLY and LATE aligned, so `EARLY − PRE` remains
    a pixel-wise enhancement map. Bilinear interpolation; uncovered pixels become 0.
    The identity configuration returns the input unchanged.
    """

    rotation_degrees: float = 0.0
    max_translation: float = 0.0
    scale_range: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rotation_degrees <= 180.0:
            raise ValueError("rotation_degrees must be in [0,180]")
        if not 0.0 <= self.max_translation < 1.0:
            raise ValueError("max_translation must be a fraction in [0,1)")
        low, high = self.scale_range
        if not 0.0 < low <= high:
            raise ValueError("scale_range must satisfy 0 < low <= high")

    @property
    def is_identity(self) -> bool:
        return (
            self.rotation_degrees == 0
            and self.max_translation == 0
            and self.scale_range == (1.0, 1.0)
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return image
        angle = math.radians((torch.rand(()).item() * 2 - 1) * self.rotation_degrees)
        low, high = self.scale_range
        scale = low + torch.rand(()).item() * (high - low)
        # affine_grid maps output to input coordinates in [-1, 1]; a shift of 2·t is t of the width.
        tx, ty = ((torch.rand(2) * 2 - 1) * 2 * self.max_translation).tolist()
        cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=image.dtype)
        grid = F.affine_grid(theta[None], [1, *image.shape], align_corners=False)
        return F.grid_sample(
            image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]


@dataclass(frozen=True)
class Compose:
    transforms: tuple

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image
