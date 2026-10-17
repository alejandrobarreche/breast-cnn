"""Compose settings for scripts and notebooks from configs/ using validated schemas."""

from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    validation_fold: int = Field(ge=0, le=4)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    seed: int = Field(ge=0, lt=2**63)
    horizontal_flip_probability: float = Field(ge=0, le=1)
    report_path: str


class EDAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seed: int = Field(ge=0)
    figures_dir: str
    report_path: str


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    experiment: str = Field(pattern=r"^[a-z0-9_]+$")
    model: str
    folds: list[int] = Field(min_length=1)
    seed: int = Field(ge=0, lt=2**63)
    epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    optimizer: Literal["adamw"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    horizontal_flip_probability: float = Field(ge=0, le=1)
    rotation_degrees: float = Field(ge=0, le=180)
    max_translation: float = Field(ge=0, lt=1)
    scale_range: list[float] = Field(min_length=2, max_length=2)
    weighted_loss: bool
    threshold: float = Field(gt=0, lt=1)
    runs_dir: str

    @field_validator("folds")
    @classmethod
    def check_folds(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value) or not set(value) <= set(range(5)):
            raise ValueError("folds must be distinct values in 0-4")
        return value


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metadata_dir: str
    image_path_aliases: dict[str, str]
    expected_image_size: list[int]

    @field_validator("expected_image_size")
    @classmethod
    def check_size(cls, value: list[int]) -> list[int]:
        if value != [256, 256]:
            raise ValueError("The DCE input contract requires [256,256]; resizing is not supported")
        return value


def _compose(project_root: Path, overrides: list[str] | None) -> dict:
    with initialize_config_dir(
        version_base=None, config_dir=str(project_root.resolve() / "configs")
    ):
        config = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_container(config, resolve=True)


def load_phase_one_config(
    project_root: Path, overrides: list[str] | None = None
) -> tuple[DataConfig, PipelineConfig, EDAConfig]:
    """Resolve Hydra overrides; relative data/output paths use project_root."""
    resolved = _compose(project_root, overrides)
    return (
        DataConfig.model_validate(resolved["data"]),
        PipelineConfig.model_validate(resolved["pipeline"]),
        EDAConfig.model_validate(resolved["eda"]),
    )


def load_train_config(
    project_root: Path, overrides: list[str] | None = None
) -> tuple[DataConfig, TrainConfig]:
    """Resolve the training protocol, e.g. overrides=["+experiment=baseline_weighted"]."""
    resolved = _compose(project_root, overrides)
    return DataConfig.model_validate(resolved["data"]), TrainConfig.model_validate(
        resolved["train"]
    )
