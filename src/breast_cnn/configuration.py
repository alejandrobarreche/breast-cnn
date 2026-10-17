"""Compose phase-one settings for scripts and notebooks using the same schema."""

from pathlib import Path

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


def load_phase_one_config(
    project_root: Path, overrides: list[str] | None = None
) -> tuple[DataConfig, PipelineConfig, EDAConfig]:
    """Resolve Hydra overrides; relative data/output paths use project_root."""
    with initialize_config_dir(
        version_base=None, config_dir=str(project_root.resolve() / "configs")
    ):
        config = compose(config_name="config", overrides=overrides or [])
    resolved = OmegaConf.to_container(config, resolve=True)
    return (
        DataConfig.model_validate(resolved["data"]),
        PipelineConfig.model_validate(resolved["pipeline"]),
        EDAConfig.model_validate(resolved["eda"]),
    )
