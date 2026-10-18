"""Predictions of the fold models of a finished run and their average."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from breast_cnn.configuration import DataConfig, TrainConfig
from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.models import ModelConfig, build_model


def load_fold_models(run_dir: Path, device: torch.device) -> dict[int, nn.Module]:
    """Build the run's architecture and load the saved weights of every configured fold."""
    saved = OmegaConf.to_container(OmegaConf.load(Path(run_dir) / "config.yaml"), resolve=True)
    model_config = ModelConfig.model_validate(saved["model"])
    models = {}
    for fold in TrainConfig.model_validate(saved["train"]).folds:
        model = build_model(model_config)
        weights = torch.load(Path(run_dir) / f"model_fold{fold}.pt", map_location="cpu")
        model.load_state_dict(weights)
        models[fold] = model.to(device).eval()
    return models


@torch.no_grad()
def predict_ensemble(
    models: dict[int, nn.Module],
    samples: pd.DataFrame,
    root: Path,
    data: DataConfig,
    device: torch.device,
    batch_size: int = 64,
) -> pd.DataFrame:
    """One row per slice: the probability of each fold model and their mean (`probability`).

    The mean of probabilities matches the per-patient aggregation used in training. Each
    image is read once and passed through every model, without augmentation.
    """
    dataset = BreastDCEDataset(samples, root, path_aliases=data.image_path_aliases)
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = batch["image"].to(device)
        frame = pd.DataFrame(
            {
                "sample_id": batch["sample_id"],
                "patient_id": batch["patient_id"],
                "slice_index": batch["slice_index"].tolist(),
                "label": batch["label"].int().tolist(),
            }
        )
        for fold, model in models.items():
            frame[f"probability_fold{fold}"] = torch.sigmoid(model(images)).float().cpu().tolist()
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    folds = [f"probability_fold{fold}" for fold in models]
    predictions["probability"] = predictions[folds].mean(axis=1)
    return predictions


class CalibratedEnsemble(nn.Module):
    """The fold models and their Platt maps as one network, for export to the web application.

    Input: float32 [B, 3, 256, 256] in [0, 1], channels PRE, EARLY, LATE (value / 255).
    Outputs: probability of each fold model [B, K], their mean [B] and the mean of the
    calibrated probabilities [B], the score compared with the threshold.
    """

    def __init__(self, models: dict[int, nn.Module], maps: dict[int, dict]) -> None:
        super().__init__()
        folds = sorted(models)
        self.models = nn.ModuleList(models[fold] for fold in folds)
        self.register_buffer("slopes", torch.tensor([maps[f]["slope"] for f in folds]))
        self.register_buffer("intercepts", torch.tensor([maps[f]["intercept"] for f in folds]))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.sigmoid(torch.stack([m(images) for m in self.models], dim=1))
        # Same clipping as calibration.apply_platt, so both give the same calibrated value.
        clipped = probabilities.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(clipped / (1 - clipped))
        calibrated = torch.sigmoid(self.slopes * logits + self.intercepts)
        return probabilities, probabilities.mean(dim=1), calibrated.mean(dim=1)
