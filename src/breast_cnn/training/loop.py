"""Explicit PyTorch training with patient-level model selection on internal folds."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from breast_cnn.configuration import DataConfig, TrainConfig
from breast_cnn.data.loaders import create_data_loaders, seed_everything
from breast_cnn.models import build_model, load_model_config
from breast_cnn.training.metrics import evaluate_predictions


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def positive_class_weight(labels: pd.Series) -> float:
    """pos_weight = N0 / N1 over the training slices of the current fold."""
    positives = int(labels.eq(1).sum())
    negatives = int(labels.eq(0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("The training split must contain both classes")
    return negatives / positives


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass of minibatch gradient descent; returns the slice-weighted mean loss."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.numel()
        count += labels.numel()
    return total / count


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, pd.DataFrame]:
    """Evaluate in eval mode; return the mean loss and one prediction row per slice."""
    model.eval()
    total, rows = 0.0, []
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        logits = model(images)
        total += criterion(logits, labels).item() * labels.numel()
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": batch["sample_id"],
                    "patient_id": batch["patient_id"],
                    "slice_index": batch["slice_index"].tolist(),
                    "label": labels.int().tolist(),
                    "logit": logits.float().cpu().tolist(),
                    "probability": torch.sigmoid(logits).float().cpu().tolist(),
                }
            )
        )
    predictions = pd.concat(rows, ignore_index=True)
    return total / len(predictions), predictions


@dataclass
class FoldResult:
    history: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict
    peak_memory_bytes: int | None


def fit_fold(
    root: Path,
    data: DataConfig,
    config: TrainConfig,
    samples: pd.DataFrame,
    fold: int,
    device: torch.device,
    weights_path: Path,
) -> FoldResult:
    """Train on four folds, validate on `fold` and keep the epoch with best patient AUC.

    Early stopping ends training after `early_stopping_patience` epochs without
    improvement. Selecting the epoch on the same fold makes its AUC slightly optimistic.
    """
    train_loader, validation_loader = create_data_loaders(
        samples,
        root,
        validation_fold=fold,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        path_aliases=data.image_path_aliases,
        horizontal_flip_probability=config.horizontal_flip_probability,
    )
    seed_everything(config.seed)
    model = build_model(load_model_config(root / config.model)).to(device)
    pos_weight = (
        positive_class_weight(train_loader.dataset.samples["pCR"]) if config.weighted_loss else None
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, best_auc, best_epoch, best_predictions = [], None, 0, None
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, predictions = predict(model, validation_loader, criterion, device)
        metrics = evaluate_predictions(predictions, config.threshold)
        patient_auc = metrics["patient"]["auc"]
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_slice_auc": metrics["slice"]["auc"],
                "validation_patient_auc": patient_auc,
                "seconds": time.perf_counter() - start,
            }
        )
        if best_predictions is None or (
            patient_auc is not None and (best_auc is None or patient_auc > best_auc)
        ):
            best_auc, best_epoch, best_predictions = patient_auc, epoch, predictions
            torch.save(model.state_dict(), weights_path)
        elif epoch - best_epoch >= config.early_stopping_patience:
            break

    best_predictions.insert(0, "fold", fold)
    return FoldResult(
        history=pd.DataFrame(history),
        predictions=best_predictions,
        metrics={
            "fold": fold,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "pos_weight": pos_weight,
            "training_slices": len(train_loader.dataset),
            "training_patients": int(train_loader.dataset.samples["patient_id"].nunique()),
            **evaluate_predictions(best_predictions, config.threshold),
        },
        peak_memory_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    )
