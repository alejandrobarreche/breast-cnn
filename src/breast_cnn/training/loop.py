"""Explicit PyTorch training with patient-level model selection on internal folds."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from breast_cnn.configuration import DataConfig, TrainConfig
from breast_cnn.data.loaders import create_data_loaders, seed_everything
from breast_cnn.data.transforms import RandomAffine
from breast_cnn.models import ModelConfig, build_model
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
) -> tuple[float, pd.DataFrame]:
    """One pass of minibatch gradient descent; returns the mean loss and running predictions.

    The predictions are those produced while training (dropout and augmentation active,
    weights changing within the epoch). They cost nothing extra and give a slightly
    pessimistic estimate of the training AUC, as Keras-style training metrics do.
    """
    model.train()
    total, rows = 0.0, []
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.numel()
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": batch["patient_id"],
                    "label": labels.int().tolist(),
                    "probability": torch.sigmoid(logits.detach()).float().cpu().tolist(),
                }
            )
        )
    predictions = pd.concat(rows, ignore_index=True)
    return total / len(predictions), predictions


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


def _rng_state(train_loader: DataLoader) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": train_loader.generator.get_state(),
    }


def _restore_rng_state(state: dict, train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["loader"])


def fit_fold(
    root: Path,
    data: DataConfig,
    config: TrainConfig,
    model_config: ModelConfig,
    samples: pd.DataFrame,
    fold: int,
    device: torch.device,
    run_dir: Path,
) -> FoldResult:
    """Train on four folds, validate on `fold` and keep the epoch with best patient AUC.

    Early stopping ends training after `early_stopping_patience` epochs without
    improvement. Selecting the epoch on the same fold makes its AUC slightly optimistic.

    After every epoch the full state is saved to `checkpoint_fold<k>.pt`; if that file
    exists, training continues from it. With `num_workers=0` the continuation is
    identical to an uninterrupted run; with workers, augmentation draws differ.
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
        affine=RandomAffine(
            config.rotation_degrees, config.max_translation, tuple(config.scale_range)
        ),
    )
    seed_everything(config.seed)
    model = build_model(model_config).to(device)
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

    checkpoint_path = run_dir / f"checkpoint_fold{fold}.pt"
    state = {
        "epoch": 0,
        "finished": False,
        "history": [],
        "best_auc": None,
        "best_epoch": 0,
        "best_train_auc": None,
        "best_predictions": None,
        "peak_memory_bytes": None,
    }
    if checkpoint_path.exists():
        # CPU first: RNG states must stay CPU byte tensors; load_state_dict moves weights.
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        _restore_rng_state(saved.pop("rng"), train_loader)
        state = saved
        print(f"Fold {fold}: resuming after epoch {state['epoch']}", flush=True)

    epoch = state["epoch"]
    while not state["finished"]:
        epoch += 1
        start = time.perf_counter()
        train_loss, train_predictions = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_auc = evaluate_predictions(train_predictions, config.threshold)["patient"]["auc"]
        validation_loss, predictions = predict(model, validation_loader, criterion, device)
        metrics = evaluate_predictions(predictions, config.threshold)
        patient_auc = metrics["patient"]["auc"]
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_patient_auc": train_auc,
            "validation_slice_auc": metrics["slice"]["auc"],
            "validation_patient_auc": patient_auc,
            "seconds": time.perf_counter() - start,
        }
        state["history"].append(row)
        improved = state["best_predictions"] is None or (
            patient_auc is not None
            and (state["best_auc"] is None or patient_auc > state["best_auc"])
        )
        if improved:
            state.update(
                best_auc=patient_auc,
                best_epoch=epoch,
                best_train_auc=train_auc,
                best_predictions=predictions.to_dict("list"),
            )
            torch.save(model.state_dict(), run_dir / f"model_fold{fold}.pt")
        state["epoch"] = epoch
        state["finished"] = (
            epoch >= config.epochs or epoch - state["best_epoch"] >= config.early_stopping_patience
        )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            state["peak_memory_bytes"] = max(peak, state["peak_memory_bytes"] or 0)
        torch.save(
            {
                **state,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": _rng_state(train_loader),
            },
            checkpoint_path,
        )
        print(_progress_line(row, config.epochs, improved), flush=True)

    best_predictions = pd.DataFrame(state["best_predictions"])
    best_predictions.insert(0, "fold", fold)
    best_auc, best_train_auc = state["best_auc"], state["best_train_auc"]
    return FoldResult(
        history=pd.DataFrame(state["history"]),
        predictions=best_predictions,
        metrics={
            "fold": fold,
            "best_epoch": state["best_epoch"],
            "epochs_run": len(state["history"]),
            "pos_weight": pos_weight,
            "train_patient_auc": best_train_auc,
            "train_validation_patient_auc_gap": (
                None if best_train_auc is None or best_auc is None else best_train_auc - best_auc
            ),
            "training_slices": len(train_loader.dataset),
            "training_patients": int(train_loader.dataset.samples["patient_id"].nunique()),
            **evaluate_predictions(best_predictions, config.threshold),
        },
        peak_memory_bytes=state["peak_memory_bytes"],
    )


def _format(value: float | None) -> str:
    return "  —  " if value is None else f"{value:.3f}"


def _progress_line(row: dict, epochs: int, improved: bool) -> str:
    return (
        f"Fold {row['fold']} · epoch {row['epoch']:>2}/{epochs} · "
        f"loss train {_format(row['train_loss'])} val {_format(row['validation_loss'])} · "
        f"patient AUC train {_format(row['train_patient_auc'])} "
        f"val {_format(row['validation_patient_auc'])} · "
        f"{row['seconds']:.0f} s{'  *best' if improved else ''}"
    )
