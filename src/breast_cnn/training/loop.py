"""Explicit PyTorch training with patient-level model selection on internal folds."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
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
    """pos_weight = N0 / N1 over the training units (slices or patients) of the current fold."""
    positives = int(labels.eq(1).sum())
    negatives = int(labels.eq(0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("The training split must contain both classes")
    return negatives / positives


@dataclass(frozen=True)
class Objective:
    """BCEWithLogitsLoss applied per slice, or per patient after pooling slice logits.

    Patient pooling: "mean" averages slice logits (every slice contributes to the
    gradient); "max" keeps the highest logit (only the most pCR-like slice receives
    gradient: a patient is predicted pCR as soon as one slice looks like pCR).
    """

    criterion: nn.BCEWithLogitsLoss
    unit: Literal["slice", "patient"]
    pooling: Literal["mean", "max"]

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        patient_ids: list[str],
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, int]:
        """Return the mean loss and the number of units (slices or patients) it averages."""
        if self.unit == "patient":
            codes, _ = pd.factorize(pd.Series(patient_ids))
            index = torch.as_tensor(codes, device=logits.device)
            count = int(codes.max()) + 1
            reduce = "mean" if self.pooling == "mean" else "amax"
            logits = torch.zeros(count, device=logits.device, dtype=logits.dtype).scatter_reduce(
                0, index, logits, reduce=reduce, include_self=False
            )
            labels = torch.zeros(count, device=labels.device, dtype=labels.dtype).scatter_reduce(
                0, index, labels, reduce="amax", include_self=False
            )
        targets = labels * (1 - label_smoothing) + label_smoothing / 2
        return self.criterion(logits, targets), labels.numel()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    objective: Objective,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.0,
    ema: AveragedModel | None = None,
) -> tuple[float, pd.DataFrame]:
    """One pass of minibatch gradient descent; returns the mean loss and running predictions.

    The predictions are those produced while training (dropout and augmentation active,
    weights changing within the epoch). They cost nothing extra and give a slightly
    pessimistic estimate of the training AUC, as Keras-style training metrics do.
    With `ema`, the moving average of the weights is updated after every optimizer step.
    """
    model.train()
    total, units, rows = 0.0, 0, []
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, count = objective(logits, labels, batch["patient_id"], label_smoothing)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)
        total += loss.item() * count
        units += count
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": batch["patient_id"],
                    "label": labels.int().tolist(),
                    "probability": torch.sigmoid(logits.detach()).float().cpu().tolist(),
                }
            )
        )
    return total / units, pd.concat(rows, ignore_index=True)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, objective: Objective, device: torch.device
) -> tuple[float, pd.DataFrame]:
    """Evaluate in eval mode; return the loss on hard labels and one row per slice."""
    model.eval()
    logits, labels, rows = [], [], []
    for batch in loader:
        batch_logits = model(batch["image"].to(device))
        logits.append(batch_logits)
        labels.append(batch["label"].to(device))
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": batch["sample_id"],
                    "patient_id": batch["patient_id"],
                    "slice_index": batch["slice_index"].tolist(),
                    "label": batch["label"].int().tolist(),
                    "logit": batch_logits.float().cpu().tolist(),
                    "probability": torch.sigmoid(batch_logits).float().cpu().tolist(),
                }
            )
        )
    predictions = pd.concat(rows, ignore_index=True)
    loss, _ = objective(torch.cat(logits), torch.cat(labels), predictions["patient_id"].tolist())
    return loss.item(), predictions


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
    paired_fold: int | None = None,
) -> FoldResult:
    """Train on four folds, validate on `fold` and keep one epoch (reported as best_epoch).

    With `paired_fold` (nested runs) the model trains on the other three folds and validates
    on both; it is named `<fold>-<paired_fold>` and its predictions keep each slice's own fold
    in `fold` and the model name in `model`.

    With `epoch_selection: best` the kept epoch has the best patient AUC and early stopping
    ends training after `early_stopping_patience` epochs without improvement; selecting the
    epoch on the same fold makes its AUC optimistic. With `last`, exactly `epochs` epochs
    are trained and the last one is kept, so the fold is not used to choose the epoch.

    After every epoch the full state is saved to `checkpoint_fold<k>.pt`; if that file
    exists, training continues from it. With `num_workers=0` the continuation is
    identical to an uninterrupted run; with workers, augmentation draws differ.

    With `lr_schedule: cosine` the learning rate decays from `learning_rate` to 0 over
    `epochs` epochs. With `ema_decay`, validation and the saved weights use the exponential
    moving average of the weights (BatchNorm statistics included); the training AUC still
    comes from the weights being optimised.
    """
    validation_folds = fold if paired_fold is None else (fold, paired_fold)
    name = str(fold) if paired_fold is None else f"{fold}-{paired_fold}"
    train_loader, validation_loader = create_data_loaders(
        samples,
        root,
        validation_fold=validation_folds,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        path_aliases=data.image_path_aliases,
        horizontal_flip_probability=config.horizontal_flip_probability,
        affine=RandomAffine(
            config.rotation_degrees, config.max_translation, tuple(config.scale_range)
        ),
        patients_per_batch=config.patients_per_batch if config.loss_unit == "patient" else None,
        train_slices_per_patient=config.train_slices_per_patient,
    )
    seed_everything(config.seed)
    model = build_model(model_config).to(device)
    training_units = train_loader.dataset.samples
    if config.loss_unit == "patient":
        training_units = training_units.drop_duplicates("patient_id")
    pos_weight = positive_class_weight(training_units["pCR"]) if config.weighted_loss else None
    objective = Objective(
        nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
        ),
        config.loss_unit,
        config.patient_pooling,
    )
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.lr_schedule == "cosine"
        else None
    )
    ema = (
        AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
        if config.ema_decay is not None
        else None
    )
    evaluated = model if ema is None else ema.module
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = run_dir / f"checkpoint_fold{name}.pt"
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
        scheduler_state, ema_state = saved.pop("scheduler"), saved.pop("ema")
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if ema is not None:
            ema.load_state_dict(ema_state)
        _restore_rng_state(saved.pop("rng"), train_loader)
        state = saved
        print(f"Fold {name}: resuming after epoch {state['epoch']}", flush=True)

    epoch = state["epoch"]
    while not state["finished"]:
        epoch += 1
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_predictions = train_one_epoch(
            model, train_loader, objective, optimizer, device, config.label_smoothing, ema
        )
        if scheduler is not None:
            scheduler.step()
        train_auc = evaluate_predictions(train_predictions, config.threshold)["patient"]["auc"]
        validation_loss, predictions = predict(evaluated, validation_loader, objective, device)
        metrics = evaluate_predictions(predictions, config.threshold)
        patient_auc = metrics["patient"]["auc"]
        row = {
            "fold": fold if paired_fold is None else name,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_patient_auc": train_auc,
            "validation_slice_auc": metrics["slice"]["auc"],
            "validation_patient_auc": patient_auc,
            "learning_rate": learning_rate,
            "seconds": time.perf_counter() - start,
        }
        state["history"].append(row)
        improved = (
            config.epoch_selection == "last"
            or state["best_predictions"] is None
            or (
                patient_auc is not None
                and (state["best_auc"] is None or patient_auc > state["best_auc"])
            )
        )
        if improved:
            state.update(
                best_auc=patient_auc,
                best_epoch=epoch,
                best_train_auc=train_auc,
                best_predictions=predictions.to_dict("list"),
            )
            torch.save(evaluated.state_dict(), run_dir / f"model_fold{name}.pt")
        state["epoch"] = epoch
        state["finished"] = epoch >= config.epochs or (
            config.epoch_selection == "best"
            and epoch - state["best_epoch"] >= config.early_stopping_patience
        )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            state["peak_memory_bytes"] = max(peak, state["peak_memory_bytes"] or 0)
        torch.save(
            {
                **state,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "ema": None if ema is None else ema.state_dict(),
                "rng": _rng_state(train_loader),
            },
            checkpoint_path,
        )
        print(_progress_line(row, config.epochs, improved), flush=True)

    best_predictions = pd.DataFrame(state["best_predictions"])
    if paired_fold is None:
        best_predictions.insert(0, "fold", fold)
    else:
        folds = validation_loader.dataset.samples.set_index("sample_id")["fold"]
        best_predictions.insert(0, "fold", best_predictions["sample_id"].map(folds).astype(int))
        best_predictions.insert(1, "model", name)
    best_auc, best_train_auc = state["best_auc"], state["best_train_auc"]
    return FoldResult(
        history=pd.DataFrame(state["history"]),
        predictions=best_predictions,
        metrics={
            "fold": fold if paired_fold is None else name,
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
