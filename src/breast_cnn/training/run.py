"""Run one experiment over the configured internal folds and write runs/<id>/."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf

from breast_cnn.configuration import DataConfig, TrainConfig
from breast_cnn.data.metadata import load_samples
from breast_cnn.models import ModelConfig, load_model_config
from breast_cnn.training.loop import fit_fold, select_device
from breast_cnn.visualization.plots import plot_learning_curves, save_figure


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(root: Path, device: torch.device) -> dict:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "uncommitted_changes": None if status is None else bool(status),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("breast-cnn", "torch", "numpy", "pandas", "scikit-learn")
        },
    }


def _summary(fold_metrics: list[dict]) -> dict:
    """Mean and sample standard deviation across folds of the main metrics."""
    summary = {}
    for name in ("train_patient_auc", "train_validation_patient_auc_gap"):
        values = [m[name] for m in fold_metrics if m[name] is not None]
        summary[name] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
    for unit in ("slice", "patient"):
        for name in ("auc", "sensitivity", "specificity", "balanced_accuracy"):
            values = [m[unit][name] for m in fold_metrics if m[unit][name] is not None]
            summary[f"{unit}_{name}"] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
    return summary


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(root: Path, data: DataConfig, config: TrainConfig) -> Path:
    """Start a new run directory (never overwriting one) and train every configured fold."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = root / config.runs_dir / f"{stamp}_{config.experiment}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = load_model_config(root / config.model)
    OmegaConf.save(
        OmegaConf.create(
            {
                "data": data.model_dump(),
                "train": config.model_dump(),
                "model": model_config.model_dump(),
            }
        ),
        run_dir / "config.yaml",
    )
    _write_json({"commit": _git(root, "rev-parse", "HEAD")}, run_dir / "environment.json")
    return _train(root, run_dir, data, config, model_config)


def resume_experiment(root: Path, run_dir: Path) -> Path:
    """Continue an interrupted run with its saved configuration and the same commit."""
    if (run_dir / "metrics.json").exists():
        raise ValueError(f"{run_dir} is complete; a finished run is never rewritten")
    saved = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    started = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))["commit"]
    current = _git(root, "rev-parse", "HEAD")
    if started != current:
        raise ValueError(f"Run started at commit {started}, but HEAD is {current}")
    return _train(
        root,
        run_dir,
        DataConfig.model_validate(saved["data"]),
        TrainConfig.model_validate(saved["train"]),
        ModelConfig.model_validate(saved["model"]),
    )


def _train(
    root: Path, run_dir: Path, data: DataConfig, config: TrainConfig, model_config: ModelConfig
) -> Path:
    device = select_device()
    samples = load_samples(root, data.metadata_dir)
    results = []
    for fold in config.folds:
        result = fit_fold(root, data, config, model_config, samples, fold, device, run_dir)
        results.append(result)
        auc = result.metrics["patient"]["auc"]
        print(
            f"Fold {fold}: best epoch {result.metrics['best_epoch']}"
            f"/{result.metrics['epochs_run']}, validation patient AUC {auc}",
            flush=True,
        )

    history = pd.concat([r.history for r in results], ignore_index=True)
    history.to_csv(run_dir / "history.csv", index=False)
    pd.concat([r.predictions for r in results], ignore_index=True).to_csv(
        run_dir / "predictions_validation.csv", index=False
    )
    fold_metrics = [r.metrics for r in results]
    _write_json(
        {
            "experiment": config.experiment,
            "scope": "internal_validation_folds",
            "aggregation": "mean_probability_per_patient",
            "selection": {"best": "best_validation_patient_auc", "last": "last_epoch"}[
                config.epoch_selection
            ],
            "summary": _summary(fold_metrics),
            "folds": fold_metrics,
        },
        run_dir / "metrics.json",
    )
    peaks = [r.peak_memory_bytes for r in results if r.peak_memory_bytes is not None]
    _write_json(
        {
            **_environment(root, device),
            "batch_size": config.batch_size,
            "total_seconds": float(history["seconds"].sum()),
            "mean_epoch_seconds": float(history["seconds"].mean()),
            "peak_gpu_memory_bytes": max(peaks) if peaks else None,
        },
        run_dir / "environment.json",
    )
    selected = {m["fold"]: m["best_epoch"] for m in fold_metrics}
    figure = plot_learning_curves(history, config.experiment, selected)
    save_figure(figure, run_dir / "figures" / "learning_curves_validation.png")
    plt.close(figure)
    return run_dir
