"""Reproduce train EDA and a real-loader check without any model training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from breast_cnn.configuration import DataConfig, EDAConfig, PipelineConfig
from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.exploration import explore_train, select_examples
from breast_cnn.data.loaders import create_data_loaders
from breast_cnn.data.metadata import load_patients, load_samples, summarize_folds
from breast_cnn.visualization.plots import plot_dce_phases, plot_train_overview, save_figure


def _provenance(root: Path, data: DataConfig) -> dict[str, Any]:
    directory = root / data.metadata_dir
    return {
        "metadata_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("samples.csv", "patients.csv", "statistics.json", "excluded_patients.csv")
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "breast-cnn",
                "torch",
                "numpy",
                "pandas",
                "pillow",
                "matplotlib",
                "hydra-core",
            )
        },
        "data_config": data.model_dump(),
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_train_eda(root: Path, data: DataConfig, config: EDAConfig) -> dict[str, Any]:
    """Save train-only tables and reproducibly selected, unaugmented examples."""
    samples = load_samples(root, data.metadata_dir)
    patients = load_patients(root, data.metadata_dir)
    exploration = explore_train(samples, patients)
    folds = summarize_folds(samples)
    examples = select_examples(exploration, config.seed)
    dataset = BreastDCEDataset(examples, root, path_aliases=data.image_path_aliases)
    figures = plot_train_overview(exploration)
    image_statistics = []
    for index, row in examples.iterrows():
        image = dataset[index]["image"]
        sample_slug = re.sub(r"[^a-z0-9]+", "_", row["sample_id"].lower()).strip("_")
        name = f"dce_phases_{sample_slug}"
        figures[name] = plot_dce_phases(
            image, sample_id=row["sample_id"], cohort=row["cohort"], label=int(row["pCR"])
        )
        image_statistics.append(
            {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "cohort": row["cohort"],
                "pCR": int(row["pCR"]),
                "split": "train",
                "phase_means": dict(
                    zip(("PRE", "EARLY", "LATE"), image.mean((1, 2)).tolist(), strict=True)
                ),
                "enhancement_mean": float((image[1] - image[0]).mean()),
                "enhancement_range": [
                    float((image[1] - image[0]).min()),
                    float((image[1] - image[0]).max()),
                ],
            }
        )
    figure_paths = []
    for name, figure in figures.items():
        path = Path(config.figures_dir) / f"{name}.png"
        save_figure(figure, root / path)
        plt.close(figure)
        figure_paths.append(path.as_posix())
    # pandas serializes missing descriptive statistics as JSON null, not NaN.
    tables = {
        name: json.loads(frame.to_json(orient="records"))
        for name, frame in exploration.tables.items()
    }
    report = {
        "scope": "public_train_only",
        "patients": len(exploration.patients),
        "slices": len(exploration.samples),
        "config": config.model_dump(),
        "provenance": _provenance(root, data),
        "tables": tables,
        "folds": json.loads(folds.to_json(orient="records")),
        "example_selection": "One seeded patient per cohort/pCR; middle available slice after sorting",
        "examples": image_statistics,
        "figures": figure_paths,
        "limitations": [
            "Slices are correlated within patients; patient is the statistical unit.",
            "Examples are illustrative, not population intensity estimates or segmentations.",
            "No clinical imputation, normalization fitting or test EDA is performed.",
            "tum_vol and acquisition summaries retain source units; no unit conversion is inferred.",
        ],
    }
    _write_report(report, root / config.report_path)
    return report


def check_data_pipeline(root: Path, data: DataConfig, config: PipelineConfig) -> dict[str, Any]:
    """Inspect a real train batch and hash two full passes over validation.

    Hash identities, targets and pixels to check order and deterministic loading.
    Test is used only by the fold integrity check; no test images are opened.
    """
    samples = load_samples(root, data.metadata_dir)
    folds = summarize_folds(samples)
    training, validation = create_data_loaders(
        samples,
        root,
        path_aliases=data.image_path_aliases,
        **config.model_dump(exclude={"report_path"}),
    )
    batch = next(iter(training))
    image = batch["image"]
    if image.shape[1:] != (3, 256, 256) or image.dtype != torch.float32:
        raise ValueError("Unexpected real-batch shape/dtype")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Real batch contains invalid pixel values")
    passes = []
    expected_ids = validation.dataset.samples["sample_id"].tolist()
    for _ in range(2):
        digest = hashlib.sha256()
        identifiers = []
        for validation_batch in validation:
            identifiers.extend(validation_batch["sample_id"])
            digest.update(json.dumps(validation_batch["sample_id"]).encode())
            digest.update(json.dumps(validation_batch["patient_id"]).encode())
            digest.update(validation_batch["label"].numpy().tobytes())
            digest.update(validation_batch["image"].numpy().tobytes())
        if identifiers != expected_ids:
            raise ValueError("Validation order/coverage mismatch")
        passes.append(digest.hexdigest())
    if passes[0] != passes[1]:
        raise ValueError("Validation was not deterministic across full passes")
    report = {
        "status": "ok",
        "scope": "train_and_internal_validation_only",
        "config": config.model_dump(),
        "provenance": _provenance(root, data),
        "train_batch": {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "range": [float(image.min()), float(image.max())],
            "label_shape": list(batch["label"].shape),
            "label_dtype": str(batch["label"].dtype),
            "sample_ids": batch["sample_id"],
            "patient_ids": batch["patient_id"],
        },
        "validation": {
            "slices": len(expected_ids),
            "patients": validation.dataset.samples["patient_id"].nunique(),
            "batches_per_pass": len(validation),
            "last_batch_size": len(validation_batch["sample_id"]),
            "passes": 2,
            "sha256_per_pass": passes,
            "ordered_complete": True,
        },
        "folds": json.loads(folds.to_json(orient="records")),
    }
    _write_report(report, root / config.report_path)
    return report
