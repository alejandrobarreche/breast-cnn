from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from breast_cnn.configuration import DataConfig, TrainConfig, load_train_config
from breast_cnn.training import aggregate_patients, binary_metrics, run_experiment
from breast_cnn.training.loop import positive_class_weight

ROOT = Path(__file__).resolve().parents[1]


def test_positive_class_weight_is_negatives_over_positives() -> None:
    assert positive_class_weight(pd.Series([0, 0, 0, 1])) == 3
    with pytest.raises(ValueError):
        positive_class_weight(pd.Series([0, 0]))


def test_binary_metrics_known_values() -> None:
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert metrics["auc"] == 0.75
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert binary_metrics(np.array([1, 1]), np.array([0.2, 0.8]), 0.5)["auc"] is None


def test_patient_aggregation_uses_mean_probability() -> None:
    predictions = pd.DataFrame(
        {"patient_id": ["a", "a", "b"], "label": [1, 1, 0], "probability": [0.2, 0.6, 0.3]}
    )
    patients = aggregate_patients(predictions)
    assert patients["probability"].tolist() == pytest.approx([0.4, 0.3])
    predictions.loc[1, "label"] = 0
    with pytest.raises(ValueError):
        aggregate_patients(predictions)


def test_experiment_configs_fix_name_and_loss() -> None:
    _, weighted = load_train_config(ROOT, ["+experiment=baseline_weighted"])
    _, unweighted = load_train_config(ROOT, ["+experiment=baseline_unweighted"])
    assert (weighted.experiment, weighted.weighted_loss) == ("baseline_weighted", True)
    assert (unweighted.experiment, unweighted.weighted_loss) == ("baseline_unweighted", False)


@pytest.fixture
def training_project(tmp_path: Path) -> Path:
    """Ten train patients (one per class in each fold), two slices each, plus one test patient."""
    rows = []
    for fold in range(5):
        for label in (0, 1):
            rows.append((f"f{fold}c{label}", "train", label, fold))
    rows.append(("t0", "test", 1, -1))
    samples = []
    for patient, split, label, fold in rows:
        for z in (1, 2):
            sample = f"{patient}_z00{z}"
            paths = {}
            for phase in ("PRE", "EARLY", "LATE"):
                relative = Path(split) / patient / f"{sample}_{phase}.png"
                (tmp_path / "data" / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (256, 256), 40 + 80 * label + z).save(tmp_path / "data" / relative)
                paths[f"path_{phase.lower()}"] = f"dataset/{relative.as_posix()}"
            samples.append(
                {
                    "sample_id": sample,
                    "patient_id": patient,
                    "split": split,
                    **paths,
                    "slice_index": z,
                    "pCR": label,
                    "fold": fold,
                }
            )
    (tmp_path / "metadata").mkdir()
    pd.DataFrame(samples).to_csv(tmp_path / "metadata/samples.csv", index=False)
    model = tmp_path / "configs/model/tiny.yaml"
    model.parent.mkdir(parents=True)
    model.write_text(
        "in_channels: 3\nkernel_size: 3\nactivation: relu\n"
        "stages:\n  - {channels: 4, convs: 1}\ndropout: 0.0\n"
    )
    return tmp_path


def test_run_experiment_writes_complete_run(training_project: Path) -> None:
    data = DataConfig(
        metadata_dir="metadata",
        image_path_aliases={"dataset": "data"},
        expected_image_size=[256, 256],
    )
    config = TrainConfig(
        experiment="smoke",
        model="configs/model/tiny.yaml",
        folds=[0, 3],
        seed=0,
        epochs=2,
        early_stopping_patience=1,
        batch_size=4,
        num_workers=0,
        optimizer="adamw",
        learning_rate=0.001,
        weight_decay=0.0,
        horizontal_flip_probability=0.5,
        rotation_degrees=10.0,
        max_translation=0.05,
        scale_range=[0.9, 1.1],
        weighted_loss=True,
        threshold=0.5,
        runs_dir="runs",
    )
    run_dir = run_experiment(training_project, data, config)

    assert run_dir.parent == training_project / "runs"
    assert run_dir.name.endswith("_smoke")
    for name in (
        "config.yaml",
        "environment.json",
        "history.csv",
        "metrics.json",
        "predictions_validation.csv",
        "model_fold0.pt",
        "model_fold3.pt",
        "figures/learning_curves_validation.png",
    ):
        assert (run_dir / name).is_file(), name

    predictions = pd.read_csv(run_dir / "predictions_validation.csv")
    assert set(predictions["patient_id"]) == {"f0c0", "f0c1", "f3c0", "f3c1"}
    assert predictions.groupby("fold").size().tolist() == [4, 4]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert [fold["fold"] for fold in metrics["folds"]] == [0, 3]
    assert metrics["folds"][0]["pos_weight"] == 1.0
    assert metrics["folds"][0]["patient"]["n"] == 2
    assert 0 <= metrics["summary"]["patient_auc"]["mean"] <= 1

    history = pd.read_csv(run_dir / "history.csv")
    assert set(history["fold"]) == {0, 3}
    assert history["train_loss"].notna().all()

    with pytest.raises(FileExistsError):
        run_dir.mkdir()
