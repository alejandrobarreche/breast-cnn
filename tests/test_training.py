from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from breast_cnn.configuration import DataConfig, TrainConfig, load_train_config
from breast_cnn.training import (
    aggregate_patients,
    binary_metrics,
    loop,
    resume_experiment,
    run_experiment,
)
from breast_cnn.training.loop import Objective, positive_class_weight

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


def smoke_config(**changes: object) -> tuple[DataConfig, TrainConfig]:
    data = DataConfig(
        metadata_dir="metadata",
        image_path_aliases={"dataset": "data"},
        expected_image_size=[256, 256],
    )
    config = dict(
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
        loss_unit="slice",
        patient_pooling="mean",
        patients_per_batch=2,
        label_smoothing=0.0,
        threshold=0.5,
        runs_dir="runs",
    )
    return data, TrainConfig(**(config | changes))


def test_every_experiment_config_composes() -> None:
    for path in sorted((ROOT / "configs/experiment").glob("*.yaml")):
        _, config = load_train_config(ROOT, [f"+experiment={path.stem}"])
        assert config.experiment == path.stem
        assert (ROOT / config.model).is_file()


def test_sgd_and_central_slices_run(training_project: Path) -> None:
    data, config = smoke_config(optimizer="sgd", learning_rate=0.01, train_slices_per_patient=1)
    run_dir = run_experiment(training_project, data, config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["folds"][0]["training_slices"] == 8  # 8 training patients x 1 slice
    assert metrics["folds"][0]["patient"]["n"] == 2  # validation keeps every slice
    assert len(pd.read_csv(run_dir / "predictions_validation.csv")) == 8


def test_last_epoch_selection_trains_every_epoch_and_keeps_the_last(
    training_project: Path,
) -> None:
    # Patience 1 would stop the default rule early; the fixed-epoch rule ignores it.
    data, config = smoke_config(epochs=3, epoch_selection="last")
    run_dir = run_experiment(training_project, data, config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")
    assert metrics["selection"] == "last_epoch"
    assert [fold["best_epoch"] for fold in metrics["folds"]] == [3, 3]
    assert history.groupby("fold")["epoch"].max().tolist() == [3, 3]
    last = history.loc[history["epoch"].eq(3)].set_index("fold")["validation_patient_auc"]
    assert [fold["patient"]["auc"] for fold in metrics["folds"]] == pytest.approx(last.tolist())


def test_run_experiment_writes_complete_run(training_project: Path) -> None:
    data, config = smoke_config()
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
    assert history["train_patient_auc"].between(0, 1).all()
    gap = metrics["folds"][0]["train_validation_patient_auc_gap"]
    assert gap == metrics["folds"][0]["train_patient_auc"] - metrics["folds"][0]["patient"]["auc"]

    with pytest.raises(FileExistsError):
        run_dir.mkdir()


@pytest.mark.parametrize("loss_unit", ["slice", "patient"])
def test_resume_continues_interrupted_run_identically(
    training_project: Path, monkeypatch: pytest.MonkeyPatch, loss_unit: str
) -> None:
    data, reference_config = smoke_config(
        runs_dir="reference", early_stopping_patience=5, loss_unit=loss_unit
    )
    reference = run_experiment(training_project, data, reference_config)

    # Interrupt during the second epoch of fold 3, after fold 0 has finished.
    original_predict, calls = loop.predict, []

    def interrupted_predict(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(loop, "predict", interrupted_predict)
    data, config = smoke_config(
        runs_dir="interrupted", early_stopping_patience=5, loss_unit=loss_unit
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(training_project, data, config)
    monkeypatch.setattr(loop, "predict", original_predict)
    (run_dir,) = (training_project / "interrupted").iterdir()
    assert not (run_dir / "metrics.json").exists()

    resume_experiment(training_project, run_dir)

    columns = ["fold", "epoch", "train_loss", "validation_loss", "validation_patient_auc"]
    expected = pd.read_csv(reference / "history.csv")[columns]
    resumed = pd.read_csv(run_dir / "history.csv")[columns]
    pd.testing.assert_frame_equal(resumed, expected)
    with pytest.raises(ValueError, match="never rewritten"):
        resume_experiment(training_project, run_dir)


def test_patient_objective_pools_slice_logits_per_patient() -> None:
    logits = torch.tensor([0.0, 2.0, -1.0, 4.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    patients = ["a", "a", "b", "b"]
    bce = nn.BCEWithLogitsLoss()
    mean_loss, count = Objective(bce, "patient", "mean")(logits, labels, patients)
    assert count == 2
    torch.testing.assert_close(mean_loss, bce(torch.tensor([1.0, 1.5]), torch.tensor([1.0, 0.0])))
    max_loss, _ = Objective(bce, "patient", "max")(logits, labels, patients)
    torch.testing.assert_close(max_loss, bce(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 0.0])))
    slice_loss, count = Objective(bce, "slice", "mean")(logits, labels, patients, 0.1)
    assert count == 4
    torch.testing.assert_close(slice_loss, bce(logits, torch.tensor([0.95, 0.95, 0.05, 0.05])))


def test_max_pooling_sends_gradient_only_to_the_highest_slice() -> None:
    logits = torch.tensor([0.0, 2.0, 1.0], requires_grad=True)
    loss, _ = Objective(nn.BCEWithLogitsLoss(), "patient", "max")(
        logits, torch.tensor([1.0, 1.0, 1.0]), ["a", "a", "a"]
    )
    loss.backward()
    assert logits.grad[0] == 0 and logits.grad[2] == 0 and logits.grad[1] != 0


def test_patient_batch_sampler_keeps_patients_whole() -> None:
    from breast_cnn.data.loaders import PatientBatchSampler

    patients = ["a", "a", "b", "c", "c", "c", "d"]
    sampler = PatientBatchSampler(patients, 2, torch.Generator().manual_seed(0))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 2
    assert sorted(i for batch in batches for i in batch) == list(range(7))
    for batch in batches:
        ids = {patients[i] for i in batch}
        assert len(ids) <= 2
        assert all(patients.count(p) == sum(patients[i] == p for i in batch) for p in ids)
    again = list(PatientBatchSampler(patients, 2, torch.Generator().manual_seed(0)))
    assert again == batches


@pytest.mark.parametrize("pooling", ["mean", "max"])
def test_patient_level_run_completes(training_project: Path, pooling: str) -> None:
    data, config = smoke_config(
        loss_unit="patient", patient_pooling=pooling, label_smoothing=0.1, runs_dir=pooling
    )
    run_dir = run_experiment(training_project, data, config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["folds"][0]["pos_weight"] == 1.0
    assert pd.read_csv(run_dir / "history.csv")["train_loss"].notna().all()
