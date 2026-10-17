from __future__ import annotations

from pathlib import Path

import pytest

from breast_cnn.data.metadata import (
    load_excluded_patients,
    load_patients,
    load_samples,
    load_statistics,
    resolve_image_path,
    select_test_samples,
    split_train_validation,
)


def test_load_all_metadata_sources(dataset_project: Path) -> None:
    assert len(load_samples(dataset_project)) == 3
    assert len(load_patients(dataset_project)) == 3
    assert load_statistics(dataset_project)["totals"]["samples"] == 3
    assert load_excluded_patients(dataset_project)["patient_id"].tolist() == ["px"]


def test_split_uses_patient_folds_and_excludes_test(dataset_project: Path) -> None:
    samples = load_samples(dataset_project)

    training, validation = split_train_validation(samples, validation_fold=0)

    assert set(training["patient_id"]) == {"p2"}
    assert set(validation["patient_id"]) == {"p1"}
    assert set(training["patient_id"]).isdisjoint(validation["patient_id"])
    assert select_test_samples(samples)["patient_id"].tolist() == ["p3"]


def test_resolve_image_path_uses_explicit_alias(dataset_project: Path) -> None:
    resolved = resolve_image_path(
        "dataset/train/p1/p1_z001_PRE.png",
        dataset_project,
        {"dataset": "data"},
    )

    assert resolved == dataset_project / "data/train/p1/p1_z001_PRE.png"


def test_resolve_image_path_rejects_ambiguous_files(dataset_project: Path) -> None:
    direct = dataset_project / "dataset/train/p1/p1_z001_PRE.png"
    direct.parent.mkdir(parents=True)
    direct.write_bytes(b"duplicate")

    with pytest.raises(ValueError, match="Ambiguous image path"):
        resolve_image_path(
            "dataset/train/p1/p1_z001_PRE.png",
            dataset_project,
            {"dataset": "data"},
        )


def test_resolve_image_path_rejects_parent_traversal(dataset_project: Path) -> None:
    with pytest.raises(ValueError, match="escapes the project root"):
        resolve_image_path("../outside.png", dataset_project)
