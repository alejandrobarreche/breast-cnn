from __future__ import annotations

from pathlib import Path

import pandas as pd

from breast_cnn.data.validation import audit_dataset


def test_quick_audit_validates_metadata_and_resolves_all_phases(dataset_project: Path) -> None:
    report = audit_dataset(dataset_project, path_aliases={"dataset": "data"})

    assert report.ok, report.errors
    assert report.summary["patients"] == 3
    assert report.summary["resolved_image_paths"] == 9
    assert report.summary["training_patients"] == 1
    assert report.summary["validation_patients"] == 1


def test_full_audit_checks_physical_images(dataset_project: Path) -> None:
    report = audit_dataset(
        dataset_project,
        path_aliases={"dataset": "data"},
        check_images=True,
    )

    assert report.ok, report.errors
    assert report.details["image_inspection"] == {
        "checked": 9,
        "formats": {"PNG": 9},
        "modes": {"L": 9},
        "dtypes": {"uint8": 9},
        "pixel_range": [64, 64],
    }


def test_audit_rejects_inconsistent_patient_label(dataset_project: Path) -> None:
    patients_path = dataset_project / "metadata/patients.csv"
    patients = pd.read_csv(patients_path)
    patients.loc[patients["pid"].eq("p2"), "pCR"] = 0
    patients.to_csv(patients_path, index=False)

    report = audit_dataset(dataset_project, path_aliases={"dataset": "data"})

    assert not report.ok
    assert any("pCR differs" in error for error in report.errors)


def test_audit_rejects_excluded_patient_in_active_metadata(dataset_project: Path) -> None:
    excluded_path = dataset_project / "metadata/excluded_patients.csv"
    excluded = pd.read_csv(excluded_path)
    excluded.loc[0, "patient_id"] = "p1"
    excluded.to_csv(excluded_path, index=False)

    report = audit_dataset(dataset_project, path_aliases={"dataset": "data"})

    assert not report.ok
    assert any("Excluded patients appear" in error for error in report.errors)
