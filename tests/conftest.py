from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def dataset_project(tmp_path: Path) -> Path:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    samples = pd.DataFrame(
        [
            {
                "sample_id": "p1_z001",
                "patient_id": "p1",
                "split": "train",
                "path_pre": "dataset/train/p1/p1_z001_PRE.png",
                "path_early": "dataset/train/p1/p1_z001_EARLY.png",
                "path_late": "dataset/train/p1/p1_z001_LATE.png",
                "slice_index": 1,
                "pCR": 0,
                "fold": 0,
            },
            {
                "sample_id": "p2_z001",
                "patient_id": "p2",
                "split": "train",
                "path_pre": "dataset/train/p2/p2_z001_PRE.png",
                "path_early": "dataset/train/p2/p2_z001_EARLY.png",
                "path_late": "dataset/train/p2/p2_z001_LATE.png",
                "slice_index": 1,
                "pCR": 1,
                "fold": 1,
            },
            {
                "sample_id": "p3_z001",
                "patient_id": "p3",
                "split": "test",
                "path_pre": "dataset/test/p3/p3_z001_PRE.png",
                "path_early": "dataset/test/p3/p3_z001_EARLY.png",
                "path_late": "dataset/test/p3/p3_z001_LATE.png",
                "slice_index": 1,
                "pCR": 0,
                "fold": -1,
            },
        ]
    )
    patients = pd.DataFrame(
        [
            {"pid": "p1", "pCR": 0, "data": "cohort_a", "test": 0, "split": "train"},
            {"pid": "p2", "pCR": 1, "data": "cohort_a", "test": 0, "split": "train"},
            {"pid": "p3", "pCR": 0, "data": "cohort_b", "test": 1, "split": "test"},
        ]
    )
    excluded = pd.DataFrame([{"patient_id": "px", "error": "unavailable phase"}])

    samples.to_csv(metadata_dir / "samples.csv", index=False)
    patients.to_csv(metadata_dir / "patients.csv", index=False)
    excluded.to_csv(metadata_dir / "excluded_patients.csv", index=False)

    statistics = {
        "scope": "test_fixture",
        "totals": {"patients": 3, "samples": 3, "excluded_patients": 1},
        "patients_by_split": {"test": 1, "train": 2},
        "samples_by_split": {"test": 1, "train": 2},
        "patients_by_split_and_pcr": {
            "test": {"pCR_0": 1},
            "train": {"pCR_0": 1, "pCR_1": 1},
        },
        "samples_by_split_and_pcr": {
            "test": {"pCR_0": 1},
            "train": {"pCR_0": 1, "pCR_1": 1},
        },
    }
    (metadata_dir / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")

    for row in samples.itertuples(index=False):
        for csv_path in (row.path_pre, row.path_early, row.path_late):
            image_path = tmp_path / "data" / Path(csv_path).relative_to("dataset")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (256, 256), color=64).save(image_path)

    return tmp_path
