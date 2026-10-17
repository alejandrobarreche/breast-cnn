"""Load and query the dataset metadata without modifying its source files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

SAMPLE_STRING_COLUMNS = {
    "sample_id": "string",
    "patient_id": "string",
    "split": "string",
    "path_pre": "string",
    "path_early": "string",
    "path_late": "string",
}
PATIENT_STRING_COLUMNS = {
    "pid": "string",
    "split": "string",
    "data": "string",
}
EXCLUDED_STRING_COLUMNS = {"patient_id": "string", "error": "string"}


def _metadata_path(project_root: Path | str, metadata_dir: Path | str, filename: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    directory = Path(metadata_dir)
    path = directory / filename if directory.is_absolute() else root / directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file does not exist: {path}")
    return path


def load_samples(
    project_root: Path | str = ".", metadata_dir: Path | str = "metadata"
) -> pd.DataFrame:
    """Load the operational sample index, preserving one row per slice."""
    path = _metadata_path(project_root, metadata_dir, "samples.csv")
    return pd.read_csv(path, dtype=SAMPLE_STRING_COLUMNS)


def load_patients(
    project_root: Path | str = ".", metadata_dir: Path | str = "metadata"
) -> pd.DataFrame:
    """Load one-row-per-patient clinical and acquisition metadata."""
    path = _metadata_path(project_root, metadata_dir, "patients.csv")
    return pd.read_csv(path, dtype=PATIENT_STRING_COLUMNS)


def load_statistics(
    project_root: Path | str = ".", metadata_dir: Path | str = "metadata"
) -> dict[str, Any]:
    """Load the published reference counts used as an integrity checksum."""
    path = _metadata_path(project_root, metadata_dir, "statistics.json")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_excluded_patients(
    project_root: Path | str = ".", metadata_dir: Path | str = "metadata"
) -> pd.DataFrame:
    """Load excluded-patient traceability metadata."""
    path = _metadata_path(project_root, metadata_dir, "excluded_patients.csv")
    return pd.read_csv(path, dtype=EXCLUDED_STRING_COLUMNS)


def split_train_validation(
    samples: pd.DataFrame, validation_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows by the existing patient-level fold assignment."""
    if (
        isinstance(validation_fold, bool)
        or not isinstance(validation_fold, int)
        or validation_fold not in range(5)
    ):
        raise ValueError(f"validation_fold must be between 0 and 4, got {validation_fold}")

    train_samples = samples.loc[samples["split"].eq("train")]
    training = train_samples.loc[train_samples["fold"].ne(validation_fold)].copy()
    validation = train_samples.loc[train_samples["fold"].eq(validation_fold)].copy()

    shared_patients = set(training["patient_id"]) & set(validation["patient_id"])
    if shared_patients:
        examples = sorted(shared_patients)[:5]
        raise ValueError(
            "Patient leakage between training and validation: "
            f"{len(shared_patients)} shared patient(s), for example {examples}"
        )

    return training.reset_index(drop=True), validation.reset_index(drop=True)


def summarize_folds(samples: pd.DataFrame) -> pd.DataFrame:
    """Verify five-fold coverage and return only public-train counts.

    Test rows are examined solely for index integrity and patient separation.
    No test images, clinical variables or model outputs are used.
    """
    from breast_cnn.data.validation import validate_sample_frame

    validate_sample_frame(samples)
    public_train = samples.loc[samples["split"].eq("train")]
    if set(public_train["fold"]) != set(range(5)):
        raise ValueError("Public train must cover all five supplied folds (0-4)")
    seen: set[str] = set()
    rows = []
    for fold in range(5):
        training, validation = split_train_validation(samples, fold)
        ids = set(validation["sample_id"])
        if seen & ids or set(training["sample_id"]) | ids != set(public_train["sample_id"]):
            raise ValueError(f"Incorrect sample coverage for fold {fold}")
        seen.update(ids)
        patients = validation.drop_duplicates("patient_id")
        rows.append(
            {
                "fold": fold,
                "training_patients": training["patient_id"].nunique(),
                "training_slices": len(training),
                "validation_patients": len(patients),
                "validation_slices": len(validation),
                "validation_patients_pcr_0": int(patients["pCR"].eq(0).sum()),
                "validation_patients_pcr_1": int(patients["pCR"].eq(1).sum()),
            }
        )
    if seen != set(public_train["sample_id"]):
        raise ValueError("Validation folds do not cover public train exactly once")
    return pd.DataFrame(rows)


def select_test_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the untouched test rows."""
    return samples.loc[samples["split"].eq("test")].copy().reset_index(drop=True)


def resolve_image_path(
    csv_path: str,
    project_root: Path | str,
    path_aliases: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one CSV image path using explicit top-level prefix aliases.

    A direct path and an aliased path are both considered. Exactly one must exist;
    accepting two candidates could silently select the wrong copy of the dataset.
    """
    root = Path(project_root).expanduser().resolve()
    relative_path = Path(csv_path)
    if relative_path.is_absolute():
        raise ValueError(f"Image paths in metadata must be relative, got: {csv_path}")
    if not relative_path.parts:
        raise ValueError("Image path in metadata cannot be empty")

    candidates = [root / relative_path]
    aliases = path_aliases or {}
    prefix = relative_path.parts[0]
    if prefix in aliases:
        alias_root = Path(aliases[prefix])
        if alias_root.is_absolute():
            raise ValueError(f"Image path alias targets must be relative, got: {alias_root}")
        candidates.append(root / alias_root / Path(*relative_path.parts[1:]))

    safe_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Image path escapes the project root: {csv_path}")
        if resolved not in safe_candidates:
            safe_candidates.append(resolved)

    existing = [candidate for candidate in safe_candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        choices = ", ".join(str(path) for path in existing)
        raise ValueError(f"Ambiguous image path {csv_path!r}; multiple files exist: {choices}")

    attempted = ", ".join(str(path) for path in safe_candidates)
    raise FileNotFoundError(f"Cannot resolve image path {csv_path!r}; tried: {attempted}")
