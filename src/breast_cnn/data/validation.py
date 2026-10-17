"""Dataset integrity checks shared by scripts and tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from breast_cnn.data.metadata import (
    load_excluded_patients,
    load_patients,
    load_samples,
    load_statistics,
    resolve_image_path,
    split_train_validation,
)

REQUIRED_SAMPLE_COLUMNS = {
    "sample_id",
    "patient_id",
    "split",
    "path_pre",
    "path_early",
    "path_late",
    "slice_index",
    "pCR",
    "fold",
}
REQUIRED_PATIENT_COLUMNS = {"pid", "split", "test", "data", "pCR"}
REQUIRED_EXCLUDED_COLUMNS = {"patient_id", "error"}
PHASE_PATH_COLUMNS = {
    "PRE": "path_pre",
    "EARLY": "path_early",
    "LATE": "path_late",
}


def validate_sample_frame(samples: pd.DataFrame) -> None:
    """Reject invalid indices before loading images or splitting patients.

    Partial frames are allowed; they need not contain all five folds or test.
    The full metadata/physical audit remains a separate prerequisite.
    """
    report = AuditReport()
    if not _check_required_columns(samples, REQUIRED_SAMPLE_COLUMNS, "samples", report):
        raise ValueError("; ".join(report.errors))
    if samples.empty:
        raise ValueError("Sample index is empty")
    if samples[list(REQUIRED_SAMPLE_COLUMNS)].isna().any().any():
        raise ValueError("Sample index contains missing required values")
    for column in ("sample_id", "patient_id", "split", *PHASE_PATH_COLUMNS.values()):
        if (
            not samples[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Sample index {column} must contain non-empty strings")
    _check_unique_identifier(samples, "sample_id", "samples", report)
    _check_phase_paths(samples, report)
    if not samples["pCR"].isin([0, 1]).all():
        report.errors.append("pCR must be binary (0 or 1)")
    if not samples["split"].isin(["train", "test"]).all():
        report.errors.append("split must be train or test")
    for split, folds in (("train", range(5)), ("test", [-1])):
        if not samples.loc[samples["split"].eq(split), "fold"].isin(folds).all():
            report.errors.append(f"Invalid fold for split={split}")
    indices = pd.to_numeric(samples["slice_index"], errors="coerce")
    if not (indices.ge(0) & indices.mod(1).eq(0)).all():
        report.errors.append("slice_index must contain non-negative integers")
    for column in ("pCR", "split", "fold"):
        if samples.groupby("patient_id")[column].nunique().ne(1).any():
            report.errors.append(f"{column} must be constant per patient (possible leakage)")
    for row in samples.itertuples(index=False):
        for phase, column in PHASE_PATH_COLUMNS.items():
            path = Path(getattr(row, column))
            if path.stem != f"{row.sample_id}_{phase}" or path.parent.name != row.patient_id:
                report.errors.append(f"Sample {row.sample_id}: inconsistent {phase} path {path}")
        prefix, separator, index = row.sample_id.rpartition("_z")
        if (
            not separator
            or prefix != row.patient_id
            or not index.isdigit()
            or int(index) != row.slice_index
        ):
            report.errors.append(f"Sample {row.sample_id}: patient_id/slice_index mismatch")
    if report.errors:
        raise ValueError("Invalid sample index: " + "; ".join(report.errors[:10]))


@dataclass(slots=True)
class AuditReport:
    """Structured result from a quick or full dataset audit."""

    summary: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = "ok" if self.ok else "failed"
        return result


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _pcr_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split, split_rows in frame.groupby("split"):
        result[str(split)] = {
            f"pCR_{int(label)}": int(count)
            for label, count in split_rows["pCR"].value_counts().sort_index().items()
        }
    return result


def _expected_statistics(
    samples: pd.DataFrame, patients: pd.DataFrame, excluded: pd.DataFrame
) -> dict[str, Any]:
    return {
        "totals": {
            "patients": len(patients),
            "samples": len(samples),
            "excluded_patients": len(excluded),
        },
        "patients_by_split": _value_counts(patients["split"]),
        "samples_by_split": _value_counts(samples["split"]),
        "patients_by_split_and_pcr": _pcr_counts(patients),
        "samples_by_split_and_pcr": _pcr_counts(samples),
    }


def _check_required_columns(
    frame: pd.DataFrame, required: set[str], name: str, report: AuditReport
) -> bool:
    missing = sorted(required - set(frame.columns))
    if missing:
        report.errors.append(f"{name} is missing required columns: {missing}")
        return False
    return True


def _check_unique_identifier(
    frame: pd.DataFrame, column: str, name: str, report: AuditReport
) -> None:
    missing_count = int(frame[column].isna().sum())
    duplicated = frame.loc[frame[column].duplicated(keep=False), column].dropna().astype(str)
    if missing_count:
        report.errors.append(f"{name}.{column} contains {missing_count} missing value(s)")
    if not duplicated.empty:
        examples = sorted(duplicated.unique())[:5]
        report.errors.append(
            f"{name}.{column} is not unique: {duplicated.nunique()} duplicated value(s), "
            f"for example {examples}"
        )


def _check_patient_consistency(
    samples: pd.DataFrame, patients: pd.DataFrame, validation_fold: int, report: AuditReport
) -> None:
    sample_ids = set(samples["patient_id"].dropna())
    patient_ids = set(patients["pid"].dropna())
    missing_from_patients = sorted(sample_ids - patient_ids)
    without_samples = sorted(patient_ids - sample_ids)
    if missing_from_patients:
        report.errors.append(
            f"{len(missing_from_patients)} sample patient(s) are absent from patients.csv; "
            f"examples: {missing_from_patients[:5]}"
        )
    if without_samples:
        report.errors.append(
            f"{len(without_samples)} patient(s) have no samples; examples: {without_samples[:5]}"
        )

    for column in ("pCR", "split", "fold"):
        inconsistent = samples.groupby("patient_id")[column].nunique(dropna=False)
        inconsistent = inconsistent.loc[inconsistent.ne(1)]
        if not inconsistent.empty:
            report.errors.append(
                f"{column} is not constant within {len(inconsistent)} patient(s); "
                f"examples: {sorted(inconsistent.index.astype(str))[:5]}"
            )

    patient_reference = patients.set_index("pid")[["pCR", "split"]]
    sample_reference = samples.groupby("patient_id")[["pCR", "split"]].first()
    comparable = sample_reference.join(patient_reference, how="inner", rsuffix="_patient")
    for column in ("pCR", "split"):
        mismatch = comparable[column].ne(comparable[f"{column}_patient"])
        if mismatch.any():
            examples = sorted(comparable.index[mismatch].astype(str))[:5]
            report.errors.append(
                f"{column} differs between samples.csv and patients.csv for "
                f"{int(mismatch.sum())} patient(s); examples: {examples}"
            )

    train_patients = set(samples.loc[samples["split"].eq("train"), "patient_id"])
    test_patients = set(samples.loc[samples["split"].eq("test"), "patient_id"])
    overlap = sorted(train_patients & test_patients)
    if overlap:
        report.errors.append(
            f"Patient leakage between train and test: {len(overlap)} patient(s); "
            f"examples: {overlap[:5]}"
        )

    try:
        training, validation = split_train_validation(samples, validation_fold)
    except ValueError as error:
        report.errors.append(str(error))
    else:
        report.summary["validation_fold"] = validation_fold
        report.summary["training_samples"] = len(training)
        report.summary["validation_samples"] = len(validation)
        report.summary["training_patients"] = int(training["patient_id"].nunique())
        report.summary["validation_patients"] = int(validation["patient_id"].nunique())


def _check_values(samples: pd.DataFrame, patients: pd.DataFrame, report: AuditReport) -> None:
    for name, frame in (("samples.csv", samples), ("patients.csv", patients)):
        invalid_splits = sorted(set(frame["split"].dropna().astype(str)) - {"train", "test"})
        if invalid_splits:
            report.errors.append(f"{name} contains invalid split values: {invalid_splits}")
        invalid_labels = sorted(set(frame["pCR"].dropna()) - {0, 1})
        if invalid_labels:
            report.errors.append(f"{name} contains non-binary pCR values: {invalid_labels}")
        if frame[["split", "pCR"]].isna().any().any():
            report.errors.append(f"{name} contains missing split or pCR values")

    train_folds = set(samples.loc[samples["split"].eq("train"), "fold"].dropna())
    invalid_train_folds = sorted(train_folds - set(range(5)))
    if invalid_train_folds:
        report.errors.append(f"Training samples contain invalid fold values: {invalid_train_folds}")
    if samples.loc[samples["split"].eq("train"), "fold"].isna().any():
        report.errors.append("Training samples contain missing fold values")

    test_folds = set(samples.loc[samples["split"].eq("test"), "fold"].dropna())
    if test_folds != {-1}:
        report.errors.append(f"Test samples must use fold -1, found: {sorted(test_folds)}")

    if "test" in patients.columns:
        expected_test = patients["split"].eq("test").astype(int)
        inconsistent_test = patients["test"].ne(expected_test)
        if inconsistent_test.any():
            report.errors.append(
                "patients.csv test flag disagrees with split for "
                f"{int(inconsistent_test.sum())} patient(s)"
            )


def _check_phase_paths(samples: pd.DataFrame, report: AuditReport) -> None:
    phase_frame = samples[list(PHASE_PATH_COLUMNS.values())]
    missing = phase_frame.isna().sum()
    for column, count in missing.items():
        if count:
            report.errors.append(f"samples.csv {column} contains {int(count)} missing path(s)")

    repeated_within_row = phase_frame.nunique(axis=1, dropna=False).ne(3)
    if repeated_within_row.any():
        report.errors.append(
            f"{int(repeated_within_row.sum())} sample(s) reuse a file across DCE phases"
        )

    all_paths = phase_frame.stack().astype(str)
    duplicated_paths = all_paths.loc[all_paths.duplicated(keep=False)]
    if not duplicated_paths.empty:
        report.errors.append(
            f"{duplicated_paths.nunique()} image path(s) are referenced more than once; "
            f"examples: {sorted(duplicated_paths.unique())[:5]}"
        )

    for phase, column in PHASE_PATH_COLUMNS.items():
        wrong_suffix = ~samples[column].astype(str).str.upper().str.endswith(f"_{phase}.PNG")
        if wrong_suffix.any():
            report.errors.append(
                f"{int(wrong_suffix.sum())} {column} value(s) do not end in _{phase}.png"
            )

    duplicate_slices = samples.duplicated(["patient_id", "slice_index"], keep=False)
    if duplicate_slices.any():
        report.errors.append(
            f"{int(duplicate_slices.sum())} row(s) duplicate a patient_id/slice_index pair"
        )


def _resolve_all_paths(
    samples: pd.DataFrame,
    project_root: Path,
    path_aliases: dict[str, str],
    report: AuditReport,
) -> list[Path]:
    resolved_paths: list[Path] = []
    failures: list[str] = []
    for column in PHASE_PATH_COLUMNS.values():
        for csv_path in samples[column].dropna().astype(str):
            try:
                resolved_paths.append(resolve_image_path(csv_path, project_root, path_aliases))
            except (FileNotFoundError, ValueError) as error:
                if len(failures) < 5:
                    failures.append(str(error))

    expected_count = len(samples) * len(PHASE_PATH_COLUMNS)
    report.summary["expected_image_paths"] = expected_count
    report.summary["resolved_image_paths"] = len(resolved_paths)
    if len(resolved_paths) != expected_count:
        report.errors.append(
            f"Could not resolve {expected_count - len(resolved_paths)} of {expected_count} image "
            f"path(s). First errors: {failures}"
        )
    return resolved_paths


def _inspect_images(paths: list[Path], expected_size: tuple[int, int], report: AuditReport) -> None:
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    minimum = 255
    maximum = 0
    failures: list[str] = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                formats[str(image.format)] += 1
                modes[image.mode] += 1
                array = np.asarray(image)
                dtypes[str(array.dtype)] += 1
                if image.format != "PNG":
                    raise ValueError(f"expected PNG, found {image.format}")
                if image.size != expected_size:
                    raise ValueError(f"expected size {expected_size}, found {image.size}")
                if image.mode != "L" or array.ndim != 2 or array.dtype != np.uint8:
                    raise ValueError(
                        f"expected 8-bit grayscale, found mode={image.mode}, "
                        f"shape={array.shape}, dtype={array.dtype}"
                    )
                minimum = min(minimum, int(array.min()))
                maximum = max(maximum, int(array.max()))
        except Exception as error:  # Pillow exposes several format-specific exceptions.
            if len(failures) < 5:
                failures.append(f"{path}: {error}")

    report.details["image_inspection"] = {
        "checked": len(paths),
        "formats": dict(sorted(formats.items())),
        "modes": dict(sorted(modes.items())),
        "dtypes": dict(sorted(dtypes.items())),
        "pixel_range": [minimum, maximum] if paths and not failures else None,
    }
    if failures:
        report.errors.append(
            f"Physical image inspection failed for at least {len(failures)} image(s). "
            f"First errors: {failures}"
        )


def audit_dataset(
    project_root: Path | str,
    *,
    metadata_dir: Path | str = "metadata",
    path_aliases: dict[str, str] | None = None,
    validation_fold: int = 0,
    check_images: bool = False,
    expected_image_size: tuple[int, int] = (256, 256),
) -> AuditReport:
    """Audit metadata, patient-level splits, paths, and optionally every image."""
    root = Path(project_root).expanduser().resolve()
    report = AuditReport()

    try:
        samples = load_samples(root, metadata_dir)
        patients = load_patients(root, metadata_dir)
        statistics = load_statistics(root, metadata_dir)
        excluded = load_excluded_patients(root, metadata_dir)
    except (FileNotFoundError, ValueError) as error:
        report.errors.append(str(error))
        return report

    sample_columns_ok = _check_required_columns(
        samples, REQUIRED_SAMPLE_COLUMNS, "samples.csv", report
    )
    patient_columns_ok = _check_required_columns(
        patients, REQUIRED_PATIENT_COLUMNS, "patients.csv", report
    )
    excluded_columns_ok = _check_required_columns(
        excluded, REQUIRED_EXCLUDED_COLUMNS, "excluded_patients.csv", report
    )
    if not (sample_columns_ok and patient_columns_ok and excluded_columns_ok):
        return report

    report.summary.update(
        {
            "patients": len(patients),
            "samples": len(samples),
            "excluded_patients": len(excluded),
            "patients_by_split": _value_counts(patients["split"]),
            "samples_by_split": _value_counts(samples["split"]),
            "full_image_check": check_images,
        }
    )
    report.details["patient_missing_values"] = {
        column: int(count) for column, count in patients.isna().sum().items() if int(count) > 0
    }
    report.details["patient_cohorts"] = _value_counts(patients["data"])
    report.details["patient_pcr_by_split"] = _pcr_counts(patients)
    report.details["sample_pcr_by_split"] = _pcr_counts(samples)
    report.details["samples_per_patient"] = {
        "min": int(samples.groupby("patient_id").size().min()),
        "median": float(samples.groupby("patient_id").size().median()),
        "max": int(samples.groupby("patient_id").size().max()),
    }

    _check_unique_identifier(samples, "sample_id", "samples.csv", report)
    _check_unique_identifier(patients, "pid", "patients.csv", report)
    _check_values(samples, patients, report)
    _check_patient_consistency(samples, patients, validation_fold, report)
    _check_phase_paths(samples, report)

    excluded_ids = set(excluded["patient_id"].dropna())
    sample_overlap = sorted(excluded_ids & set(samples["patient_id"].dropna()))
    patient_overlap = sorted(excluded_ids & set(patients["pid"].dropna()))
    if sample_overlap or patient_overlap:
        report.errors.append(
            "Excluded patients appear in active metadata: "
            f"samples={sample_overlap[:5]}, patients={patient_overlap[:5]}"
        )

    expected_statistics = _expected_statistics(samples, patients, excluded)
    report.details["computed_statistics"] = expected_statistics
    for section, computed in expected_statistics.items():
        if statistics.get(section) != computed:
            report.errors.append(
                f"statistics.json section {section!r} does not match CSV counts: "
                f"reference={statistics.get(section)!r}, computed={computed!r}"
            )

    resolved_paths = _resolve_all_paths(samples, root, path_aliases or {}, report)
    if check_images and len(resolved_paths) == report.summary["expected_image_paths"]:
        _inspect_images(resolved_paths, expected_image_size, report)

    return report
