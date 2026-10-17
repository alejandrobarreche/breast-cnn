"""Descriptive public-train analysis, without fitted preprocessing or test EDA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from breast_cnn.data.validation import validate_sample_frame


@dataclass
class TrainExploration:
    samples: pd.DataFrame
    patients: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def explore_train(samples: pd.DataFrame, patients: pd.DataFrame) -> TrainExploration:
    """Filter train before describing images or clinical/acquisition variables.

    The full index is checked for leakage; the train join is many-to-one and
    requires identical patient sets and labels. Missing clinical values remain
    missing; only their counts and distributions are reported.
    """
    validate_sample_frame(samples)
    train_samples = samples.loc[samples["split"].eq("train")].copy()
    train_patients = patients.loc[patients["split"].eq("train")].copy()
    if train_samples.empty or train_patients.empty:
        raise ValueError("Train EDA requires non-empty public train")
    if train_patients["pid"].isna().any() or train_patients["pid"].duplicated().any():
        raise ValueError("Train patients.pid must be unique and non-null")
    if set(train_samples["patient_id"]) != set(train_patients["pid"]):
        raise ValueError("Train sample/patient identifiers do not match")
    joined = train_samples.merge(
        train_patients[["pid", "pCR", "split", "data"]],
        left_on="patient_id",
        right_on="pid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_patient"),
    )
    for column in ("pCR", "split"):
        if not joined[column].eq(joined[f"{column}_patient"]).all():
            raise ValueError(f"Train {column} mismatch between samples and patients")
    if train_patients["data"].isna().any():
        raise ValueError("Train cohort (patients.data) must not be missing")
    counts = train_samples.groupby("patient_id").size().rename("n_slices")
    folds = train_samples.groupby("patient_id")["fold"].first()
    train_patients = train_patients.join(counts, on="pid").join(folds, on="pid")

    balances = []
    for unit, frame in (("patient", train_patients), ("slice", train_samples)):
        for label in (0, 1):
            count = int(frame["pCR"].eq(label).sum())
            balances.append(
                {"unit": unit, "pCR": label, "count": count, "fraction": count / len(frame)}
            )
    cohort = train_patients.groupby("data").agg(
        patients=("pid", "size"),
        pcr_positive=("pCR", "sum"),
        slices=("n_slices", "sum"),
        median_slices=("n_slices", "median"),
        min_slices=("n_slices", "min"),
    )
    cohort["pcr_fraction"] = cohort["pcr_positive"] / cohort["patients"]
    original_columns = patients.columns
    missing = train_patients[original_columns].isna().sum().rename("missing").to_frame()
    missing["total"] = len(train_patients)
    missing["fraction"] = missing["missing"] / len(train_patients)
    missing.index.name = "column"
    missing_by_cohort = (
        train_patients.groupby("data")[list(original_columns)]
        .agg(lambda column: int(column.isna().sum()))
        .drop(columns="data", errors="ignore")
    )
    numeric_columns = [
        column
        for column in ("age", "tum_vol", "n_z", "n_times", "slice_thick", "xy_spacing")
        if column in train_patients
    ]
    numeric = (
        train_patients[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
        if numeric_columns
        else pd.DataFrame()
    )
    numeric.index.name = "column"
    slice_counts = counts.value_counts().sort_index().rename("patients").to_frame()
    slice_counts.index.name = "n_slices"
    tables = {
        "class_balance": pd.DataFrame(balances),
        "cohort_summary": cohort.reset_index(),
        "slices_per_patient": slice_counts.reset_index(),
        "missing_values": missing.reset_index(),
        "missing_by_cohort": missing_by_cohort.reset_index(),
        "numeric_summary": numeric.reset_index(),
    }
    if "HR_HER2_STATUS" in train_patients:
        tables["subtype_balance"] = (
            train_patients.groupby(["data", "HR_HER2_STATUS", "pCR"], dropna=False)
            .size()
            .rename("patients")
            .reset_index()
        )
    return TrainExploration(train_samples.reset_index(drop=True), train_patients, tables)


def select_examples(exploration: TrainExploration, seed: int) -> pd.DataFrame:
    """Choose one patient per cohort/class, then the middle available slice.

    Sort identifiers before seeded sampling so source row order does not affect
    selection. These illustrative examples are not representative estimates.
    """
    if not exploration.samples["split"].eq("train").all():
        raise ValueError("Visual examples must come exclusively from train")
    rng = np.random.default_rng(seed)
    selected = []
    for (cohort, _), group in exploration.patients.groupby(["data", "pCR"], sort=True):
        patient_ids = sorted(group["pid"].tolist())
        patient_id = patient_ids[int(rng.integers(len(patient_ids)))]
        slices = exploration.samples.loc[exploration.samples["patient_id"].eq(patient_id)]
        slices = slices.sort_values(["slice_index", "sample_id"])
        row = slices.iloc[len(slices) // 2].copy()
        row["cohort"] = cohort
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)
