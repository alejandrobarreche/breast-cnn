"""Single evaluation of the final ensemble on test, with every decision fixed beforehand.

The fold models come from the final run; the calibration map and the threshold come from
the out-of-fold ensemble scores of the nested run (`calibration_validation_patient.json`).
Test labels are used only to compute metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from breast_cnn.evaluation.bootstrap import bootstrap_auc, paired_differences
from breast_cnn.evaluation.calibration import apply_platt
from breast_cnn.training.metrics import binary_metrics


def patient_test_scores(slices: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    """One row per patient: mean probability of the fold models (`raw`) and its calibration.

    `raw` is the patient mean of the slice ensemble, the score on which the nested run fitted
    the Platt map and the threshold.
    """
    columns = sorted(c for c in slices.columns if c.startswith("probability_fold"))
    grouped = slices.groupby("patient_id", sort=True)
    if (grouped["label"].nunique() > 1).any():
        raise ValueError("A patient has slices with different labels")
    patients = grouped[columns].mean()
    patients.insert(0, "label", grouped["label"].first())
    patients["raw"] = patients[columns].mean(axis=1)
    patients["calibrated"] = apply_platt(patients["raw"], calibration["platt"])
    return patients


def _threshold_metrics(labels: pd.Series, scores: pd.Series, threshold: float) -> dict:
    predicted = scores >= threshold
    return {
        **binary_metrics(labels, scores, threshold),
        "true_positives": int((predicted & labels.eq(1)).sum()),
        "false_negatives": int((~predicted & labels.eq(1)).sum()),
        "true_negatives": int((~predicted & labels.eq(0)).sum()),
        "false_positives": int((predicted & labels.eq(0)).sum()),
        "accuracy": float((predicted == labels.eq(1)).mean()),
    }


def evaluate_test(
    patients: pd.DataFrame,
    cohorts: pd.Series,
    train_patients: pd.DataFrame,
    calibration: dict,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict:
    """Test metrics at patient level for the ensemble, its fold models and two references.

    - Cohort baseline: pCR rate of each cohort among train patients.
    - Always no pCR: predicts the majority class for every patient.
    The AUC interval and the difference with the cohort baseline use a bootstrap over
    test patients stratified by cohort and class.
    """
    threshold = calibration["threshold"]
    scores = patients.join(cohorts.rename("cohort"))
    if scores["cohort"].isna().any():
        raise ValueError("Every test patient needs a cohort")
    rates = train_patients.groupby("data")["pCR"].mean()
    scores["cohort_baseline"] = scores["cohort"].map(rates)
    scores["fold"] = 0  # a single test set; bootstrap_auc averages AUC over this one fold

    resamples = bootstrap_auc(
        scores, ["raw", "cohort_baseline"], ["fold", "cohort", "label"], n_resamples, seed
    )
    observed = pd.Series(
        {c: roc_auc_score(scores["label"], scores[c]) for c in ("raw", "cohort_baseline")}
    )
    tail = (1 - confidence_level) / 2
    difference = paired_differences(observed, resamples, "cohort_baseline", confidence_level)

    labels = scores["label"]
    by_cohort = {}
    for cohort, rows in scores.groupby("cohort"):
        by_cohort[cohort] = {
            "auc": binary_metrics(rows["label"], rows["raw"], threshold)["auc"],
            **{
                k: v
                for k, v in _threshold_metrics(rows["label"], rows["calibrated"], threshold).items()
                if k != "auc"
            },
        }
    fold_columns = [c for c in scores.columns if c.startswith("probability_fold")]
    return {
        "unit": "patient",
        "patients": int(len(scores)),
        "positives": int(labels.sum()),
        "threshold": threshold,
        "confidence_level": confidence_level,
        "bootstrap_resamples": n_resamples,
        "auc": {
            "value": float(observed["raw"]),
            "lower": float(resamples["raw"].quantile(tail)),
            "upper": float(resamples["raw"].quantile(1 - tail)),
        },
        "at_threshold": _threshold_metrics(labels, scores["calibrated"], threshold),
        "cohort_baseline": {
            "auc": float(observed["cohort_baseline"]),
            "train_pcr_rate": {k: float(v) for k, v in rates.items()},
            "auc_difference": {
                "value": float(difference.loc["raw", "difference"]),
                "lower": float(difference.loc["raw", "lower"]),
                "upper": float(difference.loc["raw", "upper"]),
                "p_not_better": float(difference.loc["raw", "p_not_better"]),
            },
        },
        "always_no_pcr": _threshold_metrics(labels, pd.Series(0.0, index=labels.index), 0.5),
        "fold_models_auc": {
            c.removeprefix("probability_"): float(roc_auc_score(labels, scores[c]))
            for c in fold_columns
        },
        "by_cohort": by_cohort,
        "mean_calibrated_probability": float(np.mean(scores["calibrated"])),
    }
