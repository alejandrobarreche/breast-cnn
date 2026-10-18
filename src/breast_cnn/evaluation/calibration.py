"""Per-fold Platt calibration of patient probabilities and a specificity-constrained threshold.

Everything here is fitted on internal validation predictions only; test is never used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from breast_cnn.training.metrics import aggregate_patients, binary_metrics

EPSILON = 1e-6


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


def fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Unpenalised logistic regression of the label on logit(probability), with slope >= 0.

    A negative slope would invert the model's ranking. The log-likelihood is concave, so the
    constrained optimum is then at slope 0 with the intercept of the observed pCR rate: the
    model contributes a constant and no ranking information.
    """
    labels = np.asarray(labels)
    model = LogisticRegression(C=np.inf).fit(_logit(probabilities)[:, None], labels)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    constrained = slope < 0
    if constrained:
        slope, intercept = 0.0, float(_logit(labels.mean()))
    return {"slope": slope, "intercept": intercept, "slope_constrained": constrained}


def apply_platt(probabilities: np.ndarray, platt: dict) -> np.ndarray:
    scores = platt["slope"] * _logit(probabilities) + platt["intercept"]
    return 1 / (1 + np.exp(-scores))


def fit_fold_calibration(
    predictions: pd.DataFrame, unit: str = "patient"
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Fit one Platt map per fold model on its own validation fold, per patient or per slice.

    Returns the maps and the out-of-fold table (one row per unit) with raw and calibrated
    probability. Patient probabilities are the mean over slices.
    """
    if unit not in ("patient", "slice"):
        raise ValueError(f"unit must be patient or slice, got {unit}")
    maps, tables = {}, []
    for fold, rows in predictions.groupby("fold"):
        units = aggregate_patients(rows) if unit == "patient" else rows.copy()
        maps[int(fold)] = fit_platt(units["probability"], units["label"])
        units["calibrated"] = apply_platt(units["probability"], maps[int(fold)])
        tables.append(units.assign(fold=int(fold)))
    return maps, pd.concat(tables, ignore_index=True)


def specificity_threshold(labels: np.ndarray, scores: np.ndarray, min_specificity: float) -> float:
    """Lowest threshold whose specificity (score < threshold among negatives) is at least the
    minimum; the lowest such threshold gives the highest sensitivity under the constraint.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    negatives = scores[labels == 0]
    for threshold in np.unique(scores):
        if (negatives < threshold).mean() >= min_specificity:
            return float(threshold)
    return float(np.nextafter(scores.max(), np.inf))


def calibrate_and_threshold(
    predictions: pd.DataFrame, min_specificity: float, unit: str = "patient"
) -> dict:
    """Calibration maps, the pooled out-of-fold threshold and its validation metrics.

    `unit="slice"` serves the web application, which scores one slice at a time.
    """
    maps, patients = fit_fold_calibration(predictions, unit)
    threshold = specificity_threshold(patients["label"], patients["calibrated"], min_specificity)
    per_fold = {}
    for fold, rows in patients.groupby("fold"):
        per_fold[int(fold)] = {
            **maps[int(fold)],
            "mean_raw_probability": float(rows["probability"].mean()),
            "mean_calibrated_probability": float(rows["calibrated"].mean()),
            "pcr_rate": float(rows["label"].mean()),
            "raw_auc": binary_metrics(rows["label"], rows["probability"], threshold)["auc"],
            **binary_metrics(rows["label"], rows["calibrated"], threshold),
        }
    return {
        "unit": unit,
        "aggregation": "mean_probability_per_patient" if unit == "patient" else None,
        "calibration": "platt_per_fold_on_own_validation_fold_slope_nonnegative",
        # Test: AUC from the mean raw probability of the fold models (threshold-free);
        # sensitivity and specificity from the mean calibrated probability at `threshold`.
        "test_auc_score": "mean_raw_probability",
        "test_threshold_score": "mean_calibrated_probability",
        "min_specificity": min_specificity,
        "threshold": threshold,
        "pooled_validation": binary_metrics(patients["label"], patients["calibrated"], threshold),
        "folds": per_fold,
    }
