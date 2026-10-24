"""Platt calibration of ensemble probabilities and a specificity-constrained threshold.

Everything here is fitted on the out-of-fold predictions of a nested run (internal
validation only); test is never used.
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


def nested_ensemble_scores(predictions: pd.DataFrame, unit: str = "patient") -> pd.DataFrame:
    """Out-of-fold ensemble score of every validation unit of a nested run.

    Each slice is scored by the models that never saw its fold (one per other fold) and its
    score is their mean probability, as the final ensemble averages its fold models. Patient
    scores are the mean over slices. One row per unit with fold, label and probability.
    """
    if unit not in ("patient", "slice"):
        raise ValueError(f"unit must be patient or slice, got {unit}")
    grouped = predictions.groupby(["fold", "sample_id", "patient_id"], sort=True)
    counts = grouped["model"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Every slice needs the same number of out-of-fold models")
    slices = grouped.agg(label=("label", "first"), probability=("probability", "mean"))
    slices = slices.reset_index()
    if unit == "slice":
        return slices
    return pd.concat(
        [aggregate_patients(rows).assign(fold=int(fold)) for fold, rows in slices.groupby("fold")],
        ignore_index=True,
    )


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
    """One Platt map and the threshold on the out-of-fold ensemble scores of a nested run.

    The map and the threshold apply to the mean probability of the final fold models, the
    same kind of score. `unit="slice"` serves the web application, which scores one slice.
    """
    units = nested_ensemble_scores(predictions, unit)
    platt = fit_platt(units["probability"], units["label"])
    units["calibrated"] = apply_platt(units["probability"], platt)
    threshold = specificity_threshold(units["label"], units["calibrated"], min_specificity)
    per_fold = {}
    for fold, rows in units.groupby("fold"):
        per_fold[int(fold)] = {
            "mean_raw_probability": float(rows["probability"].mean()),
            "mean_calibrated_probability": float(rows["calibrated"].mean()),
            "pcr_rate": float(rows["label"].mean()),
            **binary_metrics(rows["label"], rows["calibrated"], threshold),
        }
    return {
        "unit": unit,
        "aggregation": "mean_probability_per_patient" if unit == "patient" else None,
        "score": "mean_probability_of_models_without_the_unit_fold",
        "models_per_unit": int(predictions.groupby("sample_id")["model"].nunique().iloc[0]),
        "calibration": "platt_on_out_of_fold_ensemble_mean_slope_nonnegative",
        "min_specificity": min_specificity,
        "platt": platt,
        "threshold": threshold,
        "pooled_validation": binary_metrics(units["label"], units["calibrated"], threshold),
        "folds": per_fold,
    }
