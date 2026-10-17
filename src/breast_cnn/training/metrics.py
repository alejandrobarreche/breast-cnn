"""Slice- and patient-level binary metrics computed from saved predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_patients(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average slice probabilities per patient; the label is constant within a patient."""
    grouped = predictions.groupby("patient_id", sort=True)
    if (grouped["label"].nunique() > 1).any():
        raise ValueError("A patient has slices with different labels")
    return grouped.agg(label=("label", "first"), probability=("probability", "mean")).reset_index()


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """ROC-AUC plus threshold-dependent rates; undefined values are None."""
    labels = np.asarray(labels).astype(int)
    predicted = np.asarray(probabilities) >= threshold
    positives, negatives = labels == 1, labels == 0
    sensitivity = predicted[positives].mean() if positives.any() else None
    specificity = (~predicted[negatives]).mean() if negatives.any() else None
    both = positives.any() and negatives.any()
    return {
        "n": int(labels.size),
        "positives": int(positives.sum()),
        "auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2) if both else None,
    }


def evaluate_predictions(predictions: pd.DataFrame, threshold: float) -> dict:
    """Metrics for the slice unit and for the patient unit (mean aggregation)."""
    patients = aggregate_patients(predictions)
    return {
        "threshold": threshold,
        "slice": binary_metrics(predictions["label"], predictions["probability"], threshold),
        "patient": binary_metrics(patients["label"], patients["probability"], threshold),
    }
