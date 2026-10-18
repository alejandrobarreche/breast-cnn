"""Paired, stratified bootstrap of the mean per-fold patient AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_auc(positive: np.ndarray, negative: np.ndarray, wp: np.ndarray, wn: np.ndarray):
    """AUC for each row of bootstrap weights; ties count one half (Mann-Whitney)."""
    wins = (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    return ((wp @ wins) * wn).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1))


def bootstrap_auc(
    scores: pd.DataFrame,
    columns: list[str],
    strata: list[str],
    n_resamples: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Mean over folds of the patient AUC for each column, in each bootstrap resample.

    Patients are resampled with replacement within every stratum (which must include fold
    and label), so each resample keeps the number of patients per fold, cohort and class.
    The same resample is applied to every column: differences between columns are paired.
    A resample is encoded as multinomial counts per patient, which turns each AUC into a
    weighted Mann-Whitney statistic.
    """
    if not {"fold", "label"} <= set(strata):
        raise ValueError("Strata must include fold and label")
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_resamples, len(scores)))
    for positions in scores.groupby(strata).indices.values():
        draws = rng.multinomial(
            len(positions), np.full(len(positions), 1 / len(positions)), size=n_resamples
        )
        counts[:, positions] = draws
    folds = scores["fold"].to_numpy()
    labels = scores["label"].to_numpy()
    result = {}
    for column in columns:
        values = scores[column].to_numpy()
        per_fold = []
        for fold in np.unique(folds):
            positive = (folds == fold) & (labels == 1)
            negative = (folds == fold) & (labels == 0)
            per_fold.append(
                _weighted_auc(
                    values[positive], values[negative], counts[:, positive], counts[:, negative]
                )
            )
        result[column] = np.mean(per_fold, axis=0)
    return pd.DataFrame(result)


def paired_differences(
    observed: pd.Series, resamples: pd.DataFrame, reference: str, level: float = 0.95
) -> pd.DataFrame:
    """Difference of each column with the reference: observed value and percentile interval.

    `p_not_better` is the fraction of resamples in which the column does not exceed the
    reference; it is a descriptive bootstrap proportion, not a corrected p-value.
    """
    differences = resamples.sub(resamples[reference], axis=0).drop(columns=reference)
    tail = (1 - level) / 2
    return pd.DataFrame(
        {
            "difference": observed.drop(reference) - observed[reference],
            "lower": differences.quantile(tail),
            "upper": differences.quantile(1 - tail),
            "p_not_better": (differences <= 0).mean(),
        }
    ).sort_values("difference")
