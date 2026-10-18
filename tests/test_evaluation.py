from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from breast_cnn.evaluation import (
    bootstrap_auc,
    cohort_baseline,
    epoch_selection,
    fold_auc,
    paired_differences,
)
from breast_cnn.evaluation.bootstrap import _weighted_auc


def test_out_of_fold_selection_ignores_the_evaluated_fold() -> None:
    # Fold 0 peaks at epoch 3, the other folds at epoch 1; fold 2 stops after epoch 2.
    curves = {0: [0.50, 0.55, 0.90], 1: [0.70, 0.60, 0.50], 2: [0.68, 0.62]}
    history = pd.DataFrame(
        [
            {"fold": fold, "epoch": epoch, "validation_patient_auc": auc}
            for fold, values in curves.items()
            for epoch, auc in enumerate(values, start=1)
        ]
    )
    folds = epoch_selection(history).set_index("fold")
    assert folds.loc[0, "best_epoch"] == 3
    assert folds.loc[0, "best"] == pytest.approx(0.90)
    assert folds.loc[0, "all_epochs"] == pytest.approx(0.65)
    assert folds.loc[0, "out_of_fold_epoch"] == 1
    assert folds.loc[0, "out_of_fold"] == pytest.approx(0.50)
    # Epoch 3 is not a candidate for any fold because fold 2 never reached it.
    assert folds["out_of_fold_epoch"].le(2).all()


def test_cohort_baseline_uses_only_other_folds() -> None:
    scores = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1, 1, 1],
            "label": [1, 1, 0, 1, 0, 0],
            "cohort": ["a", "b", "a", "a", "b", "b"],
        }
    )
    baseline = cohort_baseline(scores)
    assert baseline.tolist() == pytest.approx([0.5, 0.0, 1.0, 1.0, 1.0, 1.0])


def test_weighted_auc_with_unit_weights_matches_sklearn_with_ties() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    values = np.array([0.9, 0.4, 0.4, 0.4, 0.2, 0.9, 0.1])
    positive, negative = values[labels == 1], values[labels == 0]
    auc = _weighted_auc(positive, negative, np.ones((1, 3)), np.ones((1, 4)))
    assert auc[0] == pytest.approx(roc_auc_score(labels, values))


def test_paired_bootstrap_is_centred_on_observed_and_paired() -> None:
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"fold": np.repeat([0, 1], 100), "label": np.tile([0, 1], 100)})
    scores["a"] = scores["label"] + rng.normal(0, 1.5, len(scores))
    scores["b"] = scores["a"]
    observed = fold_auc(scores, ["a", "b"]).mean()
    resamples = bootstrap_auc(scores, ["a", "b"], ["fold", "label"], n_resamples=500)
    assert resamples["a"].mean() == pytest.approx(observed["a"], abs=0.02)
    differences = paired_differences(observed, resamples, reference="a")
    assert differences.loc["b"].tolist() == pytest.approx([0, 0, 0, 1])
    with pytest.raises(ValueError):
        bootstrap_auc(scores, ["a"], ["fold"])
