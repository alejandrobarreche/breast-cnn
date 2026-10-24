"""Analysis of finished runs: epoch selection, cohort baseline, bootstrap and ensemble."""

from breast_cnn.evaluation.bootstrap import bootstrap_auc, paired_differences
from breast_cnn.evaluation.calibration import (
    apply_platt,
    calibrate_and_threshold,
    fit_platt,
    nested_ensemble_scores,
    specificity_threshold,
)
from breast_cnn.evaluation.ensemble import (
    CalibratedEnsemble,
    load_fold_models,
    predict_ensemble,
)
from breast_cnn.evaluation.final import evaluate_test, patient_test_scores
from breast_cnn.evaluation.runs import (
    Run,
    cohort_auc,
    cohort_baseline,
    epoch_selection,
    fold_auc,
    load_runs,
    patient_scores,
    selection_summary,
)

__all__ = [
    "CalibratedEnsemble",
    "Run",
    "apply_platt",
    "bootstrap_auc",
    "calibrate_and_threshold",
    "cohort_auc",
    "cohort_baseline",
    "epoch_selection",
    "evaluate_test",
    "fit_platt",
    "fold_auc",
    "load_fold_models",
    "load_runs",
    "nested_ensemble_scores",
    "paired_differences",
    "patient_scores",
    "patient_test_scores",
    "predict_ensemble",
    "selection_summary",
    "specificity_threshold",
]
