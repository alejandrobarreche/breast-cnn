from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from breast_cnn.evaluation import (
    CalibratedEnsemble,
    apply_platt,
    bootstrap_auc,
    calibrate_and_threshold,
    cohort_baseline,
    epoch_selection,
    evaluate_test,
    fit_platt,
    fold_auc,
    nested_ensemble_scores,
    paired_differences,
    patient_test_scores,
    specificity_threshold,
)
from breast_cnn.evaluation.bootstrap import _weighted_auc
from breast_cnn.models import ModelConfig, build_model


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


def test_platt_recovers_a_known_logistic_map() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 20000)
    true = apply_platt(raw, {"slope": 0.5, "intercept": -0.8})
    labels = rng.binomial(1, true)
    platt = fit_platt(raw, labels)
    assert platt["slope"] == pytest.approx(0.5, abs=0.05)
    assert platt["intercept"] == pytest.approx(-0.8, abs=0.05)


def test_platt_slope_is_constrained_to_be_nonnegative() -> None:
    raw = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])  # higher probability, fewer positives
    platt = fit_platt(raw, labels)
    assert platt == {"slope": 0.0, "intercept": pytest.approx(0.0), "slope_constrained": True}
    assert apply_platt(raw, platt) == pytest.approx(np.full(8, 0.5))


def test_specificity_threshold_is_the_lowest_meeting_the_constraint() -> None:
    labels = np.array([0] * 10 + [1] * 3)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.85, 0.92, 0.99])
    threshold = specificity_threshold(labels, scores, 0.9)
    assert threshold == pytest.approx(0.92)  # 9 of 10 negatives below; 0.9 would give 8/10
    predicted = scores >= threshold
    assert (~predicted[labels == 0]).mean() >= 0.9
    assert predicted[labels == 1].sum() == 2


def _nested_predictions(rng: np.random.Generator) -> pd.DataFrame:
    """Three folds of 100 patients (two slices each), each scored by the two models whose
    pair excludes its fold, as a nested run over folds [0, 1, 2] writes them."""
    rows = []
    for fold in range(3):
        models = [f"{min(fold, other)}-{max(fold, other)}" for other in range(3) if other != fold]
        for patient in range(100):
            label = int(patient < 30)
            for z in range(2):
                for model in models:
                    rows.append(
                        {
                            "fold": fold,
                            "model": model,
                            "sample_id": f"{fold}_{patient}_z{z}",
                            "patient_id": f"{fold}_{patient}",
                            "label": label,
                            "probability": float(
                                np.clip(0.4 + 0.1 * label + rng.normal(0, 0.1), 0.01, 0.99)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_nested_scores_average_the_out_of_fold_models() -> None:
    predictions = _nested_predictions(np.random.default_rng(1))
    slices = nested_ensemble_scores(predictions, "slice")
    assert len(slices) == 600
    expected = predictions.groupby("sample_id")["probability"].mean()
    assert slices.set_index("sample_id")["probability"].loc[expected.index].tolist() == (
        pytest.approx(expected.tolist())
    )
    patients = nested_ensemble_scores(predictions)
    assert len(patients) == 300 and set(patients["fold"]) == {0, 1, 2}
    with pytest.raises(ValueError):
        nested_ensemble_scores(predictions.iloc[1:])  # one slice loses a model
    with pytest.raises(ValueError):
        nested_ensemble_scores(predictions, "cohort")


def test_calibration_report_fits_one_map_and_threshold_on_the_ensemble_mean() -> None:
    predictions = _nested_predictions(np.random.default_rng(1))
    report = calibrate_and_threshold(predictions, 0.9)
    assert report["models_per_unit"] == 2 and set(report["folds"]) == {0, 1, 2}
    patients = nested_ensemble_scores(predictions)
    calibrated = apply_platt(patients["probability"], report["platt"])
    # Platt keeps the ranking and, fitted by maximum likelihood, matches the pCR rate.
    assert calibrated.mean() == pytest.approx(patients["label"].mean(), abs=1e-3)
    assert report["pooled_validation"]["specificity"] >= 0.9
    negatives = calibrated[patients["label"].eq(0)]
    assert (negatives < report["threshold"]).mean() == pytest.approx(
        report["pooled_validation"]["specificity"]
    )
    slices = calibrate_and_threshold(predictions, 0.9, unit="slice")
    assert slices["unit"] == "slice" and slices["pooled_validation"]["n"] == 600


def test_test_evaluation_on_synthetic_patients() -> None:
    rng = np.random.default_rng(2)
    calibration = {"threshold": 0.3, "platt": {"slope": 0.5, "intercept": -0.4}}
    rows = []
    for patient in range(80):
        label = int(patient < 30)
        for _ in range(3):
            rows.append(
                {
                    "patient_id": f"p{patient:02d}",
                    "label": label,
                    "probability_fold0": float(
                        np.clip(0.3 + 0.2 * label + rng.normal(0, 0.1), 0.01, 0.99)
                    ),
                    "probability_fold1": float(rng.uniform(0.2, 0.8)),
                }
            )
    slices = pd.DataFrame(rows)
    patients = patient_test_scores(slices, calibration)
    expected_raw = slices.groupby("patient_id")[["probability_fold0", "probability_fold1"]].mean()
    assert patients["raw"].tolist() == pytest.approx(expected_raw.mean(axis=1).tolist())
    expected = apply_platt(expected_raw.mean(axis=1), calibration["platt"])
    assert patients["calibrated"].tolist() == pytest.approx(expected.tolist())

    cohorts = pd.Series(np.where(np.arange(80) % 2, "a", "b"), index=patients.index)
    train = pd.DataFrame({"data": ["a", "a", "b", "b"], "pCR": [1, 0, 0, 0]})
    report = evaluate_test(patients, cohorts, train, calibration, 200, 0.95, 0)
    assert report["patients"] == 80 and report["positives"] == 30
    assert report["auc"]["lower"] <= report["auc"]["value"] <= report["auc"]["upper"]
    assert report["auc"]["value"] == pytest.approx(
        roc_auc_score(patients["label"], patients["raw"])
    )
    assert report["cohort_baseline"]["train_pcr_rate"] == {"a": 0.5, "b": 0.0}
    assert report["always_no_pcr"]["sensitivity"] == 0.0
    assert report["always_no_pcr"]["accuracy"] == pytest.approx(50 / 80)
    counts = report["at_threshold"]
    assert counts["true_positives"] + counts["false_negatives"] == 30
    assert set(report["by_cohort"]) == {"a", "b"}


def _tiny_models() -> dict[int, torch.nn.Module]:
    config = ModelConfig(
        in_channels=3,
        kernel_size=3,
        activation="relu",
        stages=[{"channels": 4, "convs": 1}],
        dropout=0.0,
        input_crop=128,
    )
    models = {}
    for fold in (0, 1):
        torch.manual_seed(fold)
        models[fold] = build_model(config).eval()
    return models


def test_calibrated_ensemble_matches_the_numpy_calibration() -> None:
    platt = {"slope": 0.4, "intercept": -0.9}
    models = _tiny_models()
    ensemble = CalibratedEnsemble(models, platt).eval()
    images = torch.rand(3, 3, 256, 256)
    with torch.no_grad():
        folds, raw, calibrated, fold_calibrated = ensemble(images)
        expected = torch.stack([torch.sigmoid(models[f](images)) for f in (0, 1)], dim=1)
    assert folds.numpy() == pytest.approx(expected.numpy(), abs=1e-7)
    assert raw.numpy() == pytest.approx(expected.mean(dim=1).numpy(), abs=1e-7)
    numpy_calibrated = apply_platt(expected.mean(dim=1).numpy(), platt)
    assert calibrated.numpy() == pytest.approx(numpy_calibrated, abs=1e-6)
    numpy_folds = apply_platt(expected.numpy(), platt)
    assert fold_calibrated.numpy() == pytest.approx(numpy_folds, abs=1e-6)


def test_calibrated_ensemble_exports_to_onnx_with_parity(tmp_path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    ensemble = CalibratedEnsemble(_tiny_models(), {"slope": 0.4, "intercept": -0.9}).eval()
    path = tmp_path / "ensemble.onnx"
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        torch.onnx.export(ensemble, (image,), str(path), input_names=["images"], dynamo=True)
        expected = [t.numpy() for t in ensemble(image)]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": image.numpy()})
    for want, got in zip(expected, actual, strict=True):
        assert np.abs(want - got).max() < 1e-5
