#!/usr/bin/env python3
"""Evaluate the final ensemble once on test, with the calibration fixed on validation.

  python scripts/evaluate_test.py

Reads analysis.final_run (fold models) and analysis.calibration_report_path (maps and
threshold). Writes the patient-level report and the slice predictions; it refuses to run
if either output already exists, so that test is evaluated only once.
"""

import argparse
import json
from pathlib import Path

from breast_cnn.configuration import load_analysis_config
from breast_cnn.data.metadata import load_patients, load_samples, select_test_samples
from breast_cnn.evaluation import (
    evaluate_test,
    load_fold_models,
    patient_test_scores,
    predict_ensemble,
)
from breast_cnn.training.loop import select_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, analysis = load_analysis_config(root, args.overrides)
    report_path = root / analysis.test_report_path
    predictions_path = root / analysis.test_predictions_path
    for path in (report_path, predictions_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; test is evaluated only once")
    calibration = json.loads((root / analysis.calibration_report_path).read_text("utf-8"))
    if calibration["run"] != analysis.final_run:
        raise ValueError(f"Calibration fitted for {calibration['run']}, not {analysis.final_run}")

    samples = load_samples(root, data.metadata_dir)
    patients = load_patients(root, data.metadata_dir)
    train_ids = samples.loc[samples["split"].eq("train"), "patient_id"].unique()
    device = select_device()
    models = load_fold_models(root / analysis.final_run, device)
    slices = predict_ensemble(models, select_test_samples(samples), root, data, device)
    report = {
        "run": analysis.final_run,
        "calibration_report": analysis.calibration_report_path,
        **evaluate_test(
            patient_test_scores(slices, calibration),
            patients.set_index("pid")["data"],
            patients.loc[patients["pid"].isin(train_ids)],
            calibration,
            analysis.bootstrap_resamples,
            analysis.confidence_level,
            analysis.seed,
        ),
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    slices.to_csv(predictions_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8")

    auc, baseline = report["auc"], report["cohort_baseline"]
    at_threshold = report["at_threshold"]
    print(f"Test: {report['patients']} patients, {report['positives']} pCR")
    print(f"Patient AUC {auc['value']:.3f} [{auc['lower']:.3f}, {auc['upper']:.3f}]")
    print(f"Cohort baseline AUC {baseline['auc']:.3f}")
    print(
        f"Threshold {report['threshold']:.3f}: sensitivity {at_threshold['sensitivity']:.3f}, "
        f"specificity {at_threshold['specificity']:.3f}"
    )
    print(f"Report: {report_path.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
