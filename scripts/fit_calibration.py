#!/usr/bin/env python3
"""Fit the Platt map and the threshold of the final ensemble on internal validation only.

  python scripts/fit_calibration.py                # per patient (test evaluation)
  python scripts/fit_calibration.py --unit slice   # per slice (web application)

Reads the out-of-fold ensemble scores of analysis.nested_run, which must repeat the training
protocol of analysis.final_run. Writes analysis.calibration_report_path or
analysis.slice_calibration_report_path.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from breast_cnn.configuration import load_analysis_config
from breast_cnn.evaluation import calibrate_and_threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    parser.add_argument("--unit", choices=["patient", "slice"], default="patient")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _, analysis = load_analysis_config(root, args.overrides)
    if analysis.nested_run is None:
        parser.error("set analysis.nested_run to the finished nested run")
    final, nested = (
        OmegaConf.to_container(OmegaConf.load(root / run / "config.yaml"))
        for run in (analysis.final_run, analysis.nested_run)
    )
    differences = {
        key
        for key in final["train"].keys() | nested["train"].keys()
        if final["train"].get(key) != nested["train"].get(key)
    } - {"experiment", "nested", "num_workers"}
    if final["model"] != nested["model"] or differences or not nested["train"]["nested"]:
        raise ValueError(f"The nested run does not repeat the final protocol: {differences}")
    predictions = pd.read_csv(root / analysis.nested_run / "predictions_validation.csv")
    report = {
        "run": analysis.final_run,
        "nested_run": analysis.nested_run,
        **calibrate_and_threshold(predictions, analysis.min_specificity, args.unit),
    }
    path = root / (
        analysis.calibration_report_path
        if args.unit == "patient"
        else analysis.slice_calibration_report_path
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pooled = report["pooled_validation"]
    print(
        f"Threshold {report['threshold']:.4f} · validation sensitivity "
        f"{pooled['sensitivity']:.3f}, specificity {pooled['specificity']:.3f}"
    )
    print(f"Report: {path.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
