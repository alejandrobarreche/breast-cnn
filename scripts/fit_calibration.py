#!/usr/bin/env python3
"""Fit per-fold Platt calibration and the patient threshold on internal validation only.

  python scripts/fit_calibration.py
  python scripts/fit_calibration.py analysis.final_run=runs/<id>

Writes analysis.calibration_report_path; the test evaluation reads it unchanged.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from breast_cnn.configuration import load_analysis_config
from breast_cnn.evaluation import calibrate_and_threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _, analysis = load_analysis_config(root, args.overrides)
    run_dir = root / analysis.final_run
    predictions = pd.read_csv(run_dir / "predictions_validation.csv")
    report = {
        "run": analysis.final_run,
        **calibrate_and_threshold(predictions, analysis.min_specificity),
    }
    path = root / analysis.calibration_report_path
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
