#!/usr/bin/env python3
"""Generate train-only EDA. Accepts Hydra overrides, e.g. eda.seed=2026."""

import argparse
from pathlib import Path

from breast_cnn.configuration import load_phase_one_config
from breast_cnn.data.reporting import run_train_eda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, _, eda = load_phase_one_config(root, args.overrides)
    report = run_train_eda(root, data, eda)
    print(f"Train EDA: {report['patients']} patients, {report['slices']} slices")
    print(f"Report: {eda.report_path}; figures: {len(report['figures'])}")


if __name__ == "__main__":
    main()
