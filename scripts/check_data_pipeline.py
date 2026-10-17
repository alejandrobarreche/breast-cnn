#!/usr/bin/env python3
"""Check real train/validation loaders. Accepts Hydra overrides."""

import argparse
from pathlib import Path

from breast_cnn.configuration import load_phase_one_config
from breast_cnn.data.reporting import check_data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, pipeline, _ = load_phase_one_config(root, args.overrides)
    report = check_data_pipeline(root, data, pipeline)
    print(f"Real train batch: {report['train_batch']['shape']}, {report['train_batch']['dtype']}")
    print(f"Deterministic full validation passes: {report['validation']['passes']}")
    print(f"Report: {pipeline.report_path}")


if __name__ == "__main__":
    main()
