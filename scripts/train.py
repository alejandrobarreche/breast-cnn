#!/usr/bin/env python3
"""Train one experiment over the internal folds, e.g. `+experiment=baseline_weighted`."""

import argparse
from pathlib import Path

from breast_cnn.configuration import load_train_config
from breast_cnn.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=<name>")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, train = load_train_config(root, args.overrides)
    run_dir = run_experiment(root, data, train)
    print(f"Run: {run_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
