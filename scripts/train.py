#!/usr/bin/env python3
"""Train one experiment over the internal folds, or resume an interrupted run.

python scripts/train.py +experiment=baseline_weighted
python scripts/train.py --resume runs/<id>
"""

import argparse
from pathlib import Path

from breast_cnn.configuration import load_train_config
from breast_cnn.training import resume_experiment, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=<name>")
    parser.add_argument("--resume", type=Path, help="interrupted run directory to continue")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.resume:
        if args.overrides:
            parser.error("--resume uses the run's saved configuration; do not pass overrides")
        run_dir = resume_experiment(root, args.resume.resolve())
    else:
        data, train = load_train_config(root, args.overrides)
        run_dir = run_experiment(root, data, train)
    print(f"Run: {run_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
