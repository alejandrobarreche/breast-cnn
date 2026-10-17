#!/usr/bin/env python3
"""Train experiments over the internal folds, or resume an interrupted run.

  python scripts/train.py +experiment=baseline_weighted
  python scripts/train.py --experiments augmented_slow patient_mean patient_max
  python scripts/train.py --resume runs/<id>

With --experiments, each name in configs/experiment/ runs in turn; a failure is
reported and the next experiment still runs. Extra overrides apply to all of them.
"""

import argparse
import sys
import traceback
from pathlib import Path

from breast_cnn.configuration import load_train_config
from breast_cnn.training import resume_experiment, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=<name>")
    parser.add_argument("--experiments", nargs="+", metavar="NAME", help="run several in order")
    parser.add_argument("--resume", type=Path, help="interrupted run directory to continue")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.resume:
        if args.overrides or args.experiments:
            parser.error("--resume uses the run's saved configuration only")
        run_dir = resume_experiment(root, args.resume.resolve())
        print(f"Run: {run_dir.relative_to(root).as_posix()}")
        return
    if not args.experiments:
        data, train = load_train_config(root, args.overrides)
        run_dir = run_experiment(root, data, train)
        print(f"Run: {run_dir.relative_to(root).as_posix()}")
        return

    available = {path.stem for path in (root / "configs/experiment").glob("*.yaml")}
    unknown = [name for name in args.experiments if name not in available]
    if unknown:
        parser.error(f"unknown experiments {unknown}; available: {sorted(available)}")
    finished, failed = [], []
    for position, name in enumerate(args.experiments, start=1):
        print(f"=== Experiment {position}/{len(args.experiments)}: {name}", flush=True)
        try:
            data, train = load_train_config(root, [f"+experiment={name}", *args.overrides])
            finished.append(run_experiment(root, data, train).relative_to(root).as_posix())
        except Exception:
            traceback.print_exc()
            failed.append(name)
    print("\nFinished runs:", *finished or ["none"], sep="\n  ")
    if failed:
        print("Failed experiments:", *failed, sep="\n  ")
    if finished:
        print(f"\nTo publish them:\n  git add {' '.join(finished)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
