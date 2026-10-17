#!/usr/bin/env python3
"""Download public BreastDCEDL images into this repository's configured data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from breast_cnn.data.download import build_download_plan, download_images
from breast_cnn.data.metadata import load_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", "--hilos", type=positive_integer)
    parser.add_argument("--patients", "--pacientes", type=positive_integer)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--split", choices=["all", "train", "test"], default="all")
    selection.add_argument("--solo-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show selection without downloading")
    args = parser.parse_args()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
            config = compose(config_name="config")
        split = "test" if args.solo_test else args.split
        items = build_download_plan(
            load_samples(PROJECT_ROOT, config.data.metadata_dir),
            PROJECT_ROOT,
            dict(config.data.image_path_aliases),
            split=split,
            patients=args.patients,
        )
        partial = split != "all" or args.patients is not None
        print(f"Project: {PROJECT_ROOT}")
        print(f"Scope: {'PARTIAL selection' if partial else 'full public dataset'}")
        print(
            f"Images: {len(items)} | Existing paths: {sum(i.destination.is_file() for i in items)}"
        )
        print(f"Mapping: {items[0].remote_path} -> {items[0].destination}")
        if args.dry_run:
            print("Dry run: no network requests, writes or image validation performed.")
            return 0
        report = download_images(
            items,
            base_url=str(config.download.base_url),
            workers=args.workers or int(config.download.workers),
            timeout=float(config.download.timeout_seconds),
            expected_size=tuple(config.data.expected_image_size),
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Download configuration error: {error}", file=sys.stderr)
        return 2

    print(f"Downloaded: {report.downloaded} | Already valid: {report.skipped}")
    if report.failures:
        print(f"FAILED: {len(report.failures)} image(s). Rerun to retry missing files.")
        for failure in report.failures[:5]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    if partial:
        print("Selected images ready. This does not certify a complete dataset download.")
    else:
        print("Public images ready. Next: python scripts/audit_dataset.py --check-images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
