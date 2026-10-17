#!/usr/bin/env python3
"""Audit Breast-CNN metadata and optionally decode every source image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from breast_cnn.data.validation import AuditReport, audit_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="OmegaConf YAML configuration (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="decode every image and validate format, dimensions, dtype and range",
    )
    parser.add_argument("--validation-fold", type=int, help="override audit.validation_fold")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with initialize_config_dir(version_base=None, config_dir=str(path.parent.resolve())):
        config = compose(config_name=path.stem)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return resolved


def print_report(report: AuditReport) -> None:
    status = "OK" if report.ok else "FAILED"
    print(f"Dataset audit: {status}")
    print(
        f"Patients: {report.summary.get('patients', 0)} | "
        f"Samples: {report.summary.get('samples', 0)} | "
        f"Excluded: {report.summary.get('excluded_patients', 0)}"
    )
    if "resolved_image_paths" in report.summary:
        print(
            "Image paths: "
            f"{report.summary['resolved_image_paths']}/"
            f"{report.summary['expected_image_paths']} resolved"
        )
    if "training_patients" in report.summary:
        print(
            f"Validation fold {report.summary['validation_fold']}: "
            f"{report.summary['training_patients']} train patients / "
            f"{report.summary['validation_patients']} validation patients"
        )
    missing = report.details.get("patient_missing_values", {})
    print(f"Patient columns with missing values: {missing or 'none'}")
    if "image_inspection" in report.details:
        print(f"Physical image inspection: {report.details['image_inspection']}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        data_config = config["data"]
        audit_config = config["audit"]
        validation_fold = (
            args.validation_fold
            if args.validation_fold is not None
            else int(audit_config["validation_fold"])
        )
        report = audit_dataset(
            PROJECT_ROOT,
            metadata_dir=data_config["metadata_dir"],
            path_aliases=dict(data_config.get("image_path_aliases", {})),
            validation_fold=validation_fold,
            check_images=args.check_images,
            expected_image_size=tuple(data_config["expected_image_size"]),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"Audit configuration error: {error}", file=sys.stderr)
        return 2

    print_report(report)
    if args.output:
        output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Structured report written to: {output_path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
