#!/usr/bin/env python3
"""Export the final ensemble with its slice calibration to ONNX and check parity.

  python scripts/export_onnx.py

Reads analysis.final_run and analysis.slice_calibration_report_path. Writes the model to
analysis.onnx_path (outside Git) and a report with its SHA-256, the threshold, the largest
PyTorch-ONNX difference on validation slices and reference slices for the web parity test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from breast_cnn.configuration import load_analysis_config
from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.metadata import load_samples
from breast_cnn.evaluation import CalibratedEnsemble, load_fold_models

OUTPUTS = ("fold_probabilities", "probability", "calibrated", "fold_calibrated")
TOLERANCE = 1e-5
REFERENCE_SLICES = 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, analysis = load_analysis_config(root, args.overrides)
    calibration = json.loads((root / analysis.slice_calibration_report_path).read_text("utf-8"))
    if calibration["run"] != analysis.final_run or calibration["unit"] != "slice":
        raise ValueError("The slice calibration does not belong to the final run")
    device = torch.device("cpu")
    models = load_fold_models(root / analysis.final_run, device)
    ensemble = CalibratedEnsemble(models, calibration["platt"])
    ensemble.eval()

    path = root / analysis.onnx_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            ensemble,
            (torch.zeros(1, 3, 256, 256),),
            str(path),
            input_names=["images"],
            output_names=list(OUTPUTS),
            dynamo=True,
        )
    # One self-contained file (the exporter writes weights apart), so one checksum covers all.
    external = path.with_name(path.name + ".data")
    onnx.save_model(onnx.load(str(path)), str(path), save_as_external_data=False)
    external.unlink(missing_ok=True)
    onnx.checker.check_model(str(path))

    samples = load_samples(root, data.metadata_dir)
    validation = samples.loc[samples["split"].eq("train")]
    chosen = validation.sample(analysis.onnx_parity_slices, random_state=analysis.seed)
    dataset = BreastDCEDataset(chosen, root, path_aliases=data.image_path_aliases)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    largest = dict.fromkeys(OUTPUTS, 0.0)
    references = []
    for index in range(len(dataset)):
        item = dataset[index]
        image = item["image"][None]
        with torch.no_grad():
            expected = [tensor.numpy() for tensor in ensemble(image)]
        actual = session.run(list(OUTPUTS), {"images": image.numpy()})
        for name, want, got in zip(OUTPUTS, expected, actual, strict=True):
            largest[name] = max(largest[name], float(np.abs(want - got).max()))
        if index < REFERENCE_SLICES:
            row = chosen.iloc[index]
            references.append(
                {
                    "sample_id": item["sample_id"],
                    "paths": [row["path_pre"], row["path_early"], row["path_late"]],
                    **{n: v.ravel().tolist() for n, v in zip(OUTPUTS, expected, strict=True)},
                }
            )
    if max(largest.values()) > TOLERANCE:
        raise ValueError(f"ONNX differs from PyTorch by more than {TOLERANCE}: {largest}")

    report = {
        "run": analysis.final_run,
        "calibration_report": analysis.slice_calibration_report_path,
        "model": analysis.onnx_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "opset": onnx.load(str(path)).opset_import[0].version,
        "input": {"name": "images", "shape": [1, 3, 256, 256], "dtype": "float32"},
        "outputs": list(OUTPUTS),
        "threshold": calibration["threshold"],
        "threshold_score": "calibrated",
        "parity": {"slices": len(dataset), "max_abs_difference": largest, "tolerance": TOLERANCE},
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "references": references,
    }
    report_path = root / analysis.onnx_report_path
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8")
    print(
        f"Model: {analysis.onnx_path} ({report['bytes'] / 1e6:.1f} MB), sha256 {report['sha256'][:12]}"
    )
    print(f"Largest PyTorch-ONNX difference on {len(dataset)} slices: {max(largest.values()):.2e}")


if __name__ == "__main__":
    main()
