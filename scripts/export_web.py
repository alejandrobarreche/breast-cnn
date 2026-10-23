#!/usr/bin/env python3
"""Prepare the web application's model, manifest and example slices.

  python scripts/export_web.py

Run after scripts/export_onnx.py. Copies the ONNX model into web/public/model/ (outside
Git), copies a few test slices into web/public/examples/ and writes web/src/manifest.json:
model checksum and threshold, architecture, training settings, internal and test metrics,
the patient split (fold, pCR, cohort and slices of each train patient; counts only for test),
and the examples with the outputs ONNX Runtime gives them in Python, which the web parity
test must reproduce.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnxruntime
import pandas as pd
from omegaconf import OmegaConf
from torch import nn

from breast_cnn.configuration import load_analysis_config
from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.metadata import (
    load_excluded_patients,
    load_patients,
    load_samples,
    resolve_image_path,
)
from breast_cnn.models import ModelConfig, build_model

PHASES = ("PRE", "EARLY", "LATE")
OUTPUTS = ("fold_probabilities", "probability", "calibrated", "fold_calibrated")
EXAMPLES_PER_COHORT = 2


def _stages(config: ModelConfig) -> list[dict]:
    """Output volume and parameter count of each convolutional stage."""
    model = build_model(config)
    size = config.input_crop or 256
    rows, params = [], 0
    for layer in model.features:
        params += sum(p.numel() for p in layer.parameters())
        if isinstance(layer, nn.MaxPool2d):
            stage = config.stages[len(rows)]
            rows.append(
                {"channels": stage.channels, "convs": stage.convs, "size": size, "params": params}
            )
            size //= 2
            params = 0
    return rows


def _split(samples: pd.DataFrame, cohorts: pd.Series, excluded: int) -> dict:
    """Patient-level split: one row per train patient and counts per cohort for test."""
    patients = samples.groupby("patient_id").agg(
        split=("split", "first"),
        fold=("fold", "first"),
        pCR=("pCR", "first"),
        slices=("sample_id", "size"),
    )
    patients["cohort"] = patients.index.map(cohorts)
    names = sorted(patients["cohort"].unique())
    train = patients.loc[patients["split"].eq("train")].sort_values(["fold", "cohort", "pCR"])
    test = patients.loc[patients["split"].eq("test")]
    return {
        "cohorts": names,
        "excluded_patients": excluded,
        # [fold, pCR, cohort index, slices] per train patient
        "train": [
            [int(r.fold), int(r.pCR), names.index(r.cohort), int(r.slices)]
            for r in train.itertuples()
        ],
        "test": {
            "patients": len(test),
            "slices": int(test["slices"].sum()),
            "by_cohort": [int(test["cohort"].eq(name).sum()) for name in names],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, analysis = load_analysis_config(root, args.overrides)
    web = root / "web"
    export = json.loads((root / analysis.onnx_report_path).read_text("utf-8"))
    model_path = root / analysis.onnx_path
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != export["sha256"]:
        raise ValueError("The ONNX file does not match its export report; rerun export_onnx.py")
    slice_calibration = json.loads(
        (root / analysis.slice_calibration_report_path).read_text("utf-8")
    )
    test = json.loads((root / analysis.test_report_path).read_text("utf-8"))
    run = OmegaConf.to_container(OmegaConf.load(root / analysis.final_run / "config.yaml"))
    model_config = ModelConfig.model_validate(run["model"])
    train = run["train"]

    (web / "public/model").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, web / "public/model/ensemble.onnx")

    samples = load_samples(root, data.metadata_dir)
    cohorts = load_patients(root, data.metadata_dir).set_index("pid")["data"]
    test_samples = samples.loc[samples["split"].eq("test")].copy()
    test_samples["cohort"] = test_samples["patient_id"].map(cohorts)
    chosen = []
    for _, rows in test_samples.groupby("cohort"):
        for patient in (
            rows["patient_id"]
            .drop_duplicates()
            .sample(EXAMPLES_PER_COHORT, random_state=analysis.seed)
        ):
            slices = rows.loc[rows["patient_id"].eq(patient)].sort_values("slice_index")
            chosen.append(slices.iloc[len(slices) // 2])
    chosen = pd.DataFrame(chosen)
    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dataset = BreastDCEDataset(chosen, root, path_aliases=data.image_path_aliases)
    examples_dir = web / "public/examples"
    shutil.rmtree(examples_dir, ignore_errors=True)
    examples_dir.mkdir(parents=True)
    examples = []
    for index, row in enumerate(chosen.itertuples(index=False)):
        files = {}
        for phase in PHASES:
            source = resolve_image_path(
                getattr(row, f"path_{phase.lower()}"), root, data.image_path_aliases
            )
            files[phase] = f"examples/{row.sample_id}_{phase}.png"
            shutil.copyfile(source, web / "public" / files[phase])
        outputs = session.run(list(OUTPUTS), {"images": dataset[index]["image"][None].numpy()})
        examples.append(
            {
                "sample_id": row.sample_id,
                "cohort": row.cohort,
                "label": int(row.pCR),
                "files": files,
                "expected": {n: o.ravel().tolist() for n, o in zip(OUTPUTS, outputs, strict=True)},
            }
        )

    train_slices = samples.loc[samples["split"].eq("train")]
    manifest = {
        "model": {
            "file": "model/ensemble.onnx",
            "sha256": export["sha256"],
            "bytes": export["bytes"],
            "run": analysis.final_run,
            "input": export["input"],
            "outputs": list(OUTPUTS),
            "threshold": slice_calibration["threshold"],
            "min_specificity": slice_calibration["min_specificity"],
            "train_pcr_rate": float(train_slices["pCR"].mean()),
        },
        "architecture": {
            "input_crop": model_config.input_crop,
            "kernel_size": model_config.kernel_size,
            "stages": _stages(model_config),
            "params": sum(p.numel() for p in build_model(model_config).parameters()),
            "dropout": model_config.dropout,
            "models": len(slice_calibration["folds"]),
        },
        "training": {
            "experiment": train["experiment"],
            "loss": "BCE ponderada" if train["weighted_loss"] else "BCE",
            "optimizer": train["optimizer"],
            "learning_rate": train["learning_rate"],
            "epochs": train["epochs"],
            "rotation_degrees": train["rotation_degrees"],
            "max_translation": train["max_translation"],
            "scale_range": train["scale_range"],
            "horizontal_flip_probability": train["horizontal_flip_probability"],
        },
        "validation_slice": slice_calibration["pooled_validation"],
        "test_patient": {
            "patients": test["patients"],
            "positives": test["positives"],
            "auc": test["auc"],
            "cohort_baseline_auc": test["cohort_baseline"]["auc"],
        },
        "split": _split(samples, cohorts, len(load_excluded_patients(root, data.metadata_dir))),
        "examples": examples,
    }
    path = web / "src/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"Manifest: {path.relative_to(root).as_posix()} · {len(examples)} examples")


if __name__ == "__main__":
    main()
