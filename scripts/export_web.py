#!/usr/bin/env python3
"""Prepare the web application's model, manifest and example slices.

  python scripts/export_web.py

Run after scripts/export_onnx.py. Copies the ONNX model into web/public/model/ (outside
Git), copies a few test slices into web/public/examples/ and writes web/src/manifest.json:
model checksum and threshold, architecture, training settings, internal and test metrics,
the patient split (fold, pCR, cohort and slices of each train patient; counts only for test),
and the examples with the outputs ONNX Runtime gives them in Python, which the web parity
test must reproduce. Also writes web/src/runs.json for the experiment lab: per run, what it
changed with respect to the reference experiment, the validation patient AUC of each fold
(epoch chosen without the evaluated fold, or the last epoch for fixed-epoch runs, and the
optimistic best epoch), the mean learning curves, and the cohort-only baseline. And writes
web/src/results.json for the results charts: the final run's out-of-fold validation patients
and the test patients (label, raw and calibrated mean probability), the patient threshold, and
the per-fold history of the run whose curves fixed the number of epochs of the final model.
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
from breast_cnn.evaluation.calibration import fit_fold_calibration
from breast_cnn.evaluation.final import patient_test_scores
from breast_cnn.evaluation.runs import (
    cohort_baseline,
    epoch_selection,
    fold_auc,
    load_runs,
    patient_scores,
)
from breast_cnn.models import ModelConfig, build_model

PHASES = ("PRE", "EARLY", "LATE")
OUTPUTS = ("fold_probabilities", "probability", "calibrated", "fold_calibrated")
EXAMPLES_PER_COHORT = 2
LAB_REFERENCE = "baseline_weighted"
# Run whose per-fold curves fixed the number of epochs of the final model
EPOCH_REFERENCE = "wide_crop128_slow"
# Settings that identify a run or the machine, not the experiment
IGNORED = {"experiment", "model", "folds", "seed", "num_workers", "runs_dir", "threshold"}


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


def _settings(run: dict, defaults: dict) -> dict:
    """Training settings (missing keys take today's defaults) and the validated architecture."""
    train = {k: v for k, v in {**defaults, **run["train"]}.items() if k not in IGNORED}
    model = ModelConfig.model_validate(run["model"]).model_dump()
    return {**train, **{f"model.{k}": v for k, v in model.items()}}


def _lab(root: Path, runs_dir: str, final_run: str, patients: pd.DataFrame) -> dict:
    """Validation results of every finished run for the web lab; internal validation only."""
    runs = load_runs(root / runs_dir)
    defaults = OmegaConf.to_container(OmegaConf.load(root / "configs/config.yaml"))["train"]
    settings = {
        run.experiment: _settings(
            OmegaConf.to_container(OmegaConf.load(run.directory / "config.yaml")), defaults
        )
        for run in runs
    }
    reference = settings[LAB_REFERENCE]
    scores = patient_scores(runs[:1], patients)
    scores["cohort_baseline"] = cohort_baseline(scores)
    baseline = fold_auc(scores, ["cohort_baseline"])["cohort_baseline"]
    experiments = []
    for run in runs:
        last = run.metrics["selection"] == "last_epoch"
        folds = epoch_selection(run.history)
        last_epoch = run.history.groupby("fold")["validation_patient_auc"].last()
        honest = last_epoch.to_numpy() if last else folds["out_of_fold"].to_numpy()
        # The first runs did not log the train AUC: their train curve is left empty
        history = run.history.copy()
        if "train_patient_auc" not in history:
            history["train_patient_auc"] = float("nan")
        curves = history.groupby("epoch")[["train_patient_auc", "validation_patient_auc"]].mean()
        own = settings[run.experiment]
        experiments.append(
            {
                "name": run.experiment,
                "final": run.directory.resolve() == (root / final_run).resolve(),
                "selection": "last" if last else "out_of_fold",
                "changes": {k: v for k, v in own.items() if reference.get(k) != v},
                "auc": [float(x) for x in honest],
                "best_auc": folds["best"].tolist(),
                "curves": {
                    "epoch": curves.index.tolist(),
                    "train": [
                        None if pd.isna(x) else round(x, 4) for x in curves["train_patient_auc"]
                    ],
                    "validation": curves["validation_patient_auc"].round(4).tolist(),
                },
            }
        )
    return {
        "reference": LAB_REFERENCE,
        "reference_settings": reference,
        "cohort_baseline": baseline.tolist(),
        "experiments": experiments,
    }


def _results(root: Path, analysis, calibration: dict) -> dict:
    """Patient scores of internal validation and test, and the curves that fixed the epochs."""
    run_dir = root / analysis.final_run
    _, validation = fit_fold_calibration(pd.read_csv(run_dir / "predictions_validation.csv"))
    test = patient_test_scores(pd.read_csv(root / analysis.test_predictions_path), calibration)
    reference = next(
        r for r in load_runs(root / analysis.runs_dir) if r.experiment == EPOCH_REFERENCE
    )
    columns = ["train_loss", "validation_loss", "validation_patient_auc"]
    return {
        "threshold": calibration["threshold"],
        # [fold, label, raw, calibrated] per validation patient; [label, raw, calibrated] in test
        "validation": [
            [int(r.fold), int(r.label), float(r.probability), float(r.calibrated)]
            for r in validation.itertuples()
        ],
        "test": [[int(r.label), float(r.raw), float(r.calibrated)] for r in test.itertuples()],
        "epochs": {
            "experiment": EPOCH_REFERENCE,
            "chosen": int(OmegaConf.load(run_dir / "config.yaml").train.epochs),
            "folds": [
                {"epoch": rows["epoch"].tolist(), **{c: rows[c].round(4).tolist() for c in columns}}
                for _, rows in reference.history.groupby("fold")
            ],
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
    lab = _lab(root, analysis.runs_dir, analysis.final_run, load_patients(root, data.metadata_dir))
    calibration = json.loads((root / analysis.calibration_report_path).read_text("utf-8"))
    results = _results(root, analysis, calibration)
    (web / "src/results.json").write_text(json.dumps(results) + "\n", "utf-8")
    (web / "src/runs.json").write_text(json.dumps(lab, ensure_ascii=False) + "\n", "utf-8")
    path = web / "src/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"Manifest: {path.relative_to(root).as_posix()} · {len(examples)} examples")


if __name__ == "__main__":
    main()
