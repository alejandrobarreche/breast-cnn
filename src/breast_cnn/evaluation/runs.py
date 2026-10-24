"""Read finished runs and summarise internal validation without retraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from breast_cnn.training.metrics import aggregate_patients


@dataclass(frozen=True)
class Run:
    experiment: str
    directory: Path
    history: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict


def load_runs(runs_dir: Path) -> list[Run]:
    """Load every finished experiment run (one with metrics.json), ordered by directory name.

    Nested runs are skipped: they serve the threshold of the final model, not the comparison.
    """
    runs = []
    for directory in sorted(path for path in Path(runs_dir).iterdir() if path.is_dir()):
        if not (directory / "metrics.json").is_file():
            continue
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics["scope"] != "internal_validation_folds":
            continue
        runs.append(
            Run(
                experiment=metrics["experiment"],
                directory=directory,
                history=pd.read_csv(directory / "history.csv"),
                predictions=pd.read_csv(directory / "predictions_validation.csv"),
                metrics=metrics,
            )
        )
    names = [run.experiment for run in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"Experiment names must be unique across runs: {names}")
    return runs


def epoch_selection(history: pd.DataFrame) -> pd.DataFrame:
    """Validation patient AUC of each fold under three epoch-selection rules.

    - best: first maximum on the evaluated fold itself (the rule used in training; optimistic).
    - all_epochs: mean over every epoch run on the fold (no selection).
    - out_of_fold: epoch maximising the mean curve of the other folds, read on the evaluated
      fold. Only epochs reached by every fold are candidates, so that the mean curve always
      averages the same folds.
    """
    curves = history.pivot(index="epoch", columns="fold", values="validation_patient_auc")
    common = curves.dropna()
    rows = []
    for fold in curves.columns:
        own = curves[fold].dropna()
        others = common.drop(columns=fold).mean(axis=1)
        epoch = int(others.idxmax())
        rows.append(
            {
                "fold": int(fold),
                "best_epoch": int(own.idxmax()),
                "best": float(own.max()),
                "all_epochs": float(own.mean()),
                "out_of_fold_epoch": epoch,
                "out_of_fold": float(common.loc[epoch, fold]),
            }
        )
    return pd.DataFrame(rows)


def selection_summary(runs: list[Run]) -> pd.DataFrame:
    """Mean and sample standard deviation across folds of each selection rule, per experiment."""
    rows = []
    for run in runs:
        folds = epoch_selection(run.history)
        row = {"experiment": run.experiment}
        for rule in ("best", "all_epochs", "out_of_fold"):
            row[f"{rule}_mean"] = folds[rule].mean()
            row[f"{rule}_std"] = folds[rule].std()
        row["out_of_fold_epochs"] = folds["out_of_fold_epoch"].tolist()
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")


def patient_scores(runs: list[Run], patients: pd.DataFrame) -> pd.DataFrame:
    """One row per validation patient: fold, label, cohort and one probability per experiment.

    Probabilities are the mean over slices at the epoch stored in predictions_validation.csv.
    Every run must cover the same patients in the same folds, so that comparisons are paired.
    """
    frame = None
    for run in runs:
        folds = run.predictions.groupby("patient_id")["fold"].first()
        scores = aggregate_patients(run.predictions).set_index("patient_id")
        scores = scores.rename(columns={"probability": run.experiment}).join(folds)
        if frame is None:
            frame = scores[["fold", "label", run.experiment]]
            continue
        if not scores.index.equals(frame.index) or not scores["fold"].equals(frame["fold"]):
            raise ValueError(f"{run.experiment} covers different patients or folds")
        frame[run.experiment] = scores[run.experiment]
    cohorts = patients.set_index("pid")["data"].rename("cohort")
    return frame.join(cohorts, how="left").astype({"fold": int, "label": int})


def cohort_baseline(scores: pd.DataFrame) -> pd.Series:
    """Score each validation patient with the pCR rate of its cohort in the other folds."""
    baseline = pd.Series(np.nan, index=scores.index, name="cohort_baseline")
    for fold in scores["fold"].unique():
        training = scores.loc[scores["fold"].ne(fold)]
        rates = training.groupby("cohort")["label"].mean()
        validation = scores["fold"].eq(fold)
        baseline[validation] = scores.loc[validation, "cohort"].map(rates)
    return baseline


def fold_auc(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Patient AUC of each score column in each fold."""
    return scores.groupby("fold").apply(
        lambda rows: pd.Series({c: roc_auc_score(rows["label"], rows[c]) for c in columns}),
        include_groups=False,
    )


def cohort_auc(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Patient AUC within each cohort, pooling the out-of-fold patients of all folds."""
    return scores.groupby("cohort").apply(
        lambda rows: pd.Series({c: roc_auc_score(rows["label"], rows[c]) for c in columns}),
        include_groups=False,
    )
