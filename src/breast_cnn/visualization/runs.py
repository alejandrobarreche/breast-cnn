"""Figures comparing finished runs; the data come from breast_cnn.evaluation."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from breast_cnn.evaluation import Run
from breast_cnn.visualization.style import GRID, INK, MUTED, STYLE, format_percent, set_chart_title

DECIMAL = FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))


def plot_validation_curves(runs: list[Run], columns: int = 4) -> Figure:
    """Validation patient AUC per epoch: thin line per fold, thick line = mean of folds.

    The mean uses only epochs reached by every fold, the candidates of out-of-fold selection.
    """
    rows = math.ceil(len(runs) / columns)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            rows, columns, figsize=(3.2 * columns, 2.4 * rows + 1), sharex=True, sharey=True
        )
        figure.subplots_adjust(top=1 - 1.1 / (2.4 * rows + 1), hspace=0.45, wspace=0.08)
        figure.text(
            0.1,
            0.995,
            "AUC por paciente en validación según la época",
            fontsize=16,
            weight="bold",
            va="top",
        )
        figure.text(
            0.1,
            0.995 - 0.45 / (2.4 * rows + 1),
            "Validación interna · paciente · pCR positiva · media de cortes · "
            "línea fina = fold, gruesa = media de folds",
            color=MUTED,
            va="top",
        )
        for axis, run in zip(axes.flat, runs, strict=False):
            curves = run.history.pivot(
                index="epoch", columns="fold", values="validation_patient_auc"
            )
            axis.axhline(0.5, color=GRID, linewidth=1.5)
            for fold in curves.columns:
                axis.plot(curves.index, curves[fold], color=MUTED, linewidth=0.7, alpha=0.6)
            mean = curves.dropna().mean(axis=1)
            axis.plot(mean.index, mean, color=INK, linewidth=2)
            axis.set_title(run.experiment, fontsize=10, loc="left", pad=4)
            axis.grid(axis="y")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            axis.yaxis.set_major_formatter(DECIMAL)
        for axis in axes.flat[len(runs) :]:
            axis.set_axis_off()
        for axis in axes[-1]:
            axis.set_xlabel("Época")
        for axis in axes[:, 0]:
            axis.set_ylabel("ROC-AUC")
        axes.flat[0].set_ylim(0.35, 0.75)
    return figure


def plot_selection_rules(summary: pd.DataFrame, baseline: float) -> Figure:
    """Mean validation patient AUC of each experiment under the three epoch-selection rules."""
    summary = summary.sort_values("out_of_fold_mean")
    positions = np.arange(len(summary))
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(9, 0.32 * len(summary) + 2), layout="constrained")
        set_chart_title(
            axis,
            "AUC por paciente según la regla de selección de época",
            "Validación interna · paciente · pCR positiva · media de 5 folds",
        )
        axis.hlines(
            positions,
            summary[["all_epochs_mean", "out_of_fold_mean", "best_mean"]].min(axis=1),
            summary[["all_epochs_mean", "out_of_fold_mean", "best_mean"]].max(axis=1),
            color=GRID,
            linewidth=2,
        )
        axis.scatter(
            summary["best_mean"],
            positions,
            s=40,
            facecolor="white",
            edgecolor=MUTED,
            zorder=3,
            label="Mejor época del propio fold",
        )
        axis.scatter(
            summary["all_epochs_mean"],
            positions,
            s=22,
            color=MUTED,
            zorder=3,
            label="Media de todas las épocas",
        )
        axis.scatter(
            summary["out_of_fold_mean"],
            positions,
            s=45,
            color=INK,
            zorder=3,
            label="Época elegida con los otros folds",
        )
        axis.axvline(baseline, color=INK, linewidth=1, linestyle="--")
        axis.text(baseline, len(summary) - 0.2, " Solo cohorte", color=INK, fontsize=9)
        axis.axvline(0.5, color=GRID, linewidth=1.5)
        axis.set_yticks(positions, summary.index)
        axis.set_ylim(-0.7, len(summary) + 0.3)
        axis.set_xlabel("ROC-AUC")
        axis.xaxis.set_major_formatter(DECIMAL)
        axis.grid(axis="x")
        figure.legend(frameon=False, loc="outside lower center", ncols=3, fontsize=9)
    return figure


def plot_paired_differences(differences: pd.DataFrame, reference: str, level: float) -> Figure:
    """Observed AUC difference with the reference and its paired bootstrap interval."""
    positions = np.arange(len(differences))
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(
            figsize=(9, 0.32 * len(differences) + 1.8), layout="constrained"
        )
        set_chart_title(
            axis,
            f"Diferencia de AUC por paciente con {reference}",
            f"Validación interna · paciente · pCR positiva · media de 5 folds · "
            f"intervalo bootstrap emparejado del {format_percent(level)}",
        )
        axis.axvline(0, color=INK, linewidth=1)
        axis.hlines(positions, differences["lower"], differences["upper"], color=MUTED, linewidth=2)
        axis.scatter(differences["difference"], positions, s=40, color=INK, zorder=3)
        axis.set_yticks(positions, differences.index)
        axis.set_ylim(-0.7, len(differences) - 0.3)
        axis.set_xlabel(f"ROC-AUC del experimento − ROC-AUC de {reference}")
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:+.2f}".replace(".", ","))
        )
        axis.grid(axis="x")
    return figure
