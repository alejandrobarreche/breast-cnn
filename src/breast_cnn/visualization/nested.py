"""Figures of a nested run: out-of-fold ensemble, calibration and threshold."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from breast_cnn.visualization.style import (
    CLASS_COLORS,
    CLASS_LABELS,
    GRID,
    INK,
    MUTED,
    STYLE,
    format_percent,
    set_chart_title,
)

DECIMAL = FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))
PERCENT = FuncFormatter(lambda value, _: f"{value:.0%}")


def _decimal(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def plot_model_auc(models: pd.DataFrame, folds: pd.DataFrame) -> Figure:
    """Patient AUC per validation fold: each model alone, their mean ensemble and the cohort
    baseline.

    `models` has one row per fold and model with column auc; `folds` is indexed by fold with
    columns ensemble and cohort_baseline.
    """
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(9, 4.2), layout="constrained")
        set_chart_title(
            axis,
            "AUC por paciente de cada modelo y del ensemble",
            "Validación interna · paciente · pCR positiva · media de cortes · "
            "4 modelos sin el fold evaluado",
        )
        for fold, rows in models.groupby("fold"):
            axis.scatter(
                np.full(len(rows), fold), rows["auc"], s=26, facecolor="white", edgecolor=MUTED
            )
        axis.scatter(folds.index, folds["ensemble"], s=60, color=INK, zorder=3)
        axis.scatter(
            folds.index, folds["cohort_baseline"], s=60, marker="_", color=INK, linewidth=2
        )
        axis.scatter([], [], s=26, facecolor="white", edgecolor=MUTED, label="Modelo individual")
        axis.scatter([], [], s=60, color=INK, label="Media de los 4 modelos")
        axis.scatter([], [], s=60, marker="_", color=INK, linewidth=2, label="Solo cohorte")
        axis.axhline(0.5, color=GRID, linewidth=1.5)
        axis.set_xticks(folds.index, [f"Fold {fold}" for fold in folds.index])
        axis.set_xlim(folds.index.min() - 0.5, folds.index.max() + 0.5)
        axis.set_ylabel("ROC-AUC")
        axis.yaxis.set_major_formatter(DECIMAL)
        axis.grid(axis="y")
        figure.legend(frameon=False, loc="outside lower center", ncols=3, fontsize=9)
    return figure


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    operating_point: tuple[float, float],
    threshold: float,
) -> Figure:
    """ROC curves (false positive rate, true positive rate, AUC) and the chosen threshold."""
    styles = ({"color": INK, "linewidth": 2}, {"color": MUTED, "linewidth": 1.2, "linestyle": "--"})
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
        set_chart_title(
            axis,
            "Curva ROC del ensemble fuera de fold",
            "Validación interna · paciente · pCR positiva · 5 folds agrupados",
        )
        axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
        for (name, (fpr, tpr, auc)), style in zip(curves.items(), styles, strict=False):
            axis.plot(fpr, tpr, label=f"{name} · AUC {_decimal(auc)}", **style)
        fpr, tpr = operating_point
        axis.scatter(fpr, tpr, s=50, color=CLASS_COLORS[1], zorder=3)
        axis.annotate(
            f"Umbral {_decimal(threshold)}\nsensibilidad {format_percent(tpr)}\n"
            f"especificidad {format_percent(1 - fpr)}",
            (fpr, tpr),
            xytext=(12, -4),
            textcoords="offset points",
            fontsize=9,
            va="top",
        )
        axis.set(xlabel="1 − especificidad", ylabel="Sensibilidad", xlim=(0, 1), ylim=(0, 1))
        axis.set_aspect("equal")
        axis.xaxis.set_major_formatter(PERCENT)
        axis.yaxis.set_major_formatter(PERCENT)
        axis.grid()
        axis.legend(frameon=False, loc="lower right")
    return figure


def plot_reliability(bins: dict[str, pd.DataFrame], prevalence: float) -> Figure:
    """Observed pCR rate against mean predicted probability, per quantile bin and score."""
    styles = (
        {"facecolor": "white", "edgecolor": MUTED, "color": MUTED},
        {"facecolor": INK, "edgecolor": INK, "color": INK},
    )
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
        set_chart_title(
            axis,
            "Calibración antes y después del ajuste de Platt",
            "Validación interna · paciente · pCR positiva · deciles de probabilidad",
        )
        axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
        axis.axhline(prevalence, color=MUTED, linewidth=0.8, linestyle=":")
        limit = 0.8
        axis.text(
            limit - 0.01,
            prevalence + 0.01,
            "Proporción de pCR",
            color=MUTED,
            fontsize=9,
            ha="right",
        )
        for (name, rows), style in zip(bins.items(), styles, strict=False):
            axis.plot(rows["predicted"], rows["observed"], color=style["color"], linewidth=1)
            axis.scatter(
                rows["predicted"],
                rows["observed"],
                s=36,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                zorder=3,
                label=name,
            )
        axis.set(
            xlabel="Probabilidad media predicha",
            ylabel="Proporción observada de pCR",
            xlim=(0, limit),
            ylim=(0, limit),
        )
        axis.set_aspect("equal")
        axis.xaxis.set_major_formatter(PERCENT)
        axis.yaxis.set_major_formatter(PERCENT)
        axis.grid()
        axis.legend(frameon=False, loc="upper left")
    return figure


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Figure:
    """Histogram of calibrated patient scores per class, normalised within each class."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    edges = np.linspace(scores.min(), scores.max(), 31)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(9, 4.4), layout="constrained")
        set_chart_title(
            axis,
            "Distribución de la probabilidad calibrada por clase",
            f"Validación interna · paciente · pCR positiva · umbral {_decimal(threshold)} "
            "(especificidad ≥ 90 %)",
        )
        for label in (0, 1):
            values = scores[labels == label]
            axis.hist(
                values,
                bins=edges,
                weights=np.full(len(values), 1 / len(values)),
                histtype="step",
                linewidth=2,
                color=CLASS_COLORS[label],
                label=f"{CLASS_LABELS[label]} (n = {len(values)})",
            )
        axis.axvline(threshold, color=INK, linewidth=1.2, linestyle="--")
        axis.text(threshold, axis.get_ylim()[1], " Umbral", color=INK, fontsize=9, va="top")
        axis.set(xlabel="Probabilidad calibrada de pCR", ylabel="Proporción de pacientes")
        axis.xaxis.set_major_formatter(PERCENT)
        axis.yaxis.set_major_formatter(PERCENT)
        axis.grid(axis="y")
        figure.legend(frameon=False, loc="outside lower center", ncols=2, fontsize=9)
    return figure
