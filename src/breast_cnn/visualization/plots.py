"""Reusable train figures with direct labels and shared visual defaults."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from breast_cnn.data.exploration import TrainExploration
from breast_cnn.visualization.style import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_TEXT_COLORS,
    COHORT_LABELS,
    ENHANCEMENT_CMAP,
    GRID,
    MUTED,
    STYLE,
    VARIABLE_LABELS,
    format_count,
    format_percent,
    set_chart_title,
)


def save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=180)


def plot_class_balance(balance: pd.DataFrame, unit: str) -> Figure:
    counts = balance.loc[balance["unit"].eq(unit)].set_index("pCR")["count"].reindex([0, 1])
    total = int(counts.sum())
    unit_label = {"patient": "pacientes", "slice": "cortes"}[unit]
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8, 2.8), layout="constrained")
        set_chart_title(
            axis, "Respuesta patológica completa", f"Train · {format_count(total)} {unit_label}"
        )
        left = 0.0
        for label in (0, 1):
            fraction = counts[label] / total
            axis.barh(0, fraction, left=left, height=0.34, color=CLASS_COLORS[label])
            axis.text(
                left + fraction / 2,
                0,
                format_percent(fraction),
                ha="center",
                va="center",
                color=CLASS_TEXT_COLORS[label],
                fontsize=15,
                weight="bold",
            )
            position = 0 if label == 0 else 1
            alignment = "left" if label == 0 else "right"
            axis.text(
                position,
                -0.32,
                f"{CLASS_LABELS[label]} · pCR={label}",
                ha=alignment,
                color=CLASS_COLORS[label],
                fontsize=10,
            )
            axis.text(
                position,
                -0.49,
                format_count(int(counts[label])),
                ha=alignment,
                color=MUTED,
                fontsize=11,
            )
            left += fraction
        axis.set(xlim=(0, 1), ylim=(-0.6, 0.3))
        axis.set_axis_off()
    return figure


def plot_slice_counts(counts: pd.DataFrame, patient_count: int) -> Figure:
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8, 3.7), layout="constrained")
        set_chart_title(
            axis, "Cortes por paciente", f"Train · {format_count(patient_count)} pacientes"
        )
        positions = np.arange(len(counts))
        axis.hlines(positions, 0, counts["patients"], color=GRID, linewidth=3)
        axis.scatter(
            counts["patients"], positions, s=65, color=CLASS_COLORS[0], zorder=3, clip_on=False
        )
        for position, count in zip(positions, counts["patients"], strict=True):
            axis.annotate(
                format_count(int(count)),
                (count, position),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                fontsize=11,
            )
        axis.set_yticks(positions, [f"{count} cortes" for count in counts["n_slices"]])
        axis.set(xlim=(0, counts["patients"].max() * 1.15), xlabel="Pacientes")
        axis.set_ylim(len(counts) - 0.5, -0.5)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_count(int(value))))
        axis.grid(axis="x")
    return figure


def plot_cohort_distribution(cohorts: pd.DataFrame) -> Figure:
    cohorts = cohorts.sort_values("patients", ascending=False)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        set_chart_title(
            axis,
            "Pacientes por cohorte",
            f"Train · {format_count(int(cohorts['patients'].sum()))} pacientes",
        )
        positions = np.arange(len(cohorts))
        negative = cohorts["patients"] - cohorts["pcr_positive"]
        for label, values, left in (
            (0, negative, np.zeros(len(cohorts))),
            (1, cohorts["pcr_positive"], negative),
        ):
            axis.barh(
                positions,
                values,
                left=left,
                height=0.48,
                color=CLASS_COLORS[label],
                label=f"pCR={label}",
            )
            for position, value, start in zip(positions, values, left, strict=True):
                axis.text(
                    start + value / 2,
                    position,
                    format_count(int(value)),
                    ha="center",
                    va="center",
                    color=CLASS_TEXT_COLORS[label],
                    fontsize=10,
                )
        axis.set_yticks(positions, [COHORT_LABELS.get(name, name) for name in cohorts["data"]])
        axis.set_ylim(len(cohorts) - 0.4, -0.9)
        axis.set(xlim=(0, cohorts["patients"].max() * 1.02), xlabel="Pacientes")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.grid(axis="x")
        axis.legend(frameon=False, ncol=2, loc="upper right", handlelength=1, fontsize=10)
    return figure


def plot_missing_values(missing_values: pd.DataFrame, patient_count: int) -> Figure:
    missing = missing_values.loc[missing_values["missing"].gt(0)].sort_values(
        "missing", ascending=False
    )
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
        set_chart_title(
            axis, "Valores ausentes", f"Train · {format_count(patient_count)} pacientes"
        )
        positions = np.arange(len(missing))
        percentages = missing["fraction"] * 100
        axis.hlines(positions, 0, percentages, color=GRID, linewidth=2)
        axis.scatter(percentages, positions, s=45, color=CLASS_COLORS[0], zorder=3, clip_on=False)
        for position, fraction, count in zip(
            positions, missing["fraction"], missing["missing"], strict=True
        ):
            axis.annotate(
                f"{format_percent(fraction)} · n={count}",
                (fraction * 100, position),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )
        axis.set_yticks(positions, [VARIABLE_LABELS.get(name, name) for name in missing["column"]])
        axis.set_ylim(max(len(missing) - 0.4, 0.5), -0.6)
        axis.set_xlim(0, percentages.max() * 1.3 if len(missing) else 1)
        axis.set_xlabel("Pacientes con valor ausente (%)")
        axis.grid(axis="x")
        if missing.empty:
            axis.text(0.5, 0.5, "Sin valores ausentes", transform=axis.transAxes, ha="center")
    return figure


def plot_train_overview(exploration: TrainExploration) -> dict[str, Figure]:
    if not exploration.samples["split"].eq("train").all():
        raise ValueError("EDA figures require exclusively train samples")
    tables = exploration.tables
    patient_count = len(exploration.patients)
    return {
        "class_balance_train_patient": plot_class_balance(tables["class_balance"], "patient"),
        "class_balance_train_slice": plot_class_balance(tables["class_balance"], "slice"),
        "slices_per_patient_train_patient": plot_slice_counts(
            tables["slices_per_patient"], patient_count
        ),
        "cohort_distribution_train_patient": plot_cohort_distribution(tables["cohort_summary"]),
        "missing_values_train_patient": plot_missing_values(
            tables["missing_values"], patient_count
        ),
    }


def plot_dce_phases(image: torch.Tensor, *, sample_id: str, cohort: str, label: int) -> Figure:
    """Display phases on [0,1] and signed enhancement on [-1,1]."""
    if image.shape != (3, 256, 256) or image.dtype != torch.float32:
        raise ValueError("DCE figure expects float32 [3,256,256]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("DCE figure expects finite intensities in [0,1]")
    channels = image.detach().cpu().numpy()
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 4, figsize=(13, 4.7))
        figure.subplots_adjust(left=0.015, right=0.985, top=0.77, bottom=0.19, wspace=0.08)
        figure.text(0.02, 0.94, "Fases DCE y realce", fontsize=17, weight="bold")
        figure.text(
            0.02,
            0.87,
            f"Train · un corte · {COHORT_LABELS.get(cohort, cohort)} · {sample_id} · pCR={label}",
            fontsize=10,
            color=MUTED,
        )
        for axis, phase, array in zip(axes[:3], ("PRE", "EARLY", "LATE"), channels, strict=True):
            gray = axis.imshow(array, cmap="gray", vmin=0, vmax=1)
            axis.set_title(phase, fontsize=11, pad=8)
        difference = axes[3].imshow(
            channels[1] - channels[0], cmap=ENHANCEMENT_CMAP, vmin=-1, vmax=1
        )
        axes[3].set_title("EARLY − PRE", fontsize=11, pad=8)
        for axis in axes:
            axis.set_axis_off()
        gray_axis = figure.add_axes((0.22, 0.12, 0.3, 0.018))
        difference_axis = figure.add_axes((0.795, 0.12, 0.15, 0.018))
        for mappable, color_axis, ticks, caption in (
            (gray, gray_axis, [0, 0.5, 1], "Intensidad"),
            (difference, difference_axis, [-1, 0, 1], "Realce"),
        ):
            colorbar = figure.colorbar(
                mappable, cax=color_axis, orientation="horizontal", ticks=ticks
            )
            colorbar.outline.set_visible(False)
            colorbar.formatter = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
            colorbar.update_ticks()
            colorbar.ax.tick_params(labelsize=8)
            colorbar.set_label(caption, fontsize=9, labelpad=2)
        figure.text(0.02, 0.005, "BreastDCEDL · CC BY-NC 4.0", fontsize=8, color=MUTED)
    return figure
