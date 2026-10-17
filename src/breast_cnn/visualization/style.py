"""Shared typography, colors and labels for project figures."""

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

INK = "#24343B"
MUTED = "#6B7C83"
GRID = "#E5ECEE"
CLASS_COLORS = ("#23656E", "#DC7867")
CLASS_TEXT_COLORS = ("white", INK)
CLASS_LABELS = ("Sin respuesta completa", "Respuesta completa")
COHORT_LABELS = {"duke": "Duke", "spy1": "I-SPY1", "spy2": "I-SPY2"}
VARIABLE_LABELS = {
    "menopause": "Menopausia",
    "age": "Edad",
    "tum_vol": "Volumen tumoral",
    "race_white": "Raza blanca",
    "race_black": "Raza negra",
    "HR": "HR",
    "HER2": "HER2",
    "HR_HER2_STATUS": "Subtipo HR/HER2",
    "TripleNeg": "Triple negativo",
    "HER2pos": "HER2 positivo",
    "HRposHER2neg": "HR positivo / HER2 negativo",
}
ENHANCEMENT_CMAP = LinearSegmentedColormap.from_list(
    "enhancement", [CLASS_COLORS[0], "#FAFAF8", CLASS_COLORS[1]]
)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "ytick.major.pad": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 180,
}


def format_count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_percent(value: float) -> str:
    return f"{value:.1%}".replace(".", ",")


def set_chart_title(axis: Axes, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", pad=38)
    axis.text(0, 1.045, subtitle, transform=axis.transAxes, color=MUTED, fontsize=10)
