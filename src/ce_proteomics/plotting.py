"""
Shared plotting utilities: a single style setup and a save helper so that
every figure in the repository is written at the same resolution and format.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from . import config


def set_style(context: str = "notebook") -> None:
    """Apply the shared seaborn style used across all figures."""
    sns.set_theme(style="whitegrid", context=context)
    matplotlib.rcParams["savefig.bbox"] = "tight"


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = config.FIG_DPI,
    close: bool = True,
) -> Path:
    """
    Save ``fig`` to ``path`` (creating parent directories) and optionally close
    it. Returns the path written.
    """
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(f".{config.FIG_FORMAT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """Format a p-value for annotation: 'p<0.001' or 'p=0.023'."""
    if p < threshold:
        return f"p<{threshold:g}"
    return f"p={p:.3f}"


def significance_stars(q: float) -> str:
    """Asterisk code used in boxplot annotations."""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""
