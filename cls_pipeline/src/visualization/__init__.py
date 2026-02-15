"""
visualization/ — Modulo principale per visualizzazioni CLS Pipeline.

Sub-packages:
- png/: Grafici PNG publication-ready (matplotlib/seaborn)
- interactive/: Grafici HTML interattivi (Plotly)

Moduli standalone:
- config.py: Configurazione stile uniforme
"""

from .config import (
    COLORS,
    OKABE_ITO,
    DOMAIN_COLORS,
    FIGURE_SIZES,
    FONT_SIZES,
    DEFAULT_DPI,
    apply_matplotlib_style,
    jaccard_color,
    significance_label,
    significance_marker,
)

__all__ = [
    "COLORS",
    "OKABE_ITO",
    "DOMAIN_COLORS",
    "FIGURE_SIZES",
    "FONT_SIZES",
    "DEFAULT_DPI",
    "apply_matplotlib_style",
    "jaccard_color",
    "significance_label",
    "significance_marker",
]
