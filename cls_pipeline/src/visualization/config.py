"""
config.py — Configurazione uniforme per tutte le visualizzazioni.

Definisce costanti di stile, colori, font e parametri per garantire
coerenza tra PNG publication-ready e HTML interattivo.

Palette Okabe-Ito: progettata per essere distinguibile anche da persone
con deuteranopia/protanopia (CVD), usata nei migliori giornali scientifici.
Ref.: Okabe & Ito (2008) "Color Universal Design" https://jfly.uni-koeln.de/color/
"""
# ─── Colori principali per i due modelli ────────────────────────────
# Blu per WEIRD, arancione per Sinic: massimo contrasto anche in scala
# di grigi e per utenti CVD. Evita rosso-verde.
# ─────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Any


# =============================================================================
# Core color scheme
# =============================================================================

COLORS = {
    # Modelli principali
    "weird": "#4C72B0",      # Blu (seaborn deep)
    "sinic": "#DD8452",      # Arancione (seaborn deep)

    # Significatività statistica
    "significant": "#2ecc71",
    "not_significant": "#e74c3c",

    # Heatmap divergenti (per differenze)
    "positive": "#d73027",
    "neutral": "#f7f7f7",
    "negative": "#4575b4",

    # Sfondo e griglia
    "background": "#fafafa",
    "grid": "#e0e0e0",
    "text": "#333333",
    "text_light": "#666666",

    # Accento per header e titoli
    "accent": "#2c3e50",

    # Bordi
    "border": "#e0e0e0",
}

# =============================================================================
# Okabe-Ito colorblind-friendly palette (8 colors)
# =============================================================================
# Per visualizzazioni multi-categoria (domini, cluster, etc.)
# Ottimizzata per tutti i tipi di daltonismo e stampa in B/N

OKABE_ITO = [
    "#E69F00",  # Arancione
    "#56B4E9",  # Azzurro cielo
    "#009E73",  # Verde acqua
    "#F0E442",  # Giallo
    "#0072B2",  # Blu
    "#D55E00",  # Vermiglio
    "#CC79A7",  # Rosa
    "#999999",  # Grigio
]

# Mapping domini giuridici → colori Okabe-Ito
DOMAIN_COLORS = {
    "constitutional": "#0072B2",      # Blu
    "rights": "#D55E00",              # Vermiglio
    "civil": "#009E73",               # Verde acqua
    "criminal": "#E69F00",            # Arancione
    "governance": "#56B4E9",          # Azzurro
    "jurisprudence": "#CC79A7",       # Rosa
    "international": "#F0E442",       # Giallo
    "labor_social": "#999999",        # Grigio
    "environmental_tech": "#0072B2",  # Blu (riuso)
    "control": "#AAAAAA",             # Grigio chiaro
    "unknown": "#CCCCCC",
}


# =============================================================================
# Typography
# =============================================================================

FONT_SIZES = {
    "title": 14,
    "subtitle": 12,
    "axis": 11,
    "tick": 9,
    "annotation": 8,
    "legend": 10,
    "small": 7,
}

# Font family per matplotlib (con fallback)
FONT_FAMILY = "sans-serif"
FONT_FAMILY_MATH = "cm"  # Computer Modern per formule


# =============================================================================
# Plot dimensions
# =============================================================================

@dataclass
class FigureSize:
    """Standard figure sizes for different contexts."""
    # Singola colonna journal (3.5 in)
    single_column: tuple[float, float] = (3.5, 3.5)
    # Doppia colonna journal (7 in)
    double_column: tuple[float, float] = (7.0, 5.0)
    # Full page
    full_page: tuple[float, float] = (8.0, 10.0)
    # Wide (per heatmap, dendrogrammi)
    wide: tuple[float, float] = (10.0, 6.0)
    # Square (per scatter, network)
    square: tuple[float, float] = (7.0, 7.0)
    # Side by side (per confronti WEIRD vs Sinic)
    side_by_side: tuple[float, float] = (14.0, 6.0)


FIGURE_SIZES = FigureSize()


# =============================================================================
# DPI and output quality
# =============================================================================

DEFAULT_DPI = 300  # Publication quality
SCREEN_DPI = 150   # Per anteprima
THUMBNAIL_DPI = 72

# Formati supportati
SUPPORTED_FORMATS = ["png", "svg", "pdf"]


# =============================================================================
# Matplotlib style configuration
# =============================================================================

def get_matplotlib_style() -> dict[str, Any]:
    """Return matplotlib rcParams for publication-ready plots."""
    return {
        # Figure
        "figure.facecolor": "white",
        "figure.edgecolor": "none",
        "figure.dpi": DEFAULT_DPI,
        "figure.figsize": FIGURE_SIZES.double_column,

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["grid"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.axisbelow": True,
        "axes.labelsize": FONT_SIZES["axis"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "grid.color": COLORS["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,

        # Ticks
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend
        "legend.fontsize": FONT_SIZES["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["grid"],
        "legend.fancybox": False,

        # Font
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZES["tick"],
        "mathtext.fontset": FONT_FAMILY_MATH,

        # Savefig
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    }


def apply_matplotlib_style():
    """Apply publication-ready matplotlib style globally."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(get_matplotlib_style())


# =============================================================================
# Seaborn configuration
# =============================================================================

def get_seaborn_context() -> dict[str, Any]:
    """Return seaborn context parameters."""
    return {
        "font_scale": 1.0,
        "rc": get_matplotlib_style(),
    }


def apply_seaborn_style():
    """Apply seaborn style for consistent plots."""
    import seaborn as sns
    sns.set_theme(
        style="whitegrid",
        palette=[COLORS["weird"], COLORS["sinic"]] + OKABE_ITO,
        context="paper",
        font_scale=1.0,
    )
    # Applica anche matplotlib style
    apply_matplotlib_style()


# =============================================================================
# Colormaps
# =============================================================================

COLORMAPS = {
    # Per RDM heatmap (distanze, 0 = simile, 1 = diverso)
    "distance": "viridis",
    # Per transport plan (masse)
    "transport": "YlOrRd",
    # Per divergenza (bipolare)
    "divergence": "RdBu_r",
    # Per correlazione
    "correlation": "coolwarm",
    # Per cluster membership
    "categorical": "tab10",
    # Per Jaccard (0-1, basso = rosso, alto = verde)
    "jaccard": "RdYlGn",
}


# =============================================================================
# Jaccard color scale
# =============================================================================

def jaccard_color(j: float) -> str:
    """Return color for Jaccard value (0=bad/red, 1=good/green)."""
    if j < 0.2:
        return "#e74c3c"  # Rosso
    elif j < 0.4:
        return "#e67e22"  # Arancione
    elif j < 0.6:
        return "#f1c40f"  # Giallo
    else:
        return "#2ecc71"  # Verde


# =============================================================================
# Significance markers
# =============================================================================

def significance_marker(p: float) -> str:
    """Return significance marker for p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def significance_label(p: float) -> str:
    """Return formatted significance label."""
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.4f} **"
    elif p < 0.05:
        return f"p = {p:.4f} *"
    else:
        return f"p = {p:.4f} (n.s.)"


# =============================================================================
# HTML/Plotly colors (CSS format)
# =============================================================================

PLOTLY_COLORS = {
    "weird": "rgba(76, 114, 176, 1.0)",
    "weird_light": "rgba(76, 114, 176, 0.3)",
    "sinic": "rgba(221, 132, 82, 1.0)",
    "sinic_light": "rgba(221, 132, 82, 0.3)",
    "grid": "rgba(224, 224, 224, 1.0)",
    "background": "rgba(250, 250, 250, 1.0)",
}

# Layout Plotly di default
PLOTLY_LAYOUT = {
    "font": {"family": "Segoe UI, Helvetica, sans-serif", "size": 12},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 11,
        "font_family": "monospace",
    },
}


# =============================================================================
# Clustering thresholds
# =============================================================================

# Soglia per etichettare punti nel UMAP scatter
UMAP_LABEL_ISOLATION_THRESHOLD = 0.3  # Distanza minima per mostrare label

# Numero massimo di foglie nel dendrogramma troncato
DENDROGRAM_MAX_LEAVES = 30

# Top-K alignments da mostrare per GW
GW_TOP_K_ALIGNMENTS = 50

# Soglia percentile per transport plan
GW_TRANSPORT_PERCENTILE = 99

# Numero di vicini da mostrare in NDA
NDA_NEIGHBORS_DISPLAY = 5
