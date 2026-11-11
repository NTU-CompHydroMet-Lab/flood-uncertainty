"""Shared matplotlib style for the IEEE TGRS figures (single column 3.5 in, double column 7.16 in, 8 pt serif for all figure text)."""
import matplotlib as mpl

COL1 = 3.5    # single column width, inch
COL2 = 7.16   # double column width, inch

# Fixed colour per system (validated: scripts of the dataviz skill, light mode, CVD and normal-vision pass).
C = {
    "bef": "#2a78d6",         # DEUF, TerraMind branches (BEF)
    "multimodal": "#eb6834",  # TerraMind S1+S2, one encoder
    "unet_deuf": "#1baf7a",   # DEUF, UNet branches
    "optical": "#7a7a74",     # single branches: neutral gray
    "sar": "#7a7a74",
}
LABEL = {
    "bef/fused": "DEUF (TerraMind)",
    "unet_deuf/fused": "DEUF (UNet)",
    "multimodal": "TerraMind S1+S2",
    "bef/optical": "Optical branch",
    "bef/sar": "SAR branch",
}


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e4e0",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType (IEEE PDF eXpress)
        "ps.fonttype": 42,
    })
