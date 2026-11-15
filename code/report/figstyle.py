"""Shared style for every paper figure: one font, one warning-level palette, one set of source names.
Import and call apply() before creating figures. Fonts are embedded as TrueType (pdf.fonttype 42), never Type 3."""
import matplotlib

# warning levels: normal, advisory, warning, danger (same colours in every figure, the demo and the project page)
LCOL = ["#3f9b4f", "#e9b824", "#ec7d24", "#d0342c"]
LNAME = ["normal", "advisory", "warning", "danger"]
INK = "#1d1d1f"      # text
MUTED = "#5f6368"    # secondary text
RULE = "#c9cdd2"     # thin lines, grids
ACCENT = "#1f5fa8"   # our method / highlights
OURS_BG = "#e8f0fa"

SOURCE_NAME = {
    "itssnap": "ITS/KEC snapshots", "itslive": "ITS/KEC sequences", "hmm_pilot": "Winter pilot clips",
    "fvei_fog": "FVEI highway fog", "floodcam_nc": "FloodCam-NC", "vfloodnet_pixabay": "Web flood photos",
    "vfloodnet_seq": "Storm timelapses", "nysdot_511ny": "NYSDOT blizzard", "norway_trafficcam": "Norway road cams",
    "iowa_rwis": "Iowa DOT RWIS", "tsbow": "TSBOW (test only)",
}
OURS = ["itssnap", "itslive", "hmm_pilot"]
EXTERNAL = ["fvei_fog", "floodcam_nc", "vfloodnet_seq", "vfloodnet_pixabay", "nysdot_511ny", "norway_trafficcam",
            "iowa_rwis", "tsbow"]

# Paper text width 6.875 in, column width 3.25 in
TEXTW = 6.875
COLW = 3.25


def apply():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "axes.edgecolor": "#444444", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    })
