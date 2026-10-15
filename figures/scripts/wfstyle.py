"""Shared figure style for WitnessFlow single-panel figures (figure plan v2).

Style rules (2026-10-09): Arial 7 pt everywhere, text editable in Illustrator (TrueType / Type 42 embedding),
strokes 1 pt wherever possible, one PDF + one PNG per panel, panels saved separately (no composition).
Documented thin-line exceptions (THIN = 0.5 pt): hatch fills, dense background geometry (e.g. the 130 faint
hexagons), heat-map cell separators, reference grids, marker edges of small markers.

Usage:
    import wfstyle as S
    S.setup()
    fig = S.new("2A")              # size from SIZES (mm); override with w=, h=
    ...
    S.save(fig, "2A")              # -> figures/panels/fig2A.pdf + fig2A.png (600 dpi), font check
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

WF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(WF, "figures", "data")
DERIVED = os.path.join(WF, "figures", "derived")
RESULTS = os.path.join(WF, "results")
PANELS = os.path.join(WF, "figures", "panels")
MM = 1 / 25.4
LW = 1.0      # pt, default stroke
THIN = 0.5    # pt, documented exceptions only
FS = 7        # pt, the only font size

INK = "#222222"
MUTED = "#6E6E6E"
GRID = "#E3E3E3"
BAND = "#DCEAF7"      # complete-round node bands
ALARM = "#C0392B"     # alarms / failures / out-of-bound only
TSTAR = "#F2B701"     # T* = CCR7/CD27/CD45RA accent (always with INK edge)
LIMITS_HATCH = "////"

BACKBONE = "#D9D9D9"  # backbone slots of the M1 tube card (all figures)
CAPTIONS = os.path.join(WF, "figures", "captions")

METHOD = {  # fixed across all figures
    "classical": "#245B8A", "oa": "#245B8A", "lp_true": "#00857D", "lp_shuffled": "#D98B24",
    "uniform": "#8064A2", "random": "#858B93", "balanced_random": "#C3C7CC", "anneal": "#7A8B2E",
    "psl": "#C9A227",
}
MARKER = {
    "classical": "o", "oa": "s", "lp_true": "D", "lp_shuffled": "s", "uniform": "^",
    "random": "v", "balanced_random": "<", "anneal": "P", "psl": "*",
}
LABEL = {
    "classical": "Classical", "oa": "Orthogonal array", "lp_true": "True requirement",
    "lp_shuffled": "Shuffled requirement", "uniform": "Uniform requirement", "random": "Random",
    "balanced_random": "Balanced random", "anneal": "Annealing search", "psl": "PSL(2,19) 3-design",
}
# raw method names in the source tables -> keys above
ALIAS = {
    "classical_stack": "classical", "LJCR_stack": "classical", "OA_quadratic": "oa", "anneal_search": "anneal",
    "legal_random": "random", "random": "random", "balanced_random": "balanced_random", "PSL_3design": "psl",
    "lp_true": "lp_true", "lp_shuffled": "lp_shuffled", "worst": "uniform", "classical": "classical",
}
DATASET = {"levine32_v2": "Levine-32", "wingmen_v1": "WINGMEN"}
DATASET_MARKER = {"levine32_v2": "o", "wingmen_v1": "^"}

TIER = {1: "#1A1A1A", 2: "#7A7A7A", 3: "#D0D0D0", 4: "#FFFFFF"}   # evidence tiers I-IV (III hatched, IV outlined)
TIER_NAME = {1: "I  direct co-measurement", 2: "II  lower-order bounds only", 3: "III  model point estimate only",
             4: "IV  no reportable information"}
ORBIT = ["#4E79A7", "#59A14F", "#B07AA1", "#76B7B2", "#9C755F"]          # rose-window orbits (local, Fig 1B/2A)
SLOPE = {0: "#3A6EA5", 1: "#2A9D8F", 2: "#8E6BBF", 3: "#6A9F3A", 4: "#7A5230", "inf": "#D16BA5"}  # AG(2,5) (2B/2C/4C)
FAMILY = ["#DCE6F0", "#F6E3CC", "#DCEFD9", "#EADCEB", "#D9EEEC"]          # OA slot families (local, 2D/2E)
SEQ = LinearSegmentedColormap.from_list("wf_seq", ["#F5F8FB", "#9DBBD6", "#245B8A", "#0F2F4D"])
DIV = LinearSegmentedColormap.from_list("wf_div", ["#8C510A", "#D8B365", "#F5F5F5", "#5AB4AC", "#01665E"])

# panel sizes in mm (final figure width 180 mm); widths are fixed, heights may change by <= 20 %
SIZES = {
    "1A": (99, 75), "1B": (81, 75), "1C": (55, 80), "1D": (65, 80), "1E": (60, 80),
    "2A": (88, 100), "2B": (92, 50), "2C": (92, 50), "2D": (100, 75), "2E": (80, 75),
    "3A": (50, 100), "3B": (130, 70), "3C": (130, 30), "3D": (100, 70), "3E": (80, 70),
    "4A": (180, 55), "4B": (80, 60), "4C": (100, 60), "4D": (100, 55), "4E": (80, 55),
    "5A": (180, 45), "5B": (90, 60), "5C": (90, 60), "5D": (90, 60), "5E": (90, 60),
    # supplementary figures (suppplan_v2; full page ~180 x 220 mm); widths fixed, heights may change by <= 25 %
    "S1A": (180, 50), "S1B": (110, 125), "S1C": (70, 62), "S1D": (70, 62), "S1E": (180, 50),
    "S2A": (120, 70), "S2B": (60, 70), "S2C": (180, 85), "S2D": (80, 65), "S2E": (100, 65),
    "S3A": (180, 70), "S3B": (110, 85), "S3C": (70, 85), "S3D": (80, 60), "S3E": (100, 60),
}
SOURCE = os.path.join(WF, "figures", "source_data")


def save_source(df, panel, name=None):
    """Write the Source Data table behind a panel: figures/source_data/fig<panel>[_name].tsv"""
    os.makedirs(SOURCE, exist_ok=True)
    p = os.path.join(SOURCE, f"fig{panel}" + (f"_{name}" if name else "") + ".tsv")
    df.to_csv(p, sep="\t", index=False)
    return p


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS, "figure.labelsize": FS,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "regular",
        "axes.unicode_minus": True,
        "lines.linewidth": LW, "axes.linewidth": LW, "patch.linewidth": LW, "hatch.linewidth": THIN,
        "lines.markersize": 3.5, "lines.markeredgewidth": THIN,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": THIN, "ytick.minor.width": THIN,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelpad": 2,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.25, "legend.columnspacing": 0.8,
        "axes.grid": False, "grid.linewidth": THIN, "grid.color": GRID,
        "errorbar.capsize": 0, "scatter.edgecolors": "none",
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.facecolor": "white", "savefig.transparent": False,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    })


def new(panel, w=None, h=None, layout="constrained", **kw):
    W, H = SIZES[panel]
    return plt.figure(figsize=((w or W) * MM, (h or H) * MM), layout=layout, **kw)


def _is_axes(obj):
    return hasattr(obj, "transAxes")


def tier_tag(ax_or_fig, text, x=None, y=None, ha="right", va=None):
    """Evidence-type tag: Math / Replay / Simulation / Record / Exploratory (grey italic, top-right).
    On an Axes the default sits just above the axes; on a Figure it sits inside the top-right corner."""
    if _is_axes(ax_or_fig):
        tr, x, y, va = ax_or_fig.transAxes, 1.0 if x is None else x, 1.0 if y is None else y, va or "bottom"
    else:
        tr, x, y, va = ax_or_fig.transFigure, 0.99 if x is None else x, 0.985 if y is None else y, va or "top"
    return ax_or_fig.text(x, y, text, transform=tr, ha=ha, va=va, color=MUTED, fontsize=FS, fontstyle="italic")


def limits_tag(ax_or_fig, x=None, y=None, ha="left", va=None):
    """Uniform 'Limits' corner tag for boundary panels (hatched grey box); figure default = inside top-left."""
    if _is_axes(ax_or_fig):
        tr, x, y, va = ax_or_fig.transAxes, 0.0 if x is None else x, 1.0 if y is None else y, va or "bottom"
    else:
        tr, x, y, va = ax_or_fig.transFigure, 0.01 if x is None else x, 0.985 if y is None else y, va or "top"
    return ax_or_fig.text(x, y, "Limits", transform=tr, ha=ha, va=va, fontsize=FS, color=INK,
                          bbox=dict(boxstyle="square,pad=0.25", fc="#EDEDED", ec=MUTED, lw=THIN, hatch=LIMITS_HATCH))


def tstar_marker(ax, x, y, transform=None, ms=6, zorder=6):
    """The T* symbol: gold vector star with INK edge (Arial has no U+2605; never use '*' or a glyph)."""
    ax.plot([x], [y], marker="*", ms=ms, mfc=TSTAR, mec=INK, mew=THIN, ls="none",
            transform=transform or ax.transData, clip_on=False, zorder=zorder)


def tstar_label(ax, x, y, text="T", transform=None, ms=6, gap_pt=0.6, **kw):
    """Write `text` (default 'T') left-aligned at (x, y) and draw the T* star right after it (vector marker)."""
    import matplotlib.transforms as mtransforms
    tr = transform or ax.transData
    kw.setdefault("va", "center")
    t = ax.text(x, y, text, transform=tr, ha="left", **kw)
    fig = ax.figure
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    w_pt = bb.width * 72.0 / fig.dpi
    off = mtransforms.offset_copy(tr, fig=fig, x=w_pt + gap_pt + 0.45 * ms, y=0.0, units="points")
    tstar_marker(ax, x, y, transform=off, ms=ms, zorder=t.get_zorder() + 1)
    return t


def pdf_fonts(path):
    import fitz
    doc = fitz.open(path)
    fonts = set()
    for page in doc:
        for f in page.get_fonts(full=True):
            fonts.add((f[3], f[1]))  # (basefont, type)
    doc.close()
    return sorted(fonts)


def pdf_audit(path, tol=0.05):
    """Compliance audit of an exported panel PDF:
    - every text span is exactly 7 pt (no shrunken mathtext sub/superscripts),
    - every text span lies inside the page,
    - every stroked path is 1 pt or 0.5 pt (fills are ignored)."""
    import fitz
    doc = fitz.open(path)
    page = doc[0]
    R = page.rect
    small, outside, widths = [], [], set()
    for b in page.get_text("dict", clip=fitz.INFINITE_RECT())["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                if abs(s["size"] - FS) > tol:
                    small.append((s["text"], round(s["size"], 2)))
                x0, y0, x1, y1 = s["bbox"]
                if x0 < R.x0 or y0 < R.y0 or x1 > R.x1 or y1 > R.y1:
                    outside.append((s["text"], [round(v, 1) for v in s["bbox"]]))
    for d in page.get_drawings():
        if d.get("color") is not None and d.get("width"):
            widths.add(round(float(d["width"]), 3))
    doc.close()
    badw = sorted(w for w in widths if min(abs(w - LW), abs(w - THIN)) > 0.011)
    return small, outside, badw, sorted(widths)


def save(fig, panel, stem=None, strict=True):
    """Save figures/panels/fig<panel>.pdf and .png (600 dpi), then audit the PDF:
    Arial TrueType only; all text exactly 7 pt and inside the page; strokes 1 pt or 0.5 pt only."""
    os.makedirs(PANELS, exist_ok=True)
    stem = stem or f"fig{panel}"
    pdf = os.path.join(PANELS, stem + ".pdf")
    png = os.path.join(PANELS, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    fonts = pdf_fonts(pdf)
    bad = [f for f in fonts if "Arial" not in f[0] or f[1].lower() not in ("ttf", "truetype", "type0")]
    problems = []
    if bad:
        problems.append(f"non-Arial or non-TrueType fonts embedded: {bad}  (all: {fonts})")
    small, outside, badw, widths = pdf_audit(pdf)
    if small:
        problems.append(f"text spans not {FS} pt (e.g. mathtext sub/superscripts; write λ(T), N(G) instead): {small[:12]}")
    if outside:
        problems.append(f"text outside the page: {outside[:12]}")
    if badw:
        problems.append(f"stroke widths other than {LW}/{THIN} pt: {badw}")
    if problems and strict:
        raise RuntimeError(f"{stem}: " + " | ".join(problems))
    print(f"saved {stem}.pdf/.png  fonts={[f[0] for f in fonts]}  stroke widths={widths}" + (f"  WARN {problems}" if problems else ""))
    return pdf, png
