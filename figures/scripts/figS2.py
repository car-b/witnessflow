"""Supplementary Figure 2 (suppplan_v2, SF2 'full replay results'): single panels S2A-S2E.

Run from figures/scripts:   python figS2.py  [S2A S2B ...]
Read-only inputs : figures/derived/figS2_*.tsv written by figures/derive/sf2_{a_nodes,b_donor,c_forest,d_selfcheck,
                   e_ablation}.py (deterministic reorganisations of frozen results/ tables; every derive script asserts
                   its numbers against frozen summaries).  This script re-asserts the key numbers it prints.
Outputs          : figures/panels/figS2{A..E}.pdf/.png (wfstyle.save, strict audit) and
                   figures/source_data/figS2{A..E}[_name].tsv (wfstyle.save_source).
No new experiment, fit, optimisation, simulation, test statistic or p value is computed here.
"""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, Normalize
import wfstyle as S

S.setup()
DER = S.DERIVED
WHITE = "#FFFFFF"
C_CL, C_RD, C_BR, C_PSL, C_LP = (S.METHOD[k] for k in ("classical", "random", "balanced_random", "psl", "lp_true"))


def rd(name):
    return pd.read_csv(os.path.join(DER, name), sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])


def lum(c):
    r, g, b = to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fmt_num(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def dname(s):
    """Display form of parent names in drawn text (source keys CD4T / CD8T unchanged): 'CD4 T', 'CD8 T'."""
    return str(s).replace("CD4T", "CD4 T").replace("CD8T", "CD8 T")


class Panel:
    """Single panel with a full-figure overlay axis in millimetre coordinates."""

    def __init__(self, key, h, w=None):
        W0, H0 = S.SIZES[key]
        assert abs(h - H0) <= 0.25 * H0 + 1e-9, (key, h, H0)
        w = W0 if w is None else w           # composite fit (composeFigS2.py): rows filled to 184 mm
        assert abs(w - W0) <= 4.0 + 1e-9, (key, w, W0)
        self.key, self.W, self.H = key, w, h
        self.fig = S.new(key, w=w, h=h, layout=None)
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=-1)
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(0, self.H)
        self.ov.axis("off")
        self.top = self.fig.add_axes([0, 0, 1, 1], zorder=10)   # overlay drawn above data axes
        self.top.set_xlim(0, self.W)
        self.top.set_ylim(0, self.H)
        self.top.axis("off")
        self.top.patch.set_alpha(0)

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, top=False, **kw):
        return (self.top if top else self.ov).text(x, y, s, **kw)

    def rect(self, x, y, w, h, top=False, **kw):
        kw.setdefault("lw", 0)
        r = Rectangle((x, y), w, h, **kw)
        (self.top if top else self.ov).add_patch(r)
        return r

    def line(self, xs, ys, top=False, **kw):
        kw.setdefault("lw", S.LW)
        kw.setdefault("color", S.INK)
        kw.setdefault("solid_capstyle", "butt")
        return (self.top if top else self.ov).plot(xs, ys, **kw)

    def mark(self, x, y, top=False, **kw):
        kw.setdefault("ls", "none")
        return (self.top if top else self.ov).plot([x], [y], **kw)

    def head(self, tag, title):
        y = self.H - 0.8
        S.tier_tag(self.fig, tag, x=(self.W - 0.8) / self.W, y=y / self.H, va="top")
        self.text(0.8, y, title, ha="left", va="top", fontweight="bold")

    def tw(self, s, **kw):
        """Rendered width of a text string in mm."""
        t = self.ov.text(0, 0, s, **kw)
        bb = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        t.remove()
        return bb.width / self.fig.dpi * 25.4

    def legend_row(self, x, y, items, gap=2.2, glyph=3.2):
        """items: list of (draw(x_left, y) or None, label); returns the x after the last item."""
        for draw, lab in items:
            if draw is not None:
                draw(x, y)
                x += glyph + 0.6
            self.text(x, y, lab, ha="left", va="center")
            x += self.tw(lab) + gap
        return x

    def save(self):
        return S.save(self.fig, self.key)


def mstyle(key, ms=2.6):
    c = S.METHOD[key]
    if key == "balanced_random":
        return dict(marker=S.MARKER[key], ms=ms + 0.3, mfc=c, mec=C_RD, mew=S.THIN, ls="none")
    if key == "psl":
        return dict(marker=S.MARKER[key], ms=ms + 2.6, mfc=c, mec=S.INK, mew=S.THIN, ls="none")
    return dict(marker=S.MARKER[key], ms=ms, mfc=c, mec=WHITE, mew=S.THIN, ls="none")


# =====================================================================================================  S2A
A_NODES = {"wingmen_v1": [104, 117, 130, 136, 142, 156, 169, 180, 195, 221, 247, 260],
           "levine32_v2": [57, 64, 71, 85, 100, 120, 142, 170, 213, 284, 570]}
A_FULL = {"wingmen_v1": [130, 260], "levine32_v2": [71, 142, 213, 284]}
A_KEY = {"classical_stack": "classical", "random": "random", "balanced_random": "balanced_random", "PSL_3design": "psl"}


def a_pos(ds, r):
    i = A_NODES[ds].index(r)
    return i + (0.95 if (ds == "levine32_v2" and r == 570) else 0.0)


def fig_S2A():
    runs, seed, summ, grid = rd("figS2_A_runs.tsv"), rd("figS2_A_seed.tsv"), rd("figS2_A_summary.tsv"), rd("figS2_A_grid.tsv")
    assert len(runs) == 3240 and len(summ) == 140 and (grid.n_runs > 0).sum() == 140
    for ds in A_NODES:
        assert sorted(runs[runs.dataset == ds].r.unique()) == A_NODES[ds]
    H = 87                                      # composite: = S2B height (row a | b)
    P = Panel("S2A", H, w=122)
    P.head("Replay", "All design-node replays: coverage, half-width, reference error, minimum cells")
    COLS = [("wingmen_v1", "CD4T"), ("wingmen_v1", "CD8T"), ("levine32_v2", "T"), ("levine32_v2", "B")]
    x0, gP, gD = 11.5, 1.3, 5.4
    w = (P.W - 1.6 - x0 - 2 * gP - gD) / 4
    XS = [x0, x0 + w + gP, x0 + 2 * w + gP + gD, x0 + 3 * w + 2 * gP + gD]
    top = H - 12.2
    hm, hs, gr = 13.6, 8.2, 1.5
    ROWS = [("covered", top - hm, hm), ("half_ok", top - 2 * hm - gr, hm), ("err_ok", top - 3 * hm - 2 * gr, hm),
            ("min_cells", top - 3 * hm - 3 * gr - hs, hs)]
    OFF = {"classical_stack": -0.27, "random": 0.0, "balanced_random": 0.27, "PSL_3design": 0.27}
    OFF570 = {"classical_stack": -0.33, "random": -0.11, "balanced_random": 0.11, "PSL_3design": 0.33}
    MINY = {"wingmen_v1": (0, 450, [0, 200, 400]), "levine32_v2": (0, 360, [0, 150, 300])}
    eps = summ.drop_duplicates(["dataset", "parent"]).set_index(["dataset", "parent"]).eps
    for ci, (ds, par) in enumerate(COLS):
        nodes = A_NODES[ds]
        xmax = a_pos(ds, nodes[-1]) + (0.7 if ds == "levine32_v2" else 0.45)   # room for the PSL star
        for row, y0, hh in ROWS:
            ax = P.ax(XS[ci], y0, w, hh)
            ax.set_xlim(-0.45, xmax)
            for r in A_FULL[ds]:
                ax.axvspan(a_pos(ds, r) - 0.45, a_pos(ds, r) + 0.45, color=S.BAND, lw=0, zorder=0)
            sub = runs[(runs.dataset == ds) & (runs.parent == par)]
            sm = summ[(summ.dataset == ds) & (summ.parent == par)]
            sd = seed[(seed.dataset == ds) & (seed.parent == par)]
            for meth in ("balanced_random", "random", "classical_stack", "PSL_3design"):
                key = A_KEY[meth]
                col = S.METHOD[key]
                off = lambda r: OFF570[meth] if (ds == "levine32_v2" and r == 570) else OFF[meth]   # noqa: E731
                z = {"balanced_random": 3, "random": 4, "classical_stack": 5, "PSL_3design": 6}[meth]
                s_m = sm[sm.method == meth].sort_values("r")
                if s_m.empty:
                    continue
                vcol = {"covered": "covered", "half_ok": "half_ok", "err_ok": "err_ok", "min_cells": "min_cells"}[row]
                scale = 1.0 if row == "min_cells" else 100.0
                if meth != "PSL_3design":   # method mean line (no line across the Levine axis break)
                    s_l = s_m[s_m.r != 570]
                    ax.plot([a_pos(ds, r) + off(r) for r in s_l.r], scale * s_l[vcol].values, color=col,
                            lw=S.LW, zorder=z - 0.5, solid_joinstyle="round")
                if row != "covered":       # allocation layer (covered is a design property)
                    rr = sub[sub.method == meth]
                    ax.plot([a_pos(ds, r) + off(r) for r in rr.r], scale * rr[vcol].values, ls="none", marker="o",
                            ms=0.9, mew=0, mfc=col if meth != "balanced_random" else C_RD, alpha=0.45, zorder=z)
                ss = sd[sd.method == meth]   # design-seed layer
                ax.plot([a_pos(ds, r) + off(r) for r in ss.r], scale * ss[vcol].values,
                        zorder=z + 0.2, **mstyle(key, 2.3))
            if row == "min_cells":
                lo, hi, ticks = MINY[ds]
                ax.set_ylim(lo - 0.04 * hi, hi)
                ax.set_yticks(ticks)
                ax.tick_params(labelleft=(ci in (0, 2)))
                ax.set_xticks([a_pos(ds, r) for r in nodes])
                ax.set_xticklabels([str(r) for r in nodes], rotation=90)
                if ds == "wingmen_v1":     # r = 180 comes from the e5b supplement table
                    ax.text(a_pos(ds, 180), -5.7 / hs, "†", transform=ax.get_xaxis_transform(), ha="center",
                            va="top")
            else:
                ax.set_ylim(44, 103.5)
                ax.set_yticks([50, 75, 100])
                ax.tick_params(labelleft=(ci == 0))
                ax.set_xticks([a_pos(ds, r) for r in nodes])
                ax.tick_params(labelbottom=False)
            if ds == "levine32_v2":     # broken categorical axis before r = 570
                xb = (a_pos(ds, 284) + a_pos(ds, 570)) / 2
                ax.plot([xb - 0.12, xb + 0.12], [-0.06, 0.06], transform=ax.get_xaxis_transform(), color=S.INK,
                        lw=S.LW, clip_on=False, zorder=8)
        # column headers
        P.text(XS[ci] + w / 2, H - 9.0, f"{dname(par)} · ε = {eps[(ds, par)]:g}", ha="center", va="bottom")
    P.text((XS[0] + XS[1] + w) / 2, H - 5.6, "WINGMEN (base design 130 tubes)", ha="center", va="bottom")
    P.text((XS[2] + XS[3] + w) / 2, H - 5.6, "Levine-32 (base design 71 tubes)", ha="center", va="bottom")
    for x in ((XS[0], XS[1] + w), (XS[2], XS[3] + w)):
        P.line(x, [H - 5.9, H - 5.9], lw=S.THIN, color=S.MUTED)
    labs = {"covered": "Triples\ncovered (%)", "half_ok": "Half-width\n≤ ε (%)", "err_ok": "Error\n≤ ε (%)",
            "min_cells": "Min\ncells"}
    for row, y0, hh in ROWS:
        P.text(2.95, y0 + hh / 2, labs[row], rotation=90, ha="center", va="center", multialignment="center",
               linespacing=1.0)
    ybot = ROWS[-1][1]
    P.text((XS[0] + XS[3] + w) / 2, ybot - 9.0, "Tubes r (equally spaced nodes; shaded = complete round)",
           ha="center", va="top")
    # legend (3 lines)
    def meth(key):
        def f(x, y):
            if key != "psl":
                P.line([x, x + 3.2], [y, y], color=S.METHOD[key], lw=S.LW)
            P.mark(x + 1.6, y, **mstyle(key, 2.3))
        return f

    def dots(x, y):
        for dx, dy in ((0.9, 0.35), (1.6, -0.3), (2.3, 0.15)):
            P.mark(x + dx, y + dy, marker="o", ms=0.9, mew=0, mfc=S.INK, alpha=0.6)

    def band(x, y):
        P.rect(x + 0.6, y - 1.0, 2.0, 2.0, fc=S.BAND)

    xe = P.legend_row(0.8, 7.0, [(meth("classical"), "Classical (seed 0)"), (meth("random"), "Random (3 seeds)"),
                                 (meth("balanced_random"), "Balanced random (3 seeds)"),
                                 (meth("psl"), "PSL(2,19): r = 570 only")], gap=1.3, glyph=2.9)
    assert xe < P.W + 1.0, xe
    P.legend_row(0.8, 4.2, [(None, "Markers = design-seed means ·"), (dots, "single allocations (10 per design) ·"),
                            (None, "lines = method means")], gap=1.0)
    foot = "† r = 180 supplementary run · PSL not run at other nodes; nothing interpolated"
    P.text(0.8, 1.4, foot, ha="left", va="center", color=S.MUTED)
    assert P.tw(foot) < P.W - 1.6
    # source data
    src = runs.copy()
    src["exists"] = True
    S.save_source(src, "S2A")
    S.save_source(summ, "S2A", "summary")
    S.save_source(grid, "S2A", "grid")
    return P.save()


# =====================================================================================================  S2B
B_NODES = [130, 142, 156, 180, 221, 260]


def fig_S2B():
    C = rd("figS2_B_cells.tsv")
    ana = C[C.status == "analysed"]
    assert len(ana) == 300 and len(C) == 312
    H = 87
    P = Panel("S2B", H)
    P.head("Exploratory", "Per-donor replay, all six nodes")
    cw, rh, gB = 1.8, 1.95, 1.3
    xL = 7.2
    norm = Normalize(0.5, 1.0)
    assert ana[["half_mean", "err_mean"]].min().min() >= 0.5

    def cx(b, i, m):
        return xL + b * (12 * cw + gB) + (2 * i + m) * cw

    dy = 2.8                                   # header shift for the method key line
    ytop = H - 14.4 - dy
    yrow = {}
    y = ytop
    for par in ("CD4T", "CD8T"):
        for d in range(13):
            yrow[(par, d)] = y - rh
            y -= rh
        y -= 1.5
    # method key directly under the title: the two designs sit in adjacent columns at every node
    xe = P.legend_row(xL, H - 4.9, [(lambda x, y: P.mark(x + 1.0, y, **mstyle("classical", 2.8)), "Classical (seed 0)"),
                                    (lambda x, y: P.mark(x + 1.0, y, **mstyle("random", 2.8)), "Random (seed 0)")],
                      glyph=2.0, gap=3.0)
    assert xe < P.W, xe
    # headers
    for b, lab in enumerate(("Half-width ≤ ε(d)", "Error ≤ ε(d)")):
        P.text(cx(b, 0, 0) + 6 * cw, H - 5.0 - dy, lab, ha="center", va="top")
        for i, r in enumerate(B_NODES):
            xc = cx(b, i, 0) + cw
            if r in (130, 260):
                P.rect(cx(b, i, 0) + 0.05, H - 12.4 - dy, 2 * cw - 0.1, 4.6, fc=S.BAND)
            P.text(xc, H - 8.0 - dy, str(r), rotation=90, ha="center", va="top")
            for m, key in enumerate(("classical", "random")):
                P.mark(cx(b, i, m) + cw / 2, ytop + 1.0, **mstyle(key, 2.4))
    P.text(cx(1, 5, 1) + cw + 4.0, ytop + 1.0, "ε(d)", ha="center", va="center")
    # cells
    for _, q in C.iterrows():
        yb = yrow[(q.parent, int(q.donor))]
        for b, met in enumerate(("half", "err")):
            i = B_NODES.index(int(q.r))
            m = 0 if q.method == "classical_stack" else 1
            x = cx(b, i, m)
            if q.status != "analysed":
                if i == 0 and m == 0:     # one hatched strip per metric block (no imputed cells)
                    P.rect(x, yb, 12 * cw, rh, fc=WHITE, ec=S.MUTED, lw=S.THIN, hatch=S.LIMITS_HATCH)
                continue
            fc = S.SEQ(norm(q[f"{met}_mean"]))
            P.rect(x, yb, cw, rh, fc=fc, ec=WHITE, lw=S.THIN)
            L = (q[f"{met}_max"] - q[f"{met}_min"]) / 0.10 * rh
            assert L <= rh
            yc = yb + rh / 2
            P.line([x + cw / 2] * 2, [yc - L / 2, yc + L / 2], lw=S.THIN, color=WHITE if lum(fc) < 0.5 else S.INK)
    # excluded donor note, row labels, eps(d)
    yb6 = yrow[("CD8T", 6)]
    for b, s in ((0, "not analysed"), (1, "< 500 cells")):
        P.text(cx(b, 0, 0) + 6 * cw, yb6 + rh / 2, s, ha="center", va="center", top=True,
               bbox=dict(boxstyle="square,pad=0.1", fc=WHITE, ec="none"))
        assert P.tw(s) < 12 * cw - 3.0
    epsd = ana.drop_duplicates(["parent", "donor"]).set_index(["parent", "donor"]).eps_d
    for par in ("CD4T", "CD8T"):
        ys = [yrow[(par, d)] for d in range(13)]
        P.text(1.6, (min(ys) + max(ys) + rh) / 2, dname(par), rotation=90, ha="center", va="center")
        for d in range(13):
            P.text(xL - 0.6, yrow[(par, d)] + rh / 2, str(d), ha="right", va="center")
            if (par, d) in epsd.index:
                P.text(cx(1, 5, 1) + cw + 4.0, yrow[(par, d)] + rh / 2, f"{epsd[(par, d)]:.2f}", ha="center",
                       va="center")
            else:
                P.text(cx(1, 5, 1) + cw + 4.0, yrow[(par, d)] + rh / 2, "–", ha="center", va="center", color=S.MUTED)
    P.text(3.9, ytop + 1.0, "donor", ha="center", va="center", color=S.MUTED)
    # colour key + notes
    yk = 14.4
    nk = 50
    kw_ = 15.0
    for k in range(nk):
        P.rect(xL + k * kw_ / nk, yk, kw_ / nk + 0.02, 1.4, fc=S.SEQ(k / (nk - 1)))
    for v in (50, 75, 100):
        P.text(xL + (v - 50) / 50 * kw_, yk - 0.3, f"{v}", ha="center", va="top")
    P.text(xL + kw_ + 1.6, yk + 0.7, "mean of 3 allocations (%)", ha="left", va="center")
    P.line([xL + 0.6, xL + 0.6], [8.6, 10.0], lw=S.THIN, color=S.INK)
    P.text(xL + 1.8, 9.3, "tick = range of 3 allocations (cell = 10 pp)", ha="left", va="center")
    P.text(0.8, 3.0, "Exploratory · grid uses r = 180, not 195 · ε(d) varies\nby donor; colour is not a donor ranking",
           ha="left", va="center", linespacing=1.05)
    S.save_source(C, "S2B")
    return P.save()


# =====================================================================================================  S2C
C_EST = ["knn_block", "knn_block_within", "knn_major", "maxent", "cond_indep"]
C_EST_LAB = {"knn_block": "kNN block", "knn_block_within": "kNN within", "knn_major": "kNN major",
             "maxent": "EM fit", "cond_indep": "cond. ind."}
C_EST_KEY = {"knn_block": "b", "knn_block_within": "w", "knn_major": "m", "maxent": "e", "cond_indep": "c"}


def nice_range(vals, fits=lambda a, b: True):
    """Window x range: smallest 'nice' step with <= 4 intervals whose two end labels fit the window."""
    lo_r, hi_r = float(np.min(vals)), float(np.max(vals))
    if lo_r < 0.45 * hi_r:
        lo_r = 0.0
    for st in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5):
        lo = np.floor(lo_r / st + 1e-9) * st
        hi = min(np.ceil(hi_r / st - 1e-9) * st, 1.0)
        if hi <= lo:
            hi = lo + st
        if round((hi - lo) / st) <= 4 and fits(fmt_num(lo), fmt_num(hi)):
            return lo, hi, st
    return 0.0, 1.0, 0.25


def fig_S2C():
    R, Wd, T = rd("figS2_C_records.tsv"), rd("figS2_C_windows.tsv"), rd("figS2_C_totals.tsv")
    assert len(R) == 630 and int(R.in_figure.sum()) == 330 and len(Wd) == 33
    F = R[R.in_figure]
    assert F.ref_in_K1.all()
    H = 97
    P = Panel("S2C", H, w=184)                  # composite: full row width
    P.head("Replay (same-source)", "Audit forest: all 330 drawn claim × replicate records "
                                   "(27 protocol claims + 6 rare-B windows; 630 in Source Data)")
    # ---- layout: rows of windows (protocol, claim)
    layout = [
        [("wingmen_v1", ["C1", "C2", "C3", "C4", "C5"]), ("wingmen_v1", ["D1", "D2", "D3", "D4"])],
        [("wingmen_v1", ["E1", "E2", "E3"]), ("wingmen_v1", ["F1"]), ("wingmen_v1", ["G1", "G2"])],
        [("levine32_v2", ["H1", "H2", "H3"]), ("levine32_v2", ["M1", "M2"]),
         ("ladder", [("levine32_v2", "B1"), ("levine32_v2_rareB_1", "B1"), ("levine32_v2_rareB_0p3", "B1"),
                     ("levine32_v2_rareB_0p1", "B1")])],
        [("levine32_v2", ["T1", "T2", "T3"]), ("levine32_v2", ["N1", "N2"]),
         ("ladder", [("levine32_v2", "B2"), ("levine32_v2_rareB_1", "B2"), ("levine32_v2_rareB_0p3", "B2"),
                     ("levine32_v2_rareB_0p1", "B2")])],
    ]
    x0, slot, gG = 4.6, 15.15, 1.15          # 184 mm wide: +0.4 mm per window slot (9 slots), table +3.6 mm
    bw, cgap, ccw = 10.1, 0.35, 2.85         # bars width, gap, count-column width
    pitch = 1.3
    rowH = 22.6
    ytop0 = H - 5.6
    LEVEL = {"levine32_v2": "main", "levine32_v2_rareB_1": "1%", "levine32_v2_rareB_0p3": "0.3%",
             "levine32_v2_rareB_0p1": "0.1%"}
    cnt_norm = lambda k: S.SEQ(0.12 + 0.08 * k)   # noqa: E731
    drawn = 0
    for ri, groups in enumerate(layout):
        yt = ytop0 - ri * rowH            # top of the row block
        y_par, y_cl = yt, yt - 2.6        # text tops: group line, claim line
        yb_top = yt - 5.6                 # top of bars area
        x = x0
        for gi, (gkey, items) in enumerate(groups):
            if gi:
                x += gG
            gx0 = x
            for wi, it in enumerate(items):
                proto, claim = (it if gkey == "ladder" else (gkey, it))
                g = F[(F.protocol == proto) & (F.claim == claim)].sort_values("rep")
                assert len(g) == 10
                w = Wd[(Wd.protocol == proto) & (Wd.claim == claim)].iloc[0]
                lo, hi, st = nice_range(np.r_[g.conf_lo, g.conf_hi, g.knn_block, g.ref],
                                        fits=lambda a, b: P.tw(a) + P.tw(b) < bw - 1.2)
                sc = bw / (hi - lo)
                X = lambda v: x + (v - lo) * sc   # noqa: E731
                ybars = yb_top - 10 * pitch
                # reference line, bars
                for k, (_, q) in enumerate(g.iterrows()):
                    yc = yb_top - (k + 0.5) * pitch
                    P.rect(X(q.conf_lo), yc - 0.40, (q.conf_hi - q.conf_lo) * sc, 0.80, fc=S.TIER[3], zorder=2)
                    P.rect(X(q.ident_lo), yc - 0.18, (q.ident_hi - q.ident_lo) * sc, 0.36, fc=S.TIER[2], zorder=3)
                    if q.tier == "I":
                        P.rect(X(q.conf_lo), yc - 0.40, (q.conf_hi - q.conf_lo) * sc, 0.80, fill=False, ec=S.INK,
                               lw=S.THIN, zorder=4)
                    P.mark(X(q.knn_block), yc, marker="D", ms=2.1, mfc=S.INK, mec=WHITE, mew=S.THIN, zorder=6)
                    drawn += 1
                P.line([X(w.ref)] * 2, [ybars - 0.2, yb_top + 0.2], lw=S.THIN, color=S.INK, zorder=5)
                # axis
                P.line([x, x + bw], [ybars - 0.35] * 2, lw=S.THIN, color=S.INK)
                nt = int(round((hi - lo) / st))
                for j in range(nt + 1):
                    P.line([X(lo + j * st)] * 2, [ybars - 0.35, ybars - 0.95], lw=S.THIN, color=S.INK)
                P.text(x, ybars - 1.05, fmt_num(lo), ha="left", va="top")
                P.text(x + bw, ybars - 1.05, fmt_num(hi), ha="right", va="top")
                # count column (estimator errors > 5 pp)
                cxl = x + bw + cgap
                ch = 10 * pitch / 5
                for e_i, e in enumerate(C_EST):
                    v = w[f"gt5_{e}"]
                    yc0 = yb_top - (e_i + 1) * ch
                    if str(v) == "not computed":
                        P.rect(cxl, yc0, ccw, ch, fc=WHITE, ec=S.MUTED, lw=S.THIN, hatch=S.LIMITS_HATCH)
                        continue
                    k = int(float(v))
                    fc = WHITE if k == 0 else cnt_norm(k)
                    P.rect(cxl, yc0, ccw, ch, fc=fc, ec=S.GRID, lw=S.THIN)
                    P.text(cxl + ccw / 2, yc0 + ch / 2, str(k), ha="center", va="center",
                           color=(S.MUTED if k == 0 else (WHITE if lum(fc) < 0.5 else S.INK)))
                # headers
                if gkey == "ladder":
                    lev = LEVEL[proto]
                    head2 = f"{claim}  n {w.n_parent:,}"
                    head1 = "B (main)" if lev == "main" else f"rare-B {lev}"
                    P.text(x, y_par, head1, ha="left", va="top", color=S.MUTED)
                    assert P.tw(head2) < slot - 0.3 and P.tw(head1) < slot - 0.3
                else:
                    head2 = claim
                    if wi == 0:
                        P.text(x, y_par, dname(w.parent), ha="left", va="top", color=S.MUTED)
                    if not w.reliable:
                        P.text(x + bw + cgap + ccw, y_par, f"{w.unreliable_markers} unreliable", ha="right", va="top",
                               color=S.ALARM)
                if not w.reliable:
                    P.rect(x - 0.3, ybars - 0.15, bw + cgap + ccw + 0.6, 10 * pitch + 0.3, fill=False, ec=S.ALARM,
                           lw=S.THIN, zorder=7)
                P.text(x, y_cl, head2, ha="left", va="top")
                x += slot
            if gkey != "ladder" and len(groups) > 1:
                pass
        _ = gx0
    assert drawn == 330
    # dataset labels (left margin)
    for lab, r0, r1 in (("WINGMEN", 0, 1), ("Levine-32", 2, 3)):
        yA = ytop0 - r0 * rowH - 3.0
        yB = ytop0 - r1 * rowH - rowH + 2.5
        P.text(1.8, (yA + yB) / 2, lab, rotation=90, ha="center", va="center", fontweight="bold")
        P.line([3.3, 3.3], [yB, yA], lw=S.THIN, color=S.MUTED)
    # ---- key in the free slots of row 2
    tx = 145.0                                # right strip (totals table) starts here
    kx = x0 + 6 * slot + 2 * gG + 0.8
    ky = ytop0 - rowH - 1.0
    ex = tx - 6.4                             # example count column: row order, letters keyed to the table rows
    items = [("conf", "audit interval (K = 1)"), ("ident", "identification interval"),
             ("knn", "kNN (block) estimate"), ("ref", "reference frequency"),
             ("tier", "tier I (n_direct > 0)"), ("unrel", "unreliable marker"),
             ("cnt", "error > 5 pp (n of 10)"), ("nc", "not computed")]
    P.text(ex + ccw / 2, ky - 1.3, "rows", ha="center", va="center", color=S.MUTED)
    for e_i, e in enumerate(C_EST):
        yc = ky - 1.3 - (e_i + 1) * 2.45
        P.rect(ex, yc - 1.2, ccw, 2.4, fc=WHITE, ec=S.MUTED, lw=S.THIN)
        P.text(ex + ccw / 2, yc, C_EST_KEY[e], ha="center", va="center")
    for k, (kind, lab) in enumerate(items):
        yc = ky - 1.3 - k * 2.45
        assert kx + 5.8 + P.tw(lab) < ex - 0.6, lab
        if kind in ("conf", "tier"):
            P.rect(kx, yc - 0.40, 5.0, 0.80, fc=S.TIER[3])
            if kind == "tier":
                P.rect(kx, yc - 0.40, 5.0, 0.80, fill=False, ec=S.INK, lw=S.THIN)
        elif kind == "ident":
            P.rect(kx + 1.0, yc - 0.18, 3.0, 0.36, fc=S.TIER[2])
        elif kind == "knn":
            P.mark(kx + 2.5, yc, marker="D", ms=2.1, mfc=S.INK, mec=WHITE, mew=S.THIN)
        elif kind == "ref":
            P.line([kx + 2.5] * 2, [yc - 0.9, yc + 0.9], lw=S.THIN)
        elif kind == "unrel":
            P.rect(kx + 0.5, yc - 0.8, 4.0, 1.6, fill=False, ec=S.ALARM, lw=S.THIN)
        elif kind == "cnt":
            P.rect(kx + 1.1, yc - 1.0, ccw, 2.0, fc=cnt_norm(3), ec=S.GRID, lw=S.THIN)
            P.text(kx + 1.1 + ccw / 2, yc, "3", ha="center", va="center")
        elif kind == "nc":
            P.rect(kx + 1.1, yc - 1.0, ccw, 2.0, fc=WHITE, ec=S.MUTED, lw=S.THIN, hatch=S.LIMITS_HATCH)
        P.text(kx + 5.8, yc, lab, ha="left", va="center")
    # ---- right strip: totals per protocol (all records)
    ty = ytop0 + 0.4
    cols = [("wingmen_v1", "W"), ("levine32_v2", "L"), ("levine32_v2_rareB_1", "1"),
            ("levine32_v2_rareB_0p3", "0.3"), ("levine32_v2_rareB_0p1", "0.1")]
    T = T.set_index("protocol")
    cw_ = 4.65
    xs_ = [tx + 14.8 + cw_ / 2 + i * cw_ for i in range(5)]
    assert xs_[-1] + cw_ / 2 <= P.W - 0.5
    P.text(tx, ty, "All records per protocol", ha="left", va="top", fontweight="bold")
    P.text(tx, ty - 2.8, "W WINGMEN · L Levine-32 ·", ha="left", va="top", color=S.MUTED)
    P.text(tx, ty - 5.4, "rare-B replays at 1, 0.3, 0.1%", ha="left", va="top", color=S.MUTED)
    yh = ty - 12.4
    P.text((xs_[2] + xs_[4]) / 2, yh + 1.3, "rare-B %", ha="center", va="bottom")
    P.line([xs_[2] - cw_ / 2 + 0.3, xs_[4] + cw_ / 2 - 0.3], [yh + 1.2] * 2, lw=S.THIN, color=S.MUTED)
    for (pr, lab), xc in zip(cols, xs_):
        P.text(xc, yh, lab, ha="center", va="center")
    rows_ = [("n", "records", None)] + [("sec", "error > 5 pp", None)] + \
            [("est", C_EST_LAB[e], e) for e in C_EST] + [("sec", "ref inside", None)] + \
            [("K", k, k) for k in ("K1", "K4", "K8")] + [("sec", "median width (pp)", None)] + \
            [("W", k, k) for k in ("K1", "K4", "K8")]
    tp = 3.3                                    # table row pitch (composite: strip filled down to the forest axes)
    yy = yh - 2.8
    for kind, lab, key in rows_:
        if kind == "sec":
            P.text(tx, yy, lab, ha="left", va="center", color=S.MUTED, fontstyle="italic")
            P.line([tx + P.tw(lab, fontstyle="italic") + 0.8, xs_[-1] + cw_ / 2], [yy, yy], lw=S.THIN, color=S.GRID)
            yy -= tp
            continue
        if kind == "est":         # letter = row of every window's count column
            P.text(tx, yy, C_EST_KEY[key], ha="left", va="center", fontweight="bold")
            P.text(tx + 2.9, yy, lab, ha="left", va="center")
            assert tx + 2.9 + P.tw(lab) < xs_[0] - 1.3, lab
        else:
            P.text(tx, yy, lab, ha="left", va="center")
        for (pr, _), xc in zip(cols, xs_):
            if kind == "n":
                v = int(T.loc[pr, "n_records"])
            elif kind == "est":
                v = T.loc[pr, f"gt5_{key}"]
            elif kind == "K":
                v = T.loc[pr, f"ref_in_{key}"]
            else:
                v = T.loc[pr, f"median_conf_w_{key}"]
            if str(v) == "not computed":
                P.rect(xc - cw_ / 2 + 0.4, yy - 1.0, cw_ - 0.8, 2.0, fc=WHITE, ec=S.MUTED, lw=S.THIN,
                       hatch=S.LIMITS_HATCH)
            elif kind == "W":
                P.text(xc, yy, f"{100 * float(v):.0f}", ha="center", va="center")
            else:
                P.text(xc, yy, str(int(float(v))), ha="center", va="center")
        yy -= tp
    # assertions of the printed totals (claims table C9, inventory R3; frozen-table derived)
    assert [int(float(T.loc["wingmen_v1", f"gt5_{e}"])) for e in C_EST] == [9, 14, 33, 37, 66]
    assert [str(T.loc["levine32_v2", f"gt5_{e}"]) for e in C_EST] == ["1", "not computed", "50", "12", "11"]
    assert all(str(T.loc[p, "ref_in_K1"]) == str(T.loc[p, "n_records"]) for p, _ in cols)
    notes = ["Every stored interval contains", "the reference (K1; K4, K8 where", "computed). K4/K8: stored width",
             "and inclusion flag only, so no", "K4/K8 endpoints are drawn.",
             "Rare-B columns count all 12", "claims; only B1, B2 are drawn.",
             "x scale differs per window."]
    ng, npitch = 1.8, 3.0
    for k, s in enumerate(notes):
        P.text(tx, yy - ng - k * npitch, s, ha="left", va="center", color=S.MUTED)
        assert tx + P.tw(s) < P.W - 0.4, s
    assert yy - ng - (len(notes) - 1) * npitch > 2.0
    S.save_source(R, "S2C")
    S.save_source(Wd, "S2C", "windows")
    S.save_source(T.reset_index(), "S2C", "totals")
    return P.save()


# =====================================================================================================  S2D
D_LAB = {"null": "null (no change)", "single+0.5": "marker +0.5", "single-0.5": "marker −0.5",
         "single+0.25": "marker +0.25", "single-0.25": "marker −0.25", "wholetube+0.5": "whole tube +0.5",
         "joint_only": "joint only", "composition": "composition", "random_drop": "random drop",
         "global_drift": "global drift", "none": "no shift", "measured": "X in tube",
         "unmeasured": "X not in tube"}
D_DS = [("levine32_v2", "o", S.INK), ("wingmen_v1", "^", S.MUTED)]


def d_fmt(rate):
    v = 100 * rate
    return "100" if abs(v - 100) < 1e-9 else ("0" if abs(v) < 1e-9 else f"{v:.1f}")


def fig_S2D():
    M = rd("figS2_D_matrix.tsv")
    assert len(M) == 26
    H = 81                                      # composite: = S2E height (row d | e); 82 + 2 + 100 = 184 mm
    P = Panel("S2D", H, w=82)
    P.head("Simulation + Replay", "All self-check scenarios")
    xa0, xa1 = 20.2, 38.7                       # alarm axis
    colx = {"loc": (39.7, 53.1), "hit": (53.1, 66.5), "unch": (66.5, 81.2)}
    XF = (0.25, 0.75)                           # sub-column centres (Levine-32, WINGMEN)
    rp, bh, gb = 3.45, 3.0, 0.8                 # row pitch, block band height, gap after a block
    yh1, yh2 = H - 5.8, H - 8.9
    P.text((xa0 + xa1) / 2, yh1, "Alarm (%)", ha="center", va="center")
    P.text((xa0 + xa1) / 2, yh2, "with 95% CP", ha="center", va="center", color=S.MUTED)
    for k, lab in (("loc", "Localised (%)"), ("hit", "Hit | alarm (%)"), ("unch", "Uncheckable (%)")):
        P.text(sum(colx[k]) / 2, yh1, lab.replace(" (%)", ""), ha="center", va="center")
    for k in colx:
        a, b = colx[k]
        for j, (ds, mk, c) in enumerate(D_DS):
            P.mark(a + (b - a) * XF[j], yh2, marker=mk, ms=2.6, mfc=c, mec=c, mew=0)
    ytab = H - 11.2
    y = ytab
    yrows, bands = {}, []
    for block, lab in (("e5d", "e5d · fixed family, one perturbed tube · 200 allocations (null 1000)"),
                       ("e5k", "e5k · X+ cells depleted in 10% of tubes · 20 allocations")):
        bands.append((y - bh, lab))
        y -= bh
        scen = M[M.block == block].scenario.drop_duplicates().tolist()
        for s in scen:
            yrows[(block, s)] = y - rp / 2
            y -= rp
        y -= gb
    ybot = y + gb
    for yb, lab in bands:          # drawn above the alarm axis so its grid does not cut the labels
        P.rect(0.6, yb, P.W - 1.2, bh, fc=S.GRID, top=True)
        P.text(1.0, yb + bh / 2, lab, ha="left", va="center", top=True)
        assert P.tw(lab) < P.W - 2.0, lab
    # alarm axis
    ax = P.ax(xa0, ybot, xa1 - xa0, ytab - ybot)
    ax.set_xlim(-3, 103)
    ax.set_ylim(ybot, ytab)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.patch.set_alpha(0)
    ax.set_xticks([0, 50, 100])
    for xv in (0, 50, 100):
        ax.axvline(xv, color=S.GRID, lw=S.THIN, zorder=0)
    for _, q in M.iterrows():
        yc = yrows[(q.block, q.scenario)]
        for j, (ds, mk, c) in enumerate(D_DS):
            if q.dataset != ds:
                continue
            yy = yc + (0.5 if j == 0 else -0.5)
            if q.block == "e5d":
                ax.plot([100 * q.alarm_ci_lo, 100 * q.alarm_ci_hi], [yy, yy], color=c, lw=S.LW, solid_capstyle="butt")
                ax.plot([100 * q.alarm_rate], [yy], marker=mk, ms=2.6, mfc=c, mec=WHITE, mew=S.THIN, ls="none", zorder=4)
            else:
                ax.plot([100 * q.alarm_rate], [yy], marker=mk, ms=2.6, mfc=WHITE, mec=c, mew=S.THIN, ls="none", zorder=4)
    # count text where no CI is drawn or values are tiny
    for block, s in yrows:
        if block == "e5k" or s == "null":
            g = M[(M.block == block) & (M.scenario == s)].set_index("dataset")
            assert int(g.loc["levine32_v2", "n_alloc"]) == int(g.loc["wingmen_v1", "n_alloc"])
            t = f"{int(g.loc['levine32_v2', 'alarm_k'])} · {int(g.loc['wingmen_v1', 'alarm_k'])} of " \
                f"{int(g.loc['wingmen_v1', 'n_alloc'])}"
            ax.text(101, yrows[(block, s)], t, ha="right", va="center")
            assert P.tw(t) < (xa1 - xa0) - 1.8, t      # clears the markers at 0%
    ax.set_xlabel("")
    # row labels and metric cells
    for (block, s), yc in yrows.items():
        P.text(1.0, yc, D_LAB[s], ha="left", va="center")
        assert P.tw(D_LAB[s]) < xa0 - 1.6, s
        g = M[(M.block == block) & (M.scenario == s)].set_index("dataset")
        for k in ("loc", "hit", "unch"):
            a, b = colx[k]
            if k == "unch" and block == "e5k":
                P.text((a + b) / 2, yc, "not saved", ha="center", va="center", color=S.MUTED, fontstyle="italic")
                continue
            for j, (ds, mk, c) in enumerate(D_DS):
                q = g.loc[ds]
                xc = a + (b - a) * XF[j]
                if k == "loc":
                    if pd.isna(q.localised_rate):
                        s_, col = "n/a", S.MUTED
                    elif block == "e5k":
                        s_, col = f"{int(q.localised_k)}/20", S.INK
                    else:
                        s_, col = d_fmt(q.localised_rate), S.INK
                elif k == "hit":
                    if pd.isna(q.hit_given_alarm):
                        s_, col = "n/a", S.MUTED
                    elif block == "e5k":
                        s_, col = f"{int(q.localised_k)}/{int(q.alarm_k)}", S.INK
                    else:
                        s_, col = d_fmt(q.hit_given_alarm), S.INK
                else:
                    s_, col = f"{100 * q.uncheckable_frac:.1f}", S.INK
                P.text(xc, yc, s_, ha="center", va="center", color=col)
    for k in ("loc", "hit", "unch"):
        P.line([colx[k][0]] * 2, [ybot, ytab], lw=S.THIN, color=S.GRID)
    # bottom: axis title, dataset legend, notes
    P.text((xa0 + xa1) / 2, ybot - 3.4, "alarm rate (%)", ha="center", va="top")
    P.legend_row(xa1 + 3.0, ybot - 2.2, [(lambda x, y: P.mark(x + 0.9, y, marker="o", ms=2.6, mfc=S.INK, mec=S.INK,
                                                              mew=0), "Levine-32 T · 71 tubes")],
                 glyph=1.8)
    P.legend_row(xa1 + 3.0, ybot - 5.0, [(lambda x, y: P.mark(x + 0.9, y, marker="^", ms=2.6, mfc=S.MUTED,
                                                              mec=S.MUTED, mew=0), "WINGMEN CD4 T · 130 tubes")],
                 glyph=1.8)
    e_ = M.drop_duplicates("dataset").set_index("dataset").M_total
    assert int(e_["levine32_v2"]) == 1491 and int(e_["wingmen_v1"]) == 2730
    yn0, npitch = ybot - 9.4, 3.0
    for k, s in enumerate(["Values in % · n/a: undefined (no perturbed tube, or no alarm)",
                           "Localised: alarm located at the perturbed tube (e5k: at a shifted tube)",
                           "not saved: not stored · e5k: counts, no CI · p values not shown"]):
        P.text(0.8, yn0 - npitch * k, s, ha="left", va="center", color=S.MUTED)
        assert P.tw(s) < P.W - 1.4, (s, P.tw(s))
    assert yn0 - npitch * 2 - 1.4 > 0
    # assertions of printed key numbers (claims table C6)
    e = M.set_index(["block", "dataset", "scenario"])
    assert e.loc[("e5d", "levine32_v2", "joint_only"), "alarm_rate"] == 0.49
    assert e.loc[("e5d", "wingmen_v1", "joint_only"), "alarm_rate"] == 0.735
    assert int(e.loc[("e5d", "wingmen_v1", "null"), "alarm_k"]) == 1 and int(e.loc[("e5k", "wingmen_v1", "none"), "alarm_k"]) == 1
    S.save_source(M, "S2D")
    return P.save()


# =====================================================================================================  S2E
E_METH = [("LJCR_stack", "Classical stack", "baseline"), ("hybrid_worst", "Uniform req.", ""),
          ("hybrid_knn", "kNN req.", ""), ("hybrid_lp", "LP req.", "")]
# neutral state encoding: grey level + Arial glyph (U+2191 / U+2193 / '='); method colours stay with methods
E_STATE = {"baseline": WHITE, "higher in all": S.TIER[2], "lower in all": S.TIER[3], "identical in all": S.GRID,
           "mixed": WHITE}
E_SYM = {"higher in all": "↑", "lower in all": "↓", "identical in all": "=", "mixed": "", "baseline": ""}
E_TXT = {"higher in all": WHITE, "lower in all": S.INK, "identical in all": S.INK, "mixed": S.INK, "baseline": S.INK}
PAR_LAB = {"T": "Levine-32 T", "B": "Levine-32 B", "CD4T": "WINGMEN CD4 T", "CD8T": "WINGMEN CD8 T"}


def e_dot(share, smax):
    """Marker diameter (pt) with area proportional to the share of targets; minimum 0.6 pt for any nonzero share."""
    return max(0.6, 4.4 * np.sqrt(share / smax))


def fig_S2E():
    cells, runs, led, dl = rd("figS2_E_cells.tsv"), rd("figS2_E_runs.tsv"), rd("figS2_E_ledgers.tsv"), rd("figS2_E_dlam.tsv")
    rec = rd("figS2_E_e5f_records.tsv")
    assert len(rec) == 288 and len(cells) == 32 and len(runs) == 54 and len(led) == 18
    for par in ("CD4T", "CD8T"):   # 'mixed' includes ties: LP error at r = 142 is 0 higher / 1 equal / 8 lower
        q = cells[(cells.parent == par) & (cells.r == 142) & (cells.method == "hybrid_lp")].iloc[0]
        assert (q.err_ok_n_higher, q.err_ok_n_equal, q.err_ok_n_lower, q.err_ok_state) == (0, 1, 8, "mixed")
    H = 81
    P = Panel("S2E", H)
    P.head("Replay", "Requirement ablation (e5f) and allocation ledgers (e5j)")
    # ---------------- top: e5f matrix
    xl = 0.8
    xc0, cwid, mg = 25.6, 8.65, 1.2
    cxs = {}
    for mi, (m, lab, _) in enumerate(E_METH):
        for k, met in enumerate(("half_ok", "err_ok")):
            cxs[(m, met)] = xc0 + mi * (2 * cwid + mg) + k * cwid
    assert cxs[("hybrid_lp", "err_ok")] + cwid <= P.W - 0.6
    yh = H - 5.8
    for mi, (m, lab, sub) in enumerate(E_METH):
        xa = cxs[(m, "half_ok")]
        P.text(xa + cwid, yh, lab, ha="center", va="center")
        assert P.tw(lab) < 2 * cwid, lab
        P.text(xa + cwid / 2, yh - 2.6, "half", ha="center", va="center", color=S.MUTED)
        P.text(xa + 1.5 * cwid, yh - 2.6, "error", ha="center", va="center", color=S.MUTED)
    P.text(xl, yh - 2.6, "parent · ε", ha="left", va="center", color=S.MUTED)
    P.text(xc0 - 1.0, yh - 2.6, "r", ha="right", va="center", color=S.MUTED)
    rh = 2.95
    y = H - 10.0
    order = [("levine32_v2", "T"), ("levine32_v2", "B"), ("wingmen_v1", "CD4T"), ("wingmen_v1", "CD8T")]
    ds_span = {}
    for gi, (ds, par) in enumerate(order):
        if gi == 2:
            y -= 0.8
        for r in (142, 180):
            yb = y - rh
            sub = cells[(cells.dataset == ds) & (cells.parent == par) & (cells.r == r)]
            assert len(sub) == 4
            ds_span.setdefault(ds, []).append(yb)
            if r == 142:
                P.text(xl, yb, f"{S.DATASET[ds]}\n{dname(par)} · {sub.eps.iloc[0]:g}", ha="left", va="center",
                       linespacing=1.0)
            P.text(xc0 - 1.0, yb + rh / 2, str(r), ha="right", va="center")
            for _, q in sub.iterrows():
                for met in ("half_ok", "err_ok"):
                    st = q[f"{met}_state"]
                    x = cxs[(q.method, met)]
                    fc = E_STATE[st]
                    if st == "mixed":
                        P.rect(x + 0.1, yb + 0.1, cwid - 0.2, rh - 0.2, fc=WHITE, ec=S.MUTED, lw=S.THIN, hatch="//////")
                    elif st == "baseline":
                        P.rect(x + 0.3, yb + 0.22, cwid - 0.6, rh - 0.44, fc=WHITE, ec=S.INK, lw=S.LW)
                    else:
                        P.rect(x + 0.1, yb + 0.1, cwid - 0.2, rh - 0.2, fc=fc)
                    v = 100 * q[f"{met}_mean"]
                    s_ = f"{v:.1f}" + (f" {E_SYM[st]}" if E_SYM[st] else "")
                    P.text(x + cwid / 2, yb + rh / 2, s_, ha="center", va="center", top=True, color=E_TXT[st],
                           bbox=(dict(boxstyle="square,pad=0.05", fc=WHITE, ec="none") if st == "mixed" else None))
                    assert P.tw(s_) < cwid - 0.4, s_
            y -= rh
    ymat = y
    # state legend (two rows)
    def sw(st):
        def f(x, yy):
            if st == "mixed":
                P.rect(x, yy - 0.95, 3.4, 1.9, fc=WHITE, ec=S.MUTED, lw=S.THIN, hatch="//////")
            elif st == "baseline":
                P.rect(x + 0.2, yy - 0.75, 3.0, 1.5, fc=WHITE, ec=S.INK, lw=S.LW)
            else:
                P.rect(x, yy - 0.95, 3.4, 1.9, fc=E_STATE[st])
                P.text(x + 1.7, yy, E_SYM[st], ha="center", va="center", color=E_TXT[st])
        return f
    yl = ymat - 2.0
    xe = P.legend_row(xl, yl, [(sw("higher in all"), "higher than baseline in all paired records"),
                               (sw("lower in all"), "lower in all"), (sw("identical in all"), "equal in all")],
                      glyph=3.4, gap=1.8)
    assert xe < P.W + 1.5, xe
    xe = P.legend_row(xl, yl - 2.6, [(sw("mixed"), "mixed signs/ties: not uniformly higher, lower or equal"),
                                     (sw("baseline"), "baseline")], glyph=3.4, gap=1.8)
    assert xe < P.W + 1.5, xe
    note = "Numbers: % of targets meeting half-width ≤ ε (half) or reference error ≤ ε (error)"
    P.text(xl, yl - 5.2, note, ha="left", va="center", color=S.MUTED)
    assert P.tw(note) < P.W - 1.6
    # ---------------- bottom (i): 54 lp_true runs
    yz = yl - 8.6
    P.text(xl, yz, "(i) Gained / lost targets, 54 LP runs", ha="left", va="center", fontweight="bold")
    sub1 = "18 runs/parent vs 3 classical baselines"
    P.text(xl, yz - 2.6, sub1, ha="left", va="center", color=S.MUTED)
    assert xl + P.tw(sub1) < 55.0, P.tw(sub1)
    ax = P.ax(14.6, 13.0, 37.6, yz - 4.6 - 13.0)
    rows = [("T", "half"), ("T", "err"), ("CD4T", "half"), ("CD4T", "err"), ("CD8T", "half"), ("CD8T", "err")]
    nT = {"T": 1140, "CD4T": 2600, "CD8T": 2600}
    jit = {}
    for i, (par, crit) in enumerate(rows):
        yc = len(rows) - 1 - i
        g = runs[runs.parent == par].sort_values(["init_rep", "opt_seed", "alloc"]).reset_index(drop=True)
        for j, q in g.iterrows():
            dy = ((j * 0.6180339887) % 1.0 - 0.5) * 0.62
            for side, col in ((1, f"gained_{crit}"), (-1, f"lost_{crit}")):
                filled = bool(q.has_target_ledger)
                ax.plot([side * q[col]], [yc + dy], marker="D", ms=1.9, ls="none",
                        mfc=C_LP if filled else WHITE, mec=C_LP, mew=S.THIN, zorder=3)
        for side, col in ((1, f"gained_{crit}"), (-1, f"lost_{crit}")):
            mu = g[col].mean()
            ax.plot([side * mu] * 2, [yc - 0.42, yc + 0.42], color=S.INK, lw=S.LW, zorder=4, solid_capstyle="butt")
        jit[(par, crit)] = yc
    ax.axvline(0, color=S.INK, lw=S.THIN)
    ax.set_xlim(-270, 450)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    ax.set_yticklabels([f"{dname(par)} {'half' if c == 'half' else 'error'}" for par, c in rows])
    ax.set_xticks([-200, -100, 0, 100, 200, 300, 400])
    ax.set_xticklabels(["200", "100", "0", "100", "200", "300", "400"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    for yb in (1.5, 3.5):
        ax.axhline(yb, color=S.GRID, lw=S.THIN)
    xa0_, xw_ = 14.6, 37.6
    P.text(xa0_ + xw_ * (270 - 135) / 720, 7.9, "← lost", ha="center", va="center")
    P.text(xa0_ + xw_ * (270 + 225) / 720, 7.9, "gained →", ha="center", va="center")
    P.legend_row(xl, 4.6, [(lambda x, y: P.mark(x + 0.8, y, marker="D", ms=1.9, mfc=C_LP, mec=C_LP, mew=S.THIN),
                            "allocation 0 (= the ledgers in ii)")], glyph=1.6)
    P.legend_row(xl, 1.9, [(lambda x, y: P.mark(x + 0.8, y, marker="D", ms=1.9, mfc=WHITE, mec=C_LP, mew=S.THIN),
                            "allocations 1–2 · bar = mean of 18")], glyph=1.6)
    # ---------------- bottom (ii): dlam ledgers
    xz = 55.6
    P.text(xz, yz, "(ii) Δλ per target, 18 ledgers", ha="left", va="center", fontweight="bold")
    sub2 = "λ(matched) − λ(classical), allocation 0"
    P.text(xz, yz - 2.6, sub2, ha="left", va="center", color=S.MUTED)
    assert xz + P.tw(sub2) < P.W - 0.5
    DL = [4, 3, 2, 1, 0, -1, -2, -3, -4]       # -4 row pools dlam <= -4
    yt2 = yz - 4.4
    ph = 2.2
    xg0, cwl, gp = 63.0, 1.75, 1.6
    led = led.sort_values(["parent", "init_rep", "opt_seed"])
    order2 = ["T", "CD4T", "CD8T"]
    xcol = {}
    for pi, par in enumerate(order2):
        g = led[led.parent == par]
        for j, (_, q) in enumerate(g.iterrows()):
            xcol[(par, int(q.init_rep), int(q.opt_seed))] = xg0 + pi * (6 * cwl + gp) + j * cwl + cwl / 2
        xa = xg0 + pi * (6 * cwl + gp)
        P.text(xa + 3 * cwl, yt2 - len(DL) * ph - 0.3, dname(par), ha="center", va="top")
        P.text(xa + 3 * cwl, yt2 - len(DL) * ph - 2.8, f"n {nT[par]}", ha="center", va="top", color=S.MUTED)
        assert int(g.n_targets.iloc[0]) == nT[par]
    for i, dv in enumerate(DL):
        yc = yt2 - (i + 0.5) * ph
        P.text(xg0 - 0.8, yc, ("≤ −4" if dv == -4 else (f"+{dv}" if dv > 0 else ("0" if dv == 0 else f"−{-dv}"))),
               ha="right", va="center")
        if dv == 0:
            P.rect(xg0 - 0.3, yc - ph / 2, 3 * (6 * cwl) + 2 * gp + 0.6, ph, fc=S.GRID)
    smax = dl.share.max()
    for (par, rep, opt), xc in xcol.items():
        g = dl[(dl.parent == par) & (dl.init_rep == rep) & (dl.opt_seed == opt)].copy()
        g["bin"] = g.dlam.clip(lower=-4)
        gb = g.groupby("bin").share.sum()
        assert abs(gb.sum() - 1) < 1e-12
        for dv, sh in gb.items():
            i = DL.index(int(dv))
            yc = yt2 - (i + 0.5) * ph
            P.mark(xc, yc, marker="o", ms=e_dot(sh, smax), mfc=C_LP, mec=C_LP, mew=0)
    P.text(xz, yt2 - len(DL) * ph / 2, "Δλ", ha="left", va="center")
    xe = P.legend_row(xz, 4.4, [(None, "share of targets:")] +
                      [((lambda x, y, s=sh: P.mark(x + 0.9, y, marker="o", ms=e_dot(s, smax), mfc=C_LP, mec=C_LP,
                                                   mew=0)), f"{int(sh * 100)}%") for sh in (0.5, 0.1, 0.01)],
                      glyph=1.8, gap=1.4)
    assert xe < P.W + 1.0
    note2 = "ΣΔλ = 0 per ledger (fixed tube budget)"
    P.text(xz, 1.6, note2, ha="left", va="center", color=S.MUTED)
    assert xz + P.tw(note2) < P.W - 0.5
    # source data
    S.save_source(cells, "S2E")
    S.save_source(rec, "S2E", "e5f_records")
    S.save_source(runs, "S2E", "ledger_runs")
    S.save_source(led, "S2E", "ledgers")
    S.save_source(dl, "S2E", "dlam")
    return P.save()


PANELS = {"S2A": fig_S2A, "S2B": fig_S2B, "S2C": fig_S2C, "S2D": fig_S2D, "S2E": fig_S2E}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(PANELS)):
        PANELS[k]()
