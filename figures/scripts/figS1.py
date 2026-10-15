"""Supplementary Figure 1 (suppplan_v2 SF1): data, markers and references.  Five single panels S1A-S1E.

Run from witnessflow/figures/scripts:   python figS1.py [S1A S1B ...]
Inputs (read-only): figures/derived/figS1_*.tsv written by derive/sf1_ledger.py and derive/sf1_tables.py (both assert
every restored number against frozen values).  Outputs: figures/panels/figS1X.pdf/.png (wfstyle.save audit) and
figures/source_data/figS1X.tsv.  No new experiment, simulation, fit, test or randomness.
Source file is UTF-8 (edit with UTF-8-aware tools only; PowerShell 5.1 Get-Content/Set-Content would corrupt it).
"""
import itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import font_manager
from fontTools.ttLib import TTFont

import wfstyle as S

S.setup()
DER = S.DERIVED
_CMAP = TTFont(font_manager.findfont("Arial")).getBestCmap()
MINUS, EN, TIMES, DOT, ARR = "−", "–", "×", "·", "→"
GE, LE, EPS, PI, MU, SIGMA, APPROX, RSAQ, BULLET = "≥", "≤", "ε", "π", "μ", "Σ", "≈", "›", "•"
SPLITS = ["calibration", "initial", "followup", "reference"]
SPLIT_LABEL = {"calibration": "calibration", "initial": "initial", "followup": "follow-up", "reference": "reference"}
SPLITC = {"calibration": S.SEQ(0.12), "initial": S.SEQ(0.40), "followup": S.SEQ(0.86), "reference": S.SEQ(0.60)}
LIGHT = S.TIER[3]          # light grey (#D0D0D0) for 'not used'


def rd(name):
    return pd.read_csv(os.path.join(DER, name), sep="\t")


def fmt(n):
    return f"{int(round(n)):,}"


def dname(s):
    """Display form of parent names in drawn text (source keys CD4T / CD8T unchanged): 'CD4 T', 'CD8 T'."""
    return str(s).replace("CD4T", "CD4 T").replace("CD8T", "CD8 T")


# ---------------------------------------------------------------- layout helpers (mm coordinates)
def canvas(fig):
    W, H = fig.get_size_inches() / S.MM
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_zorder(0)
    return ax, W, H


def axmm(fig, x, y, w, h, **kw):
    W, H = fig.get_size_inches() / S.MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def tx(ax, x, y, s, ha="left", va="center", color=S.INK, **kw):
    return ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=S.FS, **kw)


def rect(ax, x, y, w, h, fc, ec="none", lw=0, z=2, **kw):
    p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    ax.add_patch(p)
    return p


def tw(fig, s, **kw):
    """Rendered width of a 7 pt string in mm."""
    t = fig.text(0, 0, s, fontsize=S.FS, **kw)
    r = fig.canvas.get_renderer()
    w = t.get_window_extent(r).width / fig.dpi * 25.4
    t.remove()
    return w


def header(ov, W, H, title, tier):
    S.tier_tag(ov, tier, x=1 - 0.6 / W, y=1 - 0.6 / H, ha="right", va="top")
    tx(ov, 0.6, H - 0.6, title, va="top", weight="bold")


def check(fig, name):
    """Self-check before saving: 7 pt, Arial glyphs, inside the figure, no overlapping texts, stroke widths."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    items, issues = [], []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip():
            continue
        if abs(t.get_fontsize() - S.FS) > 1e-6:
            issues.append(f"size {t.get_fontsize()} : {s!r}")
        miss = sorted({c for c in s.replace("\\$", "") if ord(c) not in _CMAP and c not in "\n"})
        if miss:
            issues.append(f"glyph missing {miss} : {s!r}")
        bb = t.get_window_extent(r)
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            issues.append(f"outside figure: {s!r}")
        items.append((s, bb))
    sh = 0.25 / 25.4 * fig.dpi
    for (s1, b1), (s2, b2) in itertools.combinations(items, 2):
        if (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > sh) and (min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > sh):
            issues.append(f"overlap: {s1!r} <> {s2!r}")
    allowed = {0.0, S.THIN, S.LW}
    for a in fig.findobj(Line2D):
        if a.get_visible() and round(a.get_linewidth(), 3) not in allowed and a.get_linestyle() not in ("None", ""):
            issues.append(f"line width {a.get_linewidth()}")
    for a in fig.findobj(matplotlib.patches.Patch):
        if a.get_visible() and round(a.get_linewidth(), 3) not in allowed and a.get_edgecolor()[3] > 0:
            issues.append(f"patch width {a.get_linewidth()}")
    print(f"[check {name}] {len(items)} texts, {len(issues)} issues")
    for i in issues:
        print("   ", i)
    return issues


def logfmt(v):
    return {10: "10", 100: "100", 1000: "1k", 10000: "10k", 100000: "100k", 1000000: "1M"}.get(int(v), f"{v:g}")


def legend_row(ov, fig, x, y, items, gap=2.2):
    """items: list of (draw_fn(x, y) -> width_mm, label).  Draws symbol + label left to right."""
    for draw, lab in items:
        w = draw(x, y)
        tx(ov, x + w + 0.6, y, lab, va="center")
        x += w + 0.6 + tw(fig, lab) + gap
    return x


# ================================================================ S1A cell ledger
def panel_S1A(H=52, W=184):
    F = rd("figS1_files.tsv")
    SP = rd("figS1_split.tsv")
    PA = rd("figS1_parents.tsv")
    DP = rd("figS1_donor_parent.tsv")
    RB = rd("figS1_rareB.tsv")
    # frozen-value guards (asserted in derive/sf1_ledger.py; repeated so the figure cannot drift)
    pre = F[(F.dataset == "wingmen_v1") & (F.group == "Pre")]
    post = F[(F.dataset == "wingmen_v1") & (F.group == "Post")]
    assert len(pre) == 27 and pre.used.sum() == 13 and len(post) == 27
    assert int(pre[pre.file == "PreWN08.fcs"].tot.iloc[0]) == 28060
    L = SP[SP.dataset == "levine32_v2"]
    Wd = SP[SP.dataset == "wingmen_v1"].sort_values("n_followup", ascending=False)
    assert L.n_followup.sum() == 53125 and Wd.n_followup.sum() == 159264 and L.cells_qc.sum() == 265627
    assert L.n_initial.sum() == 106250 and Wd.n_initial.sum() == 318526
    d6 = DP[(DP.parent == "CD8T") & (DP.donor == 6)].iloc[0]
    assert (d6.n_parent_followup, d6.n_parent_reference, bool(d6.included)) == (410, 384, False)
    c4 = DP[DP.parent == "CD4T"]
    assert c4.included.all() and (c4.n_parent_followup.min(), c4.n_parent_reference.min()) == (2064, 1978)
    assert max(c4.n_parent_followup.max(), c4.n_parent_reference.max()) == 13309
    assert RB.n_initial_kept.tolist() == [842, 251, 83]
    for _, r in PA.iterrows():   # pi = parent / ALL initial cells (not / sum of parents)
        tot = 106250 if r.dataset == "levine32_v2" else 318526
        assert r.n_initial_all_cells == tot and abs(r.pi_initial - r.n_initial / tot) < 1e-12

    fig = S.new("S1A", w=W, h=H, layout=None)     # 184 mm = full composite content width (FigS1 page)
    ov, W, H = canvas(fig)
    header(ov, W, H, f"Cell ledger: source files {ARR} samples {ARR} four-way split {ARR} parent populations", "Record")
    # blocks were laid out for W = 180; the extra width goes evenly into the three gaps between the four blocks
    ex = (W - 180.0) / 4.0
    dB, dD, dC = 1.4 * ex, 2.7 * ex, 4.0 * ex     # x shift of: samples block, per-donor block, parent block
    pitch = 2.22
    yh1, yh2 = H - 5.0, H - 7.4         # column-header lines
    ytop = H - 8.9                      # top edge of slot 0
    yc = lambda k: ytop - (k + 0.5) * pitch  # noqa: E731
    yb = ytop - 17 * pitch              # bottom of the slot grid
    assert yb >= 3.9, yb
    yax = yb + 0.2

    def xaxis(x0, x1, f, ticks, minor=()):
        ov.plot([x0, x1], [yax, yax], color=S.INK, lw=S.LW, solid_capstyle="butt")
        for v, s in ticks:
            ov.plot([f(v)] * 2, [yax, yax - 0.9], color=S.INK, lw=S.LW)
            tx(ov, f(v), yax - 1.3, s, ha="center", va="top")
        for v in minor:
            ov.plot([f(v)] * 2, [yax, yax - 0.55], color=S.INK, lw=S.THIN)

    # ---------------- A: WINGMEN source files (events per FCS, $TOT)
    xa0, xa1 = 7.1, 30.0
    ya0 = yb + 3.6
    axA = axmm(fig, xa0, ya0, xa1 - xa0, ytop - ya0)
    pre_s = pre.sort_values("tot", ascending=False).reset_index(drop=True)
    post_s = post.sort_values("tot", ascending=False).reset_index(drop=True)
    lo = 60
    axA.bar(np.arange(27), pre_s.tot - lo, bottom=lo, width=0.72, color=[S.INK if u else S.MUTED for u in pre_s.used], lw=0)
    axA.bar(np.arange(27) + 29.5, post_s.tot - lo, bottom=lo, width=0.72, color=LIGHT, lw=0)
    axA.set_yscale("log")
    axA.set_ylim(lo, 9e5)
    axA.set_xlim(-1, 57.5)
    axA.axhline(30000, xmin=0, xmax=(27.2 + 1) / 58.5, color=S.INK, lw=S.LW, ls=(0, (2.5, 1.5)), zorder=3)
    axA.set_yticks([100, 1000, 10000, 30000, 100000])
    axA.set_yticklabels(["100", "1k", "10k", "30k", "100k"])
    axA.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axA.set_xticks([])
    axA.spines["bottom"].set_visible(False)
    i08 = int(pre_s.index[pre_s.file == "PreWN08.fcs"][0])
    assert i08 == 13
    axA.annotate("PreWN08 28,060", xy=(i08, 28060), xytext=(24.6, 4.2e5), fontsize=S.FS, color=S.INK,
                 ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=S.THIN, color=S.INK, shrinkA=1.0, shrinkB=0.3))
    tx(ov, 0.6, yh1, "Source files (WINGMEN)", va="center")
    tx(ov, 0.6, yh2, "events per FCS file (\\$TOT)", va="center", color=S.MUTED)
    xpre_mm = xa0 + (xa1 - xa0) * (13 + 1) / 58.5
    xpost_mm = xa0 + (xa1 - xa0) * (29.5 + 13 + 1) / 58.5
    tx(ov, xpre_mm, ya0 - 1.0, f"Pre {TIMES} 27", ha="center", va="top")
    tx(ov, xpre_mm, ya0 - 3.5, "13 used", ha="center", va="top")
    tx(ov, xpost_mm, ya0 - 1.0, f"Post {TIMES} 27", ha="center", va="top")
    tx(ov, xpost_mm, ya0 - 3.5, "not used", ha="center", va="top", color=S.MUTED)

    # ---------------- B: samples, four-way split (cells after QC; linear)
    xh_b, xl, xb0, xb1, xn = (v + dB for v in (32.3, 43.8, 44.8, 82.8, 93.0))
    XMAX = 200000
    sx = lambda v: xb0 + (xb1 - xb0) * v / XMAX  # noqa: E731
    tx(ov, xh_b, yh1, "Cells per sample (20 / 40 / 20 / 20 %)", va="center")
    tx(ov, xn, yh1, "follow-up", ha="right", va="center")
    tx(ov, xn, yh2, "N(d)", ha="right", va="center", color=S.MUTED)
    rows = [("hdr", f"Levine-32 {DOT} 265,627 cells", f"{SIGMA} " + fmt(L.n_followup.sum()))]
    rows += [("row", r) for _, r in L.iterrows()]
    rows += [("hdr", f"WINGMEN {DOT} " + fmt(Wd.cells_qc.sum()) + " cells", f"{SIGMA} " + fmt(Wd.n_followup.sum()))]
    rows += [("row", r) for _, r in Wd.iterrows()]
    slot_of = {}
    for k, item in enumerate(rows):
        y = yc(k)
        if item[0] == "hdr":
            tx(ov, xh_b, y, item[1], va="center", weight="bold")
            tx(ov, xn, y, item[2], ha="right", va="center", weight="bold")
            continue
        r = item[1]
        lab = r["sample"].replace("individual", "Individual") if r.dataset == "levine32_v2" else r.file[:-4]
        slot_of[lab] = k
        tx(ov, xl, y, lab, ha="right", va="center")
        x = 0
        for sname in SPLITS:
            n = r[f"n_{sname}"]
            rect(ov, sx(x), y - 0.31 * pitch, sx(x + n) - sx(x), 0.62 * pitch, fc=SPLITC[sname], z=3)
            x += n
        assert x == r.cells_qc
        tx(ov, xn, y, fmt(r.n_followup), ha="right", va="center")
    xaxis(xb0, xb1, sx, [(v, "0" if v == 0 else f"{v // 1000}k") for v in (0, 50000, 100000, 150000, 200000)])
    # split legend (bar swatch + parent dot) and donor-imbalance note in the empty area right of the short bars
    top2 = Wd.n_followup.nlargest(2)
    share = top2.sum() / Wd.n_followup.sum()
    assert round(100 * share) == 37
    xg = 61.0 + dB
    tx(ov, xg, yc(6), "top two donors:", va="center", color=S.MUTED)
    tx(ov, xg, yc(7), f"{100 * share:.0f}% of total N(d)", va="center", color=S.MUTED)
    for j, sname in enumerate(SPLITS):      # same order as the vertical dodge of the parent dots
        y = yc(9 + j)
        rect(ov, xg, y - 0.72, 2.2, 1.44, fc=SPLITC[sname], z=3)
        ov.plot([xg + 3.4], [y], marker="o", ms=2.3, mfc=SPLITC[sname], mec=S.INK, mew=S.THIN, ls="none", zorder=6)
        tx(ov, xg + 4.6, y, SPLIT_LABEL[sname], va="center")
    tx(ov, xg, yc(13), "bar | dot (parents)", va="center", color=S.MUTED)

    # ---------------- per-donor CD8T and CD4T (e5b rule: >= 500 parent cells in follow-up AND reference)
    mlo, mhi = 250, 20000
    sub = {"CD8T": (95.8 + dD, 107.6 + dD), "CD4T": (109.6 + dD, 121.4 + dD)}
    tx(ov, 95.3 + dD, yh1, f"Per donor ({GE} 500)", va="center")
    xmid = (95.8 + 121.4) / 2 + dD
    tx(ov, xmid, (yc(1) + yc(2)) / 2 + 1.15, "no per-individual", ha="center", va="center", color=S.MUTED)
    tx(ov, xmid, (yc(1) + yc(2)) / 2 - 1.15, "replay", ha="center", va="center", color=S.MUTED)
    ov.plot([96.3 + dD], [yc(3)], marker="o", ms=1.9, mfc=S.INK, mec=S.INK, mew=0, ls="none")
    tx(ov, 97.1 + dD, yc(3), "follow-up", va="center")
    ov.plot([110.6 + dD], [yc(3)], marker="o", ms=2.6, mfc="white", mec=S.INK, mew=S.THIN, ls="none")
    tx(ov, 111.4 + dD, yc(3), "reference", va="center")
    for G, (m0, m1) in sub.items():
        mx = lambda v, m0=m0, m1=m1: m0 + (m1 - m0) * np.log10(v / mlo) / np.log10(mhi / mlo)  # noqa: E731
        tx(ov, (m0 + m1) / 2, yh2, dname(G), ha="center", va="center", color=S.MUTED)
        ov.plot([mx(500)] * 2, [yb + 0.2, ytop - 4 * pitch], color=S.INK, lw=S.LW, ls=(0, (2.5, 1.5)), zorder=4)
        DG = DP[DP.parent == G].set_index("file")
        for lab, k in slot_of.items():
            if not lab.startswith("PreWN"):
                continue
            r = DG.loc[lab + ".fcs"]
            y = yc(k)
            assert max(r.n_parent_followup, r.n_parent_reference) < mhi and min(r.n_parent_followup, r.n_parent_reference) > mlo
            ov.plot([mx(r.n_parent_reference)], [y], marker="o", ms=2.6, mfc="white", mec=S.INK, mew=S.THIN, ls="none", zorder=6)
            ov.plot([mx(r.n_parent_followup)], [y], marker="o", ms=1.9, mfc=S.INK, mec=S.INK, mew=0, ls="none", zorder=7)
            if not r.included:
                rect(ov, m0, y - pitch / 2 + 0.15, mx(500) - m0, pitch - 0.3, fc=S.GRID, ec=S.MUTED, lw=0, z=2,
                     hatch="//////")
                tx(ov, mx(500) + 0.6, y, f"{r.n_parent_followup}/{r.n_parent_reference}", va="center")
        xaxis(m0, m1, mx, [(500, "500"), (5000, "5k")], minor=(1000, 2000, 10000))

    # ---------------- C: parent populations, cells per split (log), independent dot rows; pi of the main rows
    xh_c, xc_l, xc0, xc1, xc_n, xc_p = (v + dC for v in (123.4, 131.4, 132.4, 156.4, 168.2, 179.4))
    clo, chi = 30, 200000
    cx = lambda v: xc0 + (xc1 - xc0) * np.log10(v / clo) / np.log10(chi / clo)  # noqa: E731
    tx(ov, xh_c, yh1, "Parent populations (gates overlap)", va="center")
    tx(ov, xh_c, yh2, "cells per split, log", va="center", color=S.MUTED)
    tx(ov, xc_n, yh2, "initial", ha="right", va="center", color=S.MUTED)
    tx(ov, xc_p, yh2, PI, ha="right", va="center", color=S.MUTED)
    crow = [("hdr", "Levine-32")]
    PL = PA[PA.dataset == "levine32_v2"].set_index("parent")
    PW = PA[PA.dataset == "wingmen_v1"].set_index("parent")
    for G in ["HSPC", "T", "B"]:
        crow.append(("par", G, PL.loc[G]))
    for _, r in RB.iterrows():
        crow.append(("rare", r))
    for G in ["Mono", "NK"]:
        crow.append(("par", G, PL.loc[G]))
    crow.append(("hdr", "WINGMEN"))
    for G in ["CD4T", "CD8T", "B", "NK", "Mono"]:
        crow.append(("par", G, PW.loc[G]))
    assert len(crow) <= 15
    dodge = {"calibration": 0.63, "initial": 0.21, "followup": -0.21, "reference": -0.63}
    yB = xB = None
    for k, item in enumerate(crow):
        y = yc(k)
        if item[0] == "hdr":
            tx(ov, xh_c, y, item[1], va="center", weight="bold")
            continue
        if item[0] == "par":
            G, r = item[1], item[2]
            tx(ov, xc_l, y, dname(G), ha="right", va="center")
            for sname in SPLITS:
                ov.plot([cx(r[f"n_{sname}"])], [y + dodge[sname]], marker="o", ms=2.3, mfc=SPLITC[sname], mec=S.INK,
                        mew=S.THIN, ls="none", zorder=6)
            tx(ov, xc_n, y, fmt(r.n_initial), ha="right", va="center")
            tx(ov, xc_p, y, f"{100 * r.pi_initial:.2f}%", ha="right", va="center")
            if G == "B" and r.dataset == "levine32_v2":
                yB, xB = y, cx(r.n_initial)
        else:
            r = item[1]
            pct = {0.01: "1%", 0.003: "0.3%", 0.001: "0.1%"}[round(r.target_frac, 3)]
            tx(ov, xc_l, y, "B " + pct, ha="right", va="center", color=S.MUTED)
            xi = cx(r.n_initial_kept)
            ov.plot([xB, xi], [yB + dodge["initial"], y + dodge["initial"]], color=S.MUTED, lw=S.LW,
                    ls=(0, (2.0, 1.4)), zorder=4)
            ov.plot([xi], [y + dodge["initial"]], marker="o", ms=2.3, mfc=SPLITC["initial"], mec=S.INK, mew=S.THIN,
                    ls="none", zorder=6)
            ov.plot([cx(r.n_followup_kept)], [y + dodge["followup"]], marker="o", ms=2.3, mfc=SPLITC["followup"],
                    mec=S.INK, mew=S.THIN, ls="none", zorder=6)
            tx(ov, xc_n, y, fmt(r.n_initial_kept), ha="right", va="center")
    tx(ov, xh_c, yc(15), "rareB: B thinned in initial and follow-up only", va="center", color=S.MUTED)
    xaxis(xc0, xc1, cx, [(v, logfmt(v)) for v in (100, 1000, 10000, 100000)])

    check(fig, "S1A")
    S.save(fig, "S1A")
    # ---------------- Source Data
    src = []
    for _, r in F[F.dataset == "wingmen_v1"].iterrows():
        src.append(dict(block="A_files", dataset=r.dataset, item=r.file, group=r.group, value=r.tot, unit="events ($TOT)",
                        note=r.reason))
    for _, r in SP.iterrows():
        for sname in SPLITS:
            src.append(dict(block="B_split", dataset=r.dataset, item=r["sample"] + " " + r.file, group=sname,
                            value=r[f"n_{sname}"], unit="cells after QC", note=f"events_raw={r.events_raw}; cells_qc={r.cells_qc}"))
    for _, r in DP.iterrows():
        for col, g in (("n_parent_followup", "followup"), ("n_parent_reference", "reference")):
            src.append(dict(block=f"per_donor_{r.parent}", dataset=r.dataset, item=f"donor {r.donor} {r.file}", group=g,
                            value=r[col], unit=f"{r.parent} cells", note=f"included={r.included}; {r.rule}"))
    for _, r in PA.iterrows():
        for sname in SPLITS:
            src.append(dict(block="C_parents", dataset=r.dataset, item=r.parent, group=sname, value=r[f"n_{sname}"],
                            unit="parent cells", note=f"gate {r.gate}"))
        src.append(dict(block="C_pi", dataset=r.dataset, item=r.parent, group="initial", value=r.pi_initial,
                        unit="fraction of all initial cells",
                        note=f"n_initial={r.n_initial} / all initial cells {r.n_initial_all_cells}"))
    for _, r in RB.iterrows():
        for g, col in (("initial", "n_initial_kept"), ("followup", "n_followup_kept")):
            src.append(dict(block="C_rareB", dataset="levine32_v2", item=r.protocol, group=g, value=r[col],
                            unit="B cells after thinning",
                            note=f"target_frac={r.target_frac}; seed={r.seed}; reference untouched {r.n_reference}"))
    S.save_source(pd.DataFrame(src), "S1A")


# ================================================================ S1B marker atlas
def panel_S1B(H=143.5, W=112):
    M = rd("figS1_markers.tsv")
    for pid, n in (("levine32_v2", 32), ("wingmen_v1", 37)):
        assert (M.dataset == pid).sum() == n
    assert sorted(M[~M.reliable].marker) == sorted(["CD235ab", "Flt3", "CD41", "CD61", "CTLA4", "CD123", "TIGIT"])
    assert ((M.dataset == "levine32_v2") & (M.ashman_D < 2)).sum() == 16
    assert ((M.dataset == "wingmen_v1") & (M.ashman_D < 2)).sum() == 13
    # composite: 112 + 2 + 70 = 184 mm; H = height of the S1C / S1D stack (70 + 69.5 + 3 + 1 mm)
    fig = S.new("S1B", w=W, h=H, layout=None)
    ov, W, H = canvas(fig)
    header(ov, W, H, "Marker atlas: cut, reliability and use of every marker", "Record")
    nameW = {pid: max(tw(fig, m) for m in M[M.dataset == pid].marker) + 0.3 for pid in ("levine32_v2", "wingmen_v1")}
    gapB, margin = 2.2, 0.7
    rest = (W - 2 * margin - gapB - sum(nameW.values())) / 2
    cw = dict(ann=7.2, D=7.6, w=4.3, cut=None, frac=4.0, claims=5.0)
    g = 0.40
    cw["cut"] = rest - (cw["ann"] + cw["D"] + cw["w"] + cw["frac"] + cw["claims"] + 6 * g)
    assert cw["cut"] > 10, cw
    y_hdr = H - 22.0           # bottom of the column-header zone = top of row 0
    pitch = (y_hdr - 17.2) / 37  # row pitch: the 37 WINGMEN rows end 17.2 mm above the bottom (2.40 mm at H = 128)
    yrow = lambda i: y_hdr - (i + 0.5) * pitch  # noqa: E731
    DLO, DHI, WHI, CUTHI = 1.2, 3.7, 0.8, 3.6
    src = []
    x0 = margin
    blocks = {}
    for pid, title, sub in (("levine32_v2", f"Levine-32 {DOT} 32 markers", "v2 rule, revised after unblinding"),
                            ("wingmen_v1", f"WINGMEN {DOT} 37 markers", "rule adopted before threshold fitting")):
        T = M[M.dataset == pid].sort_values("ashman_D", ascending=False).reset_index(drop=True)
        xn = x0 + nameW[pid]
        c = {}
        xx = xn + g
        for k in ("ann", "D", "w", "cut", "frac", "claims"):
            c[k] = (xx, xx + cw[k])
            xx += cw[k] + g
        blocks[pid] = (x0, xx - g)
        ann_x = [c["ann"][0] + cw["ann"] * f for f in (1 / 6, 3 / 6, 5 / 6)]
        dx = lambda v: c["D"][0] + cw["D"] * (min(max(v, DLO), DHI) - DLO) / (DHI - DLO)  # noqa: E731
        wx = lambda v: c["w"][0] + cw["w"] * v / WHI  # noqa: E731
        cx = lambda v: c["cut"][0] + cw["cut"] * min(max(v, 0), CUTHI) / CUTHI  # noqa: E731
        fx = lambda v: c["frac"][0] + cw["frac"] * v  # noqa: E731
        n = len(T)
        ybot = y_hdr - n * pitch
        tx(ov, x0, H - 6.0, title, va="center", weight="bold")
        tx(ov, x0, H - 8.6, sub, va="center", color=S.MUTED)
        for xh, lab in ((ann_x[0], "backbone"), (ann_x[1], "reliable"), ((c["D"][0] + c["D"][1]) / 2, "Ashman D"),
                        ((c["w"][0] + c["w"][1]) / 2, "w(hi)"), ((c["frac"][0] + c["frac"][1]) / 2, "zero, pos"),
                        ((c["claims"][0] + c["claims"][1]) / 2, "claims")):
            tx(ov, xh, y_hdr + 0.6, lab, ha="center", va="bottom", rotation=90)
        S.tstar_label(ov, ann_x[2] - 1.15, y_hdr + 1.9, text="T", ms=4.5)
        tx(ov, (c["cut"][0] + c["cut"][1]) / 2, y_hdr + 4.6, "cut | and", ha="center", va="center")
        tx(ov, (c["cut"][0] + c["cut"][1]) / 2, y_hdr + 2.0, f"{MU}(lo), {MU}(hi)", ha="center", va="center")
        for i in range(0, n, 2):
            rect(ov, x0 - 0.3, yrow(i) - pitch / 2, (xx - g) - x0 + 0.6, pitch, fc=S.SEQ(0.0), z=0)
        ov.plot([dx(1.5)] * 2, [ybot, y_hdr], color=S.INK, lw=S.LW, zorder=3)
        ov.plot([dx(2.0)] * 2, [ybot, y_hdr], color=S.MUTED, lw=S.LW, ls=(0, (2.0, 1.4)), zorder=3)
        ov.plot([wx(0.03)] * 2, [ybot, y_hdr], color=S.INK, lw=S.LW, zorder=3)
        for i, r in T.iterrows():
            y = yrow(i)
            tx(ov, xn, y, r.marker, ha="right", va="center")
            if r.backbone:
                ov.plot([ann_x[0]], [y], marker="s", ms=3.4, mfc=S.BACKBONE, mec=S.INK, mew=S.THIN, ls="none", zorder=5)
            if r.reliable:
                ov.plot([ann_x[1]], [y], marker="o", ms=1.7, mfc=S.INK, mec=S.INK, mew=0, ls="none", zorder=5)
            else:
                ov.plot([ann_x[1]], [y], marker="x", ms=3.0, mfc="none", mec=S.ALARM, mew=S.LW, ls="none", zorder=5)
            if r.tstar:
                S.tstar_marker(ov, ann_x[2], y, ms=4.5)
            col = S.ALARM if r.D_fail else S.INK
            if r.ashman_D > DHI:
                ov.plot([dx(DHI)], [y], marker=">", ms=2.6, mfc=S.INK, mec=S.INK, mew=0, ls="none", zorder=6)
                tx(ov, dx(DHI) - 1.0, y, f"{r.ashman_D:.1f}", ha="right", va="center", color=S.MUTED)
            else:
                ov.plot([dx(r.ashman_D)], [y], marker="o", ms=2.3, mfc=col, mec=col, mew=0, ls="none", zorder=6)
            col = S.ALARM if r.w_fail else S.INK
            ov.plot([wx(r.w_hi)], [y], marker="o", ms=2.3, mfc=col, mec=col, mew=0, ls="none", zorder=6)
            if r.mu_lo > CUTHI:
                tx(ov, c["cut"][1], y, f"{r.thr:.2f} {RSAQ}", ha="right", va="center", color=S.MUTED)
            else:
                ov.plot([cx(r.mu_lo), cx(r.mu_hi)], [y, y], color=LIGHT, lw=S.LW, zorder=4, solid_capstyle="butt")
                ov.plot([cx(r.mu_lo)], [y], marker="o", ms=1.8, mfc=S.MUTED, mec=S.MUTED, mew=0, ls="none", zorder=5)
                if r.mu_hi <= CUTHI:
                    ov.plot([cx(r.mu_hi)], [y], marker="o", ms=1.8, mfc=S.MUTED, mec=S.MUTED, mew=0, ls="none", zorder=5)
                else:
                    ov.plot([cx(CUTHI)], [y], marker=">", ms=2.2, mfc=S.MUTED, mec=S.MUTED, mew=0, ls="none", zorder=5)
                ov.plot([cx(r.thr)], [y], marker="|", ms=3.6, mec=S.INK, mew=S.LW, ls="none", zorder=7)
                if pid == "levine32_v2":
                    yv = y + 0.62
                    x1, x2 = cx(r.thr_v1), cx(r.thr)
                    ov.plot([x1], [yv], marker="o", ms=1.7, mfc="white", mec=S.MUTED, mew=S.THIN, ls="none", zorder=8)
                    if abs(x2 - x1) > 0.7:
                        ov.add_patch(FancyArrowPatch((x1 + 0.3 * np.sign(x2 - x1), yv), (x2, yv), arrowstyle="-|>",
                                                     mutation_scale=3.2, lw=S.LW, color=S.MUTED, zorder=8,
                                                     shrinkA=0, shrinkB=0))
            yh = 0.55 * pitch
            rect(ov, fx(0), y - yh / 2, fx(1) - fx(0), yh, fc=S.GRID, z=3)
            rect(ov, fx(0), y - yh / 2, fx(r.zero_frac) - fx(0), yh, fc=S.MUTED, z=4)
            rect(ov, fx(1 - r.pos_frac), y - yh / 2, fx(1) - fx(1 - r.pos_frac), yh, fc=S.SEQ(0.72), z=4)
            cl = [s for s in str(r.claims).split() if s and s != "nan"]
            flagged = [s for s in str(r.claims_unreliable_marker).split() if s and s != "nan"]
            if flagged:
                tx(ov, c["claims"][0] + 0.45, y, flagged[0], ha="left", va="center", color=S.INK,
                   bbox=dict(boxstyle="square,pad=0.12", fc="white", ec=S.ALARM, lw=S.LW))
            else:
                for j in range(len(cl)):
                    ov.plot([c["claims"][0] + 0.4 + 0.6 * j], [y], marker="o", ms=1.4, mfc=S.INK, mec=S.INK,
                            mew=0, ls="none", zorder=5)
            src.append(dict(dataset=pid, row=i + 1, marker=r.marker, backbone=r.backbone, reliable=r.reliable,
                            candidate=r.candidate, tstar=r.tstar, ashman_D=r.ashman_D, D_min=r.D_min, w_hi=r.w_hi,
                            w_min=r.w_min, thr_v2=r.thr, mu_lo=r.mu_lo, mu_hi=r.mu_hi, thr_v1=r.thr_v1,
                            zero_frac=r.zero_frac, pos_frac=r.pos_frac, claims=" ".join(cl),
                            claims_with_unreliable_marker=" ".join(flagged),
                            unreliable_initial_audit_records=r.unreliable_audit_records if flagged else ""))
        ya = ybot - 0.3
        for (a, b), ticks in ((c["D"], [(1.5, "1.5"), (3.0, "3")]), (c["w"], [(0, ""), (0.5, "0.5")]),
                              (c["cut"], [(0, ""), (1, "1"), (2, "2"), (3, "3")]), (c["frac"], [(0, "0"), (1, "1")])):
            ov.plot([a, b], [ya, ya], color=S.INK, lw=S.LW, solid_capstyle="butt")
            f = {c["D"]: dx, c["w"]: wx, c["cut"]: cx, c["frac"]: fx}[(a, b)]
            for v, s in ticks:
                ov.plot([f(v)] * 2, [ya, ya - 0.8], color=S.INK, lw=S.LW)
                if s:
                    tx(ov, f(v), ya - 1.1, s, ha="center", va="top")
        x0 = xx - g + gapB
    xL0, xL1 = blocks["levine32_v2"]
    yL = y_hdr - 32 * pitch - 6.2
    y1 = 1.5

    def sq(x, y):
        ov.plot([x + 0.9], [y], marker="s", ms=3.4, mfc=S.BACKBONE, mec=S.INK, mew=S.THIN, ls="none")
        return 1.8

    def dot(x, y):
        ov.plot([x + 0.6], [y], marker="o", ms=1.7, mfc=S.INK, mec=S.INK, mew=0, ls="none")
        return 1.2

    def cross(x, y):
        ov.plot([x + 0.9], [y], marker="x", ms=3.0, mec=S.ALARM, mew=S.LW, ls="none")
        return 1.8

    def star(x, y):
        S.tstar_marker(ov, x + 1.0, y, ms=4.5)
        return 2.0

    def cutglyph(x, y):
        ov.plot([x, x + 4.0], [y, y], color=LIGHT, lw=S.LW, solid_capstyle="butt")
        ov.plot([x, x + 4.0], [y, y], marker="o", ms=1.8, mfc=S.MUTED, mec=S.MUTED, mew=0, ls="none")
        ov.plot([x + 1.6], [y], marker="|", ms=3.6, mec=S.INK, mew=S.LW, ls="none")
        return 4.0

    def v1glyph(x, y):
        ov.plot([x + 0.4], [y], marker="o", ms=1.7, mfc="white", mec=S.MUTED, mew=S.THIN, ls="none")
        ov.add_patch(FancyArrowPatch((x + 0.7, y), (x + 3.4, y), arrowstyle="-|>", mutation_scale=3.2, lw=S.LW,
                                     color=S.MUTED, shrinkA=0, shrinkB=0))
        return 3.4

    def fracglyph(x, y):
        rect(ov, x, y - 0.7, 1.2, 1.4, fc=S.MUTED, z=4)
        rect(ov, x + 1.2, y - 0.7, 1.6, 1.4, fc=S.GRID, z=4)
        rect(ov, x + 2.8, y - 0.7, 1.4, 1.4, fc=S.SEQ(0.72), z=4)
        return 4.2

    def pip(x, y):
        for j in range(2):
            ov.plot([x + 0.4 + 0.6 * j], [y], marker="o", ms=1.4, mfc=S.INK, mec=S.INK, mew=0, ls="none")
        return 1.2

    def redbox(x, y):
        tx(ov, x + 0.3, y, "H3", va="center", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec=S.ALARM, lw=S.LW))
        return tw(fig, "H3") + 0.6

    def solid(x, y):
        ov.plot([x + 0.5, x + 0.5], [y - 1.0, y + 1.0], color=S.INK, lw=S.LW)
        return 1.0

    def dashed(x, y):
        ov.plot([x + 0.5, x + 0.5], [y - 1.0, y + 1.0], color=S.MUTED, lw=S.LW, ls=(0, (2.0, 1.4)))
        return 1.0

    # two legend groups under the Levine block (reliability | rule revision), separated by a clear gap;
    # generic encodings in two full-width lines at the bottom
    lp, ggap = 2.55, 4.1
    yg1 = yL + 0.6
    tx(ov, xL0, yg1, "Reliability", va="center", weight="bold")
    legend_row(ov, fig, xL0, yg1 - lp, [(solid, f"rule: D {GE} 1.5 and w(hi) {GE} 0.03")])
    legend_row(ov, fig, xL0, yg1 - 2 * lp, [(cross, "unreliable: fails the rule (values in red)")])
    legend_row(ov, fig, xL0, yg1 - 3 * lp, [(redbox, "claim uses an unreliable marker; 10 records")])
    yg2 = yg1 - 3 * lp - ggap
    tx(ov, xL0, yg2, "Rule revision (Levine-32)", va="center", weight="bold")
    legend_row(ov, fig, xL0, yg2 - lp, [(v1glyph, f"v1 cut {ARR} v2 cut, after unblinding")])
    legend_row(ov, fig, xL0, yg2 - 2 * lp, [(dashed, "D = 2 reference: D < 2 for 16/32 and 13/37")])
    legend_row(ov, fig, margin, y1 + lp, [(sq, "backbone"), (dot, "reliable"), (star, "CCR7/CD27/CD45RA"),
                                          (cutglyph, "cut, means (asinh)")])
    legend_row(ov, fig, margin, y1, [(fracglyph, f"zero spike ({LE} 0.05) | other | above cut"),
                                     (pip, "one dot per claim using the marker")])
    assert yg2 - 2 * lp - 1.3 > y1 + lp + 1.3, (yg2, y1)
    check(fig, "S1B")
    S.save(fig, "S1B")
    S.save_source(pd.DataFrame(src), "S1B")


# ================================================================ S1C threshold offsets
def spread(ys, mind):
    """Order-preserving 1-D label spreading (deterministic): minimum distance `mind`."""
    idx = np.argsort(ys)
    out = np.array(ys, float)
    s = out[idx]
    for _ in range(200):
        moved = False
        for k in range(1, len(s)):
            if s[k] - s[k - 1] < mind - 1e-9:
                d = (mind - (s[k] - s[k - 1])) / 2
                s[k - 1] -= d
                s[k] += d
                moved = True
        if not moved:
            break
    out[idx] = s
    return out


def panel_S1C(H=70):
    PI_ = rd("figS1_offset_pi.tsv")
    AU = rd("figS1_offset_audit.tsv")
    assert (AU.dataset == "levine32_v2").sum() == 55 and (AU.dataset == "wingmen_v1").sum() == 70 and AU.ref_in.all()
    offs = sorted(AU.offset.unique())
    assert offs == [-0.25, -0.125, 0.0, 0.125, 0.25]
    mono = PI_[(PI_.dataset == "levine32_v2") & (PI_.parent == "Mono")].sort_values("offset").pi.values
    assert round(100 * mono[0], 2) == 10.35 and round(100 * mono[-1], 2) == 1.96
    fig = S.new("S1C", h=H, layout=None)
    ov, W, H = canvas(fig)
    header(ov, W, H, "Uniform threshold offsets", "Replay")
    tx(ov, 0.6, H - 8.7, f"Gating sensitivity: Levine-32 Mono {100 * mono[0]:.2f}% {ARR} {100 * mono[-1]:.2f}%",
       va="center", color=S.MUTED)
    cell_w, pitch = 2.9, 2.3
    ytop_pi, ybot_pi = H - 11.0, H - 24.8
    ytop_m = H - 28.0
    yr = lambda i: ytop_m - (i + 0.5) * pitch  # noqa: E731
    PLO, PHI = 1.5, 55
    order = {"levine32_v2": ["HSPC", "T", "B", "Mono", "NK"], "wingmen_v1": ["CD4T", "CD8T", "B", "NK", "Mono"]}
    claim_order = {"levine32_v2": ["H1", "H2", "T1", "T2", "T3", "B1", "B2", "M1", "M2", "N1", "N2"],
                   "wingmen_v1": ["C1", "C2", "C3", "C4", "C5", "D1", "D3", "D4", "E1", "E2", "E3", "F1", "G1", "G2"]}
    WMAX = 0.5
    src = []
    xm = {"levine32_v2": 7.5, "wingmen_v1": 34.0}
    for pid, name in (("levine32_v2", "Levine-32"), ("wingmen_v1", "WINGMEN")):
        x0 = xm[pid]
        x1 = x0 + 5 * cell_w
        colx = lambda j: x0 + (j + 0.5) * cell_w  # noqa: E731
        ax = axmm(fig, x0, ybot_pi, x1 - x0, ytop_pi - ybot_pi)
        ax.set_xlim(-0.5, 4.5)
        ax.set_yscale("log")
        ax.set_ylim(PLO, PHI)
        ax.set_xticks(range(5))
        ax.tick_params(labelbottom=False)
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if pid == "levine32_v2":
            ax.set_yticklabels(["2", "5", "10", "20", "50"])
            ax.set_ylabel(f"{PI} (%)", labelpad=0.3)
        else:
            ax.set_yticklabels([])
        rep = PI_[(PI_.dataset == pid)].design_replay_parent.iloc[0]
        ends = []
        for G in order[pid]:
            s = PI_[(PI_.dataset == pid) & (PI_.parent == G)].sort_values("offset")
            assert s.offset.tolist() == offs
            emph = G == rep            # black = the parent whose design was replayed (T / CD4T)
            colr = S.INK if emph else S.MUTED
            ax.plot(range(5), 100 * s.pi, color=colr, lw=S.LW, marker="o", ms=1.6, mfc=colr, mew=0, zorder=4 if emph else 3)
            ends.append((G, 100 * s.pi.iloc[-1], colr, G == rep))
            for _, r in s.iterrows():
                src.append(dict(block="top_pi", dataset=pid, parent=G, claim="", offset=r.offset, value=r.pi,
                                unit="parent fraction (initial split)", design_replayed=G == rep, ref_in=""))

        def ymm(v):
            return ybot_pi + (ytop_pi - ybot_pi) * np.log10(v / PLO) / np.log10(PHI / PLO)
        yl = spread([ymm(e[1]) for e in ends], 2.6)
        for (G, v, colr, isrep), yy in zip(ends, yl):
            tx(ov, x1 + 1.2, yy, dname(G), va="center", color=colr)
            if abs(yy - ymm(v)) > 0.7:
                ov.plot([x1 + 0.2, x1 + 1.0], [ymm(v), yy], color=colr, lw=S.THIN)
        tx(ov, x0 - 3.2 if pid == "levine32_v2" else x0 - 3.2, H - 5.9, name, va="center", weight="bold")
        cl = claim_order[pid]
        A = AU[AU.dataset == pid]
        assert sorted(cl) == sorted(A.claim.unique())
        for i, cid in enumerate(cl):
            y = yr(i)
            tx(ov, x0 - 0.5, y, cid, ha="right", va="center")
            for j, off in enumerate(offs):
                r = A[(A.claim == cid) & (A.offset == off)].iloc[0]
                v = r.width / WMAX
                rect(ov, x0 + j * cell_w, y - pitch / 2, cell_w, pitch, fc=S.SEQ(min(v, 1.0)), ec="white", lw=S.THIN, z=3)
                dot = "white" if v > 0.5 else S.INK
                ov.plot([colx(j)], [y], marker="o", ms=1.3, mfc=dot, mec=dot, mew=0, ls="none", zorder=5)
                if r.n_direct > 0:
                    rect(ov, x0 + j * cell_w + 0.35, y - pitch / 2 + 0.3, cell_w - 0.7, pitch - 0.6, fc="none",
                         ec=S.INK, lw=S.LW, z=6)
                src.append(dict(block="bottom_audit", dataset=pid, parent=r.claim[0], claim=cid, offset=off,
                                value=r.width, unit="audit interval width (conf_hi - conf_lo)", design_replayed="",
                                ref_in=bool(r.ref_in), n_direct=r.n_direct, knn_err=r.knn_err))
        yb = yr(len(cl) - 1) - pitch / 2
        for j, s in ((0, MINUS + "0.25"), (2, "0"), (4, "+0.25")):
            tx(ov, colx(j), yb - 0.7, s, ha="center", va="top")
        if pid == "wingmen_v1":
            tx(ov, (x0 + x1) / 2, yb - 3.3, "offset (asinh)", ha="center", va="top", color=S.MUTED)
    # ---- right: per-offset dot columns of the kNN error (both datasets), same offset order as the matrices
    xk0, xk1 = 53.6, 68.0
    yk0, yk1 = yr(13) - pitch / 2, ytop_m - 3.0
    axk = axmm(fig, xk0, yk0, xk1 - xk0, yk1 - yk0)
    axk.set_xlim(-0.6, 4.6)
    axk.set_ylim(0, 5.2)
    axk.set_xticks(range(5))
    axk.set_xticklabels([MINUS + "0.25", "", "0", "", "+0.25"])
    axk.set_yticks([0, 1, 2, 3, 4, 5])
    axk.set_yticklabels(["0", "", "", "", "", "5"])
    for pid, dxo, mk, fc in (("levine32_v2", -0.17, "o", S.INK), ("wingmen_v1", 0.17, "^", "white")):
        A = AU[AU.dataset == pid]
        for j, off in enumerate(offs):
            v = 100 * A[A.offset == off].knn_err.values
            axk.plot(np.full(len(v), j + dxo), v, ls="none", marker=mk, ms=2.0, mfc=fc, mec=S.INK, mew=S.THIN, zorder=4)
    tx(ov, xk0 - 1.6, yk1 + 1.7, f"|kNN {MINUS} ref| (pp)", va="center")
    tx(ov, xk0 + (xk1 - xk0) / 2, yk0 - 4.4, "offset", ha="center", va="top", color=S.MUTED)
    ov.plot([56.3], [ytop_m + 4.9], marker="o", ms=2.0, mfc=S.INK, mec=S.INK, mew=S.THIN, ls="none")
    tx(ov, 57.2, ytop_m + 4.9, "Levine-32", va="center")
    ov.plot([56.3], [ytop_m + 2.3], marker="^", ms=2.0, mfc="white", mec=S.INK, mew=S.THIN, ls="none")
    tx(ov, 57.2, ytop_m + 2.3, "WINGMEN", va="center")
    # ---- colour bar under the Levine matrix
    xcb0, xcb1 = xm["levine32_v2"] - 3.0, xm["levine32_v2"] + 5 * cell_w + 2.5
    ycb = yr(10) - pitch / 2 - 7.4
    tx(ov, xcb0, ycb + 2.9, "audit-interval width", va="center")
    nseg = 5
    for k in range(nseg):
        rect(ov, xcb0 + (xcb1 - xcb0) * k / nseg, ycb, (xcb1 - xcb0) / nseg, 1.5, fc=S.SEQ((k + 0.5) / nseg), ec="white",
             lw=S.THIN, z=3)
    for v, s in ((0, "0"), (0.25, "0.25"), (0.5, "0.5")):
        tx(ov, xcb0 + (xcb1 - xcb0) * v / WMAX, ycb - 0.3, s, ha="center", va="top")

    def dotg(x, y):
        ov.plot([x + 0.5], [y], marker="o", ms=1.3, mfc=S.INK, mec=S.INK, mew=0, ls="none")
        return 1.0

    def boxg(x, y):
        rect(ov, x, y - 0.8, 2.2, 1.6, fc="none", ec=S.INK, lw=S.LW, z=5)
        return 2.2

    legend_row(ov, fig, 0.6, 1.5, [(dotg, "reference inside interval: 55/55, 70/70"), (boxg, "n(direct) > 0")])
    check(fig, "S1C")
    S.save(fig, "S1C")
    S.save_source(pd.DataFrame(src), "S1C")


# ================================================================ S1D reference uncertainty
def panel_S1D(H=69.5):
    RC = rd("figS1_refci.tsv")
    SE = rd("figS1_refse.tsv")
    assert len(RC) == 27 and RC.half_gt_eps.sum() == 1
    fig = S.new("S1D", h=H, layout=None)
    ov, W, H = canvas(fig)
    header(ov, W, H, "Finite-sample reference uncertainty", "Replay")
    pitch, ggap = 2.22, 0.45
    ytop = H - 10.0
    XL = 2.6
    order = {"levine32_v2": ["HSPC", "T", "B", "Mono", "NK"], "wingmen_v1": ["CD4T", "CD8T", "B", "NK", "Mono"]}
    src = []
    blk = {"levine32_v2": 0.6, "wingmen_v1": 36.2}
    ybots = []
    for pid, name in (("levine32_v2", "Levine-32"), ("wingmen_v1", "WINGMEN")):
        x0 = blk[pid]
        xpar, xn_r, xa0, xa1, xcl = x0, x0 + 14.6, x0 + 15.4, x0 + 29.2, x0 + 29.8
        ax_x = lambda v: xa0 + (xa1 - xa0) * (v + XL) / (2 * XL)  # noqa: E731
        R = RC[RC.dataset == pid]
        tx(ov, x0, H - 5.9, f"{name} {DOT} {EPS} = 0.02", va="center", weight="bold")
        tx(ov, x0, ytop + 1.6, "parent, n(ref)", va="center", color=S.MUTED)
        y = ytop
        for G in order[pid]:
            sub = R[R.parent == G].sort_values("claim")
            for k, (_, r) in enumerate(sub.iterrows()):
                yc = y - pitch / 2
                same_row = tw(fig, dname(G)) + tw(fig, fmt(r.n_ref)) + 0.8 <= xn_r - xpar
                if k == 0:
                    tx(ov, xpar, yc, dname(G), va="center")
                if (k == 0 and same_row) or (k == 1 and not same_row):
                    tx(ov, xn_r, yc, fmt(r.n_ref), ha="right", va="center", color=S.MUTED)
                tx(ov, xcl, yc, r.claim, ha="left", va="center", color=S.INK if r.reliable else S.MUTED)
                colr = S.ALARM if r.half_gt_eps else (S.SEQ(0.62) if r.reliable else LIGHT)
                rect(ov, ax_x(r.lo_pp), yc - 0.5, ax_x(r.hi_pp) - ax_x(r.lo_pp), 1.0, fc=colr, z=4)
                ov.plot([ax_x(0)] * 2, [yc - 0.72, yc + 0.72], color=S.INK, lw=S.LW, zorder=5)
                if r.half_gt_eps:
                    assert k == 1
                    tx(ov, xpar, yc, f"half > {EPS}", va="center", color=S.ALARM)
                src.append(dict(block="CP_interval", dataset=pid, parent=G, claim=r.claim, reliable=r.reliable,
                                ref=r.ref, ref_lo=r.ref_lo, ref_hi=r.ref_hi, n_ref=r.n_ref, lo_minus_ref_pp=r.lo_pp,
                                hi_minus_ref_pp=r.hi_pp, half_width=r.half, eps=r.eps, half_gt_eps=r.half_gt_eps))
                y -= pitch
            y -= ggap
        yb = y + ggap
        ybots.append(yb)
        for v in (-2, 2):
            ov.plot([ax_x(v)] * 2, [yb, ytop + 0.4], color=S.MUTED, lw=S.LW, ls=(0, (2.0, 1.4)), zorder=3)
        tx(ov, ax_x(2), ytop + 0.5, "+" + EPS, ha="center", va="bottom", color=S.MUTED)
        tx(ov, ax_x(-2), ytop + 0.5, MINUS + EPS, ha="center", va="bottom", color=S.MUTED)
        ya = yb - 0.3
        ov.plot([xa0, xa1], [ya, ya], color=S.INK, lw=S.LW, solid_capstyle="butt")
        for v, s in ((-2, MINUS + "2"), (0, "0"), (2, "+2")):
            ov.plot([ax_x(v)] * 2, [ya, ya - 0.8], color=S.INK, lw=S.LW)
            tx(ov, ax_x(v), ya - 1.1, s, ha="center", va="top")
        tx(ov, (xa0 + xa1) / 2, ya - 4.9, f"CP bound {MINUS} ref (pp)", ha="center", va="center", color=S.MUTED)
        if pid == "levine32_v2":
            # same n(ref) for M1 and M2 (Mono, 2,215): the half-width differs through the reference frequency only
            mm = R.set_index("claim").loc[["M1", "M2"]]
            assert (mm.n_ref == 2215).all()
            assert [round(100 * v, 2) for v in mm.ref] == [84.65, 53.36]
            assert [round(100 * v, 3) for v in mm.half] == [1.523, 2.099]
            y_note = ya - 7.6
            for i_, (cid, rr) in enumerate(mm.iterrows()):
                tx(ov, x0, y_note - 2.5 * i_, f"{cid} ref {100 * rr.ref:.2f}%, half {100 * rr.half:.3f} pp", va="center",
                   color=S.ALARM if rr.half_gt_eps else S.INK)
            ynote_bot = y_note - 2.5 - 1.3
    groups = [("T", "Levine-32 T", 0.05), ("CD4T", "WINGMEN CD4 T", 0.03), ("CD8T", "WINGMEN CD8 T", 0.05)]
    xh0, xh1 = 30.0, 60.0
    XMAXP = 7.5
    hx = lambda v: xh0 + (xh1 - xh0) * v / XMAXP  # noqa: E731
    yh_top = min(ybots) - 8.5
    assert ynote_bot > yh_top + 1.0 + 1.3, (ynote_bot, yh_top)
    rowh = 2.75
    tx(ov, 0.6, yh_top + 1.0, f"Target-level SE / {EPS} (ledger {EPS}), approximate", va="center", weight="bold")
    tx(ov, 69.4, yh_top + 1.0, "max", ha="right", va="center", color=S.MUTED)
    bins = np.arange(0, XMAXP + 0.25, 0.25)
    for gi, (G, lab, eps) in enumerate(groups):
        d = SE[SE.parent == G]
        assert np.isclose(d.eps.iloc[0], eps)
        v = 100 * d.se_over_eps.values
        cnt, _ = np.histogram(v, bins=bins)
        yb0 = yh_top - (gi + 1) * rowh
        hmax = rowh - 0.45
        for k, cc in enumerate(cnt):
            if cc:
                rect(ov, hx(bins[k]) + 0.05, yb0, hx(bins[k + 1]) - hx(bins[k]) - 0.1, hmax * cc / cnt.max(),
                     fc=S.SEQ(0.62), z=3)
        ov.plot([hx(0), hx(XMAXP)], [yb0, yb0], color=S.GRID, lw=S.THIN, zorder=2)
        mxv = v.max()
        ov.plot([hx(mxv)] * 2, [yb0, yb0 + hmax], color=S.INK, lw=S.LW, zorder=5)
        tx(ov, 0.6, yb0 + 0.9, lab, va="center")
        tx(ov, 22.0, yb0 + 0.9, f"{EPS} {eps:g}", va="center", color=S.MUTED)
        tx(ov, 69.4, yb0 + 0.9, f"{mxv:.2f}%", ha="right", va="center")
        for k, cc in enumerate(cnt):
            src.append(dict(block="SE_over_eps_hist", dataset=d.dataset.iloc[0], parent=G, claim="", bin_lo_pct=bins[k],
                            bin_hi_pct=bins[k + 1], count=int(cc), n_targets=len(v), n_ref=int(d.n_ref.iloc[0]),
                            eps=eps, max_se_over_eps_pct=mxv))
    ya = yh_top - 3 * rowh - 0.3
    ov.plot([hx(0), hx(XMAXP)], [ya, ya], color=S.INK, lw=S.LW, solid_capstyle="butt")
    for v in (0, 2, 4, 6):
        ov.plot([hx(v)] * 2, [ya, ya - 0.8], color=S.INK, lw=S.LW)
        tx(ov, hx(v), ya - 1.1, str(v), ha="center", va="top")
    tx(ov, hx(XMAXP / 2), ya - 4.9, f"SE / {EPS} (%), with SE {APPROX} sqrt(ref_var / n(ref))", ha="center",
       va="center", color=S.MUTED)
    check(fig, "S1D")
    S.save(fig, "S1D")
    S.save_source(pd.DataFrame(src), "S1D")


# ================================================================ S1E real batch differences
def panel_S1E(H=49.0, W=184):
    B = rd("figS1_batch.tsv")
    BM = rd("figS1_batch_marker.tsv").sort_values("column")
    BD = rd("figS1_batch_donor.tsv")
    OV = rd("figS1_label_overlap.tsv").fillna("").sort_values("column")
    assert len(B) == 111 and B.reliable.sum() == 87
    assert BM.reliable.sum() == 29 and int((BM.max_abs_median_shift >= 0.25).sum()) == 5
    assert BD.n_ge_025.tolist() == [4, 1, 3] and BD.n_reliable.tolist() == [29, 29, 29]
    assert (OV.marker.tolist() == BM.marker.tolist())
    assert (OV.levine_role != "").sum() == 14 and (OV.wingmen_role != "").sum() == 16
    fig = S.new("S1E", w=W, h=H, layout=None)     # 184 mm = full composite content width (FigS1 page)
    ov, W, H = canvas(fig)
    header(ov, W, H, "Real batch differences, CytoNorm control (same donor, two runs): magnitude reference only", "Record")
    donors = sorted(B.donor.unique())
    order = BM.marker.tolist()
    rel = B.groupby("marker").reliable.first()
    # laid out for W = 180: of the extra width, 3/4 widens the heat map and the ECDF block moves right by all of it
    ex = (W - 180.0) / 4.0
    xh0, xh1 = 15.0, 137.0 + 3.0 * ex
    hxs = lambda v: xh0 + (v - xh0) * (xh1 - xh0) / 122.0  # noqa: E731  (keys under the heat map, same scaling)
    xE = 4.0 * ex                                          # shift of the ECDF block
    cw = (xh1 - xh0) / len(order)
    colx = lambda j: xh0 + (j + 0.5) * cw  # noqa: E731
    ch, lab_gap, orow = 2.35, 0.6, 2.3
    y_title = H - 5.0
    ys_top, ys_bot = H - 6.4, H - 11.2          # per-marker max |shift| strip
    y_top_blk = ys_bot - 0.5
    lab_need = max(tw(fig, m) for m in order)
    R = lab_need + 0.6                            # label band = longest vertical label
    y_bot_top = y_top_blk - 3 * ch - lab_gap - R  # top edge of the lower heat map
    y_bot_blk = y_bot_top - 3 * ch
    assert y_bot_blk - 2.7 >= 0.3, y_bot_blk
    # labels of columns with an overlap symbol must end above the two overlap rows
    yo_top = y_bot_top + 0.3 + 2 * orow
    matched = OV[(OV.levine_role != "") | (OV.wingmen_role != "")].marker
    assert (y_top_blk - 3 * ch - lab_gap) - max(tw(fig, m) for m in matched) > yo_top + 0.3
    SH, DPP = 0.5, 20.0
    src = []
    # ---- per-marker max |median shift| over the 3 donors (29 reliable markers; others hatched)
    hmax = ys_top - ys_bot - 0.4
    for j, r in enumerate(BM.itertuples()):
        x = xh0 + j * cw
        if r.reliable:
            v = r.max_abs_median_shift
            rect(ov, x + 0.15 * cw, ys_bot, 0.7 * cw, hmax * min(v, SH) / SH, fc=S.INK if v >= 0.25 else S.MUTED, z=3)
        else:
            rect(ov, x, ys_bot, cw, ys_top - ys_bot, fc="white", ec=S.MUTED, lw=0, z=3, hatch="//////")
            rect(ov, x, ys_bot, cw, ys_top - ys_bot, fc="none", ec="white", lw=S.THIN, z=4)
        src.append(dict(block="max_abs_median_shift", donor=r.donor_of_max, marker=r.marker, column=j + 1,
                        reliable=bool(r.reliable), value=r.max_abs_median_shift, unit="asinh", pos_b1="", pos_b2=""))
    ov.plot([xh0, xh1], [ys_bot + hmax * 0.25 / SH] * 2, color=S.INK, lw=S.LW, ls=(0, (2.0, 1.4)), zorder=5)
    ov.plot([xh0, xh1], [ys_bot, ys_bot], color=S.INK, lw=S.LW, zorder=5, solid_capstyle="butt")
    tx(ov, xh0 - 0.6, ys_bot + 3.45, "max |shift|", ha="right", va="center")
    tx(ov, xh0 - 0.6, ys_bot + 0.95, f"5/29 {GE} 0.25", ha="right", va="center", color=S.MUTED)
    for ytop, col, lim, unit in ((y_top_blk, "median_shift", SH, "asinh"), (y_bot_top, "dpos_pp", DPP, "pp")):
        for i, dn in enumerate(donors):
            y = ytop - (i + 1) * ch
            tx(ov, xh0 - 0.6, y + ch / 2, dn, ha="right", va="center")
            for j, m in enumerate(order):
                r = B[(B.donor == dn) & (B.marker == m)].iloc[0]
                x = xh0 + j * cw
                if r.reliable:
                    v = (r[col] / lim + 1) / 2
                    rect(ov, x, y, cw, ch, fc=S.DIV(min(max(v, 0), 1)), ec="white", lw=S.THIN, z=3)
                else:
                    rect(ov, x, y, cw, ch, fc="white", ec=S.MUTED, lw=0, z=3, hatch="//////")
                    rect(ov, x, y, cw, ch, fc="none", ec="white", lw=S.THIN, z=4)
                src.append(dict(block=col, donor=dn, marker=m, column=j + 1, reliable=bool(r.reliable), value=r[col],
                                unit=unit, pos_b1=r.pos_b1, pos_b2=r.pos_b2))
    ylab = y_top_blk - 3 * ch - lab_gap
    for j, m in enumerate(order):
        tx(ov, colx(j), ylab, m, ha="center", va="top", rotation=90, color=S.INK if rel[m] else S.MUTED)
    # ---- marker-label overlap with the two replay panels (registered backbone / candidate roles only)
    for k, (col, name) in enumerate((("wingmen", "WINGMEN"), ("levine", "Levine-32"))):
        y = yo_top - (k + 0.5) * orow
        tx(ov, xh0 - 0.6, y, name, ha="right", va="center")
        for j, r in enumerate(OV.itertuples()):
            role = getattr(r, f"{col}_role")
            if role == "backbone":
                ov.plot([colx(j)], [y], marker="s", ms=3.0, mfc=S.BACKBONE, mec=S.INK, mew=S.THIN, ls="none", zorder=5)
            elif role == "candidate":
                ov.plot([colx(j)], [y], marker="o", ms=2.1, mfc=S.INK, mec=S.INK, mew=0, ls="none", zorder=5)
            src.append(dict(block=f"label_overlap_{col}", donor="", marker=r.marker, column=j + 1, reliable="",
                            value=role, unit=getattr(r, f"{col}_label"), pos_b1="", pos_b2=""))
    yl0 = yo_top + 1.45
    ov.plot([xh0 - 13.2], [yl0], marker="o", ms=2.1, mfc=S.INK, mec=S.INK, mew=0, ls="none")
    tx(ov, xh0 - 12.4, yl0, "candidate", va="center")
    ov.plot([xh0 - 13.2], [yl0 + 2.5], marker="s", ms=3.0, mfc=S.BACKBONE, mec=S.INK, mew=S.THIN, ls="none")
    tx(ov, xh0 - 12.4, yl0 + 2.5, "backbone", va="center")
    tx(ov, 0.6, yl0 + 5.0, "label overlap", va="center", color=S.MUTED)
    # ---- block titles and colour bars
    tx(ov, xh0, y_title, f"Median shift, run 2 {MINUS} run 1 (asinh units)", va="center")
    y_bt = y_bot_blk - 1.4
    tx(ov, xh0, y_bt, f"Positive-fraction difference, run 2 {MINUS} run 1 (pp)", va="center")
    for yb, ticks in ((y_title - 0.75, [MINUS + "0.5", "+0.5"]), (y_bt - 0.75, [MINUS + "20", "+20"])):
        xa, xb = hxs(92.0), hxs(116.0)
        nseg = 8
        for k in range(nseg):
            rect(ov, xa + (xb - xa) * k / nseg, yb, (xb - xa) / nseg, 1.5, fc=S.DIV((k + 0.5) / nseg), ec="white",
                 lw=S.THIN, z=3)
        ov.plot([(xa + xb) / 2] * 2, [yb - 0.35, yb + 1.85], color=S.INK, lw=S.THIN, zorder=5)
        tx(ov, xa - 0.8, yb + 0.75, ticks[0], ha="right", va="center")
        tx(ov, xb + 0.8, yb + 0.75, ticks[1], ha="left", va="center")
    rect(ov, hxs(126.0), y_bt - 0.75, 2.4, 1.5, fc="white", ec=S.MUTED, lw=0, z=3, hatch="//////")
    tx(ov, hxs(126.0) + 3.0, y_bt, "unreliable cut", va="center", color=S.MUTED)
    # ---- ECDF of |median shift|: all reliable records + one thin line per donor
    r = B[B.reliable].abs_median_shift.sort_values().values
    n = len(r)
    axe = axmm(fig, 148.0 + xE, 10.6, 30.0, H - 10.6 - 12.4)
    axe.set_facecolor("none")          # legend below is drawn on the canvas, behind this axes
    tx(ov, 141.0 + xE, y_title, "grey lines: offsets used in replays", va="center", color=S.MUTED)

    def ecdf_xy(v):
        v = np.sort(v)
        xs_ = np.concatenate([[0], np.repeat(v, 2), [0.55]])
        ys_ = np.concatenate([[0, 0], np.repeat(np.arange(1, len(v) + 1) / len(v), 2)])
        return xs_, ys_

    styles = {donors[0]: (0, (2.4, 1.3)), donors[1]: (0, (0.8, 1.0)), donors[2]: (0, (3.0, 1.0, 0.8, 1.0))}
    leg = [(S.INK, "solid", "all", f"{int((r >= 0.25).sum())}/{n}")]
    for dn in donors:
        v = B[B.reliable & (B.donor == dn)].abs_median_shift.values
        k25 = int((v >= 0.25).sum())
        assert len(v) == 29 and k25 == int(BD.set_index("donor").loc[dn, "n_ge_025"])
        axe.plot(*ecdf_xy(v), color=S.MUTED, lw=S.LW, ls=styles[dn], zorder=3)
        leg.append((S.MUTED, styles[dn], dn, f"{k25}/29"))
        for x_ in np.sort(v):
            src.append(dict(block="ecdf_per_donor", donor=dn, marker="", column="", reliable=True, value=x_,
                            unit="asinh", pos_b1="", pos_b2=""))
    axe.plot(*ecdf_xy(r), color=S.INK, lw=S.LW, zorder=4)
    for v in (0.25, 0.5):   # thin solid light-grey reference lines (reference-grid exception), unlike the donor dash styles
        axe.axvline(v, color=S.GRID, lw=S.THIN, ls="-", zorder=2)
        axe.text(v, 1.04, f"{v:g}", fontsize=S.FS, ha="center", va="bottom", color=S.MUTED)
    axe.set_xlim(0, 0.55)
    axe.set_ylim(0, 1.0)
    axe.set_xticks([0, 0.25, 0.5])
    axe.set_xticklabels(["0", "0.25", "0.5"])
    axe.set_yticks([0, 0.5, 1])
    axe.set_yticklabels(["0", "0.5", "1"])
    axe.set_ylabel("ECDF", labelpad=1)
    axe.set_xlabel("|median shift| (asinh)", labelpad=1)
    # legend inside the empty lower-right part of the ECDF (records >= 0.25 per line)
    ax_h, ax_w = (H - 10.6 - 12.4), 30.0
    tr = axe.transAxes
    wb = dict(boxstyle="square,pad=0.08", fc="white", ec="none", lw=0)   # white backing breaks the dashed lines
    lx = 0.12 / 0.55
    for k, (colr, ls_, lab, cnt) in enumerate(reversed(leg)):
        y = 0.08 + k * 2.45 / ax_h
        axe.plot([lx, lx + 3.4 / ax_w], [y, y], color=colr, lw=S.LW, ls=ls_, solid_capstyle="butt", transform=tr,
                 zorder=10, clip_on=False)
        axe.text(lx + 4.0 / ax_w, y, lab, transform=tr, fontsize=S.FS, ha="left", va="center", zorder=10, bbox=wb)
        axe.text(29.6 / ax_w, y, cnt, transform=tr, fontsize=S.FS, ha="right", va="center", zorder=10, bbox=wb)
    axe.text(29.6 / ax_w, 0.08 + len(leg) * 2.45 / ax_h, f"{GE} 0.25", transform=tr, fontsize=S.FS, ha="right",
             va="center", color=S.MUTED, zorder=10, bbox=wb)
    for v in r:
        src.append(dict(block="ecdf_abs_median_shift", donor="all", marker="", column="", reliable=True, value=v,
                        unit="asinh", pos_b1="", pos_b2=""))
    check(fig, "S1E")
    S.save(fig, "S1E")
    S.save_source(pd.DataFrame(src), "S1E")


PANELS = {"S1A": panel_S1A, "S1B": panel_S1B, "S1C": panel_S1C, "S1D": panel_S1D, "S1E": panel_S1E}

if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PANELS)):
        PANELS[p]()
