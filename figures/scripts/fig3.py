"""Figure 3 single panels 3A-3E (figure plan v2, design/witnessflow/figplan_v2.md).

Run from witnessflow/figures/scripts:   python fig3.py
Reads only frozen tables (figures/data, results, tools/scan/out_lat); writes display tables to
figures/derived/fig3_*.tsv and panels to figures/panels/fig3{A..E}.pdf/.png via wfstyle.save.
No new experiments / randomisation: every number on the panels is read or deterministically derived here.
"""
import ast
import os
import re
import numpy as np
import pandas as pd
import wfstyle as S

S.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402
from scipy.stats import beta as BETA  # noqa: E402

DATA, DER, RES = S.DATA, S.DERIVED, S.RESULTS
LAT = os.path.join(S.WF, "tools", "scan", "out_lat")
os.makedirs(DER, exist_ok=True)

C_CL = S.METHOD["classical"]
C_RD = S.METHOD["random"]
C_BR = S.METHOD["balanced_random"]
C_PSL = S.METHOD["psl"]
C_LIGHT = S.SEQ(0.33)          # light tint of the classical blue (wfstyle sequential map)
MS = 2.8                       # marker size (pt) for dense node series


def rd(path):
    return pd.read_csv(path, sep="\t")


def dname(s):
    """Display form of parent names in drawn text (source keys CD4T / CD8T unchanged): 'CD4 T', 'CD8 T'."""
    return re.sub(r"\bCD([48])T\b", r"CD\1 T", s)


def wder(df, name):
    df.to_csv(os.path.join(DER, name), sep="\t", index=False)


def fig_mm(fig):
    w, h = fig.get_size_inches()
    return w * 25.4, h * 25.4


def add_ax(fig, x0, y0, w, h, **kw):
    W, H = fig_mm(fig)
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def ftext(fig, x, y, s, **kw):
    W, H = fig_mm(fig)
    return fig.text(x / W, y / H, s, **kw)


def fline(fig, xs, ys, **kw):
    W, H = fig_mm(fig)
    kw.setdefault("lw", S.LW)
    kw.setdefault("color", S.INK)
    fig.add_artist(Line2D([x / W for x in xs], [y / H for y in ys], transform=fig.transFigure, **kw))


def title(fig, x, y, s):
    return ftext(fig, x, y, s, ha="left", va="top", fontweight="bold")


def tag(fig, x, y, s):
    W, H = fig_mm(fig)
    return S.tier_tag(fig, s, x=x / W, y=y / H, ha="right", va="top")


def pct(v, nd=2):
    return f"{100 * v:.{nd}f}%"


# =====================================================================================================
# shared facts (verified against frozen sources)
# =====================================================================================================
COV = {"wingmen_v1": 130, "levine32_v2": 71}         # base covering size: S(3,6,26) Steiner / LJCR C(20,6,3)
EPS = {}
for _ds in COV:
    _e = rd(os.path.join(RES, _ds, "e5b_steiner_nodes.tsv"))
    for _p, _g in _e.groupby("parent"):
        assert _g.eps.nunique() == 1
        EPS[(_ds, _p)] = float(_g.eps.iloc[0])
assert EPS == {("levine32_v2", "B"): 0.08, ("levine32_v2", "T"): 0.06,
               ("wingmen_v1", "CD4T"): 0.05, ("wingmen_v1", "CD8T"): 0.08}, EPS

SAW = rd(os.path.join(DATA, "fig3A_sawtooth.tsv"))
assert len(SAW) == 140 and list(SAW.columns) == ["dataset", "parent", "r", "method", "half_ok", "half_sd", "err_ok", "n"]
SAW["key"] = SAW.method.map(S.ALIAS)
NODES = {ds: sorted(SAW[SAW.dataset == ds].r.unique()) for ds in COV}
assert len(NODES["wingmen_v1"]) == 12 and len(NODES["levine32_v2"]) == 11
assert set(SAW[SAW.dataset == "wingmen_v1"].key) == {"classical", "random", "balanced_random"}
assert set(SAW[SAW.key == "psl"].r) == {570} and set(SAW[SAW.key == "psl"].dataset) == {"levine32_v2"}
FULL = {"wingmen_v1": [130, 260], "levine32_v2": [71, 142, 213, 284]}

THR = rd(os.path.join(DATA, "supp_threshold_design.tsv"))


def saw_val(ds, p, r, key, col="half_ok"):
    s = SAW[(SAW.dataset == ds) & (SAW.parent == p) & (SAW.r == r) & (SAW.key == key)]
    assert len(s) == 1
    return float(s[col].iloc[0])


# =====================================================================================================
# derived tables
# =====================================================================================================
def derive_3b():
    s = SAW.copy()
    s["rounds"] = s.r / s.dataset.map(COV)
    s["complete_round"] = [r in FULL[d] for d, r in zip(s.dataset, s.r)]
    s["eps"] = [EPS[(d, p)] for d, p in zip(s.dataset, s.parent)]
    wder(s[["dataset", "parent", "eps", "r", "rounds", "complete_round", "key", "method", "half_ok", "half_sd", "n"]],
         "fig3_B_series.tsv")
    g = THR.groupby(["dataset", "r", "method"])
    t = g.half_ok.agg(["min", "max", "count"]).reset_index()
    t0 = THR[THR.offset == 0.0].set_index(["dataset", "r", "method"]).half_ok
    t["offset0"] = [t0[(d, r, m)] for d, r, m in zip(t.dataset, t.r, t.method)]
    t["parent"] = t.dataset.map({"wingmen_v1": "CD4T", "levine32_v2": "T"})
    t["allocations"] = 3
    t["source"] = "figures/data/supp_threshold_design.tsv (e5g; offsets -0.25..+0.25; 3 allocations, random design seed 0)"
    wder(t, "fig3_B_threshold_ranges.tsv")
    ann = pd.DataFrame([
        ("WINGMEN CD4T r=180 classical half_ok (main curve)", saw_val("wingmen_v1", "CD4T", 180, "classical"),
         "fig3A_sawtooth.tsv (e5b_r180.tsv; mean of 10 allocations)"),
        ("WINGMEN CD4T r=180 classical half_ok (threshold run, offset 0)",
         float(t0[("wingmen_v1", 180, "classical_stack")]),
         "supp_threshold_design.tsv (e5g; mean of 3 allocations)"),
        ("Levine B r=71 classical half_ok", saw_val("levine32_v2", "B", 71, "classical"), "fig3A_sawtooth.tsv"),
        ("Levine B r=142 classical half_ok", saw_val("levine32_v2", "B", 142, "classical"), "fig3A_sawtooth.tsv"),
    ] + [(f"eps {d} {p}", v, "results/<ds>/e5b_steiner_nodes.tsv eps column") for (d, p), v in EPS.items()],
        columns=["item", "value", "source"])
    wder(ann, "fig3_B_annotations.tsv")
    return t


def derive_3c():
    # WINGMEN classical stack lambda_T distribution at all 12 nodes: exact formula, checked against the 6 saved nodes
    tab = rd(os.path.join(DATA, "fig3B_multiplicity_wingmen.tsv"))
    lam_min = {}
    e = rd(os.path.join(RES, "wingmen_v1", "e5b_steiner_nodes.tsv"))
    e = e[e.method == "classical_stack"]
    for r, g in e.groupby("r"):
        assert g.lam_min.nunique() == 1
        lam_min[int(r)] = (int(g.lam_min.iloc[0]), "results/wingmen_v1/e5b_steiner_nodes.tsv")
    e2 = rd(os.path.join(RES, "wingmen_v1", "e5b_r180.tsv"))
    e2 = e2[e2.method == "classical_stack"]
    assert e2.lam_min.nunique() == 1
    lam_min[180] = (int(e2.lam_min.iloc[0]), "results/wingmen_v1/e5b_r180.tsv")
    rows = []
    for r in NODES["wingmen_v1"]:
        if r <= 130:
            n = {0: 2600 - 20 * r, 1: 20 * r, 2: 0}
        else:
            n = {0: 0, 1: 2600 - 20 * (r - 130), 2: 20 * (r - 130)}
        saved = tab[tab.r == r]
        if len(saved):
            got = {lam: int(saved[saved.lam == lam].n_triples.sum()) for lam in (0, 1, 2)}
            assert got == n, (r, got, n)
            src = "fig3B_multiplicity_wingmen.tsv (formula agrees)"
        else:
            src = "exact formula"
        lm = min(k for k, c in n.items() if c > 0)
        assert lm == lam_min[r][0], (r, lm, lam_min[r])
        rows.append(dict(r=r, n_lam0=n[0], n_lam1=n[1], n_lam2=n[2], lam_min=lam_min[r][0],
                         share_ratio=lam_min[r][0] / r / (20 / 2600), lam_source=src, lam_min_source=lam_min[r][1]))
    m = pd.DataFrame(rows)
    wder(m, "fig3_C_multiplicity_wingmen_all_nodes.tsv")
    # Levine base designs (v = 20, 1140 triples)
    txt = open(os.path.join(LAT, "lat02_structure.txt"), encoding="utf-8").read()
    sec = txt.split("=== LJCR C(20,6,3)")[1]
    mult = ast.literal_eval(re.search(r"triple multiplicities: (\{[^}]*\})", sec).group(1))
    assert sum(mult.values()) == 1140
    hist = rd(os.path.join(LAT, "lat01_hist.tsv"))
    h71 = hist[(hist.design == "LJCR_stack(seed0)") & (hist.v == 20) & (hist.r == 71)]
    for lam, c in mult.items():
        assert abs(float(h71[h71.lam == lam].frac.iloc[0]) * 1140 - c) < 1e-6
    hp = hist[hist.design.str.startswith("PSL(2,19)")]
    assert len(hp) == 1 and int(hp.lam.iloc[0]) == 10 and float(hp.frac.iloc[0]) == 1.0 and int(hp.r.iloc[0]) == 570
    summ = rd(os.path.join(LAT, "lat01_summary.tsv"))
    eff71 = float(summ[(summ.v == 20) & (summ.design == "LJCR_stack(seed0)") & (summ.r == 71)].eff.iloc[0])
    effp = float(summ[summ.design.str.startswith("PSL(2,19)")].eff.iloc[0])
    lv = pd.DataFrame([dict(design="LJCR C(20,6,3) covering (= LJCR_stack seed0)", r=71, lam=k, n_triples=v,
                            min_share_ratio=eff71) for k, v in sorted(mult.items())]
                      + [dict(design="PSL(2,19) 3-(20,6,10) design", r=570, lam=10, n_triples=1140, min_share_ratio=effp)])
    lv["source"] = "tools/scan/out_lat/lat02_structure.txt + lat01_hist.tsv (cross-checked); ratio = lat01_summary.tsv eff"
    wder(lv, "fig3_C_levine_lambda.tsv")
    return m, lv


def derive_3e():
    rows = []
    for ds in COV:
        k = rd(os.path.join(RES, ds, "e5k_boundary.tsv"))
        a = k[k.part == "A_unequal"][["sigma", "rep", "method", "tube_size_cv", "half_ok", "err_ok", "min_cells"]].copy()
        a.insert(0, "dataset", ds)
        rows.append(a)
    reps = pd.concat(rows, ignore_index=True)
    wder(reps, "fig3_E_reps.tsv")
    sup = rd(os.path.join(DATA, "supp_boundary_unequal.tsv"))
    chk = reps.groupby(["dataset", "sigma", "method"])[["tube_size_cv", "half_ok", "err_ok"]].mean().reset_index()
    mm = sup.merge(chk, on=["dataset", "sigma", "method"], suffixes=("", "_reps"))
    assert len(mm) == 16
    for c in ("tube_size_cv", "half_ok", "err_ok"):
        assert np.allclose(mm[c], mm[c + "_reps"], atol=1e-12)
    return reps, sup


# =====================================================================================================
# 3B  sawtooth (Replay)
# =====================================================================================================
def panel_3b(thr):
    # composite size (composeFig3.py): 132 x 80 mm, same width as 3C (shared WINGMEN x axis, geo["XW"])
    W, H = 132, 80
    fig = S.new("3B", w=W, h=H, layout=None)
    XW = (12.0, 65.5)                    # WINGMEN column (mm)
    XL = (71.5, 117.0)                   # Levine, main segment
    XP = (119.0, 128.0)                  # Levine, r = 570 segment (main rows only)
    LIMW, LIML, LIMP = (98, 266), (50, 296), (546, 595)
    ROWS = {"r1": (48.9, 19.0), "r2": (27.9, 19.0), "st": (8.4, 7.5)}
    YL = (38, 103.5)
    sc = {"W": (XW[1] - XW[0]) / (LIMW[1] - LIMW[0]), "L": (XL[1] - XL[0]) / (LIML[1] - LIML[0])}

    def mk(xr, row, lim):
        y0, h = ROWS[row]
        ax = add_ax(fig, xr[0], y0, xr[1] - xr[0], h)
        ax.set_xlim(*lim)
        ax.set_ylim(*YL)
        return ax

    def bands(ax, nodes, half):
        for n in nodes:
            ax.axvspan(n - half, n + half, color=S.BAND, lw=0, zorder=0)

    def mstyle(key):
        c = S.METHOD[key]
        if key == "balanced_random":
            return dict(color=c, marker=S.MARKER[key], ms=MS + 0.3, mfc=c, mec=C_RD, mew=S.THIN)
        if key == "psl":
            return dict(ls="none", marker=S.MARKER[key], ms=7, mfc=c, mec=S.INK, mew=S.THIN)
        return dict(color=c, marker=S.MARKER[key], ms=MS, mfc=c, mec="white", mew=S.THIN)

    def curves(ax, ds, p):
        sub = SAW[(SAW.dataset == ds) & (SAW.parent == p)]
        for key, z in (("balanced_random", 3), ("random", 4), ("classical", 5), ("psl", 6)):
            s = sub[sub.key == key].sort_values("r")
            if s.empty:
                continue
            x, y = s.r.values, 100 * s.half_ok.values
            if key == "classical":
                sd = 100 * s.half_sd.values
                ax.fill_between(x, y - sd, y + sd, color=C_CL, alpha=0.28, lw=0, zorder=2)
            ax.plot(x, y, zorder=z, **mstyle(key))

    axes = {}
    for ds, pars in (("wingmen_v1", ("CD4T", "CD8T")), ("levine32_v2", ("T", "B"))):
        for row, p in zip(("r1", "r2"), pars):
            if ds == "wingmen_v1":
                a = mk(XW, row, LIMW)
                bands(a, FULL[ds], 0.9 / sc["W"])
                curves(a, ds, p)
                axes[(ds, row)] = (a,)
            else:
                a = mk(XL, row, LIML)
                b = mk(XP, row, LIMP)
                bands(a, FULL[ds], 0.9 / sc["L"])
                curves(a, ds, p)
                curves(b, ds, p)
                axes[(ds, row)] = (a, b)
            a.text(LIMW[1] - 4 if ds == "wingmen_v1" else LIML[1] - 4, 40.5, f"{dname(p)} · ε = {EPS[(ds, p)]:g}",
                   ha="right", va="bottom")
    # main-row axes cosmetics
    wt = [104, 130, 180, 260]
    lt = [71, 100, 142, 213, 284]
    for (ds, row), axs in axes.items():
        for a in axs:
            a.tick_params(labelbottom=(row == "r2"))
    for row in ("r1", "r2"):
        a = axes[("wingmen_v1", row)][0]
        a.set_xticks(wt)
        a.xaxis.set_minor_locator(FixedLocator(NODES["wingmen_v1"]))
        a.set_yticks([40, 60, 80, 100])
        a, b = axes[("levine32_v2", row)]
        a.set_xticks(lt)
        a.xaxis.set_minor_locator(FixedLocator([n for n in NODES["levine32_v2"] if n < 300]))
        a.set_yticks([40, 60, 80, 100])
        a.tick_params(labelleft=False)
        b.set_xticks([570])
        b.spines["left"].set_visible(False)
        b.tick_params(left=False, labelleft=False)
    # broken-axis marks: identical x positions on both main rows
    for row in ("r1", "r2"):
        y0 = ROWS[row][0]
        for x in (XL[1], XP[0]):
            fline(fig, [x - 0.45, x + 0.45], [y0 - 0.8, y0 + 0.8])
    # top axis: tube count / base-design size
    for ds, axs in (("wingmen_v1", axes[("wingmen_v1", "r1")]), ("levine32_v2", axes[("levine32_v2", "r1")])):
        cov = COV[ds]
        for i, a in enumerate(axs):
            sec = a.secondary_xaxis("top", functions=(lambda r, c=cov: r / c, lambda q, c=cov: q * c))
            if ds == "wingmen_v1":
                sec.set_xticks([0.8, 1, 1.5, 2], labels=["0.8", "1", "1.5", "2"])
                sec.set_xlabel(f"WINGMEN · tube count / base-design size ({cov})")
            elif i == 0:
                sec.set_xticks([1, 2, 3, 4])
                sec.set_xlabel(f"Levine-32 · tube count / base-design size ({cov})")
            else:
                sec.set_xticks([570 / cov], labels=[f"{570 / cov:.2f}"])
    # data labels
    a = axes[("wingmen_v1", "r1")][0]
    v180 = saw_val("wingmen_v1", "CD4T", 180, "classical")
    a.text(180, 100 * v180 + 4.0, pct(v180), ha="center", va="bottom")
    b = axes[("levine32_v2", "r2")][0]
    for r, tx in ((71, 92), (142, 142)):
        v = saw_val("levine32_v2", "B", r, "classical")
        b.annotate(f"{100 * v:.1f}%", xy=(r, 100 * v + 1.6), xytext=(tx, 96.5), ha="center", va="bottom",
                   arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.MUTED, shrinkA=0.5, shrinkB=0))
    ftext(fig, (XW[0] + XP[1]) / 2, 22.3, "Tubes r (fixed total cells per parent)", ha="center", va="center")
    # ---- threshold-offset strip (CD4T / T only): own sub-title, no r = 570 segment
    sw = mk(XW, "st", LIMW)
    sl = mk(XL, "st", LIML)
    bands(sw, FULL["wingmen_v1"], 0.9 / sc["W"])
    bands(sl, FULL["levine32_v2"], 0.9 / sc["L"])
    for ax, ds, dxmm in ((sw, "wingmen_v1", 0.75 / sc["W"]), (sl, "levine32_v2", 0.75 / sc["L"])):
        for meth, key, off in (("classical_stack", "classical", -dxmm), ("random", "random", dxmm)):
            tt = thr[(thr.dataset == ds) & (thr.method == meth)]
            for _, q in tt.iterrows():
                c = S.METHOD[key]
                ax.plot([q.r + off] * 2, [100 * q["min"], 100 * q["max"]], color=c, lw=S.LW, solid_capstyle="butt",
                        zorder=3)
                ax.plot(q.r + off, 100 * q.offset0, zorder=4, **{k: v for k, v in mstyle(key).items() if k != "color"})
    v0 = float(thr[(thr.dataset == "wingmen_v1") & (thr.r == 180) & (thr.method == "classical_stack")].offset0.iloc[0])
    sw.text(180 - 0.75 / sc["W"] - 2.2, 100 * v0, pct(v0), ha="right", va="center")
    sw.text(LIMW[0] + 2, YL[1] - 1, "CD4 T", ha="left", va="top")
    sl.text(LIML[0] + 2, YL[1] - 1, "T", ha="left", va="top")
    sw.set_xticks([130, 180])
    sl.set_xticks([71, 100])
    sw.set_yticks([50, 100])
    sl.set_yticks([50, 100])
    sl.tick_params(labelleft=False)
    ftext(fig, XW[0], 19.4, "Threshold-offset check · CD4 T, T only · 3 allocations", ha="left", va="top",
          fontweight="bold")
    ftext(fig, XP[1], 19.4, "bar = range, offsets ±0.25 · symbol = offset 0", ha="right", va="top")
    # labels, legend
    ftext(fig, 3.4, (ROWS["r2"][0] + ROWS["r1"][0] + ROWS["r1"][1]) / 2,
          "Triples with max half-width ≤ ε (%)\nper-pattern 95% CP, marginal; not joint",
          rotation=90, ha="center", va="center", multialignment="center", linespacing=1.1)
    hs = [(Patch(fc=C_CL, alpha=0.28, lw=0), Line2D([], [], **mstyle("classical"))),
          Line2D([], [], **mstyle("random")),
          Line2D([], [], **mstyle("balanced_random")),
          Line2D([], [], **mstyle("psl")),
          Patch(fc=S.BAND, lw=0)]
    ls = ["Classical (±SD, 10 allocations)", "Random", "Balanced random", "PSL(2,19) 3-design", "Complete round"]
    fig.legend(hs, ls, loc="center", bbox_to_anchor=(0.5, 2.3 / H), ncol=5, frameon=False, columnspacing=0.6,
               handlelength=1.15, handletextpad=0.3)
    title(fig, 0.6, H - 0.6, "Half-width attainment vs number of tubes")
    tag(fig, W - 0.6, H - 0.6, "Replay")
    S.save(fig, "3B")
    return dict(W=W, XW=XW, XL=XL, XP=XP, LIMW=LIMW, wt=wt)


# =====================================================================================================
# 3C  where the sawtooth comes from (Math)
# =====================================================================================================
def panel_3c(m, lv, geo):
    W, H = geo["W"], 36                  # same width as 3B (composite: 3B over 3C, left edges aligned)
    fig = S.new("3C", w=W, h=H, layout=None)
    XW, LIMW = geo["XW"], geo["LIMW"]
    HX0, HXW = 76.0, 51.5                # Levine histogram (mm)
    scw = (XW[1] - XW[0]) / (LIMW[1] - LIMW[0])
    # share-ratio strip
    sr = add_ax(fig, XW[0], 23.2, XW[1] - XW[0], 8.0)
    sr.set_xlim(*LIMW)
    sr.set_ylim(-0.12, 1.5)
    for n in FULL["wingmen_v1"]:
        sr.axvspan(n - 0.9 / scw, n + 0.9 / scw, color=S.BAND, lw=0, zorder=0)
    sr.axhline(1.0, color=S.GRID, lw=S.THIN, zorder=1)
    sr.plot(m.r, m.share_ratio, color=C_CL, marker="o", ms=MS, mfc=C_CL, mec="white", mew=S.THIN, zorder=4)
    sr.set_yticks([0, 1])
    sr.set_xticks(geo["wt"])
    sr.tick_params(labelbottom=False)
    sr.text(199, 1.06, "min share ratio = 130 · λ(min) / r", ha="center", va="bottom")
    s180 = float(m[m.r == 180].share_ratio.iloc[0])
    sr.text(180, s180 - 0.16, f"{s180:.2f}", ha="center", va="top")
    # stacked bars
    br = add_ax(fig, XW[0], 8.6, XW[1] - XW[0], 13.0)
    br.set_xlim(*LIMW)
    br.set_ylim(0, 133)
    r180 = m[m.r == 180].iloc[0]
    assert (int(r180.n_lam1), int(r180.n_lam2)) == (1600, 1000)
    br.text(180, 104, f"{int(r180.n_lam1)} at λ=1; {int(r180.n_lam2)} at λ=2", ha="center", va="bottom")
    for n in FULL["wingmen_v1"]:
        br.axvspan(n - 0.9 / scw, n + 0.9 / scw, color=S.BAND, lw=0, zorder=0)
    bw = 1.4 / scw
    f2, f1, f0 = (100 * m[c] / 2600 for c in ("n_lam2", "n_lam1", "n_lam0"))
    br.bar(m.r, f2, width=bw, color=C_CL, lw=0, zorder=2)
    br.bar(m.r, f1, bottom=f2, width=bw, color=C_LIGHT, lw=0, zorder=2)
    z = f0 > 0
    br.bar(m.r[z], f0[z], bottom=(f2 + f1)[z], width=bw, color="white", ec=S.MUTED, lw=0, hatch="//////", zorder=2)
    br.set_xticks(geo["wt"])
    br.xaxis.set_minor_locator(FixedLocator(NODES["wingmen_v1"]))
    br.set_yticks([0, 50, 100])
    br.spines["left"].set_bounds(0, 100)
    br.set_ylabel("Triples (%)", labelpad=1)
    ftext(fig, (XW[0] + XW[1]) / 2, 2.4, "Tubes r (WINGMEN, classical stack)", ha="center", va="center")
    # lambda legend in title row
    hs = [Patch(fc="white", ec=S.MUTED, lw=0, hatch="//////"), Patch(fc=C_LIGHT, lw=0), Patch(fc=C_CL, lw=0)]
    fig.legend(hs, ["λ(T) = 0", "λ(T) = 1", "λ(T) = 2"], loc="upper left", bbox_to_anchor=(44 / W, (H - 0.1) / H),
               ncol=3, frameon=False, handlelength=1.0, columnspacing=0.9, borderaxespad=0)
    # Levine base designs (right)
    hx = add_ax(fig, HX0, 8.6, HXW, 15.6)
    l71 = lv[lv.r == 71]
    hx.vlines(l71.lam, 0, 100 * l71.n_triples / 1140, color=C_CL, lw=S.LW, zorder=2)
    hx.plot(l71.lam, 100 * l71.n_triples / 1140, ls="none", marker="o", ms=MS + 0.4, mfc=C_CL, mec="white",
            mew=S.THIN, zorder=3)
    for lam, n in zip(l71.lam, l71.n_triples):
        hx.text(lam, 100 * n / 1140 + 5, f"{n}", ha="center", va="bottom")
    hx.vlines([10], 0, 100, color=C_PSL, lw=S.LW, zorder=2)
    hx.plot([10], [100], ls="none", marker="*", ms=7, mfc=C_PSL, mec=S.INK, mew=S.THIN, zorder=3)
    hx.text(9.55, 100, "PSL(2,19), 570 tubes: 1140", ha="right", va="center")
    e71 = float(l71.min_share_ratio.iloc[0])
    ep = float(lv[lv.r == 570].min_share_ratio.iloc[0])
    hx.text(2.5, 52, f"La Jolla covering, 71 tubes\nλ(T) = 1–6 · min ratio {e71:.2f}", ha="left", va="center",
            linespacing=1.1)
    hx.text(9.55, 83, f"λ(T) = 10 · ratio {ep:.2f}", ha="right", va="center")
    hx.set_xlim(0.3, 10.8)
    hx.set_ylim(0, 112)
    hx.set_xticks([1, 2, 3, 4, 5, 6, 10])
    hx.set_yticks([0, 50, 100])
    hx.set_ylabel("% of 1140 triples", labelpad=1)
    ftext(fig, HX0, 31.0, "Levine-32 · labels = number of triples", ha="left", va="top")
    ftext(fig, HX0, 28.2, "share ratio = (λ(T) / r) / (C(k,3) / C(v,3))", ha="left", va="top")
    ftext(fig, HX0 + HXW / 2, 2.4, "λ(T) (tubes containing the triple)", ha="center", va="center")
    title(fig, 0.6, H - 0.6, "Where the sawtooth comes from")
    tag(fig, W - 0.6, H - 0.6, "Math")
    S.save(fig, "3C")


# =====================================================================================================
# 3A  counting identity (Math, schematic)
# =====================================================================================================
def card(ax, xc, ytop, slot_h, w, target=()):
    """Vertical M1 tube card: 6 candidate slots (top) + 8 grey backbone slots."""
    for i in range(14):
        y = ytop - (i + 1) * slot_h
        if i < 6:
            fc = C_CL if i in target else C_LIGHT
        else:
            fc = S.BACKBONE
        ax.add_patch(Rectangle((xc - w / 2, y), w, slot_h, fc=fc, ec="white", lw=S.THIN))


def arrow(ax, x0, y0, x1, y1, color=S.INK, lw=S.LW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def panel_3a():
    W, H = 50, 120
    fig = S.new("3A", h=H, layout=None)
    ax = add_ax(fig, 0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    title(fig, 0.6, H - 0.6, "Counting identity")
    tag(fig, W - 0.6, H - 0.6, "Math")
    # cell pool
    ax.add_patch(FancyBboxPatch((8, 106.0), 34, 8.6, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec=S.INK, lw=S.LW))
    ax.text(25, 112.6, "N(G) cells of parent G", ha="center", va="center")
    xs = np.linspace(11, 39, 15)
    for yy, xx in ((109.6, xs), (107.7, xs[:-1] + (xs[1] - xs[0]) / 2)):
        ax.plot(xx, [yy] * len(xx), ls="none", marker="o", ms=1.3, mfc=S.MUTED, mec="none")
    # tube cards
    XC = [12, 19.5, 27, 39.5]
    lab = ["1", "2", "3", "r"]
    tgt = {1: (0, 2, 4), 3: (1, 3, 5)}
    slot_h, cw, ytop = 1.8, 3.4, 98.6
    for i, xc in enumerate(XC):
        arrow(ax, 25, 106.0, xc, ytop + 2.7)
        ax.text(xc, ytop + 1.3, lab[i], ha="center", va="center")
        card(ax, xc, ytop, slot_h, cw, tgt.get(i, ()))
    ax.text(33.25, ytop - 7 * slot_h, "…", ha="center", va="center")
    ax.text(0.6, 103.6, "equal\nsplit", ha="left", va="center", linespacing=1.05)
    ax.text(49.4, 103.6, "N(G)/r\neach", ha="right", va="center", linespacing=1.05)
    ybot = ytop - 14 * slot_h
    # side brackets
    for (i0, i1, txt) in ((0, 6, "candidates"), (6, 14, "backbone")):
        y_hi, y_lo = ytop - i0 * slot_h - 0.25, ytop - i1 * slot_h + 0.25
        ax.plot([8.9, 8.9], [y_lo, y_hi], color=S.INK, lw=S.LW)
        ax.text(7.6, (y_lo + y_hi) / 2, txt, rotation=90, ha="center", va="center")
    # target-triple key
    ax.add_patch(Rectangle((42.2, ytop - 6.3 * slot_h), 1.6, 1.6, fc=C_CL, ec="none"))
    ax.text(44.3, ytop - 6.3 * slot_h + 0.8, "T", ha="left", va="center")
    ax.text(42.2, ytop - 8.6 * slot_h, "λ(T)\n= 2", ha="left", va="center", linespacing=1.05)
    # converge
    for xc in XC:
        ax.plot([xc, 25], [ybot - 0.4, ybot - 3.6], color=S.MUTED, lw=S.LW)
    arrow(ax, 25, ybot - 3.6, 25, ybot - 5.0, color=S.MUTED)
    yb = ybot - 5.0
    ax.add_patch(FancyBboxPatch((2.5, yb - 9.4), 45, 9.4, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=S.INK, lw=S.LW))
    ax.text(25, yb - 1.9, "each cell: k candidates → C(k,3) triples", ha="center", va="center")
    ax.text(25, yb - 4.7, "Σ n(T) = N(G) · C(k,3)", ha="center", va="center", fontweight="bold")
    ax.text(25, yb - 7.4, "sum over all triples T; independent of r", ha="center", va="center")
    yb2 = yb - 10.8
    ax.add_patch(FancyBboxPatch((2.5, yb2 - 6.8), 45, 6.8, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=S.INK, lw=S.LW))
    ax.text(25, yb2 - 2.1, "E[n(T)] = N(G) · λ(T) / r", ha="center", va="center", fontweight="bold")
    ax.text(25, yb2 - 4.8, "λ(T) = number of tubes containing T", ha="center", va="center")
    yc = yb2 - 8.0
    cond = ("Holds when:\n"
            "· targets = all candidate triples\n"
            "· each cell sees exactly k candidates\n"
            "· no count is duplicated\n"
            "· equiprobable (ideal equal) split,\n"
            "  required for E[n(T)]")
    ax.text(2.5, yc, cond, ha="left", va="top", linespacing=1.1)
    # row 1: equal expected share
    y1 = 32.3
    ax.text(0.6, y1, "Equal expected share (λ(T) = λ for all T)", ha="left", va="top")
    a1 = add_ax(fig, 10, 23.1, 37.5, 5.4)
    a1.bar(range(4), [1, 1, 1, 1], width=0.5, color=C_LIGHT, lw=0)
    a1.set_xlim(-0.6, 3.6)
    a1.set_ylim(0, 1.25)
    a1.set_xticks(range(4), labels=[f"T{i + 1}" for i in range(4)])
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_ylabel("E[n(T)]", rotation=0, ha="right", va="center", labelpad=1)
    # row 2: interval widths differ with pattern frequency (illustrative CP half-widths, n = 400)
    ax.text(0.6, 16.9, "Interval widths still differ (all n = 400)\nwith pattern frequency (illustrative)", ha="left", va="top",
            linespacing=1.1)
    n = 400
    ps = [0.5, 0.25, 0.10, 0.03]
    hw = []
    for p in ps:
        k = round(p * n)
        lo = BETA.ppf(0.025, k, n - k + 1)
        hi = BETA.ppf(0.975, k + 1, n - k)
        hw.append((hi - lo) / 2)
    eps_ill = 0.045
    a2 = add_ax(fig, 10, 3.8, 37.5, 6.4)
    a2.axhspan(-eps_ill, eps_ill, color=S.GRID, lw=0, zorder=0)
    for i, (p, h) in enumerate(zip(ps, hw)):
        c = S.ALARM if h > eps_ill else S.INK
        a2.errorbar([i], [0], yerr=[[h], [h]], fmt="none", ecolor=c, elinewidth=S.LW, capsize=1.8, capthick=S.LW)
        a2.text(i + 0.14, 0, f"p {p:.2f}".replace("0.", "."), ha="left", va="center")
    a2.set_xlim(-0.6, 3.9)
    a2.set_ylim(-0.062, 0.062)
    a2.set_xticks([])
    a2.set_yticks([])
    for s in ("left", "bottom"):
        a2.spines[s].set_visible(False)
    a2.text(-0.62, eps_ill * 0.0, "±ε", ha="right", va="center")
    ax.text(0.6, 1.5, "per-pattern 95% CP, marginal; not joint", ha="left", va="center", color=S.MUTED)
    S.save(fig, "3A")
    return dict(hw=hw, ps=ps, n=n, eps_ill=eps_ill)


# =====================================================================================================
# 3D  tube split + per-donor shape
# =====================================================================================================
def panel_3d():
    # composite size (composeFig3.py): 100 x 76 mm, same height as 3E; DT = shift of the top-anchored elements
    W, H = 100, 76
    DT = H - 70
    fig = S.new("3D", w=W, h=H, layout=None)
    ts = rd(os.path.join(DATA, "fig3D_tube_split.tsv"))
    pdn = rd(os.path.join(DATA, "fig3C_per_donor_shape.tsv"))
    assert len(ts) == 20 and (ts.max_abs_count_diff == 0).all()
    assert (ts.half_ok_130 == ts.half_ok_260split).all() and (ts.err_ok_130 == ts.err_ok_260split).all()
    assert (pdn.groupby("parent").size().to_dict() == {"CD4T": 13, "CD8T": 12}) and (pdn.difference > 0).all()
    wder(pd.DataFrame([
        dict(item="tube split: rows (parent x allocation)", value=len(ts)),
        dict(item="tube split: max|dcount| == 0", value=int((ts.max_abs_count_diff == 0).sum())),
        dict(item="tube split: half_ok identical", value=int((ts.half_ok_130 == ts.half_ok_260split).sum())),
        dict(item="tube split: err_ok identical", value=int((ts.err_ok_130 == ts.err_ok_260split).sum())),
        dict(item="per donor: CD4T donors / diff>0", value=f"{(pdn.parent == 'CD4T').sum()}/"
                                                            f"{((pdn.parent == 'CD4T') & (pdn.difference > 0)).sum()}"),
        dict(item="per donor: CD8T donors / diff>0", value=f"{(pdn.parent == 'CD8T').sum()}/"
                                                            f"{((pdn.parent == 'CD8T') & (pdn.difference > 0)).sum()}"),
        dict(item="per donor: difference range (pp)", value=f"{100 * pdn.difference.min():.1f}-{100 * pdn.difference.max():.1f}"),
    ] + [dict(item=f"per donor: {par} delta = min(mean_130, mean_260) - mean_180, range (pp)",
              value=f"{100 * g.difference.min():.1f}-{100 * g.difference.max():.1f}") for par, g in pdn.groupby("parent")]),
        "fig3_D_summary.tsv")
    drng = {par: (100 * g.difference.min(), 100 * g.difference.max()) for par, g in pdn.groupby("parent")}
    assert np.allclose(pdn.difference, pdn.endpoint_min - pdn.node180)
    title(fig, 0.6, H - 0.6, "Split tubes, same evidence")
    tag(fig, 46.5, H - 0.6, "Replay")
    title(fig, 51.0, H - 0.6, "Per-donor shape")
    tag(fig, W - 0.6, H - 0.6, "Exploratory")
    fline(fig, [49.8, 49.8], [2.0, H - 1.0], color=S.GRID)
    # ---- schematic (mm coordinates)
    sx = add_ax(fig, 0, 41.5 + DT, 47.5, 23.5)   # schematic keeps its own 41.5-65 mm data frame
    sx.set_xlim(0, 47.5)
    sx.set_ylim(41.5, 65)
    sx.set_axis_off()
    sh, sw_ = 1.85, 1.08

    def hcard(xc, y):
        x0 = xc - 7 * sw_
        for i in range(14):
            sx.add_patch(Rectangle((x0 + i * sw_, y), sw_, sh, fc=C_LIGHT if i < 6 else S.BACKBONE, ec="white", lw=S.THIN))

    hcard(23.75, 61.6)
    dots = np.linspace(17.5, 30.0, 8)
    sx.plot(dots, [60.3] * 8, ls="none", marker="o", ms=1.6, mfc=S.INK, mec="none")
    sx.text(23.75 - 7 * sw_ - 1.0, 62.5, "tube i", ha="right", va="center")
    sx.text(23.75 + 7 * sw_ + 1.0, 62.5, "of 130", ha="left", va="center")
    for xc in (11.5, 36.0):
        arrow(sx, 23.75, 59.3, xc, 56.6)
    hcard(11.5, 54.4)
    hcard(36.0, 54.4)
    sx.plot(np.linspace(7.5, 15.5, 4), [53.1] * 4, ls="none", marker="o", ms=1.6, mfc=S.INK, mec="none")
    sx.plot(np.linspace(32.0, 40.0, 4), [53.1] * 4, ls="none", marker="o", ms=1.6, mfc=S.INK, mec="none")
    sx.text(11.5, 51.2, "tube i", ha="center", va="center")
    sx.text(36.0, 51.2, "tube i + 130", ha="center", va="center")
    sx.text(23.75, 55.2, "same\npanel", ha="center", va="center", linespacing=1.0)
    for xc in (11.5, 36.0):
        sx.plot([xc, 23.75], [49.8, 48.4], color=S.MUTED, lw=S.LW)
    sx.text(23.75, 47.9, "pooled counts n(T) identical", ha="center", va="top")
    sx.text(23.75, 45.4, "random half of each tube's cells → copy", ha="center", va="top", color=S.MUTED)
    # ---- max |dcount| strip + half_ok overlay
    xpos = np.r_[np.arange(10), 11.5 + np.arange(10)]
    order = ts.sort_values(["parent", "alloc"]).reset_index(drop=True)
    assert list(order.parent[:10]) == ["CD4T"] * 10
    st = add_ax(fig, 10.5, 35.0 + DT, 35.5, 3.2)
    st.set_xlim(-0.9, 21.4)
    st.set_ylim(-0.6, 1.6)
    st.plot(xpos, order.max_abs_count_diff, ls="none", marker="o", ms=MS, mfc=S.INK, mec="white", mew=S.THIN)
    st.set_yticks([0])
    st.set_xticks([])
    st.spines["bottom"].set_visible(False)
    ftext(fig, 1.0, 39.4 + DT, "max |Δcount| (2600 triples × 8 patterns)", ha="left", va="bottom")
    dp = add_ax(fig, 10.5, 16.0, 35.5, 16.5 + DT)
    dp.set_xlim(-0.9, 21.4)
    dp.set_ylim(93.0, 100.6)
    dp.plot(xpos, 100 * order.half_ok_130, ls="none", marker="o", ms=4.2, mfc="white", mec=C_CL, mew=S.LW,
            label="130 tubes")
    dp.plot(xpos, 100 * order.half_ok_260split, ls="none", marker="+", ms=3.6, mec=C_CL, mew=S.LW,
            label="260 tubes (split)")
    dp.set_yticks([96, 98, 100])
    dp.set_xticks([4.5, 16.0], labels=["CD4 T", "CD8 T"])
    dp.tick_params(axis="x", length=0)
    dp.xaxis.set_minor_locator(FixedLocator(xpos))
    dp.tick_params(axis="x", which="minor", length=1.5)
    dp.set_xlabel("Allocation (2 parents × 10)", labelpad=2)
    dp.set_ylabel("Half-width ≤ ε (%)")
    dp.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.3, borderaxespad=0.2)
    e4, e8 = EPS[("wingmen_v1", "CD4T")], EPS[("wingmen_v1", "CD8T")]
    ftext(fig, 0.6, 7.6, f"WINGMEN · CD4 T ε {e4:g} · CD8 T ε {e8:g}", ha="left", va="center")
    ftext(fig, 0.6, 4.9, "per-pattern 95% CP, marginal; not joint", ha="left", va="center")
    # ---- per-donor dumbbells
    dd = add_ax(fig, 63.0, 13.0, 35.5, 41.5 + DT)
    ys, yl = [], []
    y = 0
    for par in ("CD4T", "CD8T"):
        g = pdn[pdn.parent == par].sort_values("difference", ascending=False)
        yy = np.arange(len(g)) + y
        dd.hlines(yy, 100 * g.node180, 100 * g.endpoint_min, color=C_LIGHT, lw=S.LW, zorder=1)
        dd.plot(100 * g.endpoint_min, yy, ls="none", marker="o", ms=MS + 0.3, mfc=C_CL, mec="white", mew=S.THIN,
                zorder=3)
        dd.plot(100 * g.node180, yy, ls="none", marker="o", ms=MS + 0.3, mfc="white", mec=C_CL, mew=S.THIN, zorder=3)
        ys.append(yy.mean())
        yl.append(f"{dname(par)}\n{len(g)} donors")
        y += len(g) + 1.5
    dd.set_ylim(y - 1.5 + 0.2, -0.8)
    dd.set_yticks(ys, labels=yl)
    dd.tick_params(axis="y", length=0)
    dd.set_xlim(54, 101)
    dd.set_xticks([60, 70, 80, 90, 100])
    dd.set_xlabel("Half-width ≤ ε(d) (%)", labelpad=2)
    hs = [Line2D([], [], ls="none", marker="o", ms=MS + 0.3, mfc=C_CL, mec="white", mew=S.THIN),
          Line2D([], [], ls="none", marker="o", ms=MS + 0.3, mfc="white", mec=C_CL, mew=S.THIN)]
    dd.legend(hs, ["min(130, 260 tubes)", "180 tubes"], loc="lower left", bbox_to_anchor=(-12.0 / 35.5, 1.0),
              ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.3)
    n4 = int(((pdn.parent == "CD4T") & (pdn.difference > 0)).sum())
    n8 = int(((pdn.parent == "CD8T") & (pdn.difference > 0)).sum())
    assert (n4, n8) == (13, 12)
    ftext(fig, 51.0, 63.2 + DT, "Δ = min(130, 260) − 180 > 0 for every donor", ha="left", va="center")
    ftext(fig, 51.0, 60.5 + DT, f"CD4 T Δ {drng['CD4T'][0]:.1f}–{drng['CD4T'][1]:.1f} · "
                           f"CD8 T Δ {drng['CD8T'][0]:.1f}–{drng['CD8T'][1]:.1f} pp", ha="left", va="center")
    ftext(fig, 51.0, 2.6, "Exploratory · 180 not 195 · donor-specific ε", ha="left", va="center")
    S.save(fig, "3D")


# =====================================================================================================
# 3E  combinatorial balance must become cell balance (Simulation, Limits)
# =====================================================================================================
def panel_3e(reps, sup):
    # composite size (composeFig3.py): 82 x 76 mm, same height as 3D
    W, H = 82, 76
    fig = S.new("3E", w=W, h=H, layout=None)
    cols = (("levine32_v2", "Levine-32 T / 71 tubes", 0.06, 12.5), ("wingmen_v1", "WINGMEN CD4 T / 130 tubes", 0.05, 49.0))
    cw = 31.0
    R1, R2, R3 = (50.0, 18.0), (34.0, 14.0), (17.5, 14.5)
    # min_T n(T) (classical only): per-replicate minimum, then mean over the 10 replicates of each sigma
    mc = (reps[reps.method == "classical"].groupby(["dataset", "sigma"])
          .agg(tube_size_cv=("tube_size_cv", "mean"), min_cells_mean=("min_cells", "mean"),
               min_cells_lo=("min_cells", "min"), min_cells_hi=("min_cells", "max"), n=("min_cells", "size"))
          .reset_index())
    assert (reps[reps.method == "random"].min_cells == 0).all()
    chk = {(d, s): round(v, 1) for d, s, v in zip(mc.dataset, mc.sigma, mc.min_cells_mean)}
    assert (chk[("levine32_v2", 0.0)], chk[("levine32_v2", 0.5)]) == (267.7, 68.8)
    assert (chk[("wingmen_v1", 0.0)], chk[("wingmen_v1", 0.5)]) == (396.7, 102.6)
    mc["source"] = "results/<ds>/e5k_boundary.tsv part A, classical, min_cells (= min over triples of n(T))"
    wder(mc, "fig3_E_mincells.tsv")
    ann = []
    for ds, head, eps, x0 in cols:
        assert EPS[(ds, "T" if ds == "levine32_v2" else "CD4T")] == eps
        for row, (y0, h), col, yl, yt in (("half", R1, "half_ok", (38, 102), [40, 60, 80, 100]),
                                          ("err", R2, "err_ok", (55, 101), [60, 80, 100]),
                                          ("min", R3, "min_cells", (0, 445), [0, 200, 400])):
            a = add_ax(fig, x0, y0, cw, h)
            a.axvspan(0.25, 0.5, color=S.GRID, lw=0, zorder=0)
            sc = 1 if row == "min" else 100
            for meth in (("classical",) if row == "min" else ("random", "classical")):
                c, mk = S.METHOD[meth], S.MARKER[meth]
                rr = reps[(reps.dataset == ds) & (reps.method == meth)]
                a.plot(rr.tube_size_cv, sc * rr[col], ls="none", marker=mk, ms=1.7, mfc=c, mec="none", alpha=0.45,
                       zorder=2)
                if row == "min":
                    s = mc[mc.dataset == ds].sort_values("sigma")
                    a.plot(s.tube_size_cv, s.min_cells_mean, color=c, marker=mk, ms=3.4, mfc=c, mec="white",
                           mew=S.THIN, zorder=4)
                else:
                    s = sup[(sup.dataset == ds) & (sup.method == meth)].sort_values("sigma")
                    a.plot(s.tube_size_cv, 100 * s[col], color=c, marker=mk, ms=3.4, mfc=c, mec="white", mew=S.THIN,
                           zorder=4)
            a.set_xlim(-0.03, 1.56)
            a.set_ylim(*yl)
            a.set_yticks(yt)
            a.set_xticks([0, 0.5, 1.0, 1.5], labels=["0", "0.5", "1", "1.5"])
            if row != "min":
                a.tick_params(labelbottom=False)
            if row == "half":
                a.set_title(head, loc="center", fontweight="normal", pad=2)
                a.text(0.02, 41, f"ε = {eps:g}", ha="left", va="bottom")
            if x0 > 20:
                a.tick_params(labelleft=False)
            if row == "min":
                s = mc[mc.dataset == ds].set_index("sigma")
                v0, v5 = s.loc[0.0], s.loc[0.5]
                a.text(0.12, v0.min_cells_mean, f"{v0.min_cells_mean:.1f}", ha="left", va="center")
                a.annotate(f"{v5.min_cells_mean:.1f}", xy=(v5.tube_size_cv, v5.min_cells_mean), xytext=(1.0, 230),
                           ha="center", va="center",
                           arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.MUTED, shrinkA=1.5, shrinkB=2.0))
                if x0 < 20:
                    a.text(1.53, 440, "Minimum direct cells\nper triple", ha="right", va="top", linespacing=1.05)
                continue
            q = sup[(sup.dataset == ds) & (sup.method == "classical") & (sup.sigma == 0.5)].iloc[0]
            v = float(q[col])
            ann.append(dict(dataset=ds, metric=col, sigma=0.5, tube_size_cv=float(q.tube_size_cv), classical=v,
                            random=float(sup[(sup.dataset == ds) & (sup.method == "random") & (sup.sigma == 0.5)][col].iloc[0])))
            if row == "half":
                a.annotate(pct(v), xy=(q.tube_size_cv, 100 * v), xytext=(1.0, 88), ha="center", va="center",
                           arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.MUTED, shrinkA=1.5, shrinkB=2.0))
            else:
                a.text(q.tube_size_cv, 100 * v - 6.0, f"{100 * v:.1f}%", ha="center", va="top")
    wder(pd.DataFrame(ann), "fig3_E_annotations.tsv")
    ftext(fig, 3.6, R1[0] + R1[1] / 2, "Half-width\n≤ ε (%)", rotation=90, ha="center", va="center",
          multialignment="center", linespacing=1.05)
    ftext(fig, 3.6, R2[0] + R2[1] / 2, "Reference error\n≤ ε (%)", rotation=90, ha="center", va="center",
          multialignment="center", linespacing=1.05)
    ftext(fig, 3.6, R3[0] + R3[1] / 2, "min n(T)\n(cells)", rotation=90, ha="center", va="center",
          multialignment="center", linespacing=1.05)
    ftext(fig, 12.5 + (cols[1][3] + cw - 12.5) / 2, 11.6, "Measured tube-size CV", ha="center", va="center")
    hs = [Line2D([], [], color=C_CL, marker=S.MARKER["classical"], ms=3.4, mec="white", mew=S.THIN),
          Line2D([], [], color=C_RD, marker=S.MARKER["random"], ms=3.4, mec="white", mew=S.THIN),
          Line2D([], [], ls="none", marker="o", ms=1.7, mfc=C_CL, mec="none", alpha=0.45),
          Patch(fc=S.GRID, lw=0)]
    fig.legend(hs, ["Classical", "Random", "Replicate", "CV 0.25–0.5"], loc="center", bbox_to_anchor=(0.5, 8.1 / H),
               ncol=4, frameon=False, columnspacing=0.9)
    ftext(fig, W / 2, 4.6, "Phenotype-independent perturbation", ha="center", va="center")
    ftext(fig, W / 2, 2.0, "Half-width: per-pattern 95% CP, marginal; not joint", ha="center", va="center")
    S.limits_tag(fig)
    ftext(fig, 11.0, H - 0.6, "Balance must reach the cells", ha="left", va="top", fontweight="bold")
    tag(fig, W - 0.6, H - 0.6, "Simulation")
    S.save(fig, "3E")


if __name__ == "__main__":
    thr = derive_3b()
    m, lv = derive_3c()
    reps, sup = derive_3e()
    geo = panel_3b(thr)
    panel_3c(m, lv, geo)
    info = panel_3a()
    panel_3d()
    panel_3e(reps, sup)
    print("3A illustrative CP half-widths (n=%d):" % info["n"], dict(zip(info["ps"], np.round(info["hw"], 4))))
    print(m.to_string())
