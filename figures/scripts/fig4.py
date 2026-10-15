"""Figure 4 (figure plan v2) -- 'the same table audits evidence and checks itself': single panels 4A-4E.

Read-only inputs : results/<protocol>/initial_audit.tsv, results/*/e5k_boundary.tsv, results/cytonorm_real_batch/,
                   figures/data/{supp_calibration,fig4B_qc_fixed_family,supp_boundary_composition}.tsv,
                   figures/derived/d3b_pair_tubes.tsv (deterministic restore of the frozen e5b run, see derive/d3b_pair_tubes.py)
Display tables   : figures/derived/fig4_*.tsv   (deterministic summaries only: counts, max, median, sorting)
No new experiment, simulation, randomisation, test statistic or p-value is computed here.
Run from figures/scripts:  python fig4.py  [4A 4B ...]
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.transforms as mtrans
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import wfstyle as S

S.setup()
import wfgeom as G  # noqa: E402

RES, DATA, DER = S.RESULTS, S.DATA, S.DERIVED
TAU = 1e-9            # evidence-tier rule (figure plan v2, section 1)
INTOL = 1e-9          # tolerance of 'reference inside interval' (as in src/compare_shift.py)
WHITE = "#FFFFFF"


def tint(c, f):
    """Blend a wfstyle colour towards white (f = 1 -> colour itself, f = 0 -> white)."""
    r = np.array(to_rgb(c))
    return tuple(1 - f * (1 - r))


def lum(c):
    r, g, b = to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def dname(s):
    """Display form of parent names in drawn text (source keys CD4T / CD8T unchanged): 'CD4 T', 'CD8 T'."""
    return str(s).replace("CD4T", "CD4 T").replace("CD8T", "CD8 T")


class Panel:
    """Single panel with a full-figure overlay axis in millimetre coordinates (no constrained layout, no tight bbox)."""

    def __init__(self, key, h, w=None):
        """w: composite width (composeFig4.py fills each row to 184 mm); default = wfstyle width."""
        W0, H0 = S.SIZES[key]
        assert abs(h - H0) <= 0.2 * H0 + 1e-9, (key, h)
        self.key, self.W, self.H = key, (w or W0), h
        self.fig = S.new(key, w=self.W, h=h, layout=None)
        self.ov = self.fig.add_axes([0, 0, 1, 1])
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(0, self.H)
        self.ov.axis("off")

    def ax(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])

    def text(self, x, y, s, **kw):
        return self.ov.text(x, y, s, **kw)

    def rect(self, x, y, w, h, **kw):
        kw.setdefault("lw", 0)
        r = Rectangle((x, y), w, h, **kw)
        self.ov.add_patch(r)
        return r

    def head(self, tag, title=None, limits=False, title_line=1):
        y = self.H - 0.8
        if limits:
            S.limits_tag(self.fig, x=0.9 / self.W, y=(y - 0.3) / self.H, va="top")
        S.tier_tag(self.fig, tag, x=(self.W - 0.8) / self.W, y=y / self.H, va="top")
        if title:
            ty = y if title_line == 1 else y - 3.4
            self.text(0.8, ty, title, ha="left", va="top", fontweight="bold")

    def save(self):
        return S.save(self.fig, self.key)


def write(df, name):
    df.to_csv(os.path.join(DER, name), sep="\t", index=False)


def load_audit(d):
    return pd.read_csv(os.path.join(RES, d, "initial_audit.tsv"), sep="\t", keep_default_na=False, na_values=[""])


def evidence_tier(A):
    """Figure plan v2 four-tier rule, evaluated top-down."""
    direct = A.n_direct.fillna(0) > 0
    ok = A.lp_status.astype(str) == "ok"
    shrink = (A.conf_lo > TAU) | (A.conf_hi < 1 - TAU)
    return np.where(direct, 1, np.where(ok & shrink, 2, np.where(A.knn_block.notna(), 3, 4)))


# =====================================================================================================  4A
A4 = [("wingmen_v1", "WINGMEN"), ("levine32_v2", "Levine-32"), ("levine32_v2_rareB_1", "rareB 1%"),
      ("levine32_v2_rareB_0p3", "rareB 0.3%"), ("levine32_v2_rareB_0p1", "rareB 0.1%")]
ERR_BOUNDS_PP = [0, 1, 2, 3, 5, 10, 18]          # colour classes of |kNN - ref| in percentage points
ERR_COLS = [S.SEQ(f) for f in (0.03, 0.2, 0.36, 0.52, 0.78, 1.0)]


def data_4A():
    rows = []
    for d, lab in A4:
        A = load_audit(d)
        proto = json.load(open(os.path.join(S.WF, "protocols", f"{d}.json"), encoding="utf-8"))
        claims = [c["id"] for c in proto["claims"]]    # protocol order (grouped by parent)
        assert set(A.claim) == set(claims)
        A = A.assign(dataset=d, label=lab, tier=evidence_tier(A), corder=A.claim.map({c: i for i, c in enumerate(claims)}),
                     err=(A.knn_block - A.ref).abs(), width=A.conf_hi - A.conf_lo,
                     ref_in=(A.ref >= A.conf_lo - INTOL) & (A.ref <= A.conf_hi + INTOL),
                     ident_in_conf=(A.ident_lo >= A.conf_lo - INTOL) & (A.ident_hi <= A.conf_hi + INTOL))
        rows.append(A[["dataset", "label", "claim", "corder", "parent", "markers", "rep", "tier", "n_direct", "lp_status",
                       "conf_lo", "conf_hi", "ident_lo", "ident_hi", "ref", "knn_block", "maxent", "err", "width", "ref_in",
                       "ident_in_conf"]])
    M = pd.concat(rows, ignore_index=True)
    assert (M.tier <= 2).all() and (M.lp_status == "ok").all() and M.ident_in_conf.all()
    summ = M.groupby(["dataset", "label"], sort=False).agg(
        n=("rep", "size"), ref_in=("ref_in", "sum"), knn_gt_5pp=("err", lambda e: int((e > 0.05).sum())),
        max_err=("err", "max"), tier1=("tier", lambda t: int((t == 1).sum())), tier2=("tier", lambda t: int((t == 2).sum()))
    ).reset_index()
    s = summ.set_index("dataset")
    # key numbers of the plan / claims table C9 (asserted against the frozen tables)
    assert (s.loc["wingmen_v1", "ref_in"], s.loc["wingmen_v1", "n"]) == (150, 150)
    assert (s.loc["levine32_v2", "ref_in"], s.loc["levine32_v2", "n"]) == (120, 120)
    assert s.loc["wingmen_v1", "knn_gt_5pp"] == 9
    assert round(s.loc[[d for d, _ in A4[2:]], "max_err"].max() * 100, 2) == 17.93
    assert M.err.max() * 100 < ERR_BOUNDS_PP[-1]
    # ---- instances, fixed rules (written before looking at the plot):
    #  R1 'median width'      : records without direct evidence (n_direct = 0) of the two main protocols, sorted by
    #                           confidence-audit width (ties: dataset order, claim order, rep); element (n-1)//2.
    #  R2 'largest kNN error' : same pool, max |kNN - ref|.
    #  R3 'largest kNN error, rareB': all records of the three rareB replays, max |kNN - ref|.
    main = M[M.dataset.isin(["levine32_v2", "wingmen_v1"]) & (M.tier == 2)].copy()
    main["dorder"] = main.dataset.map({d: i for i, (d, _) in enumerate(A4)})
    srt = main.sort_values(["width", "dorder", "corder", "rep"], kind="mergesort")
    r1 = srt.iloc[(len(srt) - 1) // 2]
    r2 = main.loc[main.err.idxmax()]
    rb = M[M.dataset.str.contains("rareB")]
    r3 = rb.loc[rb.err.idxmax()]
    inst = pd.DataFrame([r1, r2, r3]).assign(rule=["median width (main protocols, n_direct = 0)",
                                                   "largest |kNN - ref| (main protocols)",
                                                   "largest |kNN - ref| (rareB replays)"])
    assert (inst.tier == 2).all()
    inst["rep_display"] = inst.rep + 1
    write(M.assign(rep_display=M.rep + 1).drop(columns="corder"), "fig4_A_matrix.tsv")
    write(summ, "fig4_A_summary.tsv")
    write(inst.drop(columns="corder"), "fig4_A_instances.tsv")
    return M, summ, inst


def err_colour(e_pp):
    for i in range(len(ERR_BOUNDS_PP) - 1):
        if e_pp < ERR_BOUNDS_PP[i + 1]:
            return ERR_COLS[i]
    return ERR_COLS[-1]


def fig_4A():
    M, summ, inst = data_4A()
    P = Panel("4A", 60, w=184)          # composite: full content width
    P.head("Replay (same-source)", "Intervals and point estimates report different things")
    cw, ch, gg, top = 1.8, 2.45, 0.55, 46.0
    s = summ.set_index("dataset")

    def matrix(d, x0, row_labels):
        A = M[M.dataset == d]
        claims = A.drop_duplicates("claim").sort_values("corder")[["claim", "parent"]].values.tolist()
        y, prev, ys = top, None, {}
        for c, p in claims:
            if prev is not None and p != prev:
                y -= gg
            ys[c] = y - ch
            if row_labels:
                P.text(x0 - 0.6, y - ch / 2, c, ha="right", va="center")
                if p != prev:
                    P.text(x0 - 4.4, y - ch / 2, dname(p), ha="right", va="center", color=S.MUTED)
            y -= ch
            prev = p
        for _, r in A.iterrows():
            x, yb = x0 + r.rep * cw, ys[r.claim]
            fc = err_colour(100 * r.err)
            P.rect(x, yb, cw, ch, fc=fc, ec=WHITE, lw=S.THIN)
            if r.tier == 1:
                P.rect(x + 0.32, yb + 0.32, cw - 0.64, ch - 0.64, fill=False, ec=S.INK, lw=S.LW)
            if r.ref_in:
                P.ov.plot(x + cw / 2, yb + ch / 2, "o", ms=1.1, mew=0, color=WHITE if lum(fc) < 0.45 else S.INK)
        xc = x0 + 5 * cw
        P.text(xc, top + 5.9, s.loc[d, "label"], ha="center", va="bottom")
        P.text(xc, top + 3.0, f"ref in {s.loc[d, 'ref_in']}/{s.loc[d, 'n']}", ha="center", va="bottom")
        P.text(xc, top + 0.3, f"kNN >5 pp: {s.loc[d, 'knn_gt_5pp']}", ha="center", va="bottom")
        return min(ys.values())

    # WINGMEN block, then Levine-32 + its three rareB replays sharing the claim rows
    xw = 11.5
    yb_w = matrix("wingmen_v1", xw, True)
    xl = [43.0 + i * (10 * cw + 1.5) for i in range(4)]
    yb_l = None
    for i, (d, _) in enumerate(A4[1:]):
        yb_l = matrix(d, xl[i], i == 0)
    for x0, yb in ((xw, yb_w), (xl[0], yb_l)):
        P.text(x0 + cw / 2, yb - 0.5, "1", ha="center", va="top")
        P.text(x0 + 9.5 * cw, yb - 0.5, "10", ha="center", va="top")
        P.text(x0 + 5 * cw, yb - 0.5, "replicate", ha="center", va="top", color=S.MUTED)
    # binned colour key; class boundary 5 pp = the +0.05 reference line
    kx, ky, kw, kh = 63.0, 6.6, 33.0, 2.0
    nb = len(ERR_COLS)
    for i, c in enumerate(ERR_COLS):
        P.rect(kx + i * kw / nb, ky, kw / nb, kh, fc=c, ec=WHITE, lw=S.THIN)
    for i, b in enumerate(ERR_BOUNDS_PP):
        P.text(kx + i * kw / nb, ky - 0.5, str(b), ha="center", va="top")
    x5 = kx + ERR_BOUNDS_PP.index(5) * kw / nb
    P.ov.plot([x5, x5], [ky - 0.6, ky + kh + 0.6], color=S.INK, lw=S.LW, solid_capstyle="butt")
    P.text(kx - 1.2, ky + kh / 2, "|kNN − ref| (pp)", ha="right", va="center")
    lx = 100.0
    P.ov.plot(lx + 0.9, 9.4, "o", ms=1.6, mew=0, color=S.INK)
    P.text(lx + 2.4, 9.4, "ref in interval", va="center")
    P.rect(lx + 0.2, 5.6, 1.4, 1.9, fill=False, ec=S.INK, lw=S.LW)
    P.text(lx + 2.4, 6.55, "tier I (direct)", va="center")

    # ---- instances
    ax = P.ax(126.5, 11.0, P.W - 129.0, 30.5)
    xmax = float(np.ceil(inst[["conf_hi", "knn_block", "maxent", "ref"]].values.max() * 10) / 10)
    for k, (_, r) in enumerate(inst.iterrows()):
        yc = 2 - k
        ax.add_patch(Rectangle((r.conf_lo, yc - 0.13), r.conf_hi - r.conf_lo, 0.26, fc=S.TIER[3], lw=0))
        ax.add_patch(Rectangle((r.ident_lo, yc - 0.065), r.ident_hi - r.ident_lo, 0.13, fc=S.TIER[2], lw=0))
        ax.plot([r.ref, r.ref], [yc - 0.21, yc + 0.21], color=S.INK, lw=S.LW, solid_capstyle="butt")
        ax.plot(r.maxent, yc, "o", ms=3.4, mfc=WHITE, mec=S.INK, mew=S.THIN)
        ax.plot(r.knn_block, yc, "D", ms=3.0, mfc=S.INK, mec=S.INK, mew=0)
        rule = {0: "median width, main", 1: "largest kNN error, main", 2: "largest kNN error, rareB"}[k]
        ds = r.label if k < 2 else r.label.replace("rareB ", "rareB ")
        ax.text(0, yc + 0.47, rule, ha="left", va="bottom", color=S.MUTED)
        ax.text(0, yc + 0.215, f"{ds} {r.claim} ({dname(r.parent)}), rep {r.rep + 1} · |kNN − ref| {100 * r.err:.1f} pp",
                ha="left", va="bottom")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.4, 2.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.arange(0, xmax + 1e-9, 0.2))
    ax.set_xticklabels([f"{t:.1f}".rstrip("0").rstrip(".") if t > 0 else "0" for t in np.arange(0, xmax + 1e-9, 0.2)])
    ax.set_xlabel("pattern frequency in parent")
    h = [Patch(fc=S.TIER[3], lw=0, label="confidence audit interval"),
         Patch(fc=S.TIER[2], lw=0, label="identification interval"),
         Line2D([], [], color=S.INK, marker="|", ms=6, mew=S.LW, ls="none", label="reference split frequency"),
         Line2D([], [], color=S.INK, marker="D", ms=3.0, mew=0, ls="none", label="kNN (block)"),
         Line2D([], [], marker="o", ms=3.4, mfc=WHITE, mec=S.INK, mew=S.THIN, ls="none", label="EM fit")]
    P.fig.legend(handles=h, loc="upper left", bbox_to_anchor=(124.5 / P.W, 55.5 / P.H), ncol=2, handlelength=1.2,
                 columnspacing=1.0, borderaxespad=0, borderpad=0)
    P.text(126.5, 1.0, "instances: fixed rules, records with n_direct = 0", ha="left", va="bottom", color=S.MUTED)
    return P.save(), M, summ, inst


# =====================================================================================================  4B
def tube_cards():
    """Observation tubes of each synthetic evidence structure, read from src/e5h_calib.py (markers 0,1,2 = A,B,C)."""
    sys.path.insert(0, os.path.join(S.WF, "src"))
    from e5h_calib import STRUCT
    cards = {s: ["".join("ABC"[i] for i in sorted(t)) for t in tubes] for s, tubes in STRUCT.items()}
    assert cards == {"direct_only": ["ABC"], "pairs+direct": ["AB", "AC", "BC", "ABC"], "pairs": ["AB", "AC", "BC"],
                     "mixed": ["AB", "C"], "singles": ["A", "B", "C"]}, cards
    return cards


def fig_4B():
    cal = pd.read_csv(os.path.join(DATA, "supp_calibration.tsv"), sep="\t")
    assert len(cal) == 40 and cal.structure.nunique() == 5 and (cal.infeasible == 0).all() and (cal.n_sims == 1000).all()
    lo = cal.loc[cal.coverage.idxmin()]
    assert (round(lo.coverage, 3), round(lo.coverage_ci_lo, 4), round(lo.coverage_ci_hi, 4)) == (0.992, 0.9843, 0.9965)
    order = ["direct_only", "pairs+direct", "pairs", "mixed", "singles"]
    glab = {"direct_only": "direct\nonly", "pairs+direct": "pairs +\ndirect", "pairs": "pairs", "mixed": "mixed\n(AB, C)",
            "singles": "singles"}
    tg = ["zero", "sparse", "common", "half"]
    pitch = 9.0
    cal["x"] = [order.index(s) * pitch + tg.index(t) * 2.0 + (-0.42 if n == 600 else 0.42)
                for s, t, n in zip(cal.structure, cal.target, cal.n_cells)]
    lo = cal.loc[cal.coverage.idxmin()]
    cards = tube_cards()
    # composite size (composeFig4.py): 82 x 70 mm, same height as 4C; DT = shift of the top-anchored block
    P = Panel("4B", 70, w=82)
    DT = P.H - 60
    P.head("Simulation", "Interval calibration in synthetic 3-marker scenarios")
    P.text(0.8, 56.3 + DT, "K = 1 · common distribution · random split · fixed-seed version", va="top", color=S.MUTED)
    P.text(0.8, 53.4 + DT, "fixed total 600 / 6000 cells split across these tubes", va="top")
    AX0, AXW = 13.0, P.W - 15.0
    xlo, xhi = -1.3, (len(order) - 1) * pitch + 6 + 1.3
    # extra height split 4 : 6 between coverage (0.95-0.98 stays empty by design) and mean width (spread data)
    a1 = P.ax(AX0, 22.0 + 0.6 * DT, AXW, 20.6 + 0.4 * DT)
    A2H = 9.8 + 0.6 * DT
    a2 = P.ax(AX0, 9.6, AXW, A2H)
    for ax in (a1, a2):
        ax.set_xlim(xlo, xhi)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    # ---- observation tube cards, one separate box per tube of each structure (from src/e5h_calib.py STRUCT);
    #      rows are filled greedily up to the group width, so 3-4 tubes wrap onto a second row
    rend = P.fig.canvas.get_renderer()

    def text_mm(s):
        t = P.ov.text(0, 0, s)
        w = t.get_window_extent(renderer=rend).width / P.fig.dpi * 25.4
        t.remove()
        return w

    ch, gap, padx, row_top, max_row = 2.9, 0.55, 0.55, 50.3 + DT, 11.6
    P.text(AX0 - 1.6, row_top - ch / 2, "tubes", ha="right", va="center", color=S.MUTED)
    for g, sname in enumerate(order):
        xc = AX0 + (g * pitch + 3.0 - xlo) / (xhi - xlo) * AXW
        widths = [text_mm(c) + 2 * padx for c in cards[sname]]
        rows, cur = [], []
        for c, w in zip(cards[sname], widths):
            if cur and sum(w_ for _, w_ in cur) + gap * len(cur) + w > max_row:
                rows.append(cur)
                cur = []
            cur.append((c, w))
        rows.append(cur)
        for k, row in enumerate(rows):
            tot = sum(w for _, w in row) + gap * (len(row) - 1)
            x = xc - tot / 2
            y = row_top - (k + 1) * ch - k * 0.5
            for c, w in row:
                P.ov.add_patch(FancyBboxPatch((x, y), w, ch, boxstyle="round,pad=0,rounding_size=0.45", fc=S.GRID,
                                              lw=0))
                P.text(x + w / 2, y + ch / 2, c, ha="center", va="center")
                x += w + gap
    a1.axhline(0.95, color=S.MUTED, lw=S.THIN, ls=(0, (4, 2)), zorder=0)
    for _, r in cal.iterrows():
        a1.plot([r.x, r.x], [r.coverage_ci_lo, r.coverage_ci_hi], color=S.INK, lw=S.LW, solid_capstyle="butt", zorder=2)
        for ax, yv in ((a1, r.coverage), (a2, r.mean_width)):
            if r.n_cells == 600:
                ax.plot(r.x, yv, "o", ms=3.2, mfc=WHITE, mec=S.INK, mew=S.LW, zorder=3)
            else:
                ax.plot(r.x, yv, "o", ms=3.4, mfc=S.INK, mec=S.INK, mew=0, zorder=3)
    a1.set_ylim(0.941, 1.0035)
    a1.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    a1.set_yticklabels(["0.95", "0.96", "0.97", "0.98", "0.99", "1.00"])
    a1.set_ylabel("coverage")
    a1.text(-1.0, 0.9493, "nominal 0.95", va="top", ha="left", color=S.MUTED)
    # lowest scenario: one line just right of the lower end of its own interval; key + CI note sit lower right
    a1.text(lo.x + 0.6, lo.coverage_ci_lo - 0.0011, f"lowest {lo.coverage:.3f} [{lo.coverage_ci_lo:.4f}, {lo.coverage_ci_hi:.4f}]",
            ha="left", va="center")
    tr = mtrans.blended_transform_factory(a1.transAxes, a1.transData)
    h = [Line2D([], [], marker="o", ms=3.2, mfc=WHITE, mec=S.INK, mew=S.LW, ls="none", label="600 cells"),
         Line2D([], [], marker="o", ms=3.4, mfc=S.INK, mew=0, ls="none", label="6000 cells")]
    a1.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, (0.9745 - 0.941) / (1.0035 - 0.941)), ncol=2,
              borderaxespad=0, handletextpad=0.2, columnspacing=0.9)
    a1.text(0.99, 0.9632, "binomial 95% CI · 1000 runs · infeasible = 0", transform=tr, ha="right", va="top",
            color=S.MUTED)
    a2.set_ylim(0, 0.78)
    a2.set_yticks([0, 0.25, 0.5, 0.75])
    a2.set_yticklabels(["0", "0.25", "0.50", "0.75"])
    a2.set_ylabel("mean width")
    tb = mtrans.blended_transform_factory(a2.transData, a2.transAxes)
    for g, sname in enumerate(order):
        a2.text(g * pitch + 3.0, -1.18 / A2H, glab[sname], transform=tb, ha="center", va="top", linespacing=1.0)
    P.text(0.8, 0.8, "within each structure, left to right: target frequency 0, 0.01, 0.2, 0.5", va="bottom",
           color=S.MUTED)
    return P.save(), cal


# =====================================================================================================  4C
def data_4C():
    v = G.ag_view(G.POINT["CCR7"])
    C, Q = v["coords"], G.POINT["CD27"]
    qa, qb = C[Q]
    thru = [ln for ln in v["lines"] if Q in ln["points"]]
    assert [ln["block"] for ln in thru] == [8, 34, 54, 63, 70, 72] and [ln["slope"] for ln in thru] == [2, 4, G.INF, 3, 0, 1]
    disp = {}  # re-centred fundamental domain (Q at the centre); coordinates are still read mod 5
    for p, z in C.items():
        if z != G.INF:
            disp[p] = (((z[0] - qa + 2) % 5) - 2, ((z[1] - qb + 2) % 5) - 2)
    others = set(range(26)) - {G.POINT["CCR7"], Q}
    part = [set(ln["points"]) - {Q} for ln in thru]
    assert all(len(s_) == 4 for s_ in part) and set().union(*part) == others and sum(map(len, part)) == 24  # 6 x 4
    pt = pd.read_csv(os.path.join(DER, "d3b_pair_tubes.tsv"), sep="\t")
    tubes = pt[pt.tube >= 0].set_index("tube")
    assert sorted(tubes.index) == sorted(ln["block"] for ln in thru)
    rows = []
    for ln in thru:
        oth = sorted(G.LABELS[p] for p in ln["points"] if p != Q)
        assert set(oth) == set(tubes.loc[ln["block"], "others"].split("/")), (ln["block"], oth)
        t = tubes.loc[ln["block"]]
        n = int(t.n)
        assert n == int(t.s00 + t.s01 + t.s10 + t.s11)
        sm = ln["slope"] if ln["slope"] == G.INF or ln["slope"] <= 2 else ln["slope"] - 5
        rows.append(dict(tube0=ln["block"], tube=ln["block"] + 1, slope=ln["slope"], slope_repr=sm,
                         others="/".join(tubes.loc[ln["block"], "others"].split("/")), n=n,
                         **{f"f{b}": t[f"s{b}"] / n for b in ("00", "01", "10", "11")},
                         **{f"k{b}": int(t[f"s{b}"]) for b in ("00", "01", "10", "11")},
                         points_ag=";".join(f"{G.LABELS[p]}@a{C[p][0]}b{C[p][1]}" for p in ln["points"])))
    ref = pt[pt.tube == -1].iloc[0]
    rows.append(dict(tube0=-1, tube=-1, slope="", slope_repr="", others="reference split", n=int(ref.n),
                     **{f"f{b}": float(ref[f"s{b}"]) for b in ("00", "01", "10", "11")}))
    D = pd.DataFrame(rows)
    write(D, "fig4_C_pair_tubes.tsv")
    return v, C, Q, (qa, qb), disp, thru, D


def wrap(y):
    return ((y + 2.5) % 5) - 2.5


def torus_steps(pts, sm):
    """Point-to-point steps of one line in the centred window (same convention as the Fig. 2B Torus class):
    a step that wraps mod 5 is split at the window edge; both halves are returned with wrapped=True (dashed)."""
    if sm == G.INF:
        pts = sorted(pts, key=lambda t: t[1])
        return [((a, b), False) for a, b in zip(pts, pts[1:])]
    pts = sorted(pts)
    out = []
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        assert xb == xa + 1
        lift = ya + sm
        if lift == yb:
            out.append((((xa, ya), (xb, yb)), False))
        else:
            assert abs(lift - yb) == 5
            edge = 2.5 if lift > 2 else -2.5
            t = (edge - ya) / sm
            out.append((((xa, ya), (xa + t, edge)), True))
            out.append((((xa + t, -edge), (xb, yb)), True))
    return out


# tube-number labels at the line ends: same anchor positions as the slope labels of Fig. 2B (centred window)
LINE_LABEL_AT = {3: (-1.25, 2.62, "center", "bottom"), G.INF: (0.0, 2.62, "center", "bottom"),
                 2: (1.25, 2.62, "center", "bottom"), 1: (2.58, 2.58, "left", "bottom"),
                 0: (2.62, 0.0, "left", "center"), 4: (2.62, -2.0, "left", "center")}


def fig_4C():
    v, C, Q, (qa, qb), disp, thru, D = data_4C()
    assert (qa, qb) == (2, 0)  # same window as the Fig. 2B Torus (x = a, Q in the middle row)
    P = Panel("4C", 70)
    P.head("Math + Replay (descriptive)", "Redundancy as a self-check: the six-line star")
    P.text(0.8, 65.4, "P = CCR7 sent to ∞; Q = CD27", va="top")
    P.text(0.8, 62.5, "6 lines through Q × 4 points = 24", va="top")
    P.text(0.8, 59.6, "6 × 4 partition (cf. 4 × 6, Fig. 2b)", va="top", color=S.MUTED)
    bx, by, bs = 7.0, 18.0, 34.0
    u = bs / 5.0                       # mm per lattice unit
    ax = P.ax(bx, by, bs, bs)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-2.5, -2.5), 5, 5, fill=False, ec=S.MUTED, lw=S.THIN, zorder=1))
    # edge identification marks as in Fig. 2B: '^' on the left/right edges (b = -1.5 / +1.5), '>>' on top/bottom
    c = 0.5 / u
    for xx, yc in ((-2.5, -1.5), (2.5, 1.5)):
        ax.plot([xx - c, xx, xx + c], [yc - c / 2, yc + c / 2, yc - c / 2], color=S.MUTED, lw=S.THIN,
                solid_joinstyle="miter", clip_on=False, zorder=1)
    xm = 2.75 - qa                     # Fig. 2B: a = 2.75
    for yy in (-2.5, 2.5):
        for d in (-0.45 / u, 0.45 / u):
            ax.plot([xm + d - c / 2, xm + d + c / 2, xm + d - c / 2], [yy - c, yy, yy + c], color=S.MUTED, lw=S.THIN,
                    solid_joinstyle="miter", clip_on=False, zorder=1)
    slope_of = {}
    for ln in thru:
        m = ln["slope"]
        sm = m if m == G.INF or m <= 2 else m - 5
        col = S.SLOPE[m]
        for p in ln["points"]:
            if p == Q:
                continue
            dx, dy = disp[p]
            assert sm == G.INF and dx == 0 or sm != G.INF and abs(wrap(sm * dx) - dy) < 1e-9, (ln["block"], p)
            slope_of[p] = m
        for ((x0, y0), (x1, y1)), wrapped in torus_steps([disp[p] for p in ln["points"]], sm):
            ax.plot([x0, x1], [y0, y1], color=col, lw=S.LW, ls=(0, (2.2, 1.4)) if wrapped else "-",
                    solid_capstyle="round", dash_capstyle="butt", clip_on=False, zorder=3)
        lx, ly, ha, va = LINE_LABEL_AT[m]
        ax.text(lx, ly, str(ln["block"] + 1), ha=ha, va=va, clip_on=False)
    ax.text(-2.5, 2.62, "tube", ha="left", va="bottom", color=S.MUTED, clip_on=False)
    for p, (dx, dy) in disp.items():
        if p == Q:
            continue
        ax.plot(dx, dy, "o", ms=4.0, mfc=S.SLOPE[slope_of[p]], mec=S.INK, mew=S.THIN, zorder=4)
    ax.plot(0, 0, "o", ms=5.6, mfc=S.INK, mec=WHITE, mew=S.THIN, zorder=5)
    ax.text(0.5, 0.06, "Q", ha="left", va="bottom", fontweight="bold", zorder=6)  # free sector, slope 0 / slope 1
    for i in range(-2, 3):
        ax.text(i, -2.5 - 0.7 / u, str((i + qa) % 5), ha="center", va="top", color=S.MUTED, clip_on=False)
        ax.text(-2.5 - 0.8 / u, i, str((i + qb) % 5), ha="right", va="center", color=S.MUTED, clip_on=False)
    P.text(bx, by - 4.4, "(a, b) mod 5; dashed = wraps mod 5", ha="left", va="top", color=S.MUTED)

    # ---- right: 6 tube rows + reference split, four-state proportions (descriptive)
    nx, rx, rw, pitch = 53.5, 55.0, 32.0, 5.05     # tube-number column (right-aligned), bars, row pitch
    tub = D[D.tube > 0].sort_values("tube")
    ref = D[D.tube == -1].iloc[0]
    top = 54.0
    levels = (0.22, 0.45, 0.7, 1.0)
    P.text(nx - 3.6, top + 6.4, "State of (CCR7, CD27) in each tube", ha="left", va="bottom")
    # state key (reference tints): 00 01 10 11
    kx = rx
    for b, lev in zip(("00", "01", "10", "11"), levels):
        P.rect(kx, top + 3.0, 2.2, 1.9, fc=tint(S.MUTED, lev), lw=0)
        P.text(kx + 2.7, top + 3.95, b, ha="left", va="center")
        kx += 7.0
    P.text(nx, top + 0.2, "tube", ha="right", va="bottom", color=S.MUTED)
    P.text(rx, top + 0.2, "other 4 markers of the tube", ha="left", va="bottom", color=S.MUTED)
    P.text(rx + rw + 0.9, top + 0.2, "cells", ha="left", va="bottom", color=S.MUTED)
    cum_ref = np.cumsum([ref.f00, ref.f01, ref.f10, ref.f11])
    bar_h = 1.85
    ys = []
    for i, (_, r) in enumerate(tub.iterrows()):
        y = top - (i + 1) * pitch + 0.2
        ys.append(y)
        col = S.SLOPE[r.slope if r.slope == G.INF else int(r.slope)]
        x = rx
        for b, lev in zip(("00", "01", "10", "11"), levels):
            w = rw * r[f"f{b}"]
            P.rect(x, y, w, bar_h, fc=tint(col, lev), ec=WHITE, lw=S.THIN)
            x += w
        P.text(nx, y + bar_h / 2 + 1.1, f"{int(r.tube)}", ha="right", va="center")
        P.text(rx, y + bar_h + 0.3, r.others, ha="left", va="bottom")
        P.text(rx + rw + 0.9, y + bar_h / 2, f"{int(r.n)}", ha="left", va="center")
    yr = ys[-1] - pitch - 0.6
    x = rx
    for b, lev in zip(("00", "01", "10", "11"), levels):
        w = rw * ref[f"f{b}"]
        P.rect(x, yr, w, bar_h, fc=tint(S.MUTED, lev), ec=WHITE, lw=S.THIN)
        x += w
    P.text(rx, yr + bar_h + 0.3, "reference split (CD4 T)", ha="left", va="bottom")
    P.text(rx + rw + 0.9, yr + bar_h / 2, f"{int(ref.n):,}", ha="left", va="center")
    for c in cum_ref[:-1]:  # reference-split boundaries marked on every bar (reference grid)
        for yy in ys + [yr]:
            P.ov.plot([rx + rw * c] * 2, [yy - 0.35, yy + bar_h + 0.35], color=S.INK, lw=S.THIN, zorder=5,
                      solid_capstyle="butt")
    P.text(rx + rw, yr - 0.6, "ticks: reference boundaries", ha="right", va="top", color=S.MUTED)
    P.text(0.8, 7.0, "Restored r = 130 allocation (WINGMEN CD4 T), no perturbation; descriptive.", va="bottom")
    P.text(0.8, 4.1, "Six non-overlapping tube samples, not independent biological replicates.", va="bottom",
           color=S.MUTED)
    P.text(0.8, 1.2, "Check rule illustrated; no test computed in this panel.", va="bottom", color=S.MUTED)
    return P.save(), D


# =====================================================================================================  4D
SCEN = [("null", "null"), ("single+0.5", "marker +0.5"), ("single-0.5", "marker −0.5"),
        ("single+0.25", "marker +0.25"), ("single-0.25", "marker −0.25"), ("wholetube+0.5", "whole tube +0.5"),
        ("joint_only", "joint only"), ("composition", "composition"), ("random_drop", "random drop"),
        ("global_drift", "global drift")]
BLIND = {"composition", "global_drift"}   # hatched: low detection in the scenarios checked


def e5k_counts():
    rows = []
    for d in ("levine32_v2", "wingmen_v1"):
        K = pd.read_csv(os.path.join(RES, d, "e5k_boundary.tsv"), sep="\t", keep_default_na=False, na_values=[""])
        K = K[K.part == "B_composition"]
        for mode, g in K.groupby("mode", sort=False):
            al = g.alarm.astype(str).str.lower().eq("true")
            rows.append(dict(dataset=d, source="e5k", scenario=f"composition_{mode}", k=int(al.sum()), n=len(g)))
    E = pd.DataFrame(rows)
    sb = pd.read_csv(os.path.join(DATA, "supp_boundary_composition.tsv"), sep="\t")
    for _, r in E.iterrows():  # cross-check with the frozen summary table
        a = sb[(sb.dataset == r.dataset) & (sb["mode"] == r.scenario.split("_", 1)[1])].alarm.iloc[0]
        assert abs(a - r.k / r.n) < 1e-12
    return E


def fig_4D():
    Q = pd.read_csv(os.path.join(DATA, "fig4B_qc_fixed_family.tsv"), sep="\t", keep_default_na=False)
    Q = Q.set_index(["dataset", "scenario"])
    assert Q.loc[("levine32_v2", "joint_only"), "alarm"] == 0.49 and Q.loc[("wingmen_v1", "joint_only"), "alarm"] == 0.735
    assert Q.loc[("levine32_v2", "joint_only"), "single_only_alarm"] == 0 == Q.loc[("wingmen_v1", "joint_only"), "single_only_alarm"]
    E = e5k_counts().set_index(["dataset", "scenario"])
    cells = []
    for d in ("levine32_v2", "wingmen_v1"):
        for sc, lab in SCEN:
            r = Q.loc[(d, sc)]
            n, MT = int(r.n), int(r.M_total)
            for col, val in (("single_only", float(r.single_only_alarm)), ("single_pair", float(r.alarm))):
                k = int(round(val * n))
                assert abs(k / n - val) < 1e-9
                txt = f"{k}/{n}" if sc == "null" else pct(val)
                cells.append(dict(dataset=d, scenario=sc, column=col, n=n, M_total=MT, k=k, rate=val, display=txt,
                                  alarm_ci=r.alarm_ci if col == "single_pair" else "",
                                  hit_given_alarm=r.hit_given_alarm if col == "single_pair" else ""))
            cells.append(dict(dataset=d, scenario=sc, column="uncheckable", n=n, M_total=MT, k=np.nan,
                              rate=float(r.uncheckable_frac_mean), display=pct(float(r.uncheckable_frac_mean)), alarm_ci="",
                              hit_given_alarm=""))
    Ct = pd.DataFrame(cells)
    write(Ct, "fig4_D_cells.tsv")
    ci_up = {d: Q.loc[(d, "null"), "alarm_ci"].strip("[]").split(",")[1] for d in ("levine32_v2", "wingmen_v1")}
    assert ci_up == {"levine32_v2": "0.0037", "wingmen_v1": "0.0056"}
    e5k_null = {d: f"{E.loc[(d, 'composition_none'), 'k']}/{E.loc[(d, 'composition_none'), 'n']}" for d in ("levine32_v2", "wingmen_v1")}
    assert e5k_null == {"levine32_v2": "0/20", "wingmen_v1": "1/20"}

    # composite size (composeFig4.py): 100 x 60 mm, same height as 4E (taller rows, header kept 4.5 mm below the top)
    P = Panel("4D", 60)
    P.head("Simulation + Replay", "What the built-in checks can and cannot detect")
    rh, top = 3.55, P.H - 13.4
    xlab = 18.4
    colx = {("levine32_v2", "single_only"): 19.6, ("levine32_v2", "single_pair"): 29.6,
            ("wingmen_v1", "single_only"): 40.9, ("wingmen_v1", "single_pair"): 50.9}
    cwid = 10.0
    ux = {"levine32_v2": 62.6, "wingmen_v1": 69.4}
    uw = 6.8
    # headers
    P.text(colx[("levine32_v2", "single_only")] + cwid, top + 6.4, "Levine-32", ha="center", va="bottom")
    P.text(colx[("wingmen_v1", "single_only")] + cwid, top + 6.4, "WINGMEN", ha="center", va="bottom")
    P.text(ux["levine32_v2"] + uw, top + 6.4, "uncheckable %", ha="center", va="bottom", color=S.MUTED)
    for d in ("levine32_v2", "wingmen_v1"):
        for col, l1, l2 in (("single_only", "single", "only"), ("single_pair", "single", "+ pair")):
            x = colx[(d, col)] + cwid / 2
            P.text(x, top + 3.3, l1, ha="center", va="bottom")
            P.text(x, top + 0.5, l2, ha="center", va="bottom")
        P.text(ux[d] + uw / 2, top + 0.5, {"levine32_v2": "Lev.", "wingmen_v1": "WIN."}[d], ha="center", va="bottom",
               color=S.MUTED)
    P.text(xlab, top + 0.5, "alarm rate (%)", ha="right", va="bottom")
    for i, (sc, lab) in enumerate(SCEN):
        y = top - (i + 1) * rh
        P.text(xlab, y + rh / 2, lab, ha="right", va="center")
        for d in ("levine32_v2", "wingmen_v1"):
            for col in ("single_only", "single_pair"):
                c = Ct[(Ct.dataset == d) & (Ct.scenario == sc) & (Ct.column == col)].iloc[0]
                fc = S.SEQ(c.rate)
                x = colx[(d, col)]
                P.rect(x, y, cwid, rh, fc=fc, ec=WHITE, lw=S.THIN)
                if sc in BLIND:
                    P.rect(x, y, cwid, rh, fill=False, hatch=S.LIMITS_HATCH, ec=S.MUTED, lw=0)
                P.text(x + cwid / 2, y + rh / 2, c.display, ha="center", va="center",
                       color=WHITE if lum(fc) < 0.45 else S.INK,
                       bbox=dict(boxstyle="square,pad=0.08", fc=fc, lw=0) if sc in BLIND else None)
            u = Ct[(Ct.dataset == d) & (Ct.scenario == sc) & (Ct.column == "uncheckable")].iloc[0]
            P.text(ux[d] + uw / 2, y + rh / 2, u.display, ha="center", va="center", color=S.MUTED)
    # joint-only row: outline + dumbbell 0 -> single + pair
    ij = [s for s, _ in SCEN].index("joint_only")
    yj = top - (ij + 1) * rh
    P.rect(colx[("levine32_v2", "single_only")] - 0.25, yj - 0.2, colx[("wingmen_v1", "single_pair")] + cwid -
           colx[("levine32_v2", "single_only")] + 0.5, rh + 0.4, fill=False, ec=S.INK, lw=S.LW)
    # dumbbell (joint only): single only (open, 0) -> single + pair (alarm colour), x = 0..100 %
    dx0, dxw = 80.0, 13.0
    yjc = yj + rh / 2
    P.text(78.0, yjc + 5.6, "joint only", ha="left", va="bottom", color=S.MUTED)
    for d, dy, lab, va in (("levine32_v2", +1.55, "Levine-32", "bottom"), ("wingmen_v1", -1.55, "WINGMEN", "top")):
        yc = yjc + dy
        v_ = float(Q.loc[(d, "joint_only"), "alarm"])
        P.ov.plot([dx0, dx0 + dxw * v_], [yc, yc], color=S.MUTED, lw=S.LW, solid_capstyle="butt")
        P.ov.plot(dx0, yc, "o", ms=3.2, mfc=WHITE, mec=S.INK, mew=S.LW)
        P.ov.plot(dx0 + dxw * v_, yc, "o", ms=3.6, mfc=S.ALARM, mew=0)
        P.text(dx0 - 1.0, yc, "0", ha="right", va="center")
        P.text(dx0 + dxw * v_ + 1.0, yc, f"{pct(v_)}%", ha="left", va="center")
        P.text(78.0, yc + (0.9 if va == "bottom" else -0.9), lab, ha="left", va=va)
    P.text(78.0, yjc - 5.6, "single → + pair", ha="left", va="top", color=S.MUTED)
    yb = top - len(SCEN) * rh
    # scenario definitions, run configuration and the separate boundary-run null (0/20, 1/20): captions/fig4.md
    notes = [
        ("Fixed test family: M_total = r × 21 = 1491 / 2730; alarm if min p < 0.001 / M_total.", S.INK),
        (f"Null: 0/1000 and 1/1000 alarmed (95% CI upper {ci_up['levine32_v2']}, {ci_up['wingmen_v1']}); "
         "non-null rows: n = 200.", S.INK),
        ("hatched = low detection in the scenarios checked", S.MUTED)]
    for k, (t, c) in enumerate(notes):
        P.text(0.8, yb - 1.0 - k * 2.8, t, va="top", color=c)
    return P.save(), Ct


# =====================================================================================================  4E
# grey scale, separated mainly by line style (two grey levels only, no method colours)
SHIFT = [("levine32_v2", "no shift", S.INK, "-"),
         ("levine32_v2_shiftAllp0p5", "all markers +0.5", S.INK, (0, (3.2, 1.6))),
         ("levine32_v2_shiftNonBBp0p5", "non-backbone +0.5", S.MUTED, (0, (4.0, 1.4, 1.0, 1.4))),
         ("levine32_v2_shiftNonBBm0p5", "non-backbone −0.5", S.MUTED, (0, (1.0, 1.3)))]


def data_4E():
    # (1) alarm matrix: global drift from e5d fixed-family table (n = 200); composition + null from e5k (n = 20)
    Q = pd.read_csv(os.path.join(DATA, "fig4B_qc_fixed_family.tsv"), sep="\t", keep_default_na=False)
    E = e5k_counts()
    rows = []
    for d in ("levine32_v2", "wingmen_v1"):
        g = Q[(Q.dataset == d) & (Q.scenario == "global_drift")].iloc[0]
        rows.append(dict(dataset=d, row="global drift, all tubes", source="e5d fixed family", k=int(round(g.alarm * g.n)), n=int(g.n)))
        for mode, lab in (("measured", "composition, marker in tube"), ("unmeasured", "composition, marker not in tube"),
                          ("none", "no shift (control)")):
            e = E[(E.dataset == d) & (E.scenario == f"composition_{mode}")].iloc[0]
            rows.append(dict(dataset=d, row=lab, source="e5k_boundary B_composition", k=int(e.k), n=int(e.n)))
    AM = pd.DataFrame(rows)
    am = AM.set_index(["dataset", "row"])
    assert (am.loc[("wingmen_v1", "composition, marker in tube"), "k"], am.loc[("wingmen_v1", "no shift (control)"), "k"]) == (3, 1)
    write(AM, "fig4_E_alarms.tsv")
    # (2) fixed 116 keys (n_direct == 0 by claim, rep), identical across the four replays (D7)
    keyset, recs = None, []
    for d, lab, _, _ in SHIFT:
        A = load_audit(d)
        A = A[A.n_direct.fillna(0) == 0]
        ks = set(zip(A.claim, A.rep))
        keyset = ks if keyset is None else keyset
        assert ks == keyset
        for _, r in A.iterrows():
            recs.append(dict(replay=d, scenario=lab, claim=r.claim, rep=int(r.rep), knn_block=r.knn_block, ref=r.ref,
                             err=abs(r.knn_block - r.ref), conf_lo=r.conf_lo, conf_hi=r.conf_hi,
                             ref_in=bool(r.conf_lo - INTOL <= r.ref <= r.conf_hi + INTOL)))
    K = pd.DataFrame(recs)
    cnt = K.groupby("scenario", sort=False).ref_in.sum()
    assert len(keyset) == 116 and cnt.tolist() == [116, 111, 109, 112], cnt
    write(K, "fig4_E_shift_keys.tsv")
    # (3) CytoNorm real-batch magnitude (reliable markers, as in src/real_batch.py)
    b = pd.read_csv(os.path.join(RES, "cytonorm_real_batch", "batch_shift.tsv"), sep="\t")
    b = b[b.reliable.astype(str).str.lower() == "true"]
    m = b.median_shift.abs()
    cy = dict(n=len(m), donors=b.donor.nunique(), markers=b.marker.nunique(), median=m.median(), p90=m.quantile(0.9),
              max=m.max())
    assert round(cy["max"], 2) == 0.49 and round(cy["p90"], 3) == 0.235
    write(pd.DataFrame([cy]), "fig4_E_cytonorm.tsv")
    return AM, K, cnt, m.values, cy


def fig_4E():
    AM, K, cnt, mshift, cy = data_4E()
    # composite size (composeFig4.py): 82 x 60 mm (row 4D | 4E filled to 184 mm); DX = shift of the right column
    P = Panel("4E", 60, w=82)
    DX = P.W - 80
    P.head("Simulation + magnitude reference", "No alarm ≠ cross-tube comparability", limits=True, title_line=2)
    # ---- alarm matrix
    rows = ["global drift, all tubes", "composition, marker in tube", "composition, marker not in tube", "no shift (control)"]
    rh, cw, x0, top = 3.0, 13.0, 41.0 + DX / 2, 49.8
    P.text(x0 + cw / 2, top + 0.5, "Levine-32", ha="center", va="bottom")
    P.text(x0 + 1.5 * cw, top + 0.5, "WINGMEN", ha="center", va="bottom")
    P.text(x0 - 1.0, top + 0.5, "check alarms / allocations", ha="right", va="bottom", color=S.MUTED)
    for i, lab in enumerate(rows):
        y = top - (i + 1) * rh
        P.text(x0 - 1.0, y + rh / 2, lab, ha="right", va="center")
        for j, d in enumerate(("levine32_v2", "wingmen_v1")):
            r = AM[(AM.dataset == d) & (AM.row == lab)].iloc[0]
            fc = S.SEQ(r.k / r.n)
            P.rect(x0 + j * cw, y, cw, rh, fc=fc, ec=WHITE, lw=S.THIN)
            P.text(x0 + j * cw + cw / 2, y + rh / 2, f"{r.k}/{r.n}", ha="center", va="center")
    yb = top - len(rows) * rh
    P.text(x0 + 2 * cw + 1.0, top - rh / 2, "Fig. 4d", ha="left", va="center", color=S.MUTED)
    # (composition-shift definition and run configuration: captions/fig4.md)
    # ---- ECDF of |kNN - ref| over the fixed 116 keys (Levine-32 only)
    ax = P.ax(10.5, 20.6, 28.5 + DX / 2, 14.8)
    xmin = 1e-3
    for d, lab, col, ls in SHIFT:
        e = np.sort(K[K.replay == d].err.values)
        y = np.arange(1, len(e) + 1) / len(e)
        xs = np.concatenate([[xmin], np.clip(e, xmin, None)])
        ys = np.concatenate([[np.mean(e < xmin)], y])
        ax.step(xs, ys, where="post", color=col, lw=S.LW, ls=ls)
    ax.axvline(0.05, color=S.TIER[3], lw=S.THIN, zorder=0)
    ax.text(0.057, 0.03, "0.05", ha="left", va="bottom", color=S.MUTED, transform=ax.get_xaxis_transform())
    ax.text(0.04, 0.97, "Levine-32", transform=ax.transAxes, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xlim(xmin, 1)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("ECDF")
    ax.set_xlabel("|kNN − ref|, fixed 116 keys (n_direct = 0)")
    h = [Line2D([], [], color=col, lw=S.LW, ls=ls, label=f"{lab}  {int(cnt[lab])}/116") for d, lab, col, ls in SHIFT]
    P.fig.legend(handles=h, loc="upper left", bbox_to_anchor=((41.5 + DX / 2) / P.W, 36.9 / P.H), borderaxespad=0, borderpad=0,
                 handlelength=2.2, title="one initial tube shifted (asinh);\nref in interval",
                 alignment="left")
    # ---- CytoNorm real-batch magnitude: separate ruler (asinh units), not matched to the simulated treatments
    ar = P.ax(10.5, 7.1, 50.0 + DX / 2, 2.2)
    for v_ in mshift:
        ar.plot([v_, v_], [0, 1], color=S.MUTED, lw=S.THIN)
    for v_, lab in ((cy["p90"], f"p90 {cy['p90']:.3f}"), (cy["max"], f"max {cy['max']:.2f}")):
        ar.plot([v_, v_], [0, 1.55], color=S.INK, lw=S.LW, clip_on=False, solid_capstyle="butt")
        ar.text(v_, 1.65, lab, ha="center", va="bottom")
    ar.set_xlim(0, 0.52)
    ar.set_ylim(0, 1)
    ar.set_yticks([])
    ar.spines["left"].set_visible(False)
    ar.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ar.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    P.text(63.0 + DX / 2, 6.6, "magnitude\nreference only", ha="left", va="center", color=S.MUTED, linespacing=1.0)
    P.text(0.8, 0.5, f"CytoNorm real batches: |median shift| (asinh), {cy['donors']} donors × {cy['markers']} markers",
           va="bottom", color=S.MUTED)
    return P.save(), AM, K, cy


PANELS = {"4A": fig_4A, "4B": fig_4B, "4C": fig_4C, "4D": fig_4D, "4E": fig_4E}

if __name__ == "__main__":
    for key in (sys.argv[1:] or list(PANELS)):
        PANELS[key]()
