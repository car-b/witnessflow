"""Figure 5 (figure plan v2): 'After coverage, design starts to face trade-offs' - five single panels 5A-5E.

Read-only inputs (frozen):
  results/wingmen_v1/ledger/CD4T_rep0_opt4_targets.tsv, CD4T_rep0_lp_true_opt4_panels.json, ledger_summary.tsv
  results/{levine32_v2,wingmen_v1}/e5i_shuffle.tsv, results/wingmen_v1/thresholds.json, protocols/wingmen_v1.json
  figures/data/fig4C_attribution_shuffle.tsv, fig4D_ledger.tsv, supp_act5_threshold.tsv
Deterministic summaries only (no LP requirement estimation, no annealing, no D6). The classical baseline tubes of 5A
are recomputed with budget.ljcr_stack(cand, 6, 180, seed=0) exactly as in src/e5j_ledger.py (deterministic, no data).
e5i records: classical stack and uniform requirement are design-free and registered once per initial design; they are
reduced to one record per distinct evaluation (identity asserted, see e5i_records).
Display tables -> figures/derived/fig5_*.tsv.  usage: python fig5.py [5A 5B 5C 5D 5E]
"""
import json, os, sys
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import wfstyle as S

S.setup()
sys.path.insert(0, os.path.join(S.WF, "src"))
import budget  # noqa: E402

os.makedirs(S.DERIVED, exist_ok=True)
KINDS = ["classical", "uniform", "lp_shuffled", "lp_true"]
RAW = {"classical": "LJCR_stack", "uniform": "worst", "lp_shuffled": "lp_shuffled", "lp_true": "lp_true"}
FACETS = [("levine32_v2", "T"), ("wingmen_v1", "CD4T"), ("wingmen_v1", "CD8T")]
FNAME = {("levine32_v2", "T"): "Levine-32 T", ("wingmen_v1", "CD4T"): "WINGMEN CD4 T", ("wingmen_v1", "CD8T"): "WINGMEN CD8 T"}
HALF_NOTE = "Half-width: per-pattern 95% CP marginal; not joint."
LEDGER = os.path.join(S.RESULTS, "wingmen_v1", "ledger")
PT = 25.4 / 72  # mm per pt


def rd(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def save(fig, panel):
    """S.save (strict audit) + an extra unclipped check: S.pdf_audit reads text with PyMuPDF's default page clip,
    so a span that runs past the page edge is truncated rather than reported; re-read without clipping."""
    import fitz
    pdf, png = S.save(fig, panel)
    doc = fitz.open(pdf)
    page = doc[0]
    R = page.rect
    bad = [(s["text"][-30:], [round(v, 1) for v in s["bbox"]])
           for b in page.get_text("dict", clip=fitz.INFINITE_RECT())["blocks"] for ln in b.get("lines", [])
           for s in ln["spans"] if s["text"].strip() and (s["bbox"][0] < R.x0 - 0.05 or s["bbox"][1] < R.y0 - 0.05
                                                         or s["bbox"][2] > R.x1 + 0.05 or s["bbox"][3] > R.y1 + 0.05)]
    doc.close()
    if bad:
        raise RuntimeError(f"fig{panel}: text runs past the page edge: {bad}")
    return pdf, png


def wsave(df, name):
    p = os.path.join(S.DERIVED, f"fig5_{name}.tsv")
    df.to_csv(p, sep="\t", index=False)
    return p


# ------------------------------------------------------------------ manual mm layout helpers
class Canvas:
    """Figure in mm coordinates (layout=None; no tight bbox)."""

    def __init__(self, panel, h=None, w=None):
        self.fig = S.new(panel, w=w, h=h, layout=None)
        self.W, self.H = (v / S.MM for v in self.fig.get_size_inches())
        self.base = self.fig.add_axes([0, 0, 1, 1])
        self.base.set_xlim(0, self.W)
        self.base.set_ylim(0, self.H)
        self.base.axis("off")
        self.base.set_zorder(-5)
        self.r = self.fig.canvas.get_renderer()

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", S.FS)
        return self.base.text(x, y, s, **kw)

    def width(self, s, **kw):
        t = self.base.text(0, 0, s, fontsize=S.FS, **kw)
        w = t.get_window_extent(renderer=self.r).width / self.fig.dpi * 25.4
        t.remove()
        return w

    def rich(self, x, y, parts, ha="left", color=S.INK, gap=0.6, **kw):
        """parts: str or dict(marker=..., ms=..., mfc=..., mec=..., mew=...) drawn inline at text height."""
        ws = []
        for p in parts:
            if isinstance(p, str):
                ws.append(self.width(p, **kw))
            else:
                ws.append(p.get("ms", 4) * PT + 2 * gap)
        tot = sum(ws)
        cur = x - (tot if ha == "right" else tot / 2 if ha == "center" else 0)
        for p, w in zip(parts, ws):
            if isinstance(p, str):
                self.base.text(cur, y, p, ha="left", va="baseline", fontsize=S.FS, color=color, **kw)
            else:
                self.base.plot([cur + w / 2], [y + 0.85], ls="none", marker=p["marker"], ms=p.get("ms", 4),
                               mfc=p.get("mfc", "none"), mec=p.get("mec", S.INK), mew=p.get("mew", S.THIN),
                               clip_on=False, zorder=6)
            cur += w
        return cur

    def line(self, xs, ys, **kw):
        kw.setdefault("lw", S.LW)
        kw.setdefault("color", S.INK)
        kw.setdefault("solid_capstyle", "butt")
        return self.base.plot(xs, ys, clip_on=False, **kw)

    def para(self, x, y, s, maxw=None, lead=3.15, **kw):
        """word-wrapped text block (measured widths); returns the baseline of the next line."""
        maxw = maxw or (self.W - x - 0.8)
        line = ""
        for w in s.split(" "):
            cand = (line + " " + w).strip()
            if line and self.width(cand, **{k: v for k, v in kw.items() if k in ("fontweight", "fontstyle")}) > maxw:
                self.text(x, y, line, va="baseline", **kw)
                y -= lead
                line = w
            else:
                line = cand
        if line:
            self.text(x, y, line, va="baseline", **kw)
            y -= lead
        return y

    def legend(self, x, y, entries, maxw=None, lead=3.15, sep=2.2):
        """entries: list of (marker dict, label); wrapped onto as many lines as needed; returns next baseline."""
        maxw = maxw or (self.W - x - 0.8)
        cur = x
        for m, lab in entries:
            w = m.get("ms", 4) * PT + 1.2 + self.width(" " + lab)
            if cur > x and cur + w > x + maxw:
                cur, y = x, y - lead
            cur = self.rich(cur, y, [m, " " + lab]) + sep
        return y - lead


STAR = dict(marker="*", ms=6.5, mfc=S.TSTAR, mec=S.INK, mew=S.THIN)


def mk(kind, filled=True, ms=4.0):
    c = S.METHOD[kind]
    return dict(marker=S.MARKER[kind], ms=ms, mfc=c if filled else "white", mec=c if not filled else S.INK,
                mew=S.LW if not filled else S.THIN)


def style_ax(ax):
    ax.tick_params(length=2.5, width=S.LW, pad=1.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(S.LW)


def pct_fmt(ax, axis="y", nd=0):
    from matplotlib.ticker import FuncFormatter
    def lab(v, _):
        x = v * 100
        d = nd if abs(round(x, nd) - x) < 1e-6 else nd + 1      # never round a tick label off its position (72.5 != 73)
        return f"{x:.{d}f}"
    f = FuncFormatter(lab)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# ------------------------------------------------------------------ shared data
def facet_meta():
    """r and eps of every facet, read from the frozen e5i tables (no hard-coding)."""
    out = {}
    for ds in ("levine32_v2", "wingmen_v1"):
        d = rd(os.path.join(S.RESULTS, ds, "e5i_shuffle.tsv"))
        for G, g in d.groupby("parent"):
            assert g.r.nunique() == 1 and g.eps.nunique() == 1
            out[(ds, G)] = (int(g.r.iloc[0]), float(g.eps.iloc[0]))
    return out


META = facet_meta()


E5I_METRICS = ["covered", "half_ok", "err_ok", "min_cells", "plan_eps", "lam_min", "lam_cv"]
DESIGN_FREE = ("LJCR_stack", "worst")   # requirement kinds that do not use the initial design


def e5i_records(registrations=False):
    """e5i records, one per distinct evaluation. The classical stack (LJCR_stack) and the uniform worst-case
    requirement (worst) do not depend on the initial design: for each (dataset, parent, opt_seed, alloc) the rows of
    initial designs 0 and 1 are the same evaluation registered twice (all metric columns identical, asserted). Each
    is reduced to one record, keeping the initial-design-0 row: per setting classical 6 -> 3 and uniform 18 -> 9.
    registrations=True returns every registered row (used only to document the frozen fig4C table, which counted the
    18 uniform registrations)."""
    rows = []
    for ds in ("levine32_v2", "wingmen_v1"):
        d = rd(os.path.join(S.RESULTS, ds, "e5i_shuffle.tsv"))
        assert set(E5I_METRICS) <= set(d.columns)
        key = ["parent", "opt_seed", "alloc"]
        for rk in DESIGN_FREE:
            g = d[d.req_kind == rk]
            assert set(g.init_rep) == {0, 1} and (g.groupby(key).size() == 2).all(), (ds, rk)
            assert (g.groupby(key)[E5I_METRICS].nunique(dropna=False) == 1).all().all(), (ds, rk)   # identical rows
        if not registrations:
            d = d[~(d.req_kind.isin(DESIGN_FREE) & (d.init_rep != 0))]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["kind"] = d.req_kind.map(S.ALIAS)
    d["shuffle"] = d.req.where(d.kind == "lp_shuffled", "").str[-1:]
    return d


def fig4c():
    f = rd(os.path.join(S.DATA, "fig4C_attribution_shuffle.tsv"))
    f["kind"] = f.req_kind.map(S.ALIAS)
    return f


# ================================================================== 5A
def derive_5A():
    t = rd(os.path.join(LEDGER, "CD4T_rep0_opt4_targets.tsv"))
    thr = json.load(open(os.path.join(S.RESULTS, "wingmen_v1", "thresholds.json"), encoding="utf-8"))
    proto = json.load(open(os.path.join(S.WF, "protocols", "wingmen_v1.json"), encoding="utf-8"))
    eps = float(proto["e3"]["eps"]["CD4T"][0])
    assert abs(eps - 0.03) < 1e-12
    markers = list(thr.keys())
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    assert len(cand) == 26 and len(t) == 2600
    r = META[("wingmen_v1", "CD4T")][0]
    base = budget.ljcr_stack(cand, 6, r, seed=0)                       # e5j: st = ljcr_stack(cand, 6, r, seed=0)
    opt = json.load(open(os.path.join(LEDGER, "CD4T_rep0_lp_true_opt4_panels.json")))
    assert len(base) == len(opt) == r
    changed = [i for i in range(r) if set(base[i]) != set(opt[i])]
    mi = {m: i for i, m in enumerate(markers)}
    rows = []
    for tag, name in (("row1", t.triple.iloc[0]), ("tstar", "CCR7/CD27/CD45RA")):
        rec = t[t.triple == name].iloc[0]
        T = {mi[m] for m in name.split("/")}
        sb = [i for i, X in enumerate(base) if T <= set(X)]
        so = [i for i, X in enumerate(opt) if T <= set(X)]
        assert len(sb) == rec.lam_base and len(so) == rec.lam_opt
        rows.append(dict(row=tag, triple=name, eps=eps, r=r, req_lam=int(rec.req_lam), ref_var=rec.ref_var,
                         lam_base=int(rec.lam_base), lam_opt=int(rec.lam_opt), n_base=int(rec.n_base), n_opt=int(rec.n_opt),
                         half_base=rec.half_base, half_opt=rec.half_opt, err_base=rec.err_base, err_opt=rec.err_opt,
                         tubes_base_0based=",".join(map(str, sb)), tubes_opt_0based=",".join(map(str, so)),
                         tubes_opt_changed=",".join(str(i) for i in so if i in changed),
                         base_tube_markers=" | ".join("/".join(markers[x] for x in sorted(base[i])) for i in sb),
                         opt_tube_markers=" | ".join("/".join(markers[x] for x in sorted(opt[i])) for i in so),
                         n_slots_changed=len(changed), n_tubes_base=len(base), n_tubes_opt=len(opt)))
    df = pd.DataFrame(rows)
    wsave(df, "A_ledger_rows")
    return df, base, opt, markers, changed


def tube_card(cv, x, y, members, target, fill, slot=0.85, h=2.3, label=None):
    """M1 tube card: 6 candidate slots (target markers filled) + 8 grey backbone slots. (x, y) = left-centre."""
    mem = sorted(members)
    for j in range(14):
        if j < 6:
            fc = fill if mem[j] in target else "white"
        else:
            fc = S.BACKBONE
        cv.base.add_patch(Rectangle((x + j * slot, y - h / 2), slot, h, fc=fc, ec=S.MUTED, lw=S.THIN, zorder=3))
    cv.base.add_patch(Rectangle((x, y - h / 2), 14 * slot, h, fc="none", ec=S.INK, lw=S.LW, zorder=4))
    if label is not None:
        cv.text(x - 0.7, y, label, ha="right", va="center_baseline")
    return x + 14 * slot


def tstar_text(cv, x, y, rest="", ms=6):
    """'T' + gold vector star (S.tstar_label) followed by `rest`; y = vertical centre (mm)."""
    S.tstar_label(cv.base, x, y, text="T", ms=ms, va="center", fontsize=S.FS)
    xr = x + cv.width("T") + (0.6 + 0.95 * ms + 0.9) * PT
    if rest:
        cv.text(xr, y, rest, va="center")
    return xr


def chip(cv, x, y, s, fc, pad=0.75, h=3.0):
    w = cv.width(s) + 2 * pad
    cv.base.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                     fc=fc, ec=S.INK, lw=S.LW, zorder=3))
    cv.text(x + w / 2, y, s, ha="center", va="center_baseline", zorder=4)
    return x + w


def panel_5A():
    df, base, opt, markers, changed = derive_5A()
    mi = {m: i for i, m in enumerate(markers)}
    R1, RT = df.iloc[0], df.iloc[1]
    eps, r = R1.eps, int(R1.r)
    cv = Canvas("5A", h=52, w=184)      # 184 mm = full composite content width (Fig5 page)
    W, H = cv.W, cv.H
    # ---------------- header
    ttl = "Evidence ledger of two targets"
    cv.text(1, H - 3.2, ttl, va="baseline", fontweight="bold")
    cv.text(1 + cv.width(ttl, fontweight="bold") + 2.0, H - 3.2,
            f"WINGMEN CD4 T · r = {r} tubes · ε = {eps:.2f} · one ledger of 2600 target triples", va="baseline",
            color=S.MUTED)
    S.tier_tag(cv.base, "Replay · single example", x=(W - 0.6) / W, y=(H - 3.2) / H, ha="right", va="baseline")
    xx = cv.rich(1, H - 7.0, [mk("classical", True), " classical stack (La Jolla, 180 tubes)   ", mk("lp_true", True),
                              " after reallocation with the matched requirement (180 tubes)   "])
    cv.line([xx + 0.5, xx + 4.0], [H - 7.0 + 0.85] * 2, ls=(0, (2.5, 1.5)))
    cv.text(xx + 5.0, H - 7.0, f"ε = {eps:.2f}", va="baseline")
    # ---------------- columns
    C = dict(tgt=(1, 34), req=(36.5, 50), tube=(55, 95.5), lam=(98.5, 116), n=(118.5, 136), half=(139, 156.5),
             err=(158, 179))
    fx = (W - 2) / 178.0                # columns laid out for W = 180; spread evenly over a wider panel
    C = {k: (1 + (a - 1) * fx, 1 + (b - 1) * fx) for k, (a, b) in C.items()}
    yH1, yH2 = H - 12.0, H - 15.0      # header baselines
    heads = dict(tgt=("Target triple", ""), req=("Requirement", "req_λ"), tube=("Supporting tubes", "classical → reallocated"),
                 lam=("Multiplicity", "λ"), n=("Direct cells", "n"), half=("Half-width", "both: max over 8 patterns"),
                 err=("Reference error", ""))
    for k, (a, b) in C.items():
        cv.text(a, yH1, heads[k][0], fontweight="bold")
        if heads[k][1]:
            cv.text(a, yH2, heads[k][1], color=S.MUTED)
    for k in ("lam", "n", "half", "err"):   # the M4 chain: requirement -> multiplicity -> cells -> half -> error
        a = C[k][0]
        cv.text(a - 2.4, yH1, "→", color=S.MUTED, ha="center")
    ytop = H - 16.8
    cv.line([1, W - 1], [ytop, ytop], lw=S.THIN, color=S.MUTED)
    y1, y2 = ytop - 6.0, ytop - 17.0            # row centres
    ysep = (y1 + y2) / 2
    cv.line([1, W - 1], [ysep - 0.2, ysep - 0.2], lw=S.THIN, color=S.GRID)
    yr = y2 - 6.6                                # ruler line
    # ---------------- target column
    for yc, R, star in ((y1, R1, False), (y2, RT, True)):
        a = C["tgt"][0]
        if star:
            tstar_text(cv, a, yc + 3.75, "· through-line target (row 2)")
        else:
            cv.text(a, yc + 3.75, "Ledger file, first row (row 1)", va="center")
        x = a
        for m in R.triple.split("/"):
            x = chip(cv, x, yc - 0.1, m, S.TSTAR if star else "white") + 0.8
        cv.text(a, yc - 4.6, f"max ref. variance {R.ref_var:.3f}", va="baseline", color=S.MUTED)
    # ---------------- requirement pips (mini ruler 1..4)
    a = C["req"][0] + 0.6
    pip, pg = 2.0, 0.9
    for yc, R in ((y1, R1), (y2, RT)):
        for j in range(4):
            cv.base.add_patch(Rectangle((a + j * (pip + pg), yc - 1.6), pip, pip,
                                        fc=S.INK if j < R.req_lam else "white", ec=S.INK, lw=S.LW, zorder=3))
        cv.text(a, yc + 1.6, f"{R.req_lam}" + ("  (highest)" if R.req_lam == 4 else ""), va="baseline")
    cv.line([a, a + 3 * (pip + pg) + pip], [yr, yr])
    for j in range(4):
        xc = a + j * (pip + pg) + pip / 2
        cv.line([xc, xc], [yr, yr - 0.9])
        cv.text(xc, yr - 3.6, str(j + 1), ha="center", va="baseline")
    # ---------------- tube cards
    a = C["tube"][0] + 5.0
    cw = 14 * 0.85
    xo = a + cw + 6.0
    for yc, R, fill in ((y1, R1, S.INK), (y2, RT, S.TSTAR)):
        T = {mi[m] for m in R.triple.split("/")}
        sb = [int(v) for v in R.tubes_base_0based.split(",")]
        so = [int(v) for v in R.tubes_opt_0based.split(",")]
        n = len(so)
        ys_o = [yc + (n - 1) / 2 * 3.0 - k * 3.0 for k in range(n)]
        yb = ys_o[0] if n > 1 else yc
        for i in sb:
            tube_card(cv, a, yb, base[i], T, fill, label=str(i + 1))
        for k, i in enumerate(so):
            tube_card(cv, xo, ys_o[k], opt[i], T, fill)
            cv.text(xo + cw + 0.7, ys_o[k], str(i + 1), ha="left", va="center_baseline")
        cv.text((a + cw + xo) / 2, yb, "→", ha="center", va="center_baseline", color=S.MUTED)
        if n > 1:
            cv.text(C["tube"][0], yb - 4.4, "this target gains", va="baseline")
            cv.text(C["tube"][0], yb - 7.3, "supporting tubes", va="baseline")
        else:
            cv.text(C["tube"][0], yb - 3.9, "no supporting tube added", va="baseline", color=S.MUTED)
    # card legend in the ruler band of the tube column
    lx = C["tube"][0]
    tube_card(cv, lx + 0.3, yr - 1.4, [0, 1, 2, 3, 4, 5], {0, 1, 2}, S.MUTED, slot=0.85, h=2.0)
    cv.text(lx + 0.3 + cw + 1.0, yr - 0.6, "6 candidate + 8 backbone", va="baseline", color=S.MUTED)
    cv.text(lx + 0.3 + cw + 1.0, yr - 3.4, "slots; filled = target", va="baseline", color=S.MUTED)

    # ---------------- numeric dumbbell columns with mini rulers
    def column(key, lo, hi, ticks, fmt, vals, epsline=False, status=False, lab=None):
        a, b = C[key]
        x1 = b - (4.2 if status else 0.8)
        X = lambda v: a + 0.5 + (v - lo) / (hi - lo) * (x1 - a - 0.5)  # noqa: E731
        for yc, (v0, v1, txt) in zip((y1, y2), vals):
            cv.line([X(lo), X(hi)], [yc - 1.2] * 2, color=S.GRID)
            if v0 != v1:
                cv.line([X(v0), X(v1)], [yc - 1.2] * 2, color=S.MUTED)
            cv.base.plot([X(v0)], [yc - 1.2], ls="none", marker="o", ms=4.8 if v0 == v1 else 4, mfc=S.METHOD["classical"],
                         mec=S.INK, mew=S.THIN, zorder=5, clip_on=False)
            cv.base.plot([X(v1)], [yc - 1.2], ls="none", marker="D", ms=3.4, mfc=S.METHOD["lp_true"], mec=S.INK,
                         mew=S.THIN, zorder=6, clip_on=False)
            cv.text(a + 0.5, yc + 1.6, txt, va="baseline", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
            if status:
                ok = v1 <= eps
                cv.text(b, yc - 1.2, "≤ ε" if ok else "> ε", ha="right", va="center_baseline",
                        color=S.INK if ok else S.ALARM)
        cv.line([X(lo), X(hi)], [yr, yr])
        for tv in ticks:
            cv.line([X(tv), X(tv)], [yr, yr - 0.9])
            cv.text(X(tv), yr - 3.6, fmt(tv), ha="center", va="baseline")
        if epsline:
            cv.line([X(eps), X(eps)], [yr, ytop - 0.6], ls=(0, (2.5, 1.5)), zorder=2)
        return X

    def arrow_txt(v0, v1, f):
        return f"{f(v0)} → {f(v1)}"

    column("lam", 0, 4, [0, 1, 2, 3, 4], lambda v: f"{v:d}",
           [(R1.lam_base, R1.lam_opt, f"{R1.lam_base} → {R1.lam_opt}  (= req)"),
            (RT.lam_base, RT.lam_opt, f"{RT.lam_base} → {RT.lam_opt}  (< req {RT.req_lam})")])
    column("n", 0, 1000, [0, 500, 1000], lambda v: f"{v:d}",
           [(R1.n_base, R1.n_opt, arrow_txt(R1.n_base, R1.n_opt, str)),
            (RT.n_base, RT.n_opt, arrow_txt(RT.n_base, RT.n_opt, str))])
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    ft = lambda v: "0" if v == 0 else f"{v:.2f}"  # noqa: E731
    column("half", 0, 0.07, [0, 0.03, 0.06], ft,
           [(R1.half_base, R1.half_opt, arrow_txt(R1.half_base, R1.half_opt, f3)),
            (RT.half_base, RT.half_opt, arrow_txt(RT.half_base, RT.half_opt, f3))], epsline=True, status=True)
    column("err", 0, 0.07, [0, 0.03, 0.06], ft,
           [(R1.err_base, R1.err_opt, arrow_txt(R1.err_base, R1.err_opt, f3)),
            (RT.err_base, RT.err_opt, arrow_txt(RT.err_base, RT.err_opt, f3))], epsline=True, status=True)
    # ---------------- footer (details -> captions/fig5.md)
    nchg = int(R1.n_slots_changed)
    yf = 4.3
    cv.text(1, yf, f"Fixed {r} → {r}-tube reallocation ({nchg} of {r} tubes changed; total unchanged). "
                   "req_λ is a planning requirement, not a guarantee.", va="baseline")
    cv.text(1, yf - 3.1, "Half-width and reference error: both max over 8 patterns; half-width from per-pattern 95% CP "
                         "intervals (not joint). Tube numbers 1-based.", va="baseline", color=S.MUTED)
    return save(cv.fig, "5A")


# ================================================================== 5B
def derive_5B():
    d = e5i_records()
    dall = e5i_records(registrations=True)
    f = fig4c()
    recs, pairs, summ = [], [], []
    for ds, G in FACETS:
        g = d[(d.dataset == ds) & (d.parent == G)]
        gall = dall[(dall.dataset == ds) & (dall.parent == G)]
        r, eps = META[(ds, G)]
        for k in KINDS:
            s = g[g.kind == k]
            sr = gall[gall.kind == k]          # every registered row of this group
            ref = f[(f.dataset == ds) & (f.parent == G) & (f.kind == k)].iloc[0]
            # means: duplicates are exact copies, so the frozen means are reproduced exactly
            assert np.isclose(s.half_ok.mean(), ref.half_ok) and np.isclose(s.err_ok.mean(), ref.err_ok)
            assert np.isclose(sr.half_ok.mean(), ref.half_ok) and np.isclose(sr.err_ok.mean(), ref.err_ok)
            if k == "uniform":
                # the frozen fig4C table counted the 18 registrations (9 distinct evaluations x 2 initial designs);
                # its n and SD are those of the registrations, not of the 9 distinct evaluations used here
                assert len(s) == 9 and len(sr) == ref.n_records == 18 and np.isclose(sr.half_ok.std(), ref.half_sd)
            else:
                assert len(s) == ref.n_records and np.isclose(s.half_ok.std(), ref.half_sd)
            summ.append(dict(dataset=ds, parent=G, r=r, eps=eps, kind=k, req_kind=RAW[k], n_records=len(s),
                             half_mean=s.half_ok.mean(), half_sd=s.half_ok.std(), err_mean=s.err_ok.mean(),
                             err_sd=s.err_ok.std(), n_registrations=len(sr),
                             half_sd_registrations=sr.half_ok.std(), err_sd_registrations=sr.err_ok.std(),
                             fig4C_n_records=int(ref.n_records), fig4C_half_sd=ref.half_sd,
                             dedup=("identical across initial designs 0-1 (asserted); initial-design-0 row kept"
                                    if RAW[k] in DESIGN_FREE else "none (distinct evaluations)")))
            recs.append(s[["dataset", "parent", "kind", "req", "init_rep", "opt_seed", "alloc", "shuffle", "half_ok",
                           "err_ok"]])
        # pairing key (dataset, parent fixed here) + init_rep, opt_seed, alloc; 3 shuffled seeds averaged per key first
        key = ["init_rep", "opt_seed", "alloc"]
        sh = g[g.kind == "lp_shuffled"].groupby(key).agg(shuf_mean=("half_ok", "mean"), shuf_mean_err=("err_ok", "mean"),
                                                         n_shuf=("half_ok", "size"))
        tr = g[g.kind == "lp_true"].set_index(key)[["half_ok", "err_ok"]].rename(columns={"half_ok": "true",
                                                                                          "err_ok": "true_err"})
        p = sh.join(tr, how="inner").reset_index()
        assert len(p) == 18 and (p.n_shuf == 3).all()
        p["diff"] = p["true"] - p["shuf_mean"]                  # delta half_ok (fraction)
        p["diff_err"] = p["true_err"] - p["shuf_mean_err"]      # delta err_ok (fraction): reference-error attainment change
        p.insert(0, "parent", G)
        p.insert(0, "dataset", ds)
        pairs.append(p)
    recs, pairs, summ = pd.concat(recs), pd.concat(pairs), pd.DataFrame(summ)
    pd_ = pairs.groupby(["dataset", "parent"]).agg(diff_mean=("diff", "mean"), diff_sd=("diff", "std"),
                                                   diff_err_mean=("diff_err", "mean"), diff_err_sd=("diff_err", "std"),
                                                   n_pairs=("diff", "size"), pairmean_shuf_sd=("shuf_mean", "std"),
                                                   pairmean_shuf_err_sd=("shuf_mean_err", "std"),
                                                   true_sd=("true", "std")).reset_index()
    summ = summ.merge(pd_, on=["dataset", "parent"], how="left")
    wsave(recs, "B_records")
    wsave(pairs, "B_pairs")
    wsave(summ, "B_summary")
    return recs, pairs, summ


def offsets(n, width):
    """deterministic within-group spread: n positions evenly in [-width/2, width/2]."""
    return np.zeros(1) if n == 1 else np.linspace(-width / 2, width / 2, n)


def facet_head(cv, x, y, ds, G, name=None):
    r, eps = META[(ds, G)]
    cv.text(x, y, name or FNAME[(ds, G)], fontweight="bold", va="baseline")
    cv.text(x, y - 3.0, f"r = {r} · ε = {eps:.2f}", va="baseline", color=S.MUTED)


def tier(cv, s):
    S.tier_tag(cv.base, s, x=(cv.W - 0.6) / cv.W, y=(cv.H - 2.8) / cv.H, ha="right", va="baseline")


def title(cv, s, x=1.0):
    """short 7 pt bold in-panel title (top-left)."""
    cv.text(x, cv.H - 2.8, s, fontweight="bold", va="baseline")


PER_TARGET = "Per target: max over 8 patterns (half-width from 95% CP per pattern; not joint)."


def mfmt(v, nd=2):
    """signed number with a typographic minus."""
    return f"{v:+.{nd}f}".replace("-", "−")


def panel_5B():
    from matplotlib.ticker import MaxNLocator
    recs, pairs, summ = derive_5B()
    cv = Canvas("5B", h=72, w=91)       # 91 + 2 + 91 = 184 mm composite row
    W, H = cv.W, cv.H
    tier(cv, "Replay")
    title(cv, "Correct correspondence adds gain")
    ax_top, ax_y0 = H - 12.0, 35.0           # main axes
    st_top, st_y0 = 31.0, 21.0               # paired-delta strip
    ax_w = 21.0
    ax_h, st_h = ax_top - ax_y0, st_top - st_y0
    xs = [10.0, 39.5, 69.0]
    key = ["init_rep", "opt_seed", "alloc"]
    # shared y range of the delta strips (all 18 pair differences, means +- SD, and zero)
    dv = 100 * np.concatenate([pairs["diff"].values, pairs["diff_err"].values])
    dlo, dhi = min(dv.min(), 0), max(dv.max(), 0)
    dpad = 0.08 * (dhi - dlo)
    for (ds, G), x0 in zip(FACETS, xs):
        ax = cv.ax(x0, ax_y0, ax_w, ax_h)
        style_ax(ax)
        R = recs[(recs.dataset == ds) & (recs.parent == G)]
        P = pairs[(pairs.dataset == ds) & (pairs.parent == G)].sort_values(key).reset_index(drop=True)
        Sm = summ[(summ.dataset == ds) & (summ.parent == G)].set_index("kind")
        o18 = offsets(18, 0.26)
        kidx = {tuple(v): i for i, v in enumerate(P[key].values)}
        c = R[R.kind == "classical"].sort_values("alloc")
        ax.plot(0 + offsets(len(c), 0.2), c.half_ok, ls="none", marker="o", ms=2.6, mfc=S.METHOD["classical"],
                mec="white", mew=S.THIN, zorder=3)
        u = R[R.kind == "uniform"].sort_values(["opt_seed", "alloc"])     # 9 distinct evaluations
        assert len(u) == 9
        ax.plot(1 + offsets(len(u), 0.26), u.half_ok, ls="none", marker="^", ms=2.6,
                mfc=S.METHOD["uniform"], mec="white", mew=S.THIN, zorder=3)
        # 54 shuffled runs (faint background, left of the seed means), three columns = three shuffle seeds
        sh = R[R.kind == "lp_shuffled"]
        osh = np.array([o18[kidx[tuple(v)]] for v in sh[key].values]) * 0.25
        ax.plot(1.70 + 0.07 * sh.shuffle.astype(int).values + osh, sh.half_ok, ls="none", marker="s", ms=1.6,
                mfc=S.METHOD["lp_shuffled"], mec="none", alpha=0.4, zorder=2)
        # 18 pairs: mean of the 3 shuffles -> true requirement (same initial design, optimizer seed, allocation)
        xm_s, xm_t = 2.08, 2.92
        for i, row in P.iterrows():
            ax.plot([xm_s + o18[i] * 0.5, xm_t + o18[i] * 0.5], [row.shuf_mean, row["true"]], color=S.MUTED,
                    lw=S.LW, alpha=0.35, zorder=2)
        ax.plot(xm_s + o18 * 0.5, P.shuf_mean, ls="none", marker="s", ms=2.4, mfc=S.METHOD["lp_shuffled"],
                mec="white", mew=S.THIN, zorder=4)
        ax.plot(xm_t + o18 * 0.5, P["true"], ls="none", marker="D", ms=2.4, mfc=S.METHOD["lp_true"], mec="white",
                mew=S.THIN, zorder=4)
        # mean +- SD over each group's own records (shuffled: SD over the 54 runs)
        for j, k in enumerate(KINDS):
            m, sd = Sm.loc[k, "half_mean"], Sm.loc[k, "half_sd"]
            xm = 3.32 if k == "lp_true" else j + (0.40 if k == "lp_shuffled" else 0.30)
            ax.plot([xm, xm], [m - sd, m + sd], color=S.INK, lw=S.LW, solid_capstyle="butt", zorder=5)
            ax.plot([xm - 0.1, xm + 0.1], [m, m], color=S.INK, lw=S.LW, solid_capstyle="butt", zorder=5)
        lo = min(R.half_ok.min(), (Sm.half_mean - Sm.half_sd).min())
        hi = max(R.half_ok.max(), (Sm.half_mean + Sm.half_sd).max())
        span = hi - lo
        ax.set_ylim(lo - 0.06 * span, hi + 0.08 * span)
        ax.set_xlim(-0.4, 3.55)
        ax.set_xticks([0, 1, 2, 3], ["C", "U", "S", "M"])
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        pct_fmt(ax, "y")
        facet_head(cv, x0 - 7.0, H - 6.4, ds, G)
        if (ds, G) == ("wingmen_v1", "CD4T"):
            ax.text(1, Sm.loc["uniform", "half_mean"] + 0.08 * span, "= classical", ha="center", va="bottom",
                    rotation=90, fontsize=S.FS, color=S.MUTED)
        # ---- paired-delta strip: true - mean(3 shuffled), two metrics, zero line
        axd = cv.ax(x0, st_y0, ax_w, st_h)
        style_ax(axd)
        axd.axhline(0, color=S.MUTED, lw=S.LW, ls=(0, (2.5, 1.5)), zorder=1)
        labs = []
        for j, (col, nm) in enumerate((("diff", "half"), ("diff_err", "ref err"))):
            v = 100 * P[col].values
            m, sd = v.mean(), v.std(ddof=1)
            axd.plot(j - 0.06 + o18 * 0.9, v, ls="none", marker="o", ms=1.9, mfc=S.MUTED, mec="none", alpha=0.75,
                     zorder=3)
            axd.plot([j + 0.30, j + 0.30], [m - sd, m + sd], color=S.INK, lw=S.LW, solid_capstyle="butt", zorder=4)
            axd.plot([j + 0.20, j + 0.40], [m, m], color=S.INK, lw=S.LW, solid_capstyle="butt", zorder=4)
            labs.append(f"{nm}\n{mfmt(m)}")
        axd.set_xlim(-0.5, 1.55)
        axd.set_ylim(dlo - dpad, dhi + dpad)
        axd.set_xticks([0.05, 1.05], labs)
        axd.tick_params(axis="x", length=0, pad=1.5)
        axd.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        axd.yaxis.set_major_formatter(plt_fmt_signed())
    cv.text(1.6, ax_y0 + ax_h / 2, "Half-width", rotation=90, ha="center", va="center")
    cv.text(4.5, ax_y0 + ax_h / 2, "attainment (%)", rotation=90, ha="center", va="center")
    cv.text(3.0, st_y0 + st_h / 2, "Δ (pp)", rotation=90, ha="center", va="center")
    y = 11.45
    y = cv.legend(1, y, [(mk("classical"), "Classical stack (3)"), (mk("uniform"), "Uniform (9)"),
                         (mk("lp_shuffled"), "Shuffled (54)"), (mk("lp_true"), "Matched (18)")])
    y = cv.para(1, y, "Δ = matched − mean of 3 shuffled seeds (18 pairs); bar: mean ± SD of Δ. ref err = "
                      "reference-error attainment change. Top: shuffled SD from 54 records.")
    cv.para(1, y, "Per target: max over 8 patterns (95% CP; not joint). Top y scales differ.", color=S.MUTED)
    return save(cv.fig, "5B")


def plt_fmt_signed():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}".replace("-", "−"))


# ================================================================== 5C
def derive_5C():
    f = rd(os.path.join(S.DATA, "fig4D_ledger.tsv"))
    ls = {ds: rd(os.path.join(S.RESULTS, ds, "ledger", "ledger_summary.tsv")) for ds in ("levine32_v2", "wingmen_v1")}
    ntg = {"levine32_v2": 1140, "wingmen_v1": 2600}
    for ds in ntg:  # number of targets = rows of a ledger file
        p = os.path.join(S.RESULTS, ds, "ledger", ("T" if ds == "levine32_v2" else "CD4T") + "_rep0_opt4_targets.tsv")
        assert len(rd(p)) == ntg[ds]
    left = []
    for ds, G in FACETS:
        row = f[(f.dataset == ds) & (f.parent == G) & (f.req == "lp_true")].iloc[0]
        l = ls[ds]
        nr = int(((l.parent == G) & (l.req == "lp_true")).sum())
        r, eps = META[(ds, G)]
        left.append(dict(dataset=ds, parent=G, r=r, eps=eps, n_targets=ntg[ds], n_runs=nr,
                         gained_half=row.gained_half, lost_half=row.lost_half, gained_err=row.gained_err,
                         lost_err=row.lost_err, gained_half_rounded=int(round(row.gained_half)),
                         lost_half_rounded=int(round(row.lost_half)), gained_err_rounded=int(round(row.gained_err)),
                         lost_err_rounded=int(round(row.lost_err))))
    left = pd.DataFrame(left)
    wsave(left, "C_left_gained_lost")
    t = rd(os.path.join(LEDGER, "CD4T_rep0_opt4_targets.tsv"))
    eps = META[("wingmen_v1", "CD4T")][1]
    gb, go = t.half_base <= eps, t.half_opt <= eps
    t["status"] = np.select([go & ~gb, ~go & gb, go & gb], ["gained", "lost", "pass_both"], "fail_both")
    t["dlam"] = t.lam_opt - t.lam_base
    # cross-check against the frozen ledger summary row of this ledger (rep 0, opt 4, alloc 0, true requirement)
    l = ls["wingmen_v1"]
    ref = l[(l.parent == "CD4T") & (l.req == "lp_true") & (l.init_rep == 0) & (l.opt_seed == 4) & (l.alloc == 0)].iloc[0]
    assert (t.status == "gained").sum() == ref.gained_half and (t.status == "lost").sum() == ref.lost_half
    ct = pd.crosstab(t.dlam, t.status).reindex(columns=["gained", "lost", "pass_both", "fail_both"], fill_value=0)
    ct["n"] = ct.sum(1)
    wsave(ct.reset_index(), "C_right_dlam_status")
    bins = np.linspace(0, 0.25, 11)
    hist = pd.DataFrame(dict(bin_lo=bins[:-1], bin_hi=bins[1:]))
    for s_ in ["all", "gained", "lost"]:
        v = t.ref_var if s_ == "all" else t.ref_var[t.status == s_]
        cnt = np.histogram(np.clip(v, 0, 0.25), bins=bins)[0]
        hist[f"n_{s_}"] = cnt
        hist[f"share_{s_}"] = cnt / len(v)
    wsave(hist, "C_right_refvar_hist")
    return left, t, ct, hist


def swatch(cv, x, y, fc, hatch=None, w=3.0, h=1.9):
    cv.base.add_patch(Rectangle((x, y + 0.85 - h / 2), w, h, fc=fc, ec="white" if hatch else "none", lw=0,
                                hatch=hatch, zorder=5))
    return x + w


def panel_5C():
    left, t, ct, hist = derive_5C()
    cv = Canvas("5C", h=72, w=91)       # 91 + 2 + 91 = 184 mm composite row
    W, H = cv.W, cv.H
    tier(cv, "Replay")
    title(cv, "Who newly passes, who drops out")
    yh = H - 7.6
    # ---------------- left: diverging gained / lost bars, mean over runs
    cv.text(1, yh, "Mean over runs, rounded", fontweight="bold", va="baseline")
    cv.text(1, yh - 3.0, "matched requirement vs classical stack", va="baseline", color=S.MUTED)
    ax_y0, ax_top = 28.2, H - 15.0     # notes end at baseline 2.0, level with the last line of 5B
    axL = cv.ax(1, ax_y0, 42.5, ax_top - ax_y0)
    axL.axis("off")
    lo_x, hi_x = -330, 560
    axL.set_xlim(lo_x, hi_x)
    ypos, rows = 0.0, []
    for i, rw in left.iterrows():
        rows.append((ypos - 1.0, rw, "half"))
        rows.append((ypos - 2.0, rw, "err"))
        axL.text(lo_x, ypos - 0.18, f"{FNAME[(rw.dataset, rw.parent)]}", ha="left", va="center", fontsize=S.FS,
                 fontweight="bold")
        axL.text(lo_x + (hi_x - lo_x) * 0.505, ypos - 0.18, f"r = {rw.r} · ε = {rw.eps:.2f}", ha="left",
                 va="center", fontsize=S.FS, color=S.MUTED)
        ypos -= 3.15
    for y, rw, crit in rows:
        g, l_ = rw[f"gained_{crit}"], rw[f"lost_{crit}"]
        hatch = None if crit == "half" else S.LIMITS_HATCH
        axL.barh(y, g, height=0.74, color=S.METHOD["lp_true"], lw=0, hatch=hatch, edgecolor="white")
        axL.barh(y, -l_, height=0.74, color=S.ALARM, lw=0, hatch=hatch, edgecolor="white")
        axL.text(g + 12, y, f"+{int(round(g))}", va="center", ha="left", fontsize=S.FS)
        axL.text(-l_ - 12, y, f"−{int(round(l_))}", va="center", ha="right", fontsize=S.FS)
    ymin = min(y for y, _, _ in rows) - 0.6
    axL.set_ylim(ymin, 0.45)
    for y, rw, crit in rows:
        if crit == "half":
            axL.plot([0, 0], [y + 0.55, y - 1.55], color=S.INK, lw=S.LW, solid_capstyle="butt")
    # bar key
    yk = ax_y0 - 3.6
    x = swatch(cv, 1, yk, S.ALARM)
    x = cv.rich(x + 0.6, yk, ["lost  "])
    x = swatch(cv, x, yk, S.METHOD["lp_true"])
    cv.rich(x + 0.6, yk, ["gained  (targets)"])
    yk2 = yk - 3.15
    x = swatch(cv, 1, yk2, S.MUTED)
    x = cv.rich(x + 0.6, yk2, ["half-width  "])
    x = swatch(cv, x, yk2, S.MUTED, hatch=S.LIMITS_HATCH)
    cv.rich(x + 0.6, yk2, ["reference error"])
    # ---------------- right: one ledger, strata of delta-lambda
    xr = 46.0
    cv.text(xr, yh, "One ledger: WINGMEN CD4 T", fontweight="bold", va="baseline")
    cv.text(xr, yh - 3.0, f"r = 180 · ε = 0.03 · {len(t)} targets", va="baseline", color=S.MUTED)
    axS = cv.ax(55.0, ax_y0, 27.0, ax_top - ax_y0 - 1.0)
    axH = cv.ax(83.0, ax_y0, 7.2, ax_top - ax_y0 - 1.0, sharey=axS)
    style_ax(axS)
    style_ax(axH)
    order = ["fail_both", "pass_both", "gained", "lost"]
    col = {"fail_both": S.GRID, "pass_both": S.MUTED, "gained": S.METHOD["lp_true"], "lost": S.ALARM}
    msz = {"fail_both": 1.1, "pass_both": 1.1, "gained": 1.4, "lost": 1.8}
    xpos = {}
    for dl, g in t.groupby("dlam"):
        idx = np.arange(len(g))
        jit = ((idx * 0.6180339887498949) % 1.0 - 0.5) * 0.72   # deterministic golden-ratio spread, file order
        xpos.update(dict(zip(g.index, dl + jit)))
    t["x"] = pd.Series(xpos)
    for s_ in order:
        g = t[t.status == s_]
        axS.plot(g.x, g.ref_var, ls="none", marker="o", ms=msz[s_], mfc=col[s_], mec="none", zorder=2 + order.index(s_))
    tr = t[t.triple == "CCR7/CD27/CD45RA"].iloc[0]
    r1 = t.iloc[0]
    S.tstar_marker(axS, tr.x, tr.ref_var, ms=6, zorder=10)
    axS.plot([r1.x], [r1.ref_var], ls="none", marker="o", ms=3.8, mfc="none", mec=S.INK, mew=S.LW, zorder=10)
    xl_, yl_ = tr.x + 0.66, 0.2695                     # T* label, kept clear of its leader line and of "row 1"
    axS.annotate("", xy=(tr.x, tr.ref_var), xytext=(xl_ - 0.06, yl_), annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.INK, shrinkA=2.0, shrinkB=4.2))
    S.tstar_label(axS, xl_, yl_, text="T", ms=6, va="center", fontsize=S.FS)
    axS.annotate("row 1", (r1.x, r1.ref_var), xytext=(2.33, 0.262), fontsize=S.FS, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.INK, shrinkA=0.5, shrinkB=2.6), annotation_clip=False)
    dls = list(ct.index)
    axS.set_xticks(dls, [f"{'+' if d > 0 else ('−' if d < 0 else '')}{abs(d)}\n{int(ct.loc[d, 'n'])}" for d in dls])
    axS.tick_params(axis="x", length=0, pad=2.0)
    axS.set_xlim(min(dls) - 0.5, max(dls) + 0.5)
    axS.set_ylim(0, 0.28)
    axS.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25], ["0", "0.05", "0.10", "0.15", "0.20", "0.25"])
    axS.set_xlabel("Δλ (reallocated − classical); n", labelpad=1.5)
    axS.set_ylabel("Max reference variance", labelpad=1.5)
    yb = (hist.bin_lo + hist.bin_hi) / 2
    hgt = hist.bin_hi - hist.bin_lo
    axH.barh(yb, hist.share_all, height=hgt, color=S.GRID, lw=0)
    ysv = np.ravel(np.column_stack([hist.bin_lo.values, hist.bin_hi.values]))
    for s_, c in (("gained", S.METHOD["lp_true"]), ("lost", S.ALARM)):
        axH.plot(np.repeat(hist[f"share_{s_}"].values, 2), ysv, color=c, lw=S.LW)
    axH.tick_params(axis="y", labelleft=False)
    axH.set_xlim(0, 0.55)
    axH.set_xticks([0, 0.5], ["0", ".5"])
    axH.set_xlabel("Share", labelpad=1.5)
    # ---------------- legend + notes
    y = ax_y0 - 13.6
    y = cv.legend(1, y, [(dict(marker="o", ms=2.6, mfc=S.METHOD["lp_true"], mec="none"),
                          f"gained ({(t.status == 'gained').sum()})"),
                         (dict(marker="o", ms=2.6, mfc=S.ALARM, mec="none"), f"lost ({(t.status == 'lost').sum()})"),
                         (dict(marker="o", ms=2.6, mfc=S.MUTED, mec="none"),
                          f"≤ ε both ({(t.status == 'pass_both').sum()})"),
                         (dict(marker="o", ms=2.6, mfc=S.GRID, mec="none"),
                          f"> ε both ({(t.status == 'fail_both').sum()})")], maxw=W - 2)
    y = cv.para(1, y, "Left (mean of 18 runs; Levine-32 1140, WINGMEN 2600 targets) and right (one ledger) are different "
                      "summaries.")
    y = cv.para(1, y, "Max reference variance: max over 8 patterns; post-hoc explanation only.")
    cv.para(1, y, "Pass: ≤ ε, max over 8 patterns (95% CP per pattern; not joint).", color=S.MUTED)
    return save(cv.fig, "5C")



# ================================================================== 5D
def panel_5D():
    from matplotlib.ticker import MaxNLocator
    recs, _, summ = derive_5B()
    D = []
    for ds, G in FACETS:
        Sm = summ[(summ.dataset == ds) & (summ.parent == G)].set_index("kind")
        D.append(dict(dataset=ds, parent=G,
                      d_half_true_vs_classical=Sm.loc["lp_true", "half_mean"] - Sm.loc["classical", "half_mean"],
                      d_err_true_vs_classical=Sm.loc["lp_true", "err_mean"] - Sm.loc["classical", "err_mean"]))
    D = pd.DataFrame(D)
    wsave(summ.merge(D, on=["dataset", "parent"]), "D_tradeoff")
    cv = Canvas("5D", h=72, w=91)       # same height as 5E (composite row); legend level with the 5E legend
    W, H = cv.W, cv.H
    tier(cv, "Replay")
    title(cv, "Half-width gain vs reference-error loss")
    ax_top = H - 17.4
    ax_y0, ax_w = 25.2, 20.0
    ax_h = ax_top - ax_y0
    xs = [10.5, 40.25, 70.0]
    for (ds, G), x0 in zip(FACETS, xs):
        ax = cv.ax(x0, ax_y0, ax_w, ax_h)
        style_ax(ax)
        Sm = summ[(summ.dataset == ds) & (summ.parent == G)].set_index("kind")
        dd = D[(D.dataset == ds) & (D.parent == G)].iloc[0]
        R = recs[(recs.dataset == ds) & (recs.parent == G)]
        for k in KINDS:   # faint per-record cloud (raw coordinates): 3 / 9 / 54 / 18 records
            g = R[R.kind == k]
            ax.plot(g.half_ok, g.err_ok, ls="none", marker=S.MARKER[k], ms=2.0, mfc=S.METHOD[k], mec="none",
                    alpha=0.35, zorder=2)
        for k in KINDS:
            m = Sm.loc[k]
            ax.plot([m.half_mean - m.half_sd, m.half_mean + m.half_sd], [m.err_mean] * 2, color=S.METHOD[k], lw=S.LW,
                    solid_capstyle="butt", zorder=3)
            ax.plot([m.half_mean] * 2, [m.err_mean - m.err_sd, m.err_mean + m.err_sd], color=S.METHOD[k], lw=S.LW,
                    solid_capstyle="butt", zorder=3)
            ax.plot([m.half_mean], [m.err_mean], ls="none", marker=S.MARKER[k], ms=3.6, mfc=S.METHOD[k], mec=S.INK,
                    mew=S.THIN, zorder=5)
        a, b = Sm.loc["classical"], Sm.loc["lp_true"]
        ax.add_patch(FancyArrowPatch((a.half_mean, a.err_mean), (b.half_mean, b.err_mean), arrowstyle="-|>",
                                     mutation_scale=6, lw=S.LW, color=S.INK, shrinkA=4, shrinkB=4, zorder=4))
        # axis limits from ALL records and all mean +- SD (no record is clipped)
        xl = [min(R.half_ok.min(), (Sm.half_mean - Sm.half_sd).min()), max(R.half_ok.max(), (Sm.half_mean + Sm.half_sd).max())]
        yl = [min(R.err_ok.min(), (Sm.err_mean - Sm.err_sd).min()), max(R.err_ok.max(), (Sm.err_mean + Sm.err_sd).max())]
        sx, sy = xl[1] - xl[0], yl[1] - yl[0]
        ax.set_xlim(xl[0] - 0.07 * sx, xl[1] + 0.07 * sx)
        ax.set_ylim(yl[0] - 0.06 * sy, yl[1] + 0.06 * sy)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        pct_fmt(ax, "x")
        pct_fmt(ax, "y")
        hx = x0 - 7.0
        facet_head(cv, hx, H - 6.4, ds, G)
        cv.text(hx, H - 12.4, f"Δ half-width +{dd.d_half_true_vs_classical * 100:.2f} pp", va="baseline")
        cv.text(hx, H - 15.4, f"Δ ref. error −{abs(dd.d_err_true_vs_classical) * 100:.2f} pp", va="baseline")
        if (ds, G) == ("wingmen_v1", "CD4T"):
            m = Sm.loc["classical"]
            ax.annotate("uniform =\nclassical", (m.half_mean, m.err_mean), xytext=(0.22, 0.10),
                        textcoords="axes fraction", fontsize=S.FS, ha="left", va="center", color=S.MUTED,
                        arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.MUTED, shrinkA=1, shrinkB=3))
    cv.text(W / 2 + 4, ax_y0 - 6.4, "Half-width ≤ ε (% of targets)", ha="center", va="baseline")
    cv.text(2.4, ax_y0 + ax_h / 2, "Reference error ≤ ε (%)", rotation=90, ha="center", va="center")
    y = ax_y0 - 10.6
    y = cv.legend(1, y, [(mk("classical"), "Classical stack (3)"), (mk("uniform"), "Uniform (9)"),
                         (mk("lp_shuffled"), "Shuffled (54)"), (mk("lp_true"), "Matched (18)")])
    y = cv.para(1, y, "Faint symbols: records; cross: mean ± SD over records. Arrow: classical → matched requirement, "
                      "same tube budget. Axes differ between facets.")
    cv.para(1, y, "Per target: max over 8 patterns (95% CP; not joint). No frontier fitted.", color=S.MUTED)
    return save(cv.fig, "5D")


# ================================================================== 5E
def derive_5E():
    raw = pd.read_csv(os.path.join(S.DATA, "supp_act5_threshold.tsv"), sep="\t", header=[0, 1])
    raw.columns = [a if (b.startswith("Unnamed") or b == "") else f"{a}_{b}" for a, b in raw.columns]
    d = raw.rename(columns={"half_ok_mean": "half_mean", "half_ok_std": "half_sd", "err_ok_mean": "err_mean",
                            "err_ok_std": "err_sd"})
    d["kind"] = d.req_kind.map(S.ALIAS)
    # n per offset group (e5l: initial design 0 only; 3 opt seeds x 3 allocations; shuffled x 3 seeds)
    nrec = []
    for ds in ("levine32_v2", "wingmen_v1"):
        e = rd(os.path.join(S.RESULTS, ds, "e5l_act5_threshold.tsv"))
        assert set(e.parent) == {"T" if ds == "levine32_v2" else "CD4T"}
        n = e.groupby(["offset", "req_kind"]).agg(size=("half_ok", "size"), mean=("half_ok", "mean"),
                                                  emean=("err_ok", "mean"), esd=("err_ok", "std")).reset_index()
        n.insert(0, "dataset", ds)
        nrec.append(n)
    nrec = pd.concat(nrec)
    d = d.merge(nrec, on=["dataset", "offset", "req_kind"])
    assert len(d) == 16 and np.allclose(d["mean"], d.half_mean) and np.allclose(d.emean, d.err_mean) \
        and np.allclose(d.esd, d.err_sd)
    d = d.drop(columns=["mean", "emean", "esd"]).rename(columns={"size": "n_records"})
    d["source"] = "e5l offset run (initial design 0)"
    # default group = Fig. 5B summary (e5i records, classical and uniform de-duplicated); err SD from the same
    # records (not 0)
    _, _, summ = derive_5B()
    dflt = summ[((summ.dataset == "levine32_v2") & (summ.parent == "T")) |
                ((summ.dataset == "wingmen_v1") & (summ.parent == "CD4T"))]
    assert dict(zip(dflt.kind, dflt.n_records)) == {"classical": 3, "uniform": 9, "lp_shuffled": 54, "lp_true": 18}
    dflt = pd.DataFrame(dict(dataset=dflt.dataset, offset=0.0, req_kind=dflt.req_kind, half_mean=dflt.half_mean,
                             half_sd=dflt.half_sd, err_mean=dflt.err_mean, err_sd=dflt.err_sd, kind=dflt.kind,
                             n_records=dflt.n_records,
                             source="default = Fig. 5B summary (e5i; initial designs 0-1; classical and uniform "
                                    "identical across designs, one record each)"))
    assert dflt.err_sd.notna().all()
    out = pd.concat([d, dflt], ignore_index=True).sort_values(["dataset", "offset", "kind"])
    wsave(out, "E_threshold")
    return out


def derive_5E_records(summary):
    """Per-record points for 5E (each record once, raw values). Offsets: e5l (initial design 0 only, shuffle seeds
    700+s); default: e5i records as in Fig. 5B/5D (initial designs 0-1, shuffle seeds 500+s+10*design; classical and
    uniform de-duplicated to one record per distinct evaluation). Group means of these records must equal the
    plotted summary."""
    rows = []
    for ds, G in (("levine32_v2", "T"), ("wingmen_v1", "CD4T")):
        e = rd(os.path.join(S.RESULTS, ds, "e5l_act5_threshold.tsv"))
        e = e[e.parent == G].assign(kind=lambda x: x.req_kind.map(S.ALIAS), init_rep=0,
                                    source="e5l offset run (initial design 0)")
        rows.append(e[["dataset", "parent", "offset", "kind", "init_rep", "opt_seed", "alloc", "half_ok", "err_ok",
                       "source"]])
    recs, _, _ = derive_5B()
    dfl = recs[((recs.dataset == "levine32_v2") & (recs.parent == "T")) |
               ((recs.dataset == "wingmen_v1") & (recs.parent == "CD4T"))].copy()
    dfl["offset"] = 0.0
    dfl["source"] = "e5i default (initial designs 0-1)"
    rows.append(dfl[["dataset", "parent", "offset", "kind", "init_rep", "opt_seed", "alloc", "half_ok", "err_ok",
                     "source"]])
    r = pd.concat(rows, ignore_index=True)
    chk = r.groupby(["dataset", "offset", "kind"]).agg(n=("half_ok", "size"), hm=("half_ok", "mean"),
                                                       em=("err_ok", "mean")).reset_index()
    chk = chk.merge(summary, on=["dataset", "offset", "kind"])
    assert len(chk) == 24 and (chk.n == chk.n_records).all() and np.allclose(chk.hm, chk.half_mean) \
        and np.allclose(chk.em, chk.err_mean)
    wsave(r, "E_records")
    return r


def panel_5E():
    from matplotlib.ticker import MaxNLocator
    d = derive_5E()
    rec = derive_5E_records(d)
    cv = Canvas("5E", h=72, w=91)       # 91 + 2 + 91 = 184 mm composite row
    W, H = cv.W, cv.H
    S.limits_tag(cv.base, x=1.2 / W, y=(H - 3.0) / H, va="baseline")
    tier(cv, "Replay · conditional")
    cv.text(1.2 + cv.width("Limits") + 3.2, H - 3.0, "Gain depends on the threshold", fontweight="bold", va="baseline")
    r2_y0, r2_h, gap, r1_h = 24.5, 14.0, 2.4, 19.0
    r1_y0 = r2_y0 + r2_h + gap
    rows = [("half", r1_y0, r1_h), ("err", r2_y0, r2_h)]
    specs = [(("levine32_v2", "T"), 10.5, 35.0), (("wingmen_v1", "CD4T"), 55.0, 35.0)]
    groups = [-0.25, 0.0, 0.25]
    off = dict(zip(KINDS, [-0.27, -0.09, 0.09, 0.27]))
    for (ds, G), x0, w in specs:
        g = d[d.dataset == ds]
        gr = rec[rec.dataset == ds]
        for met, y0, h in rows:
            ax = cv.ax(x0, y0, w, h)
            style_ax(ax)
            ax.axvspan(0.5, 1.5, color=S.GRID, lw=0, zorder=0, alpha=0.6)
            mcol, scol = f"{met}_mean", f"{met}_sd"
            for gi, o in enumerate(groups):
                for k in KINDS:
                    rw = g[(g.offset == o) & (g.kind == k)].iloc[0]
                    x = gi + off[k]
                    dflt = o == 0.0
                    # faint per-record points (randomisation repeats; each record once, raw values), left of the mean
                    pts = gr[(gr.offset == o) & (gr.kind == k)][f"{met}_ok"].values
                    xj = x - 0.075 + offsets(len(pts), 0.05)
                    ax.plot(xj, pts, ls="none", marker=S.MARKER[k], ms=1.6, mfc=S.METHOD[k], mec="none", alpha=0.45,
                            zorder=2)
                    ax.plot([x, x], [rw[mcol] - rw[scol], rw[mcol] + rw[scol]], color=S.METHOD[k], lw=S.LW,
                            solid_capstyle="butt", zorder=3)
                    ax.plot([x], [rw[mcol]], ls="none", marker=S.MARKER[k], ms=4.0,
                            mfc="white" if dflt else S.METHOD[k], mec=S.METHOD[k] if dflt else S.INK,
                            mew=S.LW if dflt else S.THIN, zorder=4)
            ax.set_xlim(-0.5, 2.5)
            ax.set_xticks([0, 1, 2], ["−0.25", "default", "+0.25"])
            ax.tick_params(axis="x", length=0, labelbottom=(met == "err"))
            ax.yaxis.set_major_locator(MaxNLocator(4 if met == "half" else 3))
            pct_fmt(ax, "y")
            lo = min((g[mcol] - g[scol]).min(), gr[f"{met}_ok"].min())      # all records and all mean +- SD
            hi = max((g[mcol] + g[scol]).max(), gr[f"{met}_ok"].max())
            span = hi - lo
            ax.set_ylim(lo - 0.07 * span, hi + 0.08 * span)
            if met == "half" and ds == "levine32_v2":
                q = g[g.offset == -0.25].set_index("kind").half_mean
                ymm = np.diff(ax.get_ylim())[0] / h          # data units per mm
                yt = q[["lp_true", "uniform", "lp_shuffled"]].max() + 7.1 * ymm
                for j, (k, lab) in enumerate((("lp_true", "M"), ("uniform", "U"), ("lp_shuffled", "S"))):
                    ax.text(0.0, yt - j * 2.75 * ymm, f"{lab} {q[k] * 100:.2f}", fontsize=S.FS, ha="center",
                            va="baseline")
                ax.text(-0.27 + 0.11, q["classical"], f"C {q['classical'] * 100:.2f}", fontsize=S.FS, ha="left",
                        va="center_baseline")
        facet_head(cv, x0 - 7.0, H - 7.0, ds, G, name=None if ds == "levine32_v2" else "WINGMEN: CD4 T only")
    for (met, y0, h), (l1, l2) in zip(rows, (("Half-width", "attainment (%)"), ("Reference-error", "attainment (%)"))):
        cv.text(1.6, y0 + h / 2, l1, rotation=90, ha="center", va="center")
        cv.text(4.5, y0 + h / 2, l2, rotation=90, ha="center", va="center")
    cv.text(W / 2 + 3, r2_y0 - 6.4, "Threshold offset (all markers)", ha="center", va="baseline")
    y = 14.6
    y = cv.legend(1, y, [(mk("classical"), "Classical (C)"), (mk("uniform"), "Uniform (U)"),
                         (mk("lp_shuffled"), "Shuffled (S)"), (mk("lp_true"), "Matched (M)")])
    y = cv.para(1, y, "Levine-32 −0.25: requirement-specific gain shrinks markedly; all above C.")
    y = cv.para(1, y, "Default (hollow) and offset groups use different configurations; not linked.")
    cv.para(1, y, "Faint symbols: records, n (C/U/S/M) = default 3/9/54/18, offsets 3/9/27/9. Bars: ±SD. Per target: "
                  "max over 8 patterns (95% CP; not joint).", color=S.MUTED)
    return save(cv.fig, "5E")


PANELS = {"5A": panel_5A, "5B": panel_5B, "5C": panel_5C, "5D": panel_5D, "5E": panel_5E}

if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PANELS)):
        PANELS[p]()
