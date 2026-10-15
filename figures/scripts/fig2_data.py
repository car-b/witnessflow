"""Deterministic display tables for Figure 2 (panels 2A-2E).  No new experiment, no randomness.

Sources (read only):
  wfgeom (LJCR C(26,6,3) blocks, rose-window orbits, WINGMEN labels, GF(25)/AG view)
  tools/scan/out_lat/lat01_summary.tsv, lat01_hist.tsv, lat02_structure.txt, lat06_derived.txt
  results/e5c.stdout.txt (frozen family list of e5c), src/e5c.py::oa_design (deterministic construction)
  results/wingmen_v1/e5c_oa_replay.tsv, e5c_multiplicities.tsv, figures/data/fig2C_orthogonal_array.tsv
Writes figures/derived/fig2_*.tsv and returns plain python structures for fig2.py.
Every number drawn in Fig. 2 is asserted here against its frozen source."""
import ast, itertools, math, os, re, sys
from collections import Counter
import numpy as np
import pandas as pd

import wfgeom as G
import wfstyle as S

WF = S.WF
OUTLAT = os.path.join(WF, "tools", "scan", "out_lat")
RES = os.path.join(WF, "results", "wingmen_v1")
os.makedirs(S.DERIVED, exist_ok=True)
sys.path.insert(0, os.path.join(WF, "src"))


def _tsv(df, name):
    df.to_csv(os.path.join(S.DERIVED, name), sep="\t", index=False)


def _txt(name):
    return open(os.path.join(OUTLAT, name), encoding="utf-8").read()


# ============================================================== 2A rose window
def rose():
    rows = []
    for bi, b in enumerate(G.BLOCKS):
        k, s = G.ORBIT[bi]
        rows.append(dict(block0=bi, tube_display=bi + 1, orbit0=k, orbit_display=f"B{k + 1}", shift=s,
                         points=" ".join(map(str, b)), markers=" ".join(G.LABELS[x] for x in b),
                         tstar=int(bi in G.TSTAR_TUBE)))
    df = pd.DataFrame(rows)
    # checks: 5 orbits x 26, all 2600 triples exactly once, 130 x C(6,3) = C(26,3) = 2600
    assert Counter(df.orbit0) == Counter({k: 26 for k in range(5)})
    tri = Counter(t for b in G.BLOCKS for t in itertools.combinations(b, 3))
    assert len(tri) == 2600 and set(tri.values()) == {1}
    assert 130 * math.comb(6, 3) == math.comb(26, 3) == 2600
    pair = Counter(p for b in G.BLOCKS for p in itertools.combinations(b, 2))
    assert set(pair.values()) == {6} and len(pair) == 325
    deg = Counter(p for b in G.BLOCKS for p in b)
    assert set(deg.values()) == {30}
    assert G.ORBIT[72] == (4, 23)
    assert set(G.BLOCKS[72]) == {(x + 23) % 26 for x in G.ROSE_BASE[4]}
    _tsv(df, "fig2_rose_blocks.tsv")
    _tsv(pd.DataFrame(dict(point=range(26), marker=G.LABELS)), "fig2_points.tsv")
    return df


def lambda_hist():
    """lat01: Steiner (LJCR_130) all lambda = 1; random 130-tube designs: mean over 200 designs."""
    summ = pd.read_csv(os.path.join(OUTLAT, "lat01_summary.tsv"), sep="\t")
    hist = pd.read_csv(os.path.join(OUTLAT, "lat01_hist.tsv"), sep="\t")
    rnd = summ[(summ.v == 26) & (summ.design == "random") & (summ.r == 130)].iloc[0]
    assert int(rnd.n_designs) == 200
    assert abs(rnd.frac0 - 0.366521) < 1e-6
    h_r = hist[(hist.v == 26) & (hist.design == "random") & (hist.r == 130)].set_index("lam")["frac"]
    h_s = hist[(hist.v == 26) & (hist.design == "LJCR_130") & (hist.r == 130)].set_index("lam")["frac"]
    assert list(h_s.index) == [1] and abs(h_s.loc[1] - 1) < 1e-12
    assert abs(h_r.loc[0] - rnd.frac0) < 1e-9
    lam = list(range(0, 6))
    r = [float(h_r.get(l, 0.0)) for l in lam[:-1]] + [float(h_r[h_r.index >= 5].sum())]
    s = [float(h_s.get(l, 0.0)) for l in lam]
    df = pd.DataFrame(dict(lam=["0", "1", "2", "3", "4", ">=5"], steiner_frac=s, random_mean_frac=r))
    df["random_n_designs"] = 200
    df["source"] = "tools/scan/out_lat/lat01_hist.tsv (design=random/LJCR_130, v=26, r=130); lat01_summary.tsv frac0"
    _tsv(df, "fig2_A_lambda_hist.tsv")
    return df, float(rnd.frac0), int(rnd.n_designs)


# ============================================================== 2B / 2C AG(2,5) view
INF_MARKER, Q_MARKER, R_MARKER = "CCR7", "CD27", "CD45RA"


def ag():
    v = G.ag_view(G.POINT[INF_MARKER])
    C = v["coords"]
    q, r = G.POINT[Q_MARKER], G.POINT[R_MARKER]
    qa, qb = C[q]
    assert (qa, qb) == (2, 0) and C[r] == (1, 4)

    def disp(z):
        """display window: x = a in 0..4, y = b shifted so that Q sits in the middle row (y in -2..2)."""
        a, b = z
        return (a, ((b - qb + 2) % 5) - 2 + qb)

    pts = []
    for p in range(26):
        z = C[p]
        if z == G.INF:
            pts.append(dict(point=p, marker=G.LABELS[p], a="inf", b="inf", x=np.nan, y=np.nan, rho_about_Q=""))
            continue
        x, y = disp(z)
        rho = G.norm(G.gsub(z, C[q])) if p != q else 0
        pts.append(dict(point=p, marker=G.LABELS[p], a=z[0], b=z[1], x=x, y=y, rho_about_Q=rho))
    P = pd.DataFrame(pts)
    lines = []
    for l in v["lines"]:
        lines.append(dict(block0=l["block"], tube_display=l["block"] + 1, slope=str(l["slope"]),
                          points=" ".join(map(str, l["points"])), markers=" ".join(G.LABELS[x] for x in l["points"]),
                          through_Q=int(q in l["points"]), through_Q_and_R=int(q in l["points"] and r in l["points"])))
    L = pd.DataFrame(lines)
    thruQ = L[L.through_Q == 1]
    assert sorted(thruQ.block0) == [8, 34, 54, 63, 70, 72]
    assert dict(zip(thruQ.block0, thruQ.slope)) == {8: "2", 34: "4", 54: "inf", 63: "3", 70: "0", 72: "1"}
    assert list(L[L.through_Q_and_R == 1].block0) == [72]
    # norm circles about Q: 4 classes x 6 points (24 other affine points), each class is one block
    circQ = [c for c in v["circles"] if c["centre"] == C[q]]
    assert len(circQ) == 4
    cls = P[(P.rho_about_Q != "") & (P.point != q)].groupby("rho_about_Q").point.apply(list).to_dict()
    assert sorted(len(x) for x in cls.values()) == [6, 6, 6, 6]
    for c in circQ:
        assert sorted(c["points"]) == sorted(cls[c["rho"]])
    circ = pd.DataFrame([dict(rho=c["rho"], block0=c["block"], tube_display=c["block"] + 1,
                              points=" ".join(map(str, sorted(c["points"]))),
                              markers=" ".join(G.LABELS[x] for x in sorted(c["points"]))) for c in circQ]).sort_values("rho")
    assert len(v["circles"]) == 100 and len({(c["centre"], c["rho"]) for c in v["circles"]}) == 100
    # multiplicity ladder through infinity: 30 / 6 / 1
    assert len(v["lines"]) == 30 and len(thruQ) == 6
    _tsv(P, "fig2_B_ag_points.tsv")
    _tsv(L, "fig2_B_ag_lines.tsv")
    _tsv(circ, "fig2_B_norm_circles_about_CD27.tsv")
    return dict(view=v, P=P, L=L, circ=circ, q=q, r=r, inf=G.POINT[INF_MARKER], disp=disp)


def levine_ladder():
    """lat02_structure.txt, LJCR C(20,6,3) (Levine 71-tube base covering): point / pair / triple multiplicity ranges."""
    t = _txt("lat02_structure.txt")
    sec = t.split("=== LJCR C(20,6,3)")[1]
    deg = ast.literal_eval(re.search(r"point degrees: (\{[^}]*\})", sec).group(1))
    pair = ast.literal_eval(re.search(r"pair multiplicities: (\{[^}]*\})", sec).group(1))
    trip = ast.literal_eval(re.search(r"triple multiplicities: (\{[^}]*\})", sec).group(1))
    lad = dict(point=(min(deg), max(deg)), pair=(min(pair), max(pair)), triple=(min(trip), max(trip)))
    assert lad == dict(point=(21, 24), pair=(5, 12), triple=(1, 6)), lad
    s26 = t.split("=== LJCR C(26,6,3)")[1].split("===")[0]
    assert "point degrees: {30: 26}" in s26 and "pair multiplicities: {6: 325}" in s26 and "triple multiplicities: {1: 2600}" in s26
    df = pd.DataFrame([dict(level=k, classical_130=dict(point=30, pair=6, triple=1)[k], levine71_min=v[0], levine71_max=v[1])
                       for k, v in lad.items()])
    df["source"] = "tools/scan/out_lat/lat02_structure.txt"
    _tsv(df, "fig2_B_multiplicity_ladder.tsv")
    return lad


def gating(A):
    """2C: gate on x = CCR7 (= infinity): only the 30 tubes through x are usable inside x+ cells."""
    x = A["inf"]
    tubes_x = [b for b in G.BLOCKS if x in b]
    others = [p for p in range(26) if p != x]
    pairs_cov = {pr for b in tubes_x for pr in itertools.combinations(sorted(set(b) - {x}), 2)}
    trip_cov = {t for b in tubes_x for t in itertools.combinations(sorted(set(b) - {x}), 3)}
    n_pairs, n_trip = math.comb(25, 2), math.comb(25, 3)
    pc = Counter(pr for b in tubes_x for pr in itertools.combinations(sorted(set(b) - {x}), 2))
    assert len(tubes_x) == 30 and len(pairs_cov) == 300 == n_pairs and set(pc.values()) == {1}
    assert len(trip_cov) == 300 and n_trip == 2300
    t = _txt("lat06_derived.txt")
    assert "witnessed pairs 300/300" in t and "witnessed triples 300/2300" in t
    assert "C(25,6,3) = 130 tubes" in t and "C(25,5,3) = 256 tubes" in t
    df = pd.DataFrame([dict(item="tubes containing x", value=30, total=130),
                       dict(item="pairs of other markers witnessed in x+", value=300, total=300),
                       dict(item="triples of other markers witnessed in x+ (collinear)", value=300, total=2300),
                       dict(item="repair x in backbone (x + 6 = 7 slots): covering certificate C(25,6,3)", value=130, total=np.nan),
                       dict(item="repair capacity 6 (x + 5): covering certificate C(25,5,3)", value=256, total=np.nan)])
    df["source"] = "recomputed from wfgeom blocks; matches tools/scan/out_lat/lat06_derived.txt"
    _tsv(df, "fig2_C_gating_counts.tsv")
    return dict(tubes_x=tubes_x, others=others, trip_cov=trip_cov)


# ============================================================== 2D orthogonal array
def oa():
    import e5c  # only the deterministic construction oa_design() is used
    log = open(os.path.join(WF, "results", "e5c.stdout.txt"), encoding="utf-8").readline()
    fams = ast.literal_eval(log.split("families: ")[1].split(" dropped")[0])
    names = [[re.sub(r"\(\d+\)$", "", m) for m in f] for f in fams]
    assert len(names) == 5 and all(len(f) == 5 for f in names)
    fam_idx = [[5 * x + y for y in range(5)] for x in range(5)]       # symbolic marker ids: family x, slot y
    tubes = e5c.oa_design(fam_idx)
    assert len(tubes) == 125
    abc = list(itertools.product(range(5), repeat=3))                  # same order as oa_design
    for (a, b, c), t in zip(abc, tubes):
        assert t == [5 * x + (a + b * x + c * x * x) % 5 for x in range(5)]
    famof = {5 * x + y: x for x in range(5) for y in range(5)}
    tgt = [t for t in itertools.combinations(range(25), 3) if len({famof[m] for m in t}) == 3]
    assert len(tgt) == 1250

    def cover(rows):
        cnt = Counter(t for r in rows for t in itertools.combinations(sorted(r), 3))
        return sum(1 for t in tgt if cnt.get(t, 0) > 0), cnt

    lin = [t for (a, b, c), t in zip(abc, tubes) if c == 0]
    n_lin, _ = cover(lin)
    n_quad, cq = cover(tubes)
    assert len(lin) == 25 and n_lin == 250 and n_quad == 1250 and all(cq[t] == 1 for t in tgt)
    single = Counter(m for t in tubes for m in t)
    pairs = Counter(p for t in tubes for p in itertools.combinations(sorted(t), 2))
    lin_pairs = Counter(p for t in lin for p in itertools.combinations(sorted(t), 2))
    xpairs = [p for p in itertools.combinations(range(25), 2) if famof[p[0]] != famof[p[1]]]
    assert set(single.values()) == {25} and {pairs[p] for p in xpairs} == {5} and {lin_pairs[p] for p in xpairs} == {1}
    # demonstration (illustrative choice, avoids the T* markers): specified f(0)=1, f(2)=2, f(4)=1
    #   -> unique f = 1 + x + x^2 = (x-2)^2 + 2 over F5
    spec = {0: 1, 2: 2, 4: 1}
    sols = [(a, b, c) for a, b, c in abc if all((a + b * x + c * x * x) % 5 == y for x, y in spec.items())]
    assert sols == [(1, 1, 1)]
    two = [(a, b, c) for a, b, c in abc if all((a + b * x + c * x * x) % 5 == y for x, y in list(spec.items())[:2])]
    one = [(a, b, c) for a, b, c in abc if (a + b * 0 + c * 0) % 5 == spec[0]]
    assert len(two) == 5 and len(one) == 25
    demo_idx = abc.index((1, 1, 1))
    demo = dict(abc=(1, 1, 1), index0=demo_idx, tube_display=demo_idx + 1,
                values=[(1 + x + x * x) % 5 for x in range(5)],
                markers=[names[x][(1 + x + x * x) % 5] for x in range(5)])
    assert not set(demo["markers"]) & set(G.TSTAR)
    pd.DataFrame([dict(specified=" ".join(f"f({x})={y}" for x, y in spec.items()), unique_abc="1 1 1",
                       tube_display=demo["tube_display"], values=" ".join(map(str, demo["values"])),
                       markers=" ".join(demo["markers"]),
                       tubes_through_first_two=" | ".join(f"{a} {b} {c}" for a, b, c in two))]).to_csv(
        os.path.join(S.DERIVED, "fig2_D_demo_tube.tsv"), sep="\t", index=False)
    # codebook ordered by (c, b, a): first 25 rows are the c = 0 'lines'
    order = sorted(range(125), key=lambda i: (abc[i][2], abc[i][1], abc[i][0]))
    code = np.array([[(abc[i][0] + abc[i][1] * x + abc[i][2] * x * x) % 5 for x in range(5)] for i in order])
    cb = pd.DataFrame(code, columns=[f"x{x}" for x in range(5)])
    cb.insert(0, "c", [abc[i][2] for i in order]); cb.insert(0, "b", [abc[i][1] for i in order])
    cb.insert(0, "a", [abc[i][0] for i in order]); cb.insert(0, "tube_display", [i + 1 for i in order])
    _tsv(cb, "fig2_D_codebook_RS53.tsv")
    fam_df = pd.DataFrame([dict(family_x=x, slot_y=y, marker=names[x][y]) for x in range(5) for y in range(5)])
    fam_df["source"] = "results/e5c.stdout.txt (frozen e5c family list; dropped CD11b)"
    _tsv(fam_df, "fig2_D_families.tsv")
    sm = pd.DataFrame([dict(design="lines a+bx (c=0)", tubes=25, triples_covered=n_lin, triples_total=1250, strength=2,
                            cross_family_pair_mult=1, single_mult=5),
                       dict(design="quadratics a+bx+cx^2", tubes=125, triples_covered=n_quad, triples_total=1250, strength=3,
                            cross_family_pair_mult=5, single_mult=25)])
    _tsv(sm, "fig2_D_strength_counts.tsv")
    return dict(names=names, abc=abc, tubes=tubes, two=two, spec=spec, demo=demo, code=code, order=order,
                n_lin=n_lin, n_quad=n_quad)


# ============================================================== 2E replay
METHODS_2E = ["OA_quadratic", "anneal_search", "balanced_random", "legal_random"]


def replay():
    raw = pd.read_csv(os.path.join(RES, "e5c_oa_replay.tsv"), sep="\t")
    D = pd.read_csv(os.path.join(S.DATA, "fig2C_orthogonal_array.tsv"), sep="\t")
    g = raw.groupby(["parent", "method"])
    agg = g[["covered", "half_ok", "err_ok"]].agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_rows"] = g.size()
    agg["n_design_seeds"] = g.seed.nunique()
    agg["n_alloc"] = g.alloc.nunique()
    agg["eps"] = g.eps.first()
    agg = agg.reset_index()
    # cross-check against the frozen D table (means and half_ok min/max)
    for _, d in D.iterrows():
        a = agg[(agg.parent == d.parent) & (agg.method == d.method)].iloc[0]
        for col in ["covered", "half_ok", "err_ok"]:
            assert abs(a[f"{col}_mean"] - d[col]) < 1e-9, (d.parent, d.method, col)
        assert abs(a.half_ok_min - d.half_min) < 1e-9 and abs(a.half_ok_max - d.half_max) < 1e-9
    assert dict(zip(agg.method, agg.n_rows)) == {"OA_quadratic": 10, "anneal_search": 30, "balanced_random": 30, "legal_random": 30}
    agg["source"] = "results/wingmen_v1/e5c_oa_replay.tsv (min/max over design seed x alloc); means = figures/data/fig2C_orthogonal_array.tsv"
    _tsv(agg, "fig2_E_replay_summary.tsv")
    mult = pd.read_csv(os.path.join(RES, "e5c_multiplicities.tsv"), sep="\t")
    gm = mult.groupby("method")
    mr = pd.DataFrame(dict(single_lo=gm.single_min.min(), single_hi=gm.single_max.max(),
                           pair_lo=gm.pair_min.min(), pair_hi=gm.pair_max.max(),
                           triple_lo=gm.triple_min.min(), triple_hi=gm.triple_max.max(),
                           covered_lo=gm.triple_covered.min(), covered_hi=gm.triple_covered.max(),
                           covered_mean=gm.triple_covered.mean(), n_designs=gm.size())).reset_index()
    b = mr.set_index("method")
    assert (b.loc["balanced_random", "single_lo"], b.loc["balanced_random", "single_hi"]) == (25, 25)
    assert (b.loc["OA_quadratic", ["single_lo", "single_hi", "pair_lo", "pair_hi", "triple_lo", "triple_hi"]].tolist()
            == [25, 25, 5, 5, 1, 1])
    mr["source"] = "results/wingmen_v1/e5c_multiplicities.tsv (range over design seeds)"
    _tsv(mr, "fig2_E_multiplicity_ranges.tsv")
    return agg, mr


if __name__ == "__main__":
    rose(); print(lambda_hist()[1:])
    A = ag(); print(A["L"][A["L"].through_Q == 1]); print(A["circ"])
    print(levine_ladder()); gating(A)
    O = oa(); print(O["demo"], O["n_lin"], O["n_quad"])
    agg, mr = replay(); print(agg.round(4).to_string()); print(mr.to_string())
