"""D3 (figure plan v2): deterministic restore of the frozen e5b run WINGMEN CD4T, r=130, classical_stack seed 0,
alloc 0, then export the 8-pattern counts of T* = CCR7/CD27/CD45RA in its single witnessing tube.

Restore rule: same inputs, order, thresholds and pattern coding as src/e5b.py; the rebuilt run must
match the frozen row of results/wingmen_v1/e5b_steiner_nodes.tsv on covered, half_ok, err_ok, min_cells, lam_min,
lam_cv, otherwise nothing is exported.  No new randomness, no new scenario.
Outputs: figures/derived/d3_tstar_tube130.tsv, figures/derived/d3_check.json"""
import itertools, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WF, "src"))
import replay, budget, audit, e5b  # noqa: E402

OUT = os.path.join(WF, "figures", "derived")
os.makedirs(OUT, exist_ok=True)
G, EPS, R, ALLOC = "CD4T", 0.05, 130, 0
TSTAR = ("CCR7", "CD27", "CD45RA")

proto = json.load(open(os.path.join(WF, "protocols", "wingmen_v1.json"), encoding="utf-8"))
res = os.path.join(WF, "results", "wingmen_v1")
Xt, markers, sample = replay.load(proto)
part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
thr = json.load(open(os.path.join(res, "thresholds.json")))
Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
pm = replay.parent_masks(Bin, markers, proto["parents"])
bb = set(proto["backbone"])
cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
v = len(markers)
tarr = np.array(list(itertools.combinations(cand, 3)))
ini, fol, ref = part == "initial", part == "followup", part == "reference"
folidx = np.flatnonzero(fol)
N = len(folidx)
piG = pm[G][ini].mean()
Rc = Bin[ref & pm[G]]
refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
gpos = np.flatnonzero(pm[G][folidx])
codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
wells = budget.ljcr_stack(cand, e5b.CAP, R, seed=0)
L = e5b.membership(wells, tarr, v)
order = np.random.default_rng(1000 * ALLOC + R).permutation(N)
well_of = np.empty(N, int)
for i, chunk in enumerate(np.array_split(order, R)):
    well_of[chunk] = i
C = e5b.well_counts(codesF, well_of[gpos], R)
m = e5b.evaluate(L, C, refp, EPS, N, piG)

frozen = pd.read_csv(os.path.join(res, "e5b_steiner_nodes.tsv"), sep="\t")
row = frozen[(frozen.parent == G) & (frozen.r == R) & (frozen.method == "classical_stack") & (frozen.seed == 0) & (frozen.alloc == ALLOC)]
assert len(row) == 1, row
row = row.iloc[0]
keys = ["covered", "half_ok", "err_ok", "min_cells", "lam_min", "lam_cv"]
check = {k: dict(rebuilt=float(m[k]), frozen=float(row[k]), match=bool(math.isclose(float(m[k]), float(row[k]), rel_tol=1e-12, abs_tol=1e-12))) for k in keys}
ok = all(c["match"] for c in check.values())

idx = {mk: i for i, mk in enumerate(markers)}
t_sorted = tuple(sorted(idx[x] for x in TSTAR))
ti = [i for i, S in enumerate(tarr) if tuple(S) == t_sorted][0]
tubes = np.flatnonzero(L[ti])
info = dict(restore_ok=ok, check=check, N_followup=int(N), n_parent_followup=int(len(gpos)), pi_G=float(piG), eps=EPS,
            tstar_triple_index=int(ti), tstar_marker_order_bits=[markers[j] for j in t_sorted],
            tstar_tubes_0based=tubes.tolist(), tube_markers=[sorted(markers[j] for j in wells[t]) for t in tubes])
if ok:
    cnt = C[tubes[0], ti, :].astype(int)
    n = int(cnt.sum())
    lo, hi = audit.cp_interval(cnt, n)
    bits = [markers[j] for j in t_sorted]
    lab = ["".join("+" if (code >> (2 - b)) & 1 else "-" for b in range(3)) for code in range(8)]
    df = pd.DataFrame(dict(code=range(8), pattern=lab, count=cnt, n_tube=n, freq=cnt / max(n, 1), cp_lo=lo, cp_hi=hi,
                           half=(hi - lo) / 2, ref=refp[ti], ref_n=len(Rc)))
    df["bits"] = "/".join(bits)
    df.to_csv(os.path.join(OUT, "d3_tstar_tube130.tsv"), sep="\t", index=False)
    info["tstar_n_in_tube"] = n
    info["tstar_maxhalf"] = float(df.half.max())
    info["tstar_maxerr"] = float(np.abs(df.freq - df.ref).max())
json.dump(info, open(os.path.join(OUT, "d3_check.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(info, indent=1))
