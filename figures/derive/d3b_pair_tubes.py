"""D3b (figure 4C): from the same deterministic restore as D3 (WINGMEN CD4T, r=130, classical_stack seed 0, alloc 0;
re-verified against the frozen e5b row), export the 4-state counts (00/01/10/11) of the marker pair
P = CCR7, Q = CD27 in each of the 6 tubes that contain both, plus the reference-split proportions.
Descriptive only: no perturbation, no test statistic, no new randomness.
Output: figures/derived/d3b_pair_tubes.tsv"""
import itertools, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WF, "src"))
import replay, budget, audit, e5b  # noqa: E402

G, EPS, R, ALLOC = "CD4T", 0.05, 130, 0
P, Q = "CCR7", "CD27"
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
row = frozen[(frozen.parent == G) & (frozen.r == R) & (frozen.method == "classical_stack") & (frozen.seed == 0) & (frozen.alloc == ALLOC)].iloc[0]
for k in ["covered", "half_ok", "err_ok", "min_cells", "lam_min", "lam_cv"]:
    assert math.isclose(float(m[k]), float(row[k]), rel_tol=1e-12, abs_tol=1e-12), k

ip, iq = markers.index(P), markers.index(Q)
cells = Bin[folidx[gpos]]
tube_of = well_of[gpos]
st = cells[:, ip].astype(int) * 2 + cells[:, iq]
rows = []
tubes = [t for t, W in enumerate(wells) if ip in W and iq in W]
assert len(tubes) == 6, tubes
for t in tubes:
    cnt = np.bincount(st[tube_of == t], minlength=4)
    others = sorted(markers[j] for j in wells[t] if j not in (ip, iq))
    rows.append(dict(tube=t, others="/".join(others), n=int(cnt.sum()), **{f"s{b:02b}": int(cnt[b]) for b in range(4)}))
rst = np.bincount(Rc[:, ip].astype(int) * 2 + Rc[:, iq], minlength=4) / len(Rc)
rows.append(dict(tube=-1, others="reference split", n=len(Rc), **{f"s{b:02b}": float(rst[b]) for b in range(4)}))
df = pd.DataFrame(rows)
df["pair"] = f"{P}/{Q} (state = {P}{Q} bits)"
df.to_csv(os.path.join(WF, "figures", "derived", "d3b_pair_tubes.tsv"), sep="\t", index=False)
print(df.to_string())
