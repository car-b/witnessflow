"""e5b supplement (spec v2): (1) the r=180 node; (2) tube-splitting control: allocate follow-up cells to the
130 Steiner tubes, then split every tube's cells at random into two tubes with the SAME panel (260 tubes).
Pooled per-triple counts, estimates and CP intervals must be identical -> 'adding tubes per se adds no
information'.  usage: python e5b_supp.py <wingmen protocol>"""
import itertools, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e5b import well_counts, membership, evaluate, cp_halfmax  # noqa: E402

PARENTS = {"CD4T": 0.05, "CD8T": 0.08}


def main(proto_path):
    proto = json.load(open(proto_path, encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows, split_rows = [], []
    for G, eps in PARENTS.items():
        piG = pm[G][ini].mean()
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        # (1) r = 180 node (same seeds convention as e5b)
        r = 180
        designs = {("classical_stack", 0): budget.ljcr_stack(cand, 6, r, seed=0)}
        for s in range(3):
            designs[("random", s)] = design.random_design(cand, 6, r, seed=s)
            designs[("balanced_random", s)] = budget.balanced_random(cand, 6, r, seed=s)
        Ls = {k: membership(w, tarr, v) for k, w in designs.items()}
        for a in range(10):
            order = np.random.default_rng(1000 * a + r).permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, r)):
                well_of[chunk] = i
            C = well_counts(codesF, well_of[gpos], r)
            for (meth, s), L in Ls.items():
                rows.append(dict(parent=G, r=r, method=meth, seed=s, alloc=a, **evaluate(L, C, refp, eps, N, piG)))
        # (2) tube splitting control
        steiner = budget.ljcr_stack(cand, 6, 130, seed=0)
        L130 = membership(steiner, tarr, v)
        L260 = np.concatenate([L130, L130], 1)  # tube i and its clone i+130 share the panel
        for a in range(10):
            rng = np.random.default_rng(9000 + a)
            order = rng.permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, 130)):
                well_of[chunk] = i
            half = rng.random(N) < 0.5
            well_split = np.where(half, well_of + 130, well_of)
            C130 = well_counts(codesF, well_of[gpos], 130)
            C260 = well_counts(codesF, well_split[gpos], 260)
            cnt130 = np.einsum("ti,itx->tx", L130, C130)
            cnt260 = np.einsum("ti,itx->tx", L260, C260)
            m130 = evaluate(L130, C130, refp, eps, N, piG)
            m260 = evaluate(L260, C260, refp, eps, N, piG)
            split_rows.append(dict(parent=G, alloc=a, max_abs_count_diff=int(np.abs(cnt130 - cnt260).max()),
                                   half_ok_130=m130["half_ok"], half_ok_260split=m260["half_ok"],
                                   err_ok_130=m130["err_ok"], err_ok_260split=m260["err_ok"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5b_r180.tsv"), sep="\t", index=False)
    sp = pd.DataFrame(split_rows)
    sp.to_csv(os.path.join(out, "e5b_tube_split.tsv"), sep="\t", index=False)
    print(df.groupby(["parent", "method"])[["covered", "half_ok", "err_ok"]].mean().round(3).to_string())
    print(sp.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
