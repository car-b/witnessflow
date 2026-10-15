"""E3c: audit-informed budgeted design.  Per-triple cell requirement from the INITIAL data
(block-kNN imputed pattern frequencies, inflated by a safety margin) instead of worst case p=0.5.
usage: python e3c.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e3b import cp_half  # noqa: E402

EPS = {"T": [0.03, 0.05, 0.07], "B": [0.05, 0.07, 0.10]}
RGRID = [48, 71, 96, 120, 142, 180, 240]
CAP, Z, MARGIN = 6, 1.96, 0.02
ALLOC_SEEDS = [0, 1, 2]
ITERS = 30000


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    mi = {m: i for i, m in enumerate(markers)}
    bbi = [mi[m] for m in proto["backbone"]]
    nonbb = [i for i in range(len(markers)) if i not in bbi]
    cand = [i for i in nonbb if thr[markers[i]]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    # initial experiment = replicate 0 of the v2 replay (same generator sequence)
    idg = proto["initial_design"]
    rng = np.random.default_rng(idg["seed"])
    perm = rng.permutation(nonbb)
    panels0 = [set(bbi) | set(perm[i * idg["capacity"]:(i + 1) * idg["capacity"]].tolist()) for i in range(idg["n_tubes"])]
    ii = np.flatnonzero(ini)
    tube_i = rng.integers(0, idg["n_tubes"], len(ii))
    Bimp = audit.knn_impute(Bin[ii], Xt[ii][:, bbi], tube_i, panels0, k=10, mode="block", seed=0)
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows = []
    for G in ["T", "B"]:
        piG = pm[G][ini].mean()
        gI = pm[G][ii]
        phat = np.stack([np.bincount(audit.codes(Bimp[gI], S), minlength=8) / gI.sum() for S in tarr])
        vT = np.minimum(0.25, (phat * (1 - phat)).max(1) + MARGIN)
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        for r in RGRID:
            alloc = {}
            for a in ALLOC_SEEDS:
                well_of = np.empty(N, int)
                order = np.random.default_rng(1000 * a + r).permutation(N)
                for i, chunk in enumerate(np.array_split(order, r)):
                    well_of[chunk] = i
                wg = well_of[gpos]
                C = np.zeros((r, len(tarr), 8), np.int32)
                for i in range(r):
                    sub = codesF[:, wg == i]
                    for x in range(8):
                        C[i, :, x] = (sub == x).sum(1)
                alloc[a] = C
            for eps in EPS[G]:
                nreq_wc = Z * Z * 0.25 / eps ** 2
                lam_wc = max(1, math.ceil(nreq_wc * r / (N * piG)))
                nreq_T = Z * Z * vT / eps ** 2
                lam_T = np.maximum(1, np.ceil(nreq_T * r / (N * piG))).astype(int)
                designs = {}
                for s in range(3):
                    designs[("random", s)] = design.random_design(cand, CAP, r, seed=s)
                st = budget.ljcr_stack(cand, CAP, r, seed=0)
                designs[("LJCR_stack", 0)] = st
                a1, _ = budget.anneal_lambda(st, cand, tarr, np.ones(len(tarr)), lam_wc, v, iters=ITERS, seed=2)
                designs[("stack+anneal_worstcase", 0)] = a1
                a2, _ = budget.anneal_lambda(st, cand, tarr, 1.0 / lam_T, lam_T, v, iters=ITERS, seed=3)
                designs[("stack+anneal_auditinformed", 0)] = a2
                for (meth, s), wells in designs.items():
                    design.check_design(wells, cand, CAP)
                    L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).astype(np.int32)
                    lam = L.sum(1)
                    for a, C in alloc.items():
                        cnt = np.einsum("ti,itx->tx", L, C)
                        n = cnt.sum(1)
                        p = cnt / np.maximum(n, 1)[:, None]
                        half = np.where(n > 0, cp_half(cnt, n).max(1), 1.0)
                        err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
                        rows.append(dict(parent=G, eps=eps, r=r, method=meth, seed=s, alloc=a,
                                         emp_half_ok=float((half <= eps).mean()),
                                         emp_err_ok=float((err <= eps).mean()),
                                         covered=float((lam >= 1).mean()),
                                         planned_auditreq=float((lam >= lam_T).mean())))
            print(f"[{time.time() - t0:.0f}s] {G} r={r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3c_auditinformed.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    for metric in ["emp_half_ok", "emp_err_ok", "covered"]:
        print(f"\n== {metric}")
        print(df.groupby(["parent", "eps", "r", "method"])[metric].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
