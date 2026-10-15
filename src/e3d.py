"""E3d: hybrid objective = coverage of every triple (lam>=1, weight W_COV) + audit-informed precision
credit (min(lam, lam_T)/lam_T).  Implemented with anneal_lambda on a duplicated target list.
Re-uses the E3c setup; only the tight regimes.  usage: python e3d.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e3b import cp_half  # noqa: E402
import e3c  # noqa: E402

EPS = {"T": [0.03, 0.05], "B": [0.05, 0.07]}
RGRID = [71, 96, 120, 142, 180, 240]
W_COV = 1.0


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
        vT = np.minimum(0.25, (phat * (1 - phat)).max(1) + e3c.MARGIN)
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        for r in RGRID:
            alloc = {}
            for a in e3c.ALLOC_SEEDS:
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
                nreq_T = e3c.Z ** 2 * vT / eps ** 2
                lam_T = np.maximum(1, np.ceil(nreq_T * r / (N * piG))).astype(int)
                st = budget.ljcr_stack(cand, e3c.CAP, r, seed=0)
                t2 = np.vstack([tarr, tarr])
                w2 = np.concatenate([1.0 / lam_T, np.full(len(tarr), W_COV)])
                l2 = np.concatenate([lam_T, np.ones(len(tarr), int)])
                hyb, _ = budget.anneal_lambda(st, cand, t2, w2, l2, v, iters=e3c.ITERS, seed=4)
                aud, _ = budget.anneal_lambda(st, cand, tarr, 1.0 / lam_T, lam_T, v, iters=e3c.ITERS, seed=3)
                for meth, wells in (("LJCR_stack", st), ("stack+anneal_auditinformed", aud), ("stack+anneal_hybrid", hyb)):
                    design.check_design(wells, cand, e3c.CAP)
                    L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).astype(np.int32)
                    lam = L.sum(1)
                    for a, C in alloc.items():
                        cnt = np.einsum("ti,itx->tx", L, C)
                        n = cnt.sum(1)
                        p = cnt / np.maximum(n, 1)[:, None]
                        half = np.where(n > 0, cp_half(cnt, n).max(1), 1.0)
                        err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
                        rows.append(dict(parent=G, eps=eps, r=r, method=meth, alloc=a,
                                         emp_half_ok=float((half <= eps).mean()), emp_err_ok=float((err <= eps).mean()),
                                         covered=float((lam >= 1).mean())))
            print(f"[{time.time() - t0:.0f}s] {G} r={r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3d_hybrid.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    for metric in ["emp_half_ok", "emp_err_ok", "covered"]:
        print(f"\n== {metric}")
        print(df.groupby(["parent", "eps", "r", "method"])[metric].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
