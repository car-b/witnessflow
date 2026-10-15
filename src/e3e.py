"""E3e: initial-design replicates x margin sensitivity x same-information baselines,
hybrid with (effectively hard) coverage.  Tight regimes only.
usage: python e3e.py <protocol.json>
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e3b import cp_half  # noqa: E402

EPS = {"T": [0.03, 0.05], "B": [0.05, 0.07]}
RGRID = [96, 142, 180]
CAP, Z = 6, 1.96
MARGINS = [0.0, 0.02, 0.05]
INIT_REPS = [0, 1, 2]
ALLOC_SEEDS = [0, 1, 2]
ITERS = 30000
W_COV = 100.0   # coverage credit >> any precision credit (max 1 per triple): effectively hard


def initial_designs(proto, nonbb, bbi, n_ini):
    """Same generator sequences as replay.py (panels rng; independent tube rng)."""
    idg = proto["initial_design"]
    rng = np.random.default_rng(idg["seed"])
    rng_tube = np.random.default_rng(idg["seed"] + 1)
    out = []
    for rep in range(max(INIT_REPS) + 1):
        perm = rng.permutation(nonbb)
        panels = [set(bbi) | set(perm[i * idg["capacity"]:(i + 1) * idg["capacity"]].tolist()) for i in range(idg["n_tubes"])]
        tubes = rng_tube.integers(0, idg["n_tubes"], n_ini)
        out.append((panels, tubes))
    return out


def main(proto_path):
    global EPS
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    if "e3" in proto:  # dataset-specific parents / precision targets
        EPS = {k: v for k, v in proto["e3"]["eps"].items()}
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
    ii = np.flatnonzero(ini)
    inits = initial_designs(proto, nonbb, bbi, len(ii))
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows = []
    for G in EPS:
        piG = pm[G][ini].mean()
        gI = pm[G][ii]
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        phats, vlp = {}, {}
        for rep in INIT_REPS:
            panels, tubes = inits[rep]
            Bimp = audit.knn_impute(Bin[ii], Xt[ii][:, bbi], tubes, panels, k=10, mode="block", seed=rep)
            phats[rep] = np.stack([np.bincount(audit.codes(Bimp[gI], S), minlength=8) / gI.sum() for S in tarr])
            # audit (LP) based robust variance: max p(1-p) inside each pattern's confidence interval
            zero = np.zeros(len(ii), int)
            vv = np.zeros(len(tarr))
            for t, S in enumerate(tarr):
                ev = audit.evidence(Bin[ii], tuple(S), gI, zero, 1, tubes, panels)
                lo, hi = audit.conf_bounds_all(ev, 1)
                vv[t] = audit.max_binom_var(lo, hi).max()
            vlp[rep] = vv
            print(f"[{time.time() - t0:.0f}s] {G} LP audit variances rep {rep}: median {np.median(vv):.3f}", flush=True)
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
            st = budget.ljcr_stack(cand, CAP, r, seed=0)
            for eps in EPS[G]:
                for rep in INIT_REPS:
                    for mg in MARGINS:
                        vT = np.minimum(0.25, (phats[rep] * (1 - phats[rep])).max(1) + mg)
                        need = Z * Z * vT / eps ** 2
                        lam_T = np.maximum(1, np.ceil(need * r / (N * piG))).astype(int)
                        cpw = N * piG / r
                        designs = {"LJCR_stack": st}
                        designs["anneal_audit"], _ = budget.anneal_lambda(st, cand, tarr, 1.0 / lam_T, lam_T, v, iters=ITERS, seed=3)
                        t2 = np.vstack([tarr, tarr])
                        w2 = np.concatenate([1.0 / lam_T, np.full(len(tarr), W_COV)])
                        l2 = np.concatenate([lam_T, np.ones(len(tarr), int)])
                        # temperature on the precision-credit scale (coverage acts as a near-hard constraint)
                        designs["hybrid_hardcov"], _ = budget.anneal_lambda(st, cand, t2, w2, l2, v, iters=ITERS, seed=4,
                                                                            t0=float(np.mean(1.0 / lam_T)))
                        designs["greedy_audit"] = budget.greedy_lambda(cand, CAP, r, tarr, 1.0 / lam_T, lam_T, v)
                        designs["packing_audit"] = budget.packing_deficit(cand, CAP, r, tarr, 1.0 / lam_T, need, cpw)
                        if mg == MARGINS[1]:  # LP-audit requirement has no margin: compute once per rep
                            need_lp = Z * Z * vlp[rep] / eps ** 2
                            lam_lp = np.maximum(1, np.ceil(need_lp * r / (N * piG))).astype(int)
                            w3 = np.concatenate([1.0 / lam_lp, np.full(len(tarr), W_COV)])
                            l3 = np.concatenate([lam_lp, np.ones(len(tarr), int)])
                            designs["hybrid_LPaudit"], _ = budget.anneal_lambda(st, cand, t2, w3, l3, v, iters=ITERS, seed=5,
                                                                                t0=float(np.mean(1.0 / lam_lp)))
                        for meth, wells in designs.items():
                            design.check_design(wells, cand, CAP)
                            L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).astype(np.int32)
                            lam = L.sum(1)
                            for a, C in alloc.items():
                                cnt = np.einsum("ti,itx->tx", L, C)
                                n = cnt.sum(1)
                                p = cnt / np.maximum(n, 1)[:, None]
                                half = np.where(n > 0, cp_half(cnt, n).max(1), 1.0)
                                err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
                                rows.append(dict(parent=G, eps=eps, r=r, init_rep=rep, margin=mg, method=meth, alloc=a,
                                                 half_ok=float((half <= eps).mean()), err_ok=float((err <= eps).mean()),
                                                 covered=float((lam >= 1).mean())))
                print(f"[{time.time() - t0:.0f}s] {G} r={r} eps={eps}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3e_robust.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    for metric in ["half_ok", "err_ok", "covered"]:
        print(f"\n== {metric}  (mean over init reps x alloc seeds; margin=0.02)")
        sub = df[df.margin == 0.02]
        print(sub.groupby(["parent", "eps", "r", "method"])[metric].mean().unstack("method").round(3).to_string())
    print("\n== margin sensitivity: half_ok / err_ok for hybrid_hardcov")
    h = df[df.method == "hybrid_hardcov"].groupby(["parent", "eps", "r", "margin"])[["half_ok", "err_ok", "covered"]].mean()
    print(h.unstack("margin").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
