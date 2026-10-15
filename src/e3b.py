"""E3b: budgeted witness design with empirical precision.

Adds: strong baselines (La Jolla stack + same-budget anneal; deficit packing), 3 shared cell-
allocation seeds, and for every design the *empirical* 8-pattern estimate from the follow-up cells:
max Clopper-Pearson half-width and max absolute error vs the independent reference split.
usage: python e3b.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402

PARENTS = ["T", "B"]
EPS = [0.05, 0.07, 0.10]
RGRID = [24, 48, 71, 96, 120, 142, 180, 240]
CAP = 6
Z = 1.96
ALLOC_SEEDS = [0, 1, 2]
ANNEAL_ITERS = 30000


def cp_half(k, n):
    k = np.asarray(k, float)
    n = np.broadcast_to(np.asarray(n, float)[..., None], k.shape) if np.ndim(n) < np.ndim(k) else np.asarray(n, float)
    lo = np.where(k > 0, beta.ppf(0.025, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(0.975, k + 1, n - k), 1.0)
    return (np.nan_to_num(hi, nan=1.0) - np.nan_to_num(lo, nan=0.0)) / 2


def main(proto_path):
    global PARENTS, EPS
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    eps_by_parent = None
    if "e3" in proto:  # dataset-specific parents; add one looser precision level for the feasible regime
        eps_by_parent = {G: sorted(set(v + [round(max(v) + 0.02, 2)])) for G, v in proto["e3"]["eps"].items()}
        PARENTS = list(eps_by_parent)
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
    w = np.ones(len(tarr))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows = []
    for G in PARENTS:
        piG = pm[G][ini].mean()
        # reference 8-pattern frequencies per triple (independent split)
        Rc = Bin[ref & pm[G]]
        refp = np.zeros((len(tarr), 8))
        for t, S in enumerate(tarr):
            refp[t] = np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc)
        # follow-up parent cells: codes per triple
        gpos = np.flatnonzero(pm[G][folidx])          # positions within folidx that are in G
        Fc = Bin[folidx[gpos]]
        codesF = np.stack([audit.codes(Fc, S).astype(np.uint8) for S in tarr])  # [T, nG]
        for r in RGRID:
            # per allocation seed: well of each follow-up cell, then per-well per-triple pattern counts
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
            for eps in (eps_by_parent[G] if eps_by_parent else EPS):
                nreq = Z * Z * 0.25 / eps ** 2
                lam_req = max(1, math.ceil(nreq * r / (N * piG)))
                cpw = N * piG / r
                designs = {}
                for s in range(3):
                    designs[("random", s)] = design.random_design(cand, CAP, r, seed=s)
                    designs[("balanced_random", s)] = budget.balanced_random(cand, CAP, r, seed=s)
                st = budget.ljcr_stack(cand, CAP, r, seed=0)
                if st is not None:
                    designs[("LJCR_stack", 0)] = st
                    a_, _ = budget.anneal_lambda(st, cand, tarr, w, lam_req, v, iters=ANNEAL_ITERS, seed=2)
                    designs[("LJCR_stack+anneal", 0)] = a_
                g = budget.greedy_lambda(cand, CAP, r, tarr, w, lam_req, v)
                a2, _ = budget.anneal_lambda(g, cand, tarr, w, lam_req, v, iters=ANNEAL_ITERS, seed=1)
                designs[("greedy+anneal", 0)] = a2
                designs[("packing_deficit", 0)] = budget.packing_deficit(cand, CAP, r, tarr, w, nreq, cpw)
                for (meth, s), wells in designs.items():
                    design.check_design(wells, cand, CAP)
                    L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).astype(np.int32)  # [T, r]
                    lam = L.sum(1)
                    for a, C in alloc.items():
                        cnt = np.einsum("ti,itx->tx", L, C)
                        n = cnt.sum(1)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            p = cnt / np.maximum(n, 1)[:, None]
                        half = np.where(n > 0, cp_half(cnt, n).max(1), 1.0)
                        err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
                        rows.append(dict(parent=G, eps=eps, r=r, lam_req=lam_req, method=meth, seed=s, alloc=a,
                                         planned=float((lam >= lam_req).mean()),
                                         witnessed=float((n >= nreq).mean()),
                                         emp_half_ok=float((half <= eps).mean()),
                                         emp_err_ok=float((err <= eps).mean()),
                                         err_med=float(np.median(err)), half_med=float(np.median(half)),
                                         covered=float((lam >= 1).mean()), min_cells=float(n.min())))
            print(f"[{time.time() - t0:.0f}s] {G} r={r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3b_budget.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    for metric in ["witnessed", "emp_half_ok", "emp_err_ok"]:
        print(f"\n== {metric} (mean over design seeds x allocation seeds)")
        print(df.groupby(["parent", "eps", "r", "method"])[metric].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
