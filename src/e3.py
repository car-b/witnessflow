"""E3: screen-scale witness design under a fixed follow-up cell budget (Levine-32 replay).

Planning uses only the initial split (parent fractions); evaluation uses the real follow-up cells,
split evenly over the r wells.  usage: python e3.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget  # noqa: E402

PARENTS = ["T", "B"]
EPS = [0.05, 0.07, 0.10]
RGRID = [24, 48, 71, 96, 120, 142, 180, 240]
CAP = 6
Z = 1.96


def main(proto_path):
    t0 = time.time()
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
    w = np.ones(len(tarr))
    ini, fol = part == "initial", part == "followup"
    N = int(fol.sum())
    rows = []
    for G in PARENTS:
        piG = pm[G][ini].mean()  # planning: initial split only
        for eps in EPS:
            nreq = Z * Z * 0.25 / eps ** 2
            for r in RGRID:
                lam_req = max(1, math.ceil(nreq * r / (N * piG)))
                designs = {}
                for s in range(3):
                    designs[("random", s)] = budget.design.random_design(cand, CAP, r, seed=s)
                    designs[("balanced_random", s)] = budget.balanced_random(cand, CAP, r, seed=s)
                    st = budget.ljcr_stack(cand, CAP, r, seed=s)
                    if st is not None:
                        designs[("LJCR_stack", s)] = st
                g = budget.greedy_lambda(cand, CAP, r, tarr, w, lam_req, v)
                designs[("greedy_lambda", 0)] = g
                a, _ = budget.anneal_lambda(g, cand, tarr, w, lam_req, v, iters=30000, seed=1)
                designs[("greedy+anneal_lambda", 0)] = a
                # replay: real follow-up cells split evenly over r wells (same split for every design)
                order = np.random.default_rng(r).permutation(np.flatnonzero(fol))
                cpar = np.array([pm[G][chunk].sum() for chunk in np.array_split(order, r)])
                for (meth, s), wells in designs.items():
                    budget.design.check_design(wells, cand, CAP)
                    L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1)  # [T, r]
                    lam = L.sum(1)
                    nT = L @ cpar
                    rows.append(dict(parent=G, eps=eps, r=r, lam_req=lam_req, method=meth, seed=s,
                                     planned=float((lam >= lam_req).mean()),
                                     witnessed=float((nT >= nreq).mean()),
                                     covered=float((lam >= 1).mean()),
                                     min_cells=float(nT.min()), med_cells=float(np.median(nT)),
                                     uniform_cells=float(N * piG * len(list(itertools.combinations(range(CAP), 3))) / len(tarr))))
                print(f"[{time.time() - t0:.0f}s] {G} eps={eps} r={r} lam_req={lam_req}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3_budget.tsv"), sep="\t", index=False)
    s = df.groupby(["parent", "eps", "r", "method"]).witnessed.mean().unstack("method")
    pd.set_option("display.width", 220)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
