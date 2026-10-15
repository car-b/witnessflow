"""Requirement-label shuffle control: same hybrid optimizer and paired optimizer seeds; requirement
vector = (a) LP audit (true assignment), (b) the SAME values randomly permuted across triples (3 shuffles),
(c) uniform worst case.  If (a) > (b), knowing WHICH triple needs more evidence matters; if (a) ~ (b), the gain
comes from the requirement value distribution only.  usage: python e5i_shuffle.py <protocol.json>"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, audit  # noqa: E402
from e3e import initial_designs  # noqa: E402
from e5b import well_counts, membership, evaluate  # noqa: E402

SET = {"levine32_v2": [("T", 0.05, 142)], "wingmen_v1": [("CD4T", 0.03, 180), ("CD8T", 0.05, 180)]}
INIT_REPS, OPT_SEEDS, SHUFFLES, ALLOC = [0, 1], [4, 5, 6], [0, 1, 2], [0, 1, 2]
ITERS, W_COV, Z = 30000, 100.0, 1.96


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    out = os.path.join(ROOT, "results", pid)
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
    t2 = np.vstack([tarr, tarr])
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    ii = np.flatnonzero(ini)
    inits = initial_designs(proto, nonbb, bbi, len(ii))
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows = []
    for G, eps, r in SET[pid]:
        piG = pm[G][ini].mean()
        gI = pm[G][ii]
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        st = budget.ljcr_stack(cand, 6, r, seed=0)
        allocC = {}
        for a in ALLOC:
            order = np.random.default_rng(1000 * a + r).permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, r)):
                well_of[chunk] = i
            allocC[a] = well_counts(codesF, well_of[gpos], r)
        for rep in INIT_REPS:
            panels, tubes = inits[rep]
            zero = np.zeros(len(ii), int)
            v_lp = np.zeros(len(tarr))
            for t, S in enumerate(tarr):
                lo, hi = audit.conf_bounds_all(audit.evidence(Bin[ii], tuple(S), gI, zero, 1, tubes, panels), 1)
                v_lp[t] = audit.max_binom_var(lo, hi).max()
            lam_lp = np.maximum(1, np.ceil(Z * Z * v_lp / eps ** 2 * r / (N * piG))).astype(int)
            lam_wc = np.full(len(tarr), max(1, int(np.ceil(Z * Z * 0.25 / eps ** 2 * r / (N * piG)))))
            reqs = {"lp_true": lam_lp, "worst": lam_wc}
            for s in SHUFFLES:
                reqs[f"lp_shuffled{s}"] = np.random.default_rng(500 + s + 10 * rep).permutation(lam_lp)
            print(f"[{time.time() - t0:.0f}s] {G} rep {rep}: lam_lp mean {lam_lp.mean():.2f}, frac>min {np.mean(lam_lp > lam_lp.min()):.2f}", flush=True)
            for name, lam in reqs.items():
                w2 = np.concatenate([1.0 / lam, np.full(len(tarr), W_COV)])
                l2 = np.concatenate([lam, np.ones(len(tarr), int)])
                for os_ in OPT_SEEDS:
                    wells, _ = budget.anneal_lambda(st, cand, t2, w2, l2, v, iters=ITERS, seed=os_,
                                                    t0=float(np.mean(1.0 / lam)))
                    L = membership(wells, tarr, v)
                    for a, C in allocC.items():
                        m = evaluate(L, C, refp, eps, N, piG)
                        rows.append(dict(dataset=pid, parent=G, eps=eps, r=r, init_rep=rep, req=name,
                                         req_kind=name.rstrip("012"), opt_seed=os_, alloc=a, **m))
            Lst = membership(st, tarr, v)
            for a, C in allocC.items():
                rows.append(dict(dataset=pid, parent=G, eps=eps, r=r, init_rep=rep, req="none", req_kind="LJCR_stack",
                                 opt_seed=-1, alloc=a, **evaluate(Lst, C, refp, eps, N, piG)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5i_shuffle.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["parent", "req_kind"])[["half_ok", "err_ok", "covered"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
