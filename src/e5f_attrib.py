"""P2 attribution ablation (spec v2): identical hybrid optimizer (hard coverage W=100, same start = La Jolla
stack, same optimizer seed, same temperature rule t0 = mean(1/lam_req)), ONLY the requirement vector differs:
  worst : lam_req = ceil(1.96^2*0.25/eps^2 * r/(N pi))  (same for all triples)
  knn   : from block-kNN imputed initial data, max_x p(1-p) + 0.02
  lp    : from LP confidence intervals of the initial data, max_x max_{p in [L,U]} p(1-p)
Requirements use the initial split only.  usage: python e5f_attrib.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e3e import initial_designs  # noqa: E402
from e5b import well_counts, membership, evaluate  # noqa: E402

EPS = {"levine32_v2": {"T": [0.05], "B": [0.07]}, "wingmen_v1": {"CD4T": [0.03], "CD8T": [0.05]}}
RGRID = [142, 180]
INIT_REPS = [0, 1, 2]
ALLOC = [0, 1, 2]
ITERS, SEED, W_COV, Z = 30000, 4, 100.0, 1.96


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
    for G, eps_list in EPS[pid].items():
        piG = pm[G][ini].mean()
        gI = pm[G][ii]
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        var = {}
        for rep in INIT_REPS:
            panels, tubes = inits[rep]
            Bimp = audit.knn_impute(Bin[ii], Xt[ii][:, bbi], tubes, panels, k=10, mode="block", seed=rep)
            ph = np.stack([np.bincount(audit.codes(Bimp[gI], S), minlength=8) / gI.sum() for S in tarr])
            v_knn = np.minimum(0.25, (ph * (1 - ph)).max(1) + 0.02)
            zero = np.zeros(len(ii), int)
            v_lp = np.zeros(len(tarr))
            for t, S in enumerate(tarr):
                lo, hi = audit.conf_bounds_all(audit.evidence(Bin[ii], tuple(S), gI, zero, 1, tubes, panels), 1)
                v_lp[t] = audit.max_binom_var(lo, hi).max()
            var[rep] = {"worst": np.full(len(tarr), 0.25), "knn": v_knn, "lp": v_lp}
            print(f"[{time.time() - t0:.0f}s] {pid} {G} rep {rep} requirements ready", flush=True)
        for r in RGRID:
            st = budget.ljcr_stack(cand, 6, r, seed=0)
            allocC = {}
            for a in ALLOC:
                order = np.random.default_rng(1000 * a + r).permutation(N)
                well_of = np.empty(N, int)
                for i, chunk in enumerate(np.array_split(order, r)):
                    well_of[chunk] = i
                allocC[a] = well_counts(codesF, well_of[gpos], r)
            for eps in eps_list:
                for rep in INIT_REPS:
                    designs = {"LJCR_stack": st}
                    reqstats = {}
                    for name, vv in var[rep].items():
                        lam = np.maximum(1, np.ceil(Z * Z * vv / eps ** 2 * r / (N * piG))).astype(int)
                        reqstats[name] = (float(lam.mean()), float((lam > 1).mean()))
                        w2 = np.concatenate([1.0 / lam, np.full(len(tarr), W_COV)])
                        l2 = np.concatenate([lam, np.ones(len(tarr), int)])
                        designs[f"hybrid_{name}"], _ = budget.anneal_lambda(st, cand, t2, w2, l2, v, iters=ITERS,
                                                                            seed=SEED, t0=float(np.mean(1.0 / lam)))
                    for meth, wells in designs.items():
                        L = membership(wells, tarr, v)
                        rs = reqstats.get(meth.replace("hybrid_", ""), (np.nan, np.nan))
                        for a, C in allocC.items():
                            m = evaluate(L, C, refp, eps, N, piG)
                            rows.append(dict(dataset=pid, parent=G, eps=eps, r=r, init_rep=rep, alloc=a, method=meth,
                                             req_mean_lam=rs[0], req_frac_gt1=rs[1], **m))
                print(f"[{time.time() - t0:.0f}s] {G} r={r} eps={eps} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5f_attribution.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["parent", "eps", "r", "method"])[["covered", "half_ok", "err_ok", "req_mean_lam", "req_frac_gt1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
