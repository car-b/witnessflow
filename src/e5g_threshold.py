"""Threshold sensitivity (protocol offsets -0.25..+0.25 asinh, applied to every marker threshold).
For each offset: parent sizes; (i) audit validity on the primary claims for the first initial design
(reference inside confidence bounds; block-kNN error for claims without direct support); (ii) the key design
contrast of e5b (classical vs random at a block-count node and a between-node) with the same fixed eps.
usage: python e5g_threshold.py <protocol.json>"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e5b import well_counts, membership, evaluate  # noqa: E402

OFFSETS = [-0.25, -0.125, 0.0, 0.125, 0.25]
NODES = {"levine32_v2": {"G": "T", "eps": 0.06, "r": [71, 100]}, "wingmen_v1": {"G": "CD4T", "eps": 0.05, "r": [130, 180]}}


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    out = os.path.join(ROOT, "results", pid)
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr0 = json.load(open(os.path.join(out, "thresholds.json")))
    mi = {m: i for i, m in enumerate(markers)}
    bb = [mi[m] for m in proto["backbone"]]
    nonbb = [i for i in range(len(markers)) if i not in bb]
    cand = [i for i in nonbb if thr0[markers[i]]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    idg = proto["initial_design"]
    rng = np.random.default_rng(idg["seed"])
    rng_t = np.random.default_rng(idg["seed"] + 1)
    perm = rng.permutation(nonbb)
    panels0 = [set(bb) | set(perm[i * idg["capacity"]:(i + 1) * idg["capacity"]].tolist()) for i in range(idg["n_tubes"])]
    ii = np.flatnonzero(ini)
    tube = np.full(len(Xt), -1)
    tube[ii] = rng_t.integers(0, idg["n_tubes"], len(ii))
    claims = []
    for c in proto["claims"]:
        if any(not thr0[m]["reliable"] for m in c["markers"]):
            continue
        trip = sorted(zip([mi[m] for m in c["markers"]], c["pattern"]))
        claims.append(dict(c, S=tuple(t for t, _ in trip), sigma=int(sum(b << (2 - q) for q, (_, b) in enumerate(trip)))))
    nd = NODES[pid]
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    rows, crow = [], []
    for off in OFFSETS:
        tv = np.array([thr0[m]["thr"] + off for m in markers])
        Bin = (Xt > tv).astype(np.uint8)
        pm = replay.parent_masks(Bin, markers, proto["parents"])
        sizes = {f"pi_{g}": float(pm[g][ini].mean()) for g in pm}
        # (i) audit validity, initial design 0
        idx = ii
        Bimp = audit.knn_impute(Bin[idx], Xt[idx][:, bb], tube[idx], panels0, k=10, mode="block", seed=0)
        for c in claims:
            inG = pm[c["parent"]][idx]
            ev = audit.evidence(Bin[idx], c["S"], inG, np.zeros(len(idx), int), 1, tube[idx], panels0)
            rb = audit.bound_claim(ev, 1, c["sigma"])
            d = audit.direct_estimate(ev, c["sigma"])
            Rg = pm[c["parent"]] & ref
            refv = float((audit.codes(Bin[Rg], c["S"]) == c["sigma"]).mean())
            knn = audit.imputed_frequency(Bimp, c["S"], inG, c["sigma"])
            crow.append(dict(dataset=pid, offset=off, claim=c["id"], ref=refv, conf_lo=rb["conf_lo"], conf_hi=rb["conf_hi"],
                             n_direct=d["n_direct"], knn=knn, ref_in=bool(rb["conf_lo"] - 1e-9 <= refv <= rb["conf_hi"] + 1e-9),
                             knn_err=abs(knn - refv)))
        # (ii) design contrast
        G, eps = nd["G"], nd["eps"]
        piG = pm[G][ini].mean()
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        for r in nd["r"]:
            Ls = {"classical_stack": membership(budget.ljcr_stack(cand, 6, r, seed=0), tarr, v),
                  "random": membership(design.random_design(cand, 6, r, seed=0), tarr, v)}
            for a in range(3):
                order = np.random.default_rng(1000 * a + r).permutation(N)
                well_of = np.empty(N, int)
                for i, chunk in enumerate(np.array_split(order, r)):
                    well_of[chunk] = i
                C = well_counts(codesF, well_of[gpos], r)
                for meth, L in Ls.items():
                    rows.append(dict(dataset=pid, offset=off, parent=G, eps=eps, r=r, method=meth, alloc=a,
                                     **evaluate(L, C, refp, eps, N, piG), **sizes))
        print(f"[{time.time() - t0:.0f}s] {pid} offset {off:+.3f} done; {G} pi={piG:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "e5g_threshold_design.tsv"), sep="\t", index=False)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(os.path.join(out, "e5g_threshold_audit.tsv"), sep="\t", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["offset", "r", "method"])[["half_ok", "err_ok"]].mean().unstack("method").round(3).to_string())
    nd_ = cdf[cdf.n_direct == 0]
    print(nd_.groupby("offset").agg(ref_in=("ref_in", "mean"), knn_err_med=("knn_err", "median"),
                                    knn_err_max=("knn_err", "max"), n=("claim", "size")).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
