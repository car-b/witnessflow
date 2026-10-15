"""Rare-parent stress: B claims as B cells are thinned (initial+followup), reference untouched."""
import os
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs = {"B~21%": "levine32_v2", "B=1%": "levine32_v2_rareB_1", "B=0.3%": "levine32_v2_rareB_0p3",
        "B=0.1%": "levine32_v2_rareB_0p1"}
pd.set_option("display.width", 250)
rows, er = [], []
for k, d in runs.items():
    ia = pd.read_csv(os.path.join(R, d, "initial_audit.tsv"), sep="\t")
    ia = ia[ia.claim.isin(["B1", "B2"])].copy()
    ia["conf_w"] = ia.conf_hi - ia.conf_lo
    for b in ["maxent", "knn_major", "knn_block"]:
        ia[f"err_{b}"] = (ia[b] - ia.ref).abs()
        ia[f"out_{b}"] = (ia[b] < ia.conf_lo - 1e-9) | (ia[b] > ia.conf_hi + 1e-9)
    nd = ia[ia.n_direct == 0]
    for cl, g in nd.groupby("claim"):
        rows.append(dict(run=k, claim=cl, n_parent=g.n_parent.mean(), conf_w=g.conf_w.mean(),
                         ref_in=((g.ref >= g.conf_lo - 1e-9) & (g.ref <= g.conf_hi + 1e-9)).mean(),
                         knnB_err_mean=g.err_knn_block.mean(), knnB_err_max=g.err_knn_block.max(),
                         maxent_err_max=g.err_maxent.max(), knnM_err_max=g.err_knn_major.max(),
                         knnB_out=g.out_knn_block.mean(), knnM_out=g.out_knn_major.mean()))
    fu = pd.read_csv(os.path.join(R, d, "followup.tsv"), sep="\t")
    fu = fu[fu.claim.isin(["B1", "B2"])]
    best = fu[fu.method.isin(["ilp", "greedy+anneal", "naive_packing"])].groupby(["r", "method"]).resolved.mean().unstack()
    rnd = fu[fu.method == "random"].groupby("r").resolved.mean()
    er.append((k, best.round(2), rnd.round(2)))
t = pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
print("\n== follow-up: fraction of B1/B2 claim-instances resolved (half-width <= eps) after follow-up")
for k, b, rnd in er:
    print(f"-- {k}  (random: {rnd.to_dict()})")
    print(b.to_string())
