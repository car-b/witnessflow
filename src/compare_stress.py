"""Compare imputation error vs audit across backbone sizes (common claims only, primary claims only)."""
import os, sys
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs = {"bb8": "levine32_v2", "bb6": "levine32_v2_bb6", "bb4": "levine32_v2_bb4"}
dfs = {}
for k, d in runs.items():
    ia = pd.read_csv(os.path.join(R, d, "initial_audit.tsv"), sep="\t")
    ia = ia[ia.reliable.astype(bool)]
    ia["conf_w"] = ia.conf_hi - ia.conf_lo
    ia["ident_w"] = ia.ident_hi - ia.ident_lo
    for b in ["maxent", "cond_indep", "knn_major", "knn_block"]:
        ia[f"err_{b}"] = (ia[b] - ia.ref).abs()
        ia[f"out_{b}"] = (ia[b] < ia.conf_lo - 1e-9) | (ia[b] > ia.conf_hi + 1e-9)
        # worst-case error given the audit interval: R = max(|a-L|, |a-U|)
        ia[f"risk_{b}"] = np.maximum((ia[b] - ia.conf_lo).abs(), (ia[b] - ia.conf_hi).abs())
    dfs[k] = ia[ia.n_direct == 0]
common = set.intersection(*[set(d.claim) for d in dfs.values()])
print("common claims:", sorted(common))
rows = []
for k, d in dfs.items():
    for scope, dd in (("common", d[d.claim.isin(common)]), ("all", d)):
        r = {"backbone": k, "scope": scope, "n": len(dd), "conf_w": dd.conf_w.mean(), "ident_w": dd.ident_w.mean(),
             "ref_in": ((dd.ref >= dd.conf_lo - 1e-9) & (dd.ref <= dd.conf_hi + 1e-9)).mean()}
        for b in ["maxent", "knn_major", "knn_block"]:
            r[f"{b}_med"] = dd[f"err_{b}"].median()
            r[f"{b}_max"] = dd[f"err_{b}"].max()
            r[f"{b}_gt5pp"] = (dd[f"err_{b}"] > 0.05).mean()
            r[f"{b}_out"] = dd[f"out_{b}"].mean()
        rows.append(r)
pd.set_option("display.width", 250)
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
print("\nper-claim block-kNN error (mean over reps), common claims")
pc = pd.concat({k: d[d.claim.isin(common)].groupby("claim").err_knn_block.mean() for k, d in dfs.items()}, axis=1)
pc2 = pd.concat({k: d[d.claim.isin(common)].groupby("claim").err_maxent.mean() for k, d in dfs.items()}, axis=1)
print(pd.concat({"knn_block": pc, "maxent": pc2}, axis=1).round(3).to_string())
