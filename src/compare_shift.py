"""Cross-tube shift stress: does the audit / tube-consistency check notice a shifted tube, and what happens
to imputation error and interval validity?  Claims without direct support, primary claims only."""
import os
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs = {"no shift": "levine32_v2_shift0", "all +0.5": "levine32_v2_shiftAllp0p5",
        "nonBB +0.5": "levine32_v2_shiftNonBBp0p5", "nonBB -0.5": "levine32_v2_shiftNonBBm0p5"}
rows = []
for k, d in runs.items():
    f = os.path.join(R, d, "initial_audit.tsv")
    if not os.path.exists(f):
        print("missing", d); continue
    ia = pd.read_csv(f, sep="\t")
    ia = ia[ia.reliable.astype(bool)]
    nd = ia[ia.n_direct == 0].copy()
    nd["conf_w"] = nd.conf_hi - nd.conf_lo
    inb = (nd.ref >= nd.conf_lo - 1e-9) & (nd.ref <= nd.conf_hi + 1e-9)
    r = {"run": k, "n": len(nd), "ref_in_bounds": inb.mean(), "lp_incompatible": (nd.lp_status == "incompatible_margins").mean(),
         "conf_w": nd.conf_w.mean()}
    for b in ["maxent", "knn_block", "knn_major"]:
        e = (nd[b] - nd.ref).abs()
        r[f"{b}_med"] = e.median(); r[f"{b}_max"] = e.max(); r[f"{b}_gt5pp"] = (e > 0.05).mean()
    if "tube_minp" in nd:
        r["flag_tube_p<1e-3"] = (nd.tube_minp < 1e-3).mean()
        r["parent_frac_range"] = nd.parent_frac_range.mean()
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
