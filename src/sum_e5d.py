"""Merge e5d worker files and summarize alarm / localization rates with Clopper-Pearson 95% CIs."""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import beta


def ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "e5d_qc_w*of*.tsv")))
df = pd.concat([pd.read_csv(f, sep="\t", keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)
df.to_csv(os.path.join(d, "e5d_qc_merged.tsv"), sep="\t", index=False)
order = ["null", "single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only",
         "composition", "random_drop", "global_drift"]
rows = []
for s in order:
    g = df[df.scenario == s]
    if g.empty:
        continue
    n = len(g)
    r = {"scenario": s, "n": n}
    for col in ["single_alarm", "single_loc", "pair_alarm", "pair_loc", "any_alarm"]:
        k = int(g[col].fillna(False).astype(bool).sum())
        lo, hi = ci(k, n)
        r[col] = f"{k / n:.3f} [{lo:.3f},{hi:.3f}]"
    rows.append(r)
pd.set_option("display.width", 250)
print(f"{len(files)} worker files, {len(df)} replicates")
print(pd.DataFrame(rows).to_string(index=False))
