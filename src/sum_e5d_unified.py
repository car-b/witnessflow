"""e5d re-summary with ONE Bonferroni family per allocation: alarm if min raw p over all single
and pair tests < 0.001 / (M_single + M_pair).  Raw p recovered from saved Bonferroni-scaled minima.
Reports: alarm rate, localisation success (alarm AND min-p test at the true tube and touching the affected
set -- for pair tests this does not identify WHICH marker), conditional hit rate given alarm, mis-localisation.
usage: python sum_e5d_unified.py <results_dir>"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import beta


def ci(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


d = sys.argv[1]
df = pd.read_csv(os.path.join(d, "e5d_qc_merged.tsv"), sep="\t", keep_default_na=False, na_values=[""])
for m in ("single", "pair"):
    df[f"{m}_praw"] = df[f"{m}_pmin_bonf"] / df[f"{m}_M"]
M = df["single_M"] + df["pair_M"]
best_is_single = df["single_praw"] <= df["pair_praw"]
pmin = np.minimum(df["single_praw"], df["pair_praw"])
df["u_alarm"] = pmin < 1e-3 / M
df["u_loc"] = df["u_alarm"] & np.where(best_is_single, df["single_loc"].astype(bool), df["pair_loc"].astype(bool))
# single-only family for the ablation (M_single only)
df["s_alarm"] = df["single_praw"] < 1e-3 / df["single_M"]
order = ["null", "single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only",
         "composition", "random_drop", "global_drift"]
rows = []
for s in order:
    g = df[df.scenario == s]
    n = len(g)
    ka, kl, ks = int(g.u_alarm.sum()), int(g.u_loc.sum()), int(g.s_alarm.sum())
    lo, hi = ci(ka, n)
    rows.append(dict(scenario=s, n=n, single_only_alarm=round(ks / n, 3),
                     unified_alarm=f"{ka / n:.3f} [{lo:.3f},{hi:.3f}]", localised=round(kl / n, 3),
                     hit_given_alarm=round(kl / ka, 3) if ka else np.nan,
                     mislocalised_given_alarm=round((ka - kl) / ka, 3) if ka else np.nan))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(d, "e5d_unified_summary.tsv"), sep="\t", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
