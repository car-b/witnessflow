"""e5d final summary: ONE fixed Bonferroni family = all prespecified checks per allocation,
M_total = r*(C(6,1) + C(6,2)) = r*21 (Levine 71 -> 1491; WINGMEN 130 -> 2730).  Uncheckable tests keep their
status and never trigger an alarm.  Reports alarm rate with Clopper-Pearson 95% CI, localisation success,
hit rate given alarm, uncheckable fraction; localisation is 'n/a' for the null scenario.
usage: python sum_e5d_fixed.py <results_dir> <r>"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import beta


def ci(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


d, r = sys.argv[1], int(sys.argv[2])
MT = r * 21
df = pd.read_csv(os.path.join(d, "e5d_qc_merged.tsv"), sep="\t", keep_default_na=False, na_values=[""])
for m in ("single", "pair"):
    df[f"{m}_praw"] = df[f"{m}_pmin_bonf"] / df[f"{m}_M"]
best_single = df["single_praw"] <= df["pair_praw"]
pmin = np.minimum(df["single_praw"], df["pair_praw"])
df["alarm"] = pmin < 1e-3 / MT
df["loc"] = df["alarm"] & np.where(best_single, df["single_loc"].astype(bool), df["pair_loc"].astype(bool))
df["s_alarm"] = df["single_praw"] < 1e-3 / (r * 6)  # single-only family (ablation)
df["uncheckable_frac"] = 1 - (df["single_M"] + df["pair_M"]) / MT
order = ["null", "single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only",
         "composition", "random_drop", "global_drift"]
rows = []
for s in order:
    g = df[df.scenario == s]
    n, ka, kl = len(g), int(g.alarm.sum()), int(g["loc"].sum())
    lo, hi = ci(ka, n)
    rows.append(dict(scenario=s, n=n, M_total=MT, single_only_alarm=round(g.s_alarm.mean(), 3),
                     alarm=round(ka / n, 3), alarm_ci=f"[{lo:.4f},{hi:.4f}]",
                     localised="n/a" if s == "null" else round(kl / n, 3),
                     hit_given_alarm="n/a" if (s == "null" or ka == 0) else round(kl / ka, 3),
                     uncheckable_frac_mean=round(g.uncheckable_frac.mean(), 4)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(d, "e5d_fixed_family_summary.tsv"), sep="\t", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
