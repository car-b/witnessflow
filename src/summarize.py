"""Compact summaries of a replay run.  usage: python summarize.py <results_dir>"""
import json, os, sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
d = sys.argv[1]
thr = json.load(open(os.path.join(d, "thresholds.json")))
tdf = pd.DataFrame(thr).T
print("== thresholds: unreliable markers:", list(tdf.index[~tdf.reliable.astype(bool)]))
cols = [c for c in ["thr", "ashman_D", "pos_frac", "zero_frac"] if c in tdf.columns]
print(tdf[cols].astype(float).round(2).T.to_string())

ref = pd.read_csv(os.path.join(d, "reference_claims.tsv"), sep="\t")
print("\n== reference (R split, full panel) + Q1 on reference table")
cols = ["claim", "parent", "markers", "n_ref", "ref", "ref_lo", "ref_hi", "q1_H3", "q1_delta_pp", "q1_p_exact"]
print(ref[cols].round(4).to_string(index=False))

ia = pd.read_csv(os.path.join(d, "initial_audit.tsv"), sep="\t")
ia["conf_w"] = ia.conf_hi - ia.conf_lo
ia["ident_w"] = ia.ident_hi - ia.ident_lo
BASES = [b for b in ["maxent", "cond_indep", "knn_major", "knn_block", "knn_block_within"] if b in ia.columns]
for b in BASES:
    ia[f"err_{b}"] = (ia[b] - ia.ref).abs()
    ia[f"out_{b}"] = (ia[b] < ia.conf_lo - 1e-9) | (ia[b] > ia.conf_hi + 1e-9)
ia["has_direct"] = ia.n_direct > 0
print("\n== initial audit (mean over replicate designs)")
g = ia.groupby("claim").agg(direct_frac=("has_direct", "mean"), pairs=("observed_pairs", "mean"),
                            conf_w=("conf_w", "mean"), ident_w=("ident_w", "mean"), ref=("ref", "first"),
                            e_maxent=("err_maxent", "mean"), e_ci=("err_cond_indep", "mean"),
                            e_knnM=("err_knn_major", "mean"), e_knnB=("err_knn_block", "mean"))
print(g.round(3).to_string())
nd = ia[~ia.has_direct]
print("\n== claims WITHOUT direct support: errors (abs, freq units)")
for b in BASES:
    e = nd[f"err_{b}"]
    print(f"  {b:11s} median={e.median():.3f}  p90={e.quantile(.9):.3f}  max={e.max():.3f}  "
          f">0.05: {(e > 0.05).mean():.2f}  outside_conf_bounds: {nd[f'out_{b}'].mean():.2f}")
print(f"  ref inside conf bounds: {((nd.ref >= nd.conf_lo - 1e-9) & (nd.ref <= nd.conf_hi + 1e-9)).mean():.3f}"
      f"   mean conf width {nd.conf_w.mean():.3f}  mean ident width {nd.ident_w.mean():.3f}")

ks = sorted(int(c.split("K")[-1]) for c in ia.columns if c.startswith("conf_w_K"))
if ks:
    print("\n== micro-strata sensitivity (claims without direct support): mean widths, ref coverage")
    print(f"  K=1: conf {nd.conf_w.mean():.3f} ident {nd.ident_w.mean():.3f}")
    for K in ks:
        print(f"  K={K}: conf {nd[f'conf_w_K{K}'].mean():.3f} ident {nd[f'ident_w_K{K}'].mean():.3f} "
              f"ref_in {nd[f'ref_in_K{K}'].mean():.3f}")

fp = os.path.join(d, "followup.tsv")
if os.path.exists(fp):
    fu = pd.read_csv(fp, sep="\t")
    s = fu.groupby(["r", "method", "rep", "frep"]).agg(resolved=("resolved", "sum"), correct=("correct", "sum"),
                                                         wcov=("weighted_cov", "first")).reset_index()
    t = s.groupby(["r", "method"]).agg(resolved=("resolved", "mean"), correct=("correct", "mean"),
                                       wcov=("wcov", "mean")).reset_index()
    print("\n== follow-up: claims resolved (half-width <= eps) / correct, mean over reps")
    print(t.pivot(index="method", columns="r", values="resolved").round(2).to_string())
    print(t.pivot(index="method", columns="r", values="correct").round(2).to_string())
