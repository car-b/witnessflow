"""Part 5 (quick probe): 2-fold coverings of triples, v=20, k=6, with fewer than 142 = 2 x 71 blocks.
Fixed-budget value of a design for 'all triples' = min_T lam_T / r; a 2-fold covering with r blocks gives
2/r (> 1/71 iff r < 142).  Probe: annealing (budget.anneal_lambda, lam_req=2, uniform weights) started
from the La Jolla stack truncated to r.  Schoenheim-type bound for lambda=2: ceil(20/6*ceil(19/5*ceil(2*18/4))) = 117.
Output: out_lat/lat08_lambda2.tsv"""
import os, time
import numpy as np
import pandas as pd
from lat_common import OUT, budget, triples

v, k = 20, 6
cand = list(range(v))
tarr = triples(v)
rows = []
t0 = time.time()
for r in (140, 136, 132, 128, 124):
    best = None
    for seed in range(2):
        st = budget.ljcr_stack(cand, k, r, seed=seed)
        wells, score = budget.anneal_lambda(st, cand, tarr, np.ones(len(tarr)), 2, v, iters=400_000, seed=seed)
        lam = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).sum(1)
        res = dict(r=r, seed=seed, frac_lam_ge2=float((lam >= 2).mean()), lam_min=int(lam.min()),
                   min_share=float(lam.min() / r), eff_vs_3design=float(lam.min() / r / (20 / 1140)))
        rows.append(res)
        print(res, f"[{time.time() - t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "lat08_lambda2.tsv"), sep="\t", index=False)
print(df.to_string(index=False))
