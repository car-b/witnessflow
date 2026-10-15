"""Diagnose marker distributions (calibration split only): zero spike + GMM on non-zero values."""
import json, os, sys
import numpy as np
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import replay

proto = json.load(open(sys.argv[1], encoding="utf-8"))
Xt, markers, sample = replay.load(proto)
part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
C = Xt[part == "calibration"]
rng = np.random.default_rng(0)
print(f"{'marker':9s} zero%  nzD   w_hi  thr_nz  pos%(nz-rule)  q50   q90   q99")
for j, m in enumerate(markers):
    x = C[:, j]
    z = x <= 0.05
    nz = x[~z]
    xs = nz if len(nz) <= 100000 else rng.choice(nz, 100000, replace=False)
    g = GaussianMixture(2, random_state=0, n_init=3).fit(xs.reshape(-1, 1))
    mu = g.means_.ravel(); sd = np.sqrt(g.covariances_.ravel()); w = g.weights_
    lo, hi = np.argsort(mu)
    D = np.sqrt(2) * (mu[hi] - mu[lo]) / np.sqrt(sd[hi] ** 2 + sd[lo] ** 2)
    grid = np.linspace(mu[lo], mu[hi], 4001).reshape(-1, 1)
    post = g.predict_proba(grid)[:, hi]
    thr = float(grid[np.argmax(post >= 0.5), 0])
    pos = (x > thr).mean()
    q = np.quantile(x, [.5, .9, .99])
    print(f"{m:9s} {z.mean()*100:5.1f} {D:5.2f} {w[hi]:5.2f} {thr:7.3f} {pos*100:9.1f}     {q[0]:5.2f} {q[1]:5.2f} {q[2]:5.2f}")
