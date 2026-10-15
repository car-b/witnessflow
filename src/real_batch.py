"""Real batch differences in the CytoNorm example (same panel, same donor, two runs).
For each donor and marker: median shift (asinh/5 units) batch2-batch1, positive-fraction difference at a
pooled 2-GMM cut, chi-square p; null = random half-splits within one batch file.
usage: python real_batch.py"""
import glob, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fcs, replay  # noqa: E402

files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "CytoNorm", "*.fcs")))
data = {}
skip = {"Time", "Event_length", "beadDist", "DNA1", "DNA2"}
for f in files:
    X, ch, mk, meta = fcs.read_fcs(f)
    keep = [i for i, (c, m) in enumerate(zip(ch, mk)) if m != c and m not in skip and not m.startswith("BC")]
    names = [mk[i] for i in keep]
    m = re.search(r"(PTLG\d+)_.*_(\d)\.fcs", os.path.basename(f))
    data[(m.group(1), int(m.group(2)))] = (np.arcsinh(X[:, keep] / 5.0), names)
names = next(iter(data.values()))[1]
pooled = np.vstack([v[0] for v in data.values()])
thr = replay.thresholds_v2(pooled, names, {"zero_cut": 0.05, "D_min": 1.5, "w_min": 0.03})
rng = np.random.default_rng(0)
rows = []
for donor in sorted({d for d, _ in data}):
    if (donor, 1) not in data or (donor, 2) not in data:
        continue
    A, B = data[(donor, 1)][0], data[(donor, 2)][0]
    for j, mname in enumerate(names):
        t = thr[mname]["thr"]
        pa, pb = (A[:, j] > t), (B[:, j] > t)
        tab = np.array([[pa.sum(), (~pa).sum()], [pb.sum(), (~pb).sum()]])
        p = chi2_contingency(tab)[1] if tab.min(0).min() >= 0 and (tab.sum(0) > 0).all() else 1.0
        # null: random half split of batch 1
        idx = rng.permutation(len(A))
        h1, h2 = A[idx[: len(A) // 2], j] > t, A[idx[len(A) // 2:], j] > t
        tab0 = np.array([[h1.sum(), (~h1).sum()], [h2.sum(), (~h2).sum()]])
        p0 = chi2_contingency(tab0)[1] if (tab0.sum(0) > 0).all() else 1.0
        rows.append(dict(donor=donor, marker=mname, reliable=thr[mname]["reliable"],
                         median_shift=float(np.median(B[:, j]) - np.median(A[:, j])),
                         pos_b1=pa.mean(), pos_b2=pb.mean(), dpos=pb.mean() - pa.mean(), p=p, p_null=p0))
df = pd.DataFrame(rows)
out = os.path.join(ROOT, "results", "cytonorm_real_batch")
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "batch_shift.tsv"), sep="\t", index=False)
r = df[df.reliable]
print(f"donors={r.donor.nunique()} reliable markers={r.marker.nunique()}  (cells per file ~{len(A)})")
print(f"|median shift| (asinh): median={r.median_shift.abs().median():.3f}  p90={r.median_shift.abs().quantile(.9):.3f}  max={r.median_shift.abs().max():.3f}")
print(f"|dpos| (pos-fraction diff): median={r.dpos.abs().median():.3f}  p90={r.dpos.abs().quantile(.9):.3f}  max={r.dpos.abs().max():.3f}")
print(f"flagged p<1e-3: batches {np.mean(r.p < 1e-3):.2f}  vs random split of one file {np.mean(r.p_null < 1e-3):.2f}")
top = r.reindex(r.dpos.abs().sort_values(ascending=False).index).head(12)
print(top[["donor", "marker", "median_shift", "pos_b1", "pos_b2", "dpos", "p"]].round(3).to_string(index=False))
