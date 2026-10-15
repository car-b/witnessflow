"""Part 5 (small): does the replication built into a covering design give per-well drift detection 'for free'?
Levine v2 follow-up split, 71 wells = La Jolla C(20,6,3) (relabelled as in budget.ljcr_stack seed 0), cells
split evenly (allocation seed 0 rule of e3b).  In each rep one random well gets a shift (asinh units) on
  S1: all 6 of its candidate markers +0.5,  S2: one random candidate marker +0.5,  S3: one marker -0.5,
  S0: no shift (false-alarm check);
cells re-gated with the global thresholds.  Test: for every (well, marker in well) a 2x2 chi-square of the
positive fraction in that well vs all other wells measuring that marker (overall, no parent split);
alarm if Bonferroni p (x number of tests = 426) < 1e-3; 'hit' if the alarm is on the shifted well.
Compare: initial 4-tube disjoint partition (v2 design) has no shared candidate marker -> nothing testable.
Output: out_lat/lat07_bridge_free.tsv"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from lat_common import OUT, WF, budget
import replay  # noqa: E402

REPS = 40


def main():
    proto = json.load(open(os.path.join(WF, "protocols", "levine32_v2.json"), encoding="utf-8"))
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(WF, "results", proto["protocol_id"], "thresholds.json")))
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    th = np.array([thr[m]["thr"] for m in markers])
    fol = np.flatnonzero(part == "followup")
    X0 = Xt[fol][:, cand]
    thc = th[cand]
    r = 71
    wells = budget.ljcr_stack(cand, 6, r, seed=0)
    W = [sorted(cand.index(m) for m in w) for w in wells]          # local indices 0..19
    N = len(fol)
    well_of = np.empty(N, int)
    for i, ch in enumerate(np.array_split(np.random.default_rng(0 + r).permutation(N), r)):
        well_of[ch] = i
    reps_of = np.array([sum(j in w for w in W) for j in range(len(cand))])
    rng = np.random.default_rng(7)
    rows = []
    for rep in range(REPS):
        w0 = int(rng.integers(r))
        one = int(rng.choice(W[w0]))
        for scen, (delta, which) in {"S0_none": (0.0, []), "S1_all6_+0.5": (0.5, W[w0]),
                                     "S2_one_+0.5": (0.5, [one]), "S3_one_-0.5": (-0.5, [one])}.items():
            X = X0.copy()
            sel = well_of == w0
            for j in which:
                X[sel, j] += delta
            Bm = X > thc
            pos = np.zeros((r, len(cand)))
            tot = np.bincount(well_of, minlength=r).astype(float)
            for j in range(len(cand)):
                pos[:, j] = np.bincount(well_of, weights=Bm[:, j], minlength=r)
            ps, keys = [], []
            for i, w in enumerate(W):
                for j in w:
                    others = [k for k, ww in enumerate(W) if j in ww and k != i]
                    a, n1 = pos[i, j], tot[i]
                    b, n2 = pos[others, j].sum(), tot[others].sum()
                    tab = np.array([[a, n1 - a], [b, n2 - b]])
                    p = chi2_contingency(tab)[1] if tab.min() >= 0 and (tab.sum(0) > 0).all() else 1.0
                    ps.append(p)
                    keys.append(i)
            ps = np.array(ps) * len(ps)
            alarm = ps.min() < 1e-3
            hit = bool((ps[np.array(keys) == w0] < 1e-3).any())
            rows.append(dict(rep=rep, scenario=scen, alarm_any=bool(alarm), alarm_on_shifted_well=hit,
                             n_tests=len(ps), cells_per_well=float(tot.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lat07_bridge_free.tsv"), sep="\t", index=False)
    print(f"each candidate marker is measured in {reps_of.min()}-{reps_of.max()} of the {r} wells; "
          f"~{N / r:.0f} cells/well; {len(df.n_tests.unique())} test count {df.n_tests.iloc[0]}")
    print(df.groupby("scenario")[["alarm_any", "alarm_on_shifted_well"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
