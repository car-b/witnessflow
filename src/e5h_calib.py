"""Interval coverage calibration of the audit confidence bounds.
Synthetic 3-marker data, one stratum.  For each scenario: 1000 simulations; each draws a true 8-cell
distribution with the target pattern fixed at p_target and the rest Dirichlet(1)-distributed, simulates
cells split evenly over the tubes of the evidence structure, computes the audit's 95% (Bonferroni within
claim) confidence bounds for the target pattern, and records whether the truth is covered.
usage: python e5h_calib.py <structure> [n_sims]
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import audit  # noqa: E402

STRUCT = {"pairs": [{0, 1}, {0, 2}, {1, 2}], "singles": [{0}, {1}, {2}], "pairs+direct": [{0, 1}, {0, 2}, {1, 2}, {0, 1, 2}],
          "mixed": [{0, 1}, {2}], "direct_only": [{0, 1, 2}]}
TARGETS = {"zero": 0.0, "sparse": 0.01, "common": 0.20, "half": 0.50}
NCELLS = [600, 6000]
TARGET_CODE = 7


def target_bounds(ev):
    rows = audit._constraint_rows(ev, 1)
    if not rows:
        return 0.0, 1.0
    R = np.vstack([r for r, _, _ in rows])
    k = np.array([c for _, c, _ in rows])
    n = np.array([nn for _, _, nn in rows])
    lo, hi = audit.cp_interval(k, n, 0.05 / len(rows))
    obj = np.zeros(8)
    obj[TARGET_CODE] = 1
    out = []
    for sgn in (1, -1):
        res = linprog(sgn * obj, A_ub=np.vstack([R, -R]), b_ub=np.concatenate([hi, -lo]),
                      A_eq=np.ones((1, 8)), b_eq=[1.0], bounds=(0, 1), method="highs")
        if res.status != 0:
            return np.nan, np.nan
        out.append(sgn * res.fun)
    return out[0], out[1]


def main(structure, n_sims=1000):
    panels = STRUCT[structure]
    seed = 20261009 + list(STRUCT).index(structure)  # explicit, process-independent seed
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for tname, pt in TARGETS.items():
        for ncell in NCELLS:
            cov, widths, infeas = 0, [], 0
            for s in range(n_sims):
                rest = rng.dirichlet(np.ones(7)) * (1 - pt)
                p = np.insert(rest, TARGET_CODE, pt) if TARGET_CODE == 7 else None
                x = rng.choice(8, ncell, p=p)
                B = audit.PAT[x]
                tube = np.arange(ncell) % len(panels)
                rng.shuffle(tube)
                ev = audit.evidence(B, (0, 1, 2), np.ones(ncell, bool), np.zeros(ncell, int), 1, tube, panels)
                lo, hi = target_bounds(ev)
                if np.isnan(lo):
                    infeas += 1
                    continue
                cov += (lo - 1e-9 <= pt <= hi + 1e-9)
                widths.append(hi - lo)
            # coverage over ALL simulations (an infeasible LP counts as a failure), with exact binomial 95% CI
            lo_ci = beta.ppf(0.025, cov, n_sims - cov + 1) if cov > 0 else 0.0
            hi_ci = beta.ppf(0.975, cov + 1, n_sims - cov) if cov < n_sims else 1.0
            rows.append(dict(structure=structure, target=tname, p_target=pt, n_cells=ncell, n_sims=n_sims, seed=seed,
                             infeasible=infeas, coverage=cov / n_sims, coverage_ci_lo=lo_ci, coverage_ci_hi=hi_ci,
                             mean_width=float(np.mean(widths)) if widths else np.nan))
            print(f"[{time.time() - t0:.0f}s] {structure} {tname} n={ncell}: coverage {rows[-1]['coverage']:.3f} "
                  f"width {rows[-1]['mean_width']:.3f} infeasible {infeas}", flush=True)
    out = os.path.join(ROOT, "results", "calibration")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"e5h_{structure}.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
