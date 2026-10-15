"""P1 / e5d: consistency checks that come for free with a covering design (frozen spec v2).

Design: classical covering (Levine 71 = LJCR C(20,6,3); WINGMEN 130 = S(3,6,26)); follow-up cells randomly
split evenly over the r tubes; each tube measures backbone + its 6 candidate markers.
Checks (per allocation):
  single: for every (tube, marker in tube): 2x2 table [this tube vs all OTHER tubes measuring the marker] x
          [pos, neg]; Fisher exact when any expected count < 5 else chi-square.
  pair:   for every (tube, pair in tube): 2x4 table [this tube vs other tubes measuring both] x joint states.
Alarm if min p < 0.001 / M  (M = number of tests performed in that mode).  False alarm = any alarm in a null
allocation.  Detection = alarm whose minimum-p test lies at the truly shifted tube (and marker, for single).
Scenarios (candidate markers only; asinh units; re-gated with global thresholds):
  null | single +0.5 | single -0.5 | single +0.25 | single -0.25 | wholetube +0.5 | joint-only (within one
  tube, permute one marker across cells: marginals kept, joints changed) | composition (drop 30% of the
  tube's T/CD4T cells) | random-drop control (drop the same number of random cells) | global drift (+0.5 on
  one marker in ALL tubes; expected undetectable).
usage: python e5d.py <protocol.json> [n_null] [n_shift]
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, chi2 as chi2dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, design  # noqa: E402

SETUP = {"levine32_v2": {"v": 20, "parent": "T"}, "wingmen_v1": {"v": 26, "parent": "CD4T"}}
ALPHA = 1e-3


def p_2xk(tab):
    tab = np.asarray(tab, float)
    keep = tab.sum(0) > 0
    tab = tab[:, keep]
    if tab.shape[1] < 2 or (tab.sum(1) == 0).any():
        return np.nan
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    if tab.shape[1] == 2 and (exp < 5).any():
        return float(fisher_exact(tab)[1])
    if (exp < 5).any():
        return np.nan  # 'not checkable'
    return float(chi2_contingency(tab, correction=False)[1])


def run_checks(Bt, tube_of, wells, mode):
    """Bt: binary matrix of the follow-up cells (candidate columns indexed by global marker id).
    Returns list of (p, tube, items)."""
    res = []
    r = len(wells)
    if mode == "single":
        for i, X in enumerate(wells):
            ins = tube_of == i
            for m in X:
                oth = np.isin(tube_of, [j for j in range(r) if j != i and m in wells[j]])
                a = Bt[ins, m]
                b = Bt[oth, m]
                p = p_2xk([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
                res.append((p, i, (m,)))
    else:
        for i, X in enumerate(wells):
            ins = tube_of == i
            for m1, m2 in itertools.combinations(sorted(X), 2):
                oth = np.isin(tube_of, [j for j in range(r) if j != i and m1 in wells[j] and m2 in wells[j]])
                ca = np.bincount(Bt[ins, m1] * 2 + Bt[ins, m2], minlength=4)
                cb = np.bincount(Bt[oth, m1] * 2 + Bt[oth, m2], minlength=4)
                res.append((p_2xk([ca, cb]), i, (m1, m2)))
    return res


def main(proto_path, n_null=1000, n_shift=200):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    out = os.path.join(ROOT, "results", pid)
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    tv = np.array([thr[m]["thr"] for m in markers])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    blocks = design.ljcr_blocks(len(cand), 6, 3)
    wells = [set(cand[x] for x in b) for b in blocks]
    r = len(wells)
    fol = np.flatnonzero(part == "followup")
    Xf = Xt[fol]
    Bf = (Xf > tv).astype(np.int64)
    pmask = replay.parent_masks(Bf.astype(np.uint8), markers, proto["parents"])[SETUP[pid]["parent"]]
    scen = ["null"] * n_null
    for s in ["single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only",
              "composition", "random_drop", "global_drift"]:
        scen += [s] * n_shift
    rows = []
    for k, sname in enumerate(scen):
        if k % N_WORKERS != WORKER:
            continue
        rng = np.random.default_rng(20261009 + k)  # per-replicate RNG: results independent of worker count
        order = rng.permutation(len(fol))
        tube_of = np.empty(len(fol), int)
        for i, chunk in enumerate(np.array_split(order, r)):
            tube_of[chunk] = i
        X = Xf
        B = Bf
        keep = np.ones(len(fol), bool)
        ti = int(rng.integers(r))
        mk = int(rng.choice(sorted(wells[ti])))
        truth_items = None
        DELTA = {"single+0.5": 0.5, "single-0.5": -0.5, "single+0.25": 0.25, "single-0.25": -0.25,
                 "wholetube+0.5": 0.5, "global_drift": 0.5}
        if sname in DELTA:
            delta = DELTA[sname]
            X = Xf.copy()
            if sname == "wholetube+0.5":
                cols = sorted(wells[ti])
                sel = tube_of == ti
                X[np.ix_(sel, cols)] += delta
                truth_items = set(cols)
            elif sname == "global_drift":
                X[:, mk] += delta
                truth_items = {mk}
            else:
                sel = tube_of == ti
                X[sel, mk] += delta
                truth_items = {mk}
            B = (X > tv).astype(np.int64)
        elif sname == "joint_only":
            B = Bf.copy()
            sel = np.flatnonzero(tube_of == ti)
            B[sel, mk] = B[rng.permutation(sel), mk]
            truth_items = {mk}
        elif sname in ("composition", "random_drop"):
            sel = np.flatnonzero(tube_of == ti)
            if sname == "composition":
                g = sel[pmask[sel]]
                drop = rng.choice(g, int(0.3 * len(g)), replace=False)
            else:
                ng = int(0.3 * pmask[sel].sum())
                drop = rng.choice(sel, ng, replace=False)
            keep[drop] = False
            truth_items = None
        tb = tube_of[keep]
        Bk = B[keep]
        rec = {"scenario": sname, "rep": k, "tube": ti, "marker": markers[mk]}
        for mode in ("single", "pair"):
            res = [x for x in run_checks(Bk, tb, wells, mode) if not np.isnan(x[0])]
            M = len(res)
            if not res:
                rec[f"{mode}_alarm"] = False
                continue
            pmin, tmin, items = min(res, key=lambda x: x[0])
            alarm = pmin < ALPHA / M
            loc = alarm and tmin == ti and (truth_items is None or bool(set(items) & truth_items))
            rec.update({f"{mode}_alarm": bool(alarm), f"{mode}_loc": bool(loc), f"{mode}_M": M,
                        f"{mode}_pmin_bonf": float(min(1.0, pmin * M))})
        rec["any_alarm"] = rec.get("single_alarm", False) or rec.get("pair_alarm", False)
        rows.append(rec)
        if k % 100 == 0:
            print(f"[{time.time() - t0:.0f}s] {k}/{len(scen)} {sname}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"e5d_qc_w{WORKER}of{N_WORKERS}.tsv"), sep="\t", index=False)
    s = df.groupby("scenario").agg(n=("rep", "size"), single_alarm=("single_alarm", "mean"),
                                   single_loc=("single_loc", "mean"), pair_alarm=("pair_alarm", "mean"),
                                   pair_loc=("pair_loc", "mean"), any_alarm=("any_alarm", "mean")).round(3)
    print(s.to_string())


WORKER, N_WORKERS = 0, 1

if __name__ == "__main__":
    # usage: e5d.py <proto> [n_null] [n_shift] [worker] [n_workers]
    if len(sys.argv) > 5:
        WORKER, N_WORKERS = int(sys.argv[4]), int(sys.argv[5])
    main(sys.argv[1], *(int(x) for x in sys.argv[2:4]))
