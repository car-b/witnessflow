"""Minimal bridge-marker validation, Levine v2 initial split.

Designs (same 4 tubes x 6 non-backbone slots = same reagent cost):
  nobridge : random disjoint partition of 24 non-backbone markers (as v2)
  bridge   : 1 prespecified bridge marker in every tube + 5 unique per tube (20) -> 3 markers unmeasured
Bridge marker rule (fixed before running): the reliable non-backbone marker with the largest Ashman D
on the calibration split.
Scenarios on tube 0 cells (asinh units, re-gated with global thresholds):
  S0 none | S1 +0.5 on all non-backbone markers incl. bridge | S2 +0.5 on non-backbone markers except
  the bridge | S3 -0.5 on all non-backbone incl. bridge
Alarm = any chi-square test (4 tubes) of positive fraction for a shared marker (backbone markers and,
if present, the bridge; overall and within each parent) with p < 1e-3 / n_tests (Bonferroni).
usage: python bridge.py <protocol.json>"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay  # noqa: E402

REPS = 20
SCEN = {"S0_none": (0.0, "all"), "S1_bridge_hit": (0.5, "all"), "S2_bridge_spared": (0.5, "except_bridge"),
        "S3_bridge_hit_neg": (-0.5, "all")}


def alarm(Bm, tube, shared, pm_list):
    tests, ps = 0, []
    for m in shared:
        for msk in pm_list:
            tab = []
            for j in range(4):
                s = (tube == j) & msk
                k = Bm[s, m].sum()
                tab.append([k, s.sum() - k])
            tab = np.array(tab, float)
            if (tab.sum(0) == 0).any() or (tab.sum(1) == 0).any():
                continue
            ps.append(chi2_contingency(tab)[1])
            tests += 1
    if not ps:
        return np.nan, 0
    return float(min(ps) * tests), tests


def main(proto_path):
    proto = json.load(open(proto_path, encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    mi = {m: i for i, m in enumerate(markers)}
    bb = [mi[m] for m in proto["backbone"]]
    nonbb = [i for i in range(len(markers)) if i not in bb]
    rel = [i for i in nonbb if thr[markers[i]]["reliable"]]
    bridge = max(rel, key=lambda i: thr[markers[i]]["ashman_D"])
    ini = np.flatnonzero(part == "initial")
    claims_m = [[mi[m] for m in c["markers"]] for c in proto["claims"]]
    rng = np.random.default_rng(11)
    rows = []
    for rep in range(REPS):
        tube = rng.integers(0, 4, len(ini))
        others = [i for i in nonbb if i != bridge]
        perm = rng.permutation(nonbb)
        designs = {"nobridge": [set(perm[k * 6:(k + 1) * 6].tolist()) for k in range(4)]}
        perm2 = rng.permutation(others)
        designs["bridge"] = [{bridge} | set(perm2[k * 5:(k + 1) * 5].tolist()) for k in range(4)]
        for dname, pan in designs.items():
            measured = set().union(*pan)
            unmeasured = [i for i in nonbb if i not in measured]
            claims_unmeasured = sum(any(m in unmeasured for m in cm) for cm in claims_m)
            shared = list(bb) + ([bridge] if dname == "bridge" else [])
            for sname, (delta, which) in SCEN.items():
                Xs = Xt[ini].copy()
                sel = tube == 0
                for m in pan[0]:
                    if which == "except_bridge" and m == bridge:
                        continue
                    Xs[sel, m] += delta
                Bm = np.stack([Xs[:, j] > thr[markers[j]]["thr"] for j in range(len(markers))], 1)
                pmask = replay.parent_masks(Bm.astype(np.uint8), markers, proto["parents"])
                pm_list = [np.ones(len(ini), bool)] + list(pmask.values())
                p, nt = alarm(Bm, tube, shared, pm_list)
                rows.append(dict(rep=rep, design=dname, scenario=sname, alarm=bool(p < 1e-3) if nt else False,
                                 p_bonf=p, n_tests=nt, n_unmeasured_markers=len(unmeasured),
                                 claims_with_unmeasured_marker=claims_unmeasured))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "bridge_test.tsv"), sep="\t", index=False)
    print(f"bridge marker = {markers[bridge]} (Ashman D {thr[markers[bridge]]['ashman_D']:.2f}); reps={REPS}")
    t = df.groupby(["design", "scenario"]).agg(alarm_rate=("alarm", "mean"),
                                               unmeasured=("n_unmeasured_markers", "mean"),
                                               claims_lost=("claims_with_unmeasured_marker", "mean")).round(3)
    print(t.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
