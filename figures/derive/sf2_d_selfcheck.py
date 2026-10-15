"""SF2d derive: every self-check scenario x stored metric (e5d fixed family + e5k composition runs).

Inputs (frozen): results/{levine32_v2, wingmen_v1}/e5d_fixed_family_summary.tsv,
                 figures/data/fig4B_qc_fixed_family.tsv (must equal the two summaries),
                 results/{levine32_v2, wingmen_v1}/e5k_boundary.tsv (part B_composition, 20 allocations per mode).
Rules (supplementary figure plan v2): no p values are recovered (pmin_bonf was stored as min(1, pmin*M), truncated);
e5d alarm CIs are the stored Clopper-Pearson strings; e5k is reported only as stored counts k/20 (no new CI);
conditional localisation is 'n/a' for the null / no-shift scenarios and for scenarios without alarms; a metric that
was never saved is 'not saved'; nothing is filled with 0.
e5k localisation = alarm whose minimum-p check sits at one of the shifted tubes (`alarm_at_shifted_tube`; the marker
is not checked), e5d localisation = stored `localised` (alarm at the perturbed tube and touching the perturbed marker
where defined, src/e5d.py).
Output: figures/derived/figS2_D_matrix.tsv
"""
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(WF, "results")
DATA = os.path.join(WF, "figures", "data")
DER = os.path.join(WF, "figures", "derived")
DS = ["levine32_v2", "wingmen_v1"]
E5D = ["null", "single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only",
       "composition", "random_drop", "global_drift"]
E5K = ["none", "measured", "unmeasured"]
R_TUBES = {"levine32_v2": 71, "wingmen_v1": 130}


def rd(*p):
    # 'null' is a scenario name (not NA); 'n/a' is the stored marker of undefined localisation
    return pd.read_csv(os.path.join(*p), sep="\t", keep_default_na=False, na_values=["", "n/a"])


def main():
    fam = rd(DATA, "fig4B_qc_fixed_family.tsv")
    rows = []
    for d in DS:
        s = rd(RES, d, "e5d_fixed_family_summary.tsv")
        assert list(s.scenario) == E5D
        f = fam[fam.dataset == d].drop(columns="dataset").reset_index(drop=True)
        assert len(f) == 10
        for c in s.columns:
            a, b = s[c], f[c]
            if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), (d, c)
            else:
                assert (a.astype(str) == b.astype(str)).all(), (d, c)
        assert set(s.M_total) == {R_TUBES[d] * 21}
        for _, q in s.iterrows():
            n = int(q.n)
            k_alarm = q.alarm * n
            assert abs(k_alarm - round(k_alarm)) < 1e-6
            lo, hi = map(float, re.findall(r"[\d.]+", q.alarm_ci))
            loc_na = q.scenario == "null"
            hit_na = loc_na or round(k_alarm) == 0
            assert pd.isna(q.localised) == loc_na and pd.isna(q.hit_given_alarm) == hit_na
            k_loc = None if loc_na else q.localised * n
            if k_loc is not None:
                assert abs(k_loc - round(k_loc)) < 1e-6
            rows.append(dict(
                block="e5d", dataset=d, scenario=q.scenario, n_alloc=n, M_total=int(q.M_total),
                alarm_k=int(round(k_alarm)), alarm_rate=float(q.alarm), alarm_ci_lo=lo, alarm_ci_hi=hi,
                alarm_ci_source="stored (Clopper-Pearson 95%, e5d_fixed_family_summary.tsv)",
                single_only_alarm=float(q.single_only_alarm),
                localised_k=None if loc_na else int(round(k_loc)), localised_rate=np.nan if loc_na else float(q.localised),
                localised_status="n/a (no perturbed tube)" if loc_na else "stored",
                hit_given_alarm=np.nan if hit_na else float(q.hit_given_alarm),
                hit_status=("n/a (null scenario)" if loc_na else "n/a (no alarm)") if hit_na else "stored",
                uncheckable_frac=float(q.uncheckable_frac_mean), uncheckable_status="stored (mean over allocations)"))
        k = rd(RES, d, "e5k_boundary.tsv")
        b = k[k.part == "B_composition"].copy()
        assert len(b) == 60 and list(b["mode"].drop_duplicates()) == E5K
        b["alarm"] = b.alarm.astype(str).str.lower().map({"true": True, "false": False})
        b["alarm_at_shifted_tube"] = b.alarm_at_shifted_tube.astype(str).str.lower().map({"true": True, "false": False})
        assert b.alarm.notna().all() and b.alarm_at_shifted_tube.notna().all()
        assert not (b.alarm_at_shifted_tube & ~b.alarm).any()
        for m in E5K:
            g = b[b["mode"] == m]
            assert len(g) == 20
            ka, kl = int(g.alarm.sum()), int(g.alarm_at_shifted_tube.sum())
            none = m == "none"
            rows.append(dict(
                block="e5k", dataset=d, scenario=m, n_alloc=20, M_total=R_TUBES[d] * 21, alarm_k=ka,
                alarm_rate=ka / 20, alarm_ci_lo=np.nan, alarm_ci_hi=np.nan,
                alarm_ci_source="not computed (stored counts only; no new CI)", single_only_alarm=np.nan,
                localised_k=None if none else kl, localised_rate=np.nan if none else kl / 20,
                localised_status="n/a (no shifted tube)" if none else "count of stored alarm_at_shifted_tube",
                hit_given_alarm=np.nan if (none or ka == 0) else kl / ka,
                hit_status=("n/a (no-shift control)" if none else "n/a (no alarm)") if (none or ka == 0)
                else "count ratio of stored flags",
                uncheckable_frac=np.nan, uncheckable_status="not saved"))
    M = pd.DataFrame(rows)
    # assertions against claims table C6 / Fig. 4d-e captions
    def get(block, d, s, col):
        v = M[(M.block == block) & (M.dataset == d) & (M.scenario == s)][col]
        assert len(v) == 1
        return v.iloc[0]
    assert get("e5d", "levine32_v2", "null", "alarm_k") == 0 and get("e5d", "wingmen_v1", "null", "alarm_k") == 1
    assert get("e5d", "levine32_v2", "joint_only", "alarm_rate") == 0.49
    assert get("e5d", "wingmen_v1", "joint_only", "alarm_rate") == 0.735
    assert get("e5k", "levine32_v2", "none", "alarm_k") == 0 and get("e5k", "wingmen_v1", "none", "alarm_k") == 1
    assert get("e5k", "wingmen_v1", "measured", "alarm_k") == 3 and get("e5k", "wingmen_v1", "measured", "localised_k") == 3
    assert get("e5k", "levine32_v2", "measured", "alarm_k") == 0
    sc = pd.read_csv(os.path.join(DATA, "supp_boundary_composition.tsv"), sep="\t")
    for _, q in sc.iterrows():   # frozen e5k summary (means over 20 allocations)
        assert abs(get("e5k", q.dataset, q["mode"], "alarm_rate") - q.alarm) < 1e-12
        lk = get("e5k", q.dataset, q["mode"], "localised_k")
        assert abs((0 if lk is None or pd.isna(lk) else lk) / 20 - q.alarm_at_shifted_tube) < 1e-12
    M.to_csv(os.path.join(DER, "figS2_D_matrix.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(M[["block", "dataset", "scenario", "alarm_k", "n_alloc", "alarm_rate", "alarm_ci_lo", "alarm_ci_hi",
             "localised_k", "localised_rate", "hit_given_alarm", "hit_status", "uncheckable_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
