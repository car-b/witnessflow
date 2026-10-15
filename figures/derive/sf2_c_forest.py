"""SF2c derive: full initial-audit forest (630 claim x replicate records), deterministic reorganisation only.

Inputs (frozen): results/{levine32_v2, wingmen_v1, levine32_v2_rareB_1, levine32_v2_rareB_0p3,
                 levine32_v2_rareB_0p1}/initial_audit.tsv, results/<protocol>/thresholds.json, protocols/<protocol>.json;
                 figures/derived/fig4_A_summary.tsv (Fig. 4a display table, assertions only).
Derived quantities: |estimate - ref| for the five frozen estimators (knn_block, knn_block_within, knn_major, maxent,
cond_indep), the > 5 pp flag, 'reference inside the K = 1 confidence audit interval' (tolerance 1e-9, as in
src/compare_shift.py), the evidence tier (I: n_direct > 0; II otherwise, asserted lp_status = ok and a shrunken
interval).  K4 / K8 are used only as stored (width and inclusion flag; no endpoints exist).  Fields that were never
computed stay empty ('not computed'): knn_block_within for Levine-32 protocols, K8 for WINGMEN.  Nothing is
re-fitted.
Outputs: figures/derived/figS2_C_records.tsv (630), figS2_C_windows.tsv (33 drawn windows), figS2_C_totals.tsv
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(WF, "results")
DER = os.path.join(WF, "figures", "derived")
PROTO = [("wingmen_v1", "WINGMEN", "main"), ("levine32_v2", "Levine-32", "main"),
         ("levine32_v2_rareB_1", "rare-B 1%", "rareB"), ("levine32_v2_rareB_0p3", "rare-B 0.3%", "rareB"),
         ("levine32_v2_rareB_0p1", "rare-B 0.1%", "rareB")]
EST = ["knn_block", "knn_block_within", "knn_major", "maxent", "cond_indep"]
INTOL, TAU, CUT = 1e-9, 1e-9, 0.05


def main():
    recs = []
    for d, lab, kind in PROTO:
        A = pd.read_csv(os.path.join(RES, d, "initial_audit.tsv"), sep="\t", keep_default_na=False, na_values=[""])
        proto = json.load(open(os.path.join(WF, "protocols", f"{d}.json"), encoding="utf-8"))
        thr = json.load(open(os.path.join(RES, d, "thresholds.json"), encoding="utf-8"))
        claims = [c["id"] for c in proto["claims"]]
        assert set(A.claim) == set(claims) and len(A) == 10 * len(claims)
        assert (A.lp_status == "ok").all()
        A = A.assign(protocol=d, label=lab, kind=kind, claim_order=A.claim.map({c: i for i, c in enumerate(claims)}),
                     rep_display=A.rep + 1)
        A["unreliable_markers"] = ["/".join(m for m in s.split("/") if not thr[m]["reliable"]) for s in A.markers]
        assert ((A.unreliable_markers != "") == (~A.reliable.astype(bool))).all()
        direct = A.n_direct.fillna(0) > 0
        shrink = (A.conf_lo > TAU) | (A.conf_hi < 1 - TAU)
        assert (direct | shrink).all()
        A["tier"] = np.where(direct, "I", "II")
        for e in EST:
            if e in A.columns:
                A[f"err_{e}"] = (A[e] - A.ref).abs()
                A[f"gt5_{e}"] = A[f"err_{e}"] > CUT
            else:
                A[e] = np.nan
                A[f"err_{e}"] = np.nan
                A[f"gt5_{e}"] = pd.NA
        A["conf_w_K1"] = A.conf_hi - A.conf_lo
        A["ref_in_K1"] = (A.ref >= A.conf_lo - INTOL) & (A.ref <= A.conf_hi + INTOL)
        A["ident_in_conf"] = (A.ident_lo >= A.conf_lo - INTOL) & (A.ident_hi <= A.conf_hi + INTOL)
        for k in ("K4", "K8"):
            if f"conf_w_{k}" not in A.columns:
                A[f"conf_w_{k}"] = np.nan
                A[f"ident_w_{k}"] = np.nan
                A[f"ref_in_{k}"] = pd.NA
        A["in_figure"] = True if kind == "main" else (A.parent == "B")
        recs.append(A)
    R = pd.concat(recs, ignore_index=True)
    assert len(R) == 630 and int(R.in_figure.sum()) == 330
    # field asymmetry: Levine protocols have no knn_block_within, WINGMEN has no K8
    lev = R.protocol.str.startswith("levine")
    assert R.loc[lev, "knn_block_within"].isna().all() and R.loc[~lev, "knn_block_within"].notna().all()
    assert R.loc[~lev, "conf_w_K8"].isna().all() and R.loc[lev, "conf_w_K8"].notna().all()
    assert R.ref_in_K1.all() and R.ident_in_conf.all()
    assert R.loc[R.ref_in_K4.notna(), "ref_in_K4"].astype(bool).all() and R.ref_in_K4.notna().all()
    assert R.loc[lev, "ref_in_K8"].astype(bool).all()
    rb = R[R.kind == "rareB"]
    assert set(rb[rb.in_figure].claim) == {"B1", "B2"}
    # assertions against the Fig. 4a display table and claims table C9
    s4 = pd.read_csv(os.path.join(DER, "fig4_A_summary.tsv"), sep="\t").set_index("dataset")
    for d, g in R.groupby("protocol"):
        assert int(g.ref_in_K1.sum()) == int(s4.loc[d, "ref_in"]) == len(g) == int(s4.loc[d, "n"])
        assert int(g.gt5_knn_block.sum()) == int(s4.loc[d, "knn_gt_5pp"])
        assert int((g.tier == "I").sum()) == int(s4.loc[d, "tier1"])
        assert abs(g.err_knn_block.max() - s4.loc[d, "max_err"]) < 1e-12
    assert round(100 * R[R.kind == "rareB"].err_knn_block.max(), 2) == 17.93
    assert int(R[R.protocol == "wingmen_v1"].gt5_knn_block.sum()) == 9
    assert set(R[~R.reliable.astype(bool)].claim) == {"H3", "D2"}
    assert R[~R.reliable.astype(bool)].groupby("protocol").size().to_dict() == {
        "levine32_v2": 10, "levine32_v2_rareB_0p1": 10, "levine32_v2_rareB_0p3": 10, "levine32_v2_rareB_1": 10,
        "wingmen_v1": 10}

    # ---- drawn windows (27 main claims + rare-B B1/B2 x 3 levels)
    F = R[R.in_figure]
    rows = []
    for (d, c), g in F.groupby(["protocol", "claim"], sort=False):
        assert len(g) == 10
        r = dict(protocol=d, label=g.label.iloc[0], kind=g.kind.iloc[0], claim=c, parent=g.parent.iloc[0],
                 markers=g.markers.iloc[0], claim_order=int(g.claim_order.iloc[0]), n_records=10,
                 n_parent=int(g.n_parent.iloc[0]), n_ref=int(g.n_ref.iloc[0]), ref=float(g.ref.iloc[0]),
                 reliable=bool(g.reliable.iloc[0]), unreliable_markers=g.unreliable_markers.iloc[0],
                 tier1=int((g.tier == "I").sum()))
        assert g.ref.nunique() == 1
        for e in EST:
            r[f"gt5_{e}"] = "not computed" if g[e].isna().all() else int(g[f"gt5_{e}"].astype(bool).sum())
        for k in ("K1", "K4", "K8"):
            r[f"ref_in_{k}"] = "not computed" if g[f"ref_in_{k}"].isna().all() else int(g[f"ref_in_{k}"].astype(bool).sum())
        rows.append(r)
    W = pd.DataFrame(rows)
    assert len(W) == 33
    # ---- per-protocol totals (all records of each protocol, incl. the rare-B claims not drawn)
    tot = []
    for d, lab, kind in PROTO:
        g = R[R.protocol == d]
        t = dict(protocol=d, label=lab, n_records=len(g), n_drawn=int(g.in_figure.sum()),
                 tier1=int((g.tier == "I").sum()), unreliable_records=int((~g.reliable.astype(bool)).sum()),
                 max_err_knn_block=float(g.err_knn_block.max()))
        for e in EST:
            t[f"gt5_{e}"] = "not computed" if g[e].isna().all() else int(g[f"gt5_{e}"].astype(bool).sum())
        for k in ("K1", "K4", "K8"):
            t[f"ref_in_{k}"] = "not computed" if g[f"ref_in_{k}"].isna().all() else int(g[f"ref_in_{k}"].astype(bool).sum())
        for k in ("K1", "K4", "K8"):   # stored confidence-interval widths (K4/K8: width only, no endpoints)
            t[f"median_conf_w_{k}"] = "not computed" if g[f"conf_w_{k}"].isna().all() else float(g[f"conf_w_{k}"].median())
        tot.append(t)
    T = pd.DataFrame(tot)
    tw = T.set_index("protocol")
    assert [tw.loc["wingmen_v1", f"gt5_{e}"] for e in EST] == [9, 14, 33, 37, 66]
    assert [tw.loc["levine32_v2", f"gt5_{e}"] for e in EST] == [1, "not computed", 50, 12, 11]
    keep = ["protocol", "label", "kind", "claim", "claim_order", "parent", "markers", "unreliable_markers", "reliable",
            "rep", "rep_display", "in_figure", "tier", "n_direct", "n_parent", "n_events", "lp_status",
            "conf_lo", "conf_hi", "conf_w_K1", "ident_lo", "ident_hi", "ref", "ref_lo", "ref_hi", "n_ref"] + EST + \
           [f"err_{e}" for e in EST] + [f"gt5_{e}" for e in EST] + \
           ["ref_in_K1", "conf_w_K4", "ident_w_K4", "ref_in_K4", "conf_w_K8", "ident_w_K8", "ref_in_K8"]
    R[keep].to_csv(os.path.join(DER, "figS2_C_records.tsv"), sep="\t", index=False)
    W.to_csv(os.path.join(DER, "figS2_C_windows.tsv"), sep="\t", index=False)
    T.to_csv(os.path.join(DER, "figS2_C_totals.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    print("conf width range", R.conf_w_K1.min().round(4), R.conf_w_K1.max().round(4))


if __name__ == "__main__":
    main()
