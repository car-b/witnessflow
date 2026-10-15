"""SF1 / S1B-S1E display tables (suppplan_v2 1b-1e): re-arrangement, differences and summaries of frozen records only.
No fitting, simulation, resampling or test.  Every number with a frozen counterpart is asserted.

S1B  results/{levine32_v2,wingmen_v1}/thresholds.json (+ levine32_v1 cuts), protocols (backbone, claims, rule),
     initial_audit.tsv (reliable flag per record)                                   -> figS1_markers.tsv
S1C  results/*/e5g_threshold_design.tsv (pi_* per offset), e5g_threshold_audit.tsv   -> figS1_offset_pi.tsv,
                                                                                      figS1_offset_audit.tsv
S1D  results/*/reference_claims.tsv (CP 95 % interval of the reference frequency); ledger/*_targets.tsv ref_var
     (= max over the 8 patterns of p(1-p), reference split) with n_ref of the parent and the ledger eps
     (src/e5i_shuffle.py SET)                                                       -> figS1_refci.tsv, figS1_refse.tsv
S1E  results/cytonorm_real_batch/batch_shift.tsv (p columns dropped)                -> figS1_batch.tsv
Check summary -> figS1_tables_check.json"""
import glob, json, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(WF, "results")
OUT = os.path.join(WF, "figures", "derived")
os.makedirs(OUT, exist_ok=True)
DS = ["levine32_v2", "wingmen_v1"]
TSTAR = {"CCR7", "CD27", "CD45RA"}
chk = {}


def proto(pid):
    return json.load(open(os.path.join(WF, "protocols", pid + ".json"), encoding="utf-8"))


# ------------------------------------------------------------------ S1B marker atlas
rows = []
v1 = json.load(open(os.path.join(RES, "levine32_v1", "thresholds.json")))
for pid in DS:
    P = proto(pid)
    T = json.load(open(os.path.join(RES, pid, "thresholds.json")))
    rule = P["threshold_rule"]
    A = pd.read_csv(os.path.join(RES, pid, "initial_audit.tsv"), sep="\t")
    bb = set(P["backbone"])
    claims = {c["id"]: c["markers"] for c in P["claims"]}
    for m, d in T.items():
        ok = bool(d["ashman_D"] >= rule["D_min"] and d["w_hi"] >= rule["w_min"])
        assert ok == bool(d["reliable"]), (pid, m)          # stored flag = documented rule (D and w_hi)
        used = [cid for cid, mk in claims.items() if m in mk]
        flag = [cid for cid in used if not d["reliable"]]
        n_unrel = {cid: int((~A[A.claim == cid].reliable.astype(bool)).sum()) for cid in flag}
        rows.append(dict(dataset=pid, marker=m, backbone=m in bb, reliable=bool(d["reliable"]),
                         candidate=(m not in bb) and bool(d["reliable"]),
                         tstar=(pid == "wingmen_v1" and m in TSTAR),
                         ashman_D=d["ashman_D"], D_fail=d["ashman_D"] < rule["D_min"], w_hi=d["w_hi"],
                         w_fail=d["w_hi"] < rule["w_min"], thr=d["thr"], mu_lo=d["mu_lo"], mu_hi=d["mu_hi"],
                         zero_frac=d["zero_frac"], pos_frac=d["pos_frac"],
                         thr_v1=v1[m]["thr"] if pid == "levine32_v2" else np.nan,
                         claims=" ".join(used), claims_unreliable_marker=" ".join(flag),
                         unreliable_audit_records=";".join(f"{k}:{v}" for k, v in n_unrel.items()),
                         D_min=rule["D_min"], w_min=rule["w_min"], zero_cut=rule["zero_cut"]))
M = pd.DataFrame(rows)
for pid, n, nb, nu, nc, nd2 in (("levine32_v2", 32, 8, 4, 20, 16), ("wingmen_v1", 37, 8, 3, 26, 13)):
    s = M[M.dataset == pid]
    assert len(s) == n and s.backbone.sum() == nb and (~s.reliable).sum() == nu and s.candidate.sum() == nc
    assert (s.ashman_D < 2.0).sum() == nd2, (pid, (s.ashman_D < 2.0).sum())
    assert not (s.backbone & ~s.reliable).any()
assert sorted(M[~M.reliable].marker) == sorted(["CD235ab", "Flt3", "CD41", "CD61", "CTLA4", "CD123", "TIGIT"])
fl = M[M.claims_unreliable_marker != ""]
assert sorted(zip(fl.marker, fl.claims_unreliable_marker, fl.unreliable_audit_records)) == \
    [("Flt3", "H3", "H3:10"), ("TIGIT", "D2", "D2:10")]
lv1 = M[M.dataset == "levine32_v2"].set_index("marker").thr_v1
for m, x in {"CD4": 0.020, "CD133": 0.016, "CD15": 0.018, "CD14": 0.022, "CD117": 0.026}.items():
    assert round(lv1[m], 3) == x, (m, lv1[m])
# zero spike, negative remainder and positive fraction are disjoint (thr > zero_cut) -> stack to 1
assert (M.thr > M.zero_cut).all() and ((M.zero_frac + M.pos_frac) <= 1 + 1e-12).all()
M.to_csv(os.path.join(OUT, "figS1_markers.tsv"), sep="\t", index=False)
chk["S1B"] = dict(rows="32+37", backbone="8+8", unreliable="4+3", candidates="20+26", D_lt_2="16/32, 13/37",
                  flagged={"Flt3": "H3 10 records", "TIGIT": "D2 10 records"},
                  amendment_text_says="D=2 would drop 13/32 (frozen data: 16/32; 13/37 is WINGMEN)")

# ------------------------------------------------------------------ S1C threshold offsets
pis, auds = [], []
for pid in DS:
    D = pd.read_csv(os.path.join(RES, pid, "e5g_threshold_design.tsv"), sep="\t")
    A = pd.read_csv(os.path.join(RES, pid, "e5g_threshold_audit.tsv"), sep="\t")
    assert len(D) == 60 and D.parent.nunique() == 1
    pc = [c for c in D.columns if c.startswith("pi_")]
    g = D.groupby("offset")[pc].agg(["min", "max"])
    for c in pc:
        assert np.allclose(g[(c, "min")], g[(c, "max")])          # one value per offset
        for off, val in D.groupby("offset")[c].first().items():
            pis.append(dict(dataset=pid, offset=off, parent=c[3:], pi=val, design_replay_parent=D.parent.iloc[0],
                            design_replayed=(c[3:] == D.parent.iloc[0])))
    assert A.ref_in.astype(bool).all()
    A = A.assign(width=A.conf_hi - A.conf_lo, knn_err_pp=100 * A.knn_err)
    assert np.allclose(A.knn_err, (A.knn - A.ref).abs())
    auds.append(A)
PI = pd.DataFrame(pis)
AU = pd.concat(auds, ignore_index=True)
assert (AU.dataset == "levine32_v2").sum() == 55 and (AU.dataset == "wingmen_v1").sum() == 70
assert AU[AU.dataset == "levine32_v2"].claim.nunique() == 11 and AU[AU.dataset == "wingmen_v1"].claim.nunique() == 14
mono = PI[(PI.dataset == "levine32_v2") & (PI.parent == "Mono")].set_index("offset").pi
assert round(100 * mono[-0.25], 2) == 10.35 and round(100 * mono[0.25], 2) == 1.96
PI.to_csv(os.path.join(OUT, "figS1_offset_pi.tsv"), sep="\t", index=False)
AU.to_csv(os.path.join(OUT, "figS1_offset_audit.tsv"), sep="\t", index=False)
chk["S1C"] = dict(design_rows="60+60", audit_rows="55+70", claims="11+14", ref_in_all=True,
                  levine_mono_pct=[round(100 * mono[-0.25], 2), round(100 * mono[0.25], 2)],
                  width_range=[round(float(AU.width.min()), 4), round(float(AU.width.max()), 4)],
                  knn_err_pp_max=round(float(AU.knn_err_pp.max()), 2))

# ------------------------------------------------------------------ S1D reference uncertainty
rc = []
for pid in DS:
    R = pd.read_csv(os.path.join(RES, pid, "reference_claims.tsv"), sep="\t")
    R = R.assign(dataset=pid, eps=proto(pid)["claim_goal"]["epsilon"], half=(R.ref_hi - R.ref_lo) / 2,
                 lo_pp=100 * (R.ref_lo - R.ref), hi_pp=100 * (R.ref_hi - R.ref))
    rc.append(R[["dataset", "claim", "parent", "markers", "reliable", "ref", "ref_lo", "ref_hi", "n_ref", "half",
                 "lo_pp", "hi_pp", "eps"]])
RC = pd.concat(rc, ignore_index=True)
RC["half_gt_eps"] = RC.half > RC.eps
assert len(RC) == 27
assert RC.half_gt_eps.sum() == 1 and RC[RC.half_gt_eps].claim.iloc[0] == "M2"
assert round(float(RC[RC.claim == "M2"].half.iloc[0]), 6) == 0.020989
RC.to_csv(os.path.join(OUT, "figS1_refci.tsv"), sep="\t", index=False)
# M1 / M2 share n(ref) = 2,215 (Mono); the half-width differs only through the reference frequency
mm = RC[(RC.dataset == "levine32_v2") & RC.claim.isin(["M1", "M2"])].set_index("claim")
assert (mm.n_ref == 2215).all()
assert round(100 * mm.loc["M1", "ref"], 2) == 84.65 and round(100 * mm.loc["M2", "ref"], 2) == 53.36
assert round(100 * mm.loc["M1", "half"], 3) == 1.523 and round(100 * mm.loc["M2", "half"], 3) == 2.099
# ledger eps (src/e5i_shuffle.py SET, also the eps column of ledger_summary.tsv)
SET = {"levine32_v2": [("T", 0.05, 142)], "wingmen_v1": [("CD4T", 0.03, 180), ("CD8T", 0.05, 180)]}
se_rows, mx = [], {}
for pid, lst in SET.items():
    LS = pd.read_csv(os.path.join(RES, pid, "ledger", "ledger_summary.tsv"), sep="\t")
    nref = pd.read_csv(os.path.join(RES, pid, "reference_claims.tsv"), sep="\t").groupby("parent").n_ref.first()
    for G, eps, r in lst:
        assert set(LS[LS.parent == G].eps) == {eps} and set(LS[LS.parent == G].r) == {r}
        fs = sorted(glob.glob(os.path.join(RES, pid, "ledger", f"{G}_rep*_opt*_targets.tsv")))
        assert len(fs) == 6
        base = pd.read_csv(fs[0], sep="\t")[["triple", "ref_var"]]
        for f in fs[1:]:
            o = pd.read_csv(f, sep="\t")[["triple", "ref_var"]]
            assert (o.triple == base.triple).all() and np.array_equal(o.ref_var.values, base.ref_var.values)
        assert np.isclose(base.ref_var.mean(), LS[LS.parent == G].all_mean_refvar.iloc[0])
        n = int(nref[G])
        se = np.sqrt(base.ref_var / n)
        d = base.assign(dataset=pid, parent=G, n_ref=n, eps=eps, r=r, se=se, se_over_eps=se / eps,
                        source=os.path.relpath(fs[0], WF).replace("\\", "/"))
        se_rows.append(d)
        mx[G] = round(100 * float(d.se_over_eps.max()), 2)
SE = pd.concat(se_rows, ignore_index=True)
assert mx == {"T": 6.75, "CD4T": 6.86, "CD8T": 6.5}, mx
assert len(SE[SE.parent == "T"]) == 1140 and len(SE[SE.parent == "CD4T"]) == 2600 == len(SE[SE.parent == "CD8T"])
SE.to_csv(os.path.join(OUT, "figS1_refse.tsv"), sep="\t", index=False)
chk["S1D"] = dict(claims=27, M2_half=0.020989, half_gt_eps=["M2"], max_se_over_eps_pct=mx,
                  n_targets={"T": 1140, "CD4T": 2600, "CD8T": 2600},
                  note="SE = sqrt(ref_var / n_ref), ref_var = max_pattern p(1-p): approximation")

# ------------------------------------------------------------------ S1E real batch differences
B = pd.read_csv(os.path.join(RES, "cytonorm_real_batch", "batch_shift.tsv"), sep="\t")
assert len(B) == 111 and B.donor.nunique() == 3 and B.marker.nunique() == 37 and int(B.reliable.sum()) == 87
assert np.allclose(B.dpos, B.pos_b2 - B.pos_b1)
rel_by_marker = B.groupby("marker").reliable.agg(["min", "max"])
assert (rel_by_marker["min"] == rel_by_marker["max"]).all()   # reliability is per marker (pooled cut)
B = B.drop(columns=["p", "p_null"]).assign(abs_median_shift=lambda x: x.median_shift.abs(), dpos_pp=lambda x: 100 * x.dpos)
r = B[B.reliable]
q = dict(n=len(r), median=round(float(r.abs_median_shift.median()), 3), p90=round(float(r.abs_median_shift.quantile(.9)), 3),
         max=round(float(r.abs_median_shift.max()), 3),
         n_ge_025=int((r.abs_median_shift >= 0.25).sum()), n_ge_05=int((r.abs_median_shift >= 0.5).sum()),
         max_abs_dpos_pp=round(float(r.dpos_pp.abs().max()), 1))
F4 = pd.read_csv(os.path.join(OUT, "fig4_E_cytonorm.tsv"), sep="\t").iloc[0]    # Fig 4E display table
assert int(F4.n) == 87 and int(F4.markers) == 29
assert np.isclose(r.abs_median_shift.quantile(.9), F4.p90) and np.isclose(r.abs_median_shift.max(), F4["max"])
assert np.isclose(r.abs_median_shift.median(), F4["median"])
B.to_csv(os.path.join(OUT, "figS1_batch.tsv"), sep="\t", index=False)
# column order of the heat maps (mean signed median shift over the 3 donors, descending; ties by name)
mean_shift = B.groupby("marker").median_shift.mean()
order = sorted(mean_shift.index, key=lambda m: (-mean_shift[m], m))
# per-marker maximum |median shift| over donors, reliable markers only (the other 8 stay hatched)
BM = pd.DataFrame(dict(marker=order, column=range(1, len(order) + 1)))
BM["reliable"] = BM.marker.map(rel_by_marker["min"]).astype(bool)
g = B.groupby("marker")
BM["mean_median_shift"] = BM.marker.map(mean_shift)
BM["max_abs_median_shift"] = BM.marker.map(g.abs_median_shift.max())
BM["donor_of_max"] = BM.marker.map(B.loc[g.abs_median_shift.idxmax()].set_index("marker").donor)
BM.loc[~BM.reliable, ["max_abs_median_shift", "donor_of_max"]] = [np.nan, ""]
nmk = int((BM.max_abs_median_shift >= 0.25).sum())
assert BM.reliable.sum() == 29 and nmk == 5, nmk                    # 5/29 markers (not 8/87 records)
BM["ge_025"] = BM.max_abs_median_shift >= 0.25
BM.to_csv(os.path.join(OUT, "figS1_batch_marker.tsv"), sep="\t", index=False)
# per-donor ECDF summary (reliable records)
dsum = []
for dn, s in r.groupby("donor"):
    dsum.append(dict(donor=dn, n_reliable=len(s), n_ge_025=int((s.abs_median_shift >= 0.25).sum()),
                     n_ge_05=int((s.abs_median_shift >= 0.5).sum()), median=float(s.abs_median_shift.median()),
                     max=float(s.abs_median_shift.max())))
DS_ = pd.DataFrame(dsum)
assert DS_.n_reliable.tolist() == [29, 29, 29] and DS_.n_ge_025.tolist() == [4, 1, 3] and DS_.n_ge_05.sum() == 0
DS_.to_csv(os.path.join(OUT, "figS1_batch_donor.tsv"), sep="\t", index=False)
# marker-label overlap with the two replay panels (labels only; no assay equivalence is implied)
SPELL = {"hladr": "hla-dr"}        # the only explicit spelling map; CD8a/CD8, CD66/CD66b and CD235ab_CD61 are NOT merged


def norm(m):
    k = m.lower()
    return SPELL.get(k, k)


ov_rows = []
roles = {}
for pid in DS:
    s = M[M.dataset == pid]
    roles[pid] = {norm(x.marker): ("backbone" if x.backbone else "candidate", x.marker)
                  for x in s.itertuples() if x.backbone or x.candidate}   # registered roles only
for m in order:
    row = dict(marker=m, column=order.index(m) + 1, reliable_cytonorm=bool(rel_by_marker["min"][m]))
    for pid, col in (("wingmen_v1", "wingmen"), ("levine32_v2", "levine")):
        hit = roles[pid].get(norm(m))
        row[f"{col}_role"] = hit[0] if hit else ""
        row[f"{col}_label"] = hit[1] if hit else ""
    ov_rows.append(row)
OV = pd.DataFrame(ov_rows)
cnt = {c: OV[f"{c}_role"].value_counts().to_dict() for c in ("wingmen", "levine")}
assert cnt["levine"] == {"": 23, "candidate": 7, "backbone": 7}, cnt
assert cnt["wingmen"] == {"": 21, "candidate": 8, "backbone": 8}, cnt
assert OV.loc[OV.marker == "CD123", "wingmen_role"].iloc[0] == ""       # unreliable in WINGMEN: not a candidate
assert OV.loc[OV.marker == "HLADR", "levine_label"].iloc[0] == "HLA-DR"
for m in ("CD66", "CD235ab_CD61"):                                        # not merged / not split
    assert (OV.loc[OV.marker == m, ["levine_role", "wingmen_role"]] == "").all(axis=None), m
assert OV.loc[OV.marker == "CD8a", "levine_role"].iloc[0] == ""           # Levine 'CD8' not merged with CD8a
OV.to_csv(os.path.join(OUT, "figS1_label_overlap.tsv"), sep="\t", index=False)
chk["S1E"] = dict(rows=111, reliable=87, unreliable=24, reliable_markers=int(rel_by_marker["min"].sum()), abs_shift=q,
                  markers_max_ge_025="5/29", donors_ge_025={d: f"{k}/29" for d, k in zip(DS_.donor, DS_.n_ge_025)},
                  label_overlap=dict(levine="14 = 7 candidate + 7 backbone", wingmen="16 = 8 candidate + 8 backbone",
                                     spelling_map="HLADR <-> HLA-DR only"))
chk["S1D"]["M1_M2"] = dict(n_ref=2215, ref_pct=[84.65, 53.36], half_pp=[1.523, 2.099])
json.dump(chk, open(os.path.join(OUT, "figS1_tables_check.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(chk, indent=1))
