"""SF1 / S1A cell ledger (suppplan_v2 1a): deterministic restore of the frozen four-way split and parent counts.

Rule: same loaders, split function, seeds, thresholds and parent gates as src/replay.py (replay.load / replay.split
with protocol split seeds; frozen results/<pid>/thresholds.json cuts; replay.parent_masks).  Raw event counts are
read from the FCS $TOT header only.  Nothing is fitted, simulated or tested.  Every restored number that has a frozen
counterpart is asserted against it:
  - follow-up totals 53,125 (Levine-32) and 159,264 (WINGMEN)      [lat06_parent_frontier.tsv N_followup]
  - reference parent counts                                      [reference_claims.tsv n_ref]
  - initial parent counts                                        [initial_audit.tsv n_parent]
  - calibration / reference parent counts                        [explore_q1_all_triples.tsv calibration_N / reference_N]
  - per-donor follow-up totals N(d)                              [e5b_per_donor.tsv N_d]
  - per-donor inclusion rule of e5b (>= 500 parent cells in the donor's follow-up AND reference split) reproduces the
    donor sets of e5b_per_donor.tsv; CD8T donor 6 must be the only exclusion
  - rareB thinning (protocol `downsample`, replay.py rule) gives initial B counts = rareB initial_audit n_parent
  - WINGMEN: 27 Pre files, 13 with >= 30,000 events (min_events), PreWN08 = 28,060
Outputs (figures/derived/): figS1_files.tsv, figS1_split.tsv, figS1_parents.tsv, figS1_donor_parent.tsv,
figS1_rareB.tsv, figS1_ledger_check.json"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WF, "src"))
import replay, datasets, fcs  # noqa: E402

OUT = os.path.join(WF, "figures", "derived")
RES = os.path.join(WF, "results")
SPLITS = ["calibration", "initial", "followup", "reference"]
os.makedirs(OUT, exist_ok=True)
check = {}


def tot_header(path):
    """$TOT from the FCS TEXT segment only (no data read)."""
    with open(path, "rb") as f:
        head = f.read(58)
        t0, t1 = int(head[10:18]), int(head[18:26])
        f.seek(0)
        buf = f.read(t1 + 1)
    return int(fcs._parse_text(buf[t0:t1 + 1])["$TOT"])


def proto(pid):
    return json.load(open(os.path.join(WF, "protocols", pid + ".json"), encoding="utf-8"))


def frozen_parent_counts(pid):
    A = pd.read_csv(os.path.join(RES, pid, "initial_audit.tsv"), sep="\t")
    Rf = pd.read_csv(os.path.join(RES, pid, "reference_claims.tsv"), sep="\t")
    Q = pd.read_csv(os.path.join(RES, pid, "explore_q1_all_triples.tsv"), sep="\t",
                    usecols=["parent", "reference_N", "calibration_N"])
    ini = A.groupby("parent").n_parent.agg(["min", "max"])
    assert (ini["min"] == ini["max"]).all()
    ref = Rf.groupby("parent").n_ref.agg(["min", "max"])
    assert (ref["min"] == ref["max"]).all()
    q = Q.groupby("parent")[["reference_N", "calibration_N"]].agg(["min", "max"])
    for c in ("reference_N", "calibration_N"):
        assert (q[(c, "min")] == q[(c, "max")]).all()
    return ini["min"].astype(int).to_dict(), ref["min"].astype(int).to_dict(), \
        q[("calibration_N", "min")].astype(int).to_dict(), q[("reference_N", "min")].astype(int).to_dict()


def binarize(Xt, markers, pid):
    thr = json.load(open(os.path.join(RES, pid, "thresholds.json")))
    return np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)


L6 = pd.read_csv(os.path.join(WF, "tools", "scan", "out_lat", "lat06_parent_frontier.tsv"), sep="\t")
NF = L6.groupby("dataset").N_followup.first().to_dict()

files_rows, split_rows, parent_rows, donor_rows, rare_rows = [], [], [], [], []

# ------------------------------------------------------------------ Levine-32 (one FCS, 2 individuals)
pid = "levine32_v2"
P = proto(pid)
fpath = os.path.join(WF, P["dataset"]["file"])
tot_L = tot_header(fpath)
files_rows.append(dict(dataset=pid, file=os.path.basename(fpath), group="single file", tot=tot_L, used=True,
                       reason="all events (no event filter in protocol)"))
Xt, markers, sample = replay.load(P)
assert len(Xt) == tot_L
part = replay.split(len(Xt), sample, P["split"], P["split"]["seed"])
for s in np.unique(sample):
    m = sample == s
    split_rows.append(dict(dataset=pid, sample=f"individual {int(s)}", file=os.path.basename(fpath), events_raw=int(m.sum()),
                           cells_qc=int(m.sum()), **{f"n_{k}": int((m & (part == k)).sum()) for k in SPLITS}))
nfol = int((part == "followup").sum())
assert nfol == 53125 == NF[pid], nfol
Bin = binarize(Xt, markers, pid)
pm = replay.parent_masks(Bin, markers, P["parents"])
ini_f, ref_f, cal_q, ref_q = frozen_parent_counts(pid)
for G, msk in pm.items():
    row = dict(dataset=pid, parent=G, gate=L6[(L6.dataset == pid) & (L6.parent == G)].gate.iloc[0],
               **{f"n_{k}": int((msk & (part == k)).sum()) for k in SPLITS})
    if G in ini_f:
        assert row["n_initial"] == ini_f[G], (G, row, ini_f[G])
        assert row["n_reference"] == ref_f[G], (G, row, ref_f[G])
    assert row["n_calibration"] == cal_q[G] and row["n_reference"] == ref_q[G], (G, row)
    n_ini_all = int((part == "initial").sum())
    assert n_ini_all == 106250, n_ini_all            # denominator of pi: all initial cells (not the sum of parents)
    pi_ini = row["n_initial"] / n_ini_all
    assert abs(pi_ini - L6[(L6.dataset == pid) & (L6.parent == G)].pi.iloc[0]) < 5e-5
    row["pi_initial"] = pi_ini
    row["n_initial_all_cells"] = n_ini_all
    parent_rows.append(row)
check[pid] = dict(tot=tot_L, n_individuals=int(len(np.unique(sample))), followup=nfol, initial_all_cells=106250,
                  asserted=["followup=53125", "n_ref (reference_claims)", "n_parent (initial_audit)",
                            "calibration_N/reference_N (explore_q1)", "pi (lat06, 4 dp)"])

# rareB thinning (replay.py `downsample` rule; kept counts depend only on n_keep, the RNG picks which cells)
for rp in ["levine32_v2_rareB_1", "levine32_v2_rareB_0p3", "levine32_v2_rareB_0p1"]:
    PR = proto(rp)
    ds = PR["downsample"]
    assert PR["split"] == P["split"] and PR["parents"] == P["parents"]
    rngd = np.random.default_rng(ds.get("seed", 0))
    keep = np.ones(len(Xt), bool)
    kept = {}
    for sp_name in ("initial", "followup"):
        insp = part == sp_name
        g = np.flatnonzero(insp & pm[ds["parent"]])
        n_non = int((insp & ~pm[ds["parent"]]).sum())
        f = ds["target_frac"]
        n_keep = min(len(g), int(round(f * n_non / (1 - f))))
        drop = rngd.choice(g, len(g) - n_keep, replace=False)
        keep[drop] = False
        kept[sp_name] = int((keep & pm[ds["parent"]] & insp).sum())
    A = pd.read_csv(os.path.join(RES, rp, "initial_audit.tsv"), sep="\t")
    fz = A[A.parent == ds["parent"]].n_parent.unique()
    assert len(fz) == 1 and int(fz[0]) == kept["initial"], (rp, kept, fz)
    Rr = pd.read_csv(os.path.join(RES, rp, "reference_claims.tsv"), sep="\t")
    assert int(Rr[Rr.parent == "B"].n_ref.iloc[0]) == ref_f["B"]  # reference split untouched
    rare_rows.append(dict(protocol=rp, parent=ds["parent"], target_frac=ds["target_frac"], seed=ds["seed"],
                          n_initial_full=int((pm["B"] & (part == "initial")).sum()), n_initial_kept=kept["initial"],
                          n_followup_full=int((pm["B"] & (part == "followup")).sum()), n_followup_kept=kept["followup"],
                          n_reference=ref_f["B"]))
check["rareB_initial_kept"] = {r["protocol"]: r["n_initial_kept"] for r in rare_rows}
del Xt, Bin, pm

# ------------------------------------------------------------------ WINGMEN (one FCS per sample)
pid = "wingmen_v1"
P = proto(pid)
ds = P["dataset"]
for grp in ("Pre", "Post"):
    for f in sorted(glob.glob(os.path.join(WF, ds["dir"], f"{grp}WN*.fcs"))):
        t = tot_header(f)
        used = grp == "Pre" and t >= ds["min_events"]
        files_rows.append(dict(dataset=pid, file=os.path.basename(f), group=grp, tot=t, used=used,
                               reason="used" if used else ("Post-treatment files not in protocol (pattern PreWN*.fcs)"
                                                           if grp == "Post" else f"< min_events {ds['min_events']}")))
FD = pd.DataFrame(files_rows)
pre = FD[(FD.dataset == pid) & (FD.group == "Pre")]
assert len(pre) == 27 and int(pre.used.sum()) == 13
assert int(pre[pre.file == "PreWN08.fcs"].tot.iloc[0]) == 28060
assert len(FD[(FD.dataset == pid) & (FD.group == "Post")]) == 27
Xt, markers, sample, names = datasets.load_fcs_dir(ds)
assert [n + ".fcs" for n in names] == pre[pre.used].file.tolist()
part = replay.split(len(Xt), sample, P["split"], P["split"]["seed"])
nfol = int((part == "followup").sum())
assert nfol == 159264 == NF[pid], nfol
E5 = pd.read_csv(os.path.join(RES, pid, "e5b_per_donor.tsv"), sep="\t")
Nd_frozen = E5.groupby("donor").N_d.agg(["min", "max"])
assert (Nd_frozen["min"] == Nd_frozen["max"]).all()
tot_by_file = dict(zip(FD.file, FD.tot))
for s in np.unique(sample):
    m = sample == s
    fn = names[int(s)] + ".fcs"
    row = dict(dataset=pid, sample=f"donor {int(s)}", file=fn, events_raw=tot_by_file[fn], cells_qc=int(m.sum()),
               **{f"n_{k}": int((m & (part == k)).sum()) for k in SPLITS})
    assert row["n_followup"] == int(Nd_frozen.loc[int(s), "min"]), (s, row)
    split_rows.append(row)
Bin = binarize(Xt, markers, pid)
pm = replay.parent_masks(Bin, markers, P["parents"])
ini_f, ref_f, cal_q, ref_q = frozen_parent_counts(pid)
for G, msk in pm.items():
    row = dict(dataset=pid, parent=G, gate=L6[(L6.dataset == pid) & (L6.parent == G)].gate.iloc[0],
               **{f"n_{k}": int((msk & (part == k)).sum()) for k in SPLITS})
    assert row["n_initial"] == ini_f[G] and row["n_reference"] == ref_f[G], (G, row)
    assert row["n_calibration"] == cal_q[G] and row["n_reference"] == ref_q[G], (G, row)
    n_ini_all = int((part == "initial").sum())
    assert n_ini_all == 318526, n_ini_all
    pi_ini = row["n_initial"] / n_ini_all
    assert abs(pi_ini - L6[(L6.dataset == pid) & (L6.parent == G)].pi.iloc[0]) < 5e-5
    row["pi_initial"] = pi_ini
    row["n_initial_all_cells"] = n_ini_all
    parent_rows.append(row)
# per-donor parent counts and the e5b inclusion rule (>= 500 in follow-up AND reference)
fol, ref = part == "followup", part == "reference"
for G in ("CD4T", "CD8T"):
    frozen_donors = set(E5[E5.parent == G].donor.unique())
    rule_donors = set()
    for s in np.unique(sample):
        n_f = int((pm[G] & fol & (sample == s)).sum())
        n_r = int((pm[G] & ref & (sample == s)).sum())
        inc = n_f >= 500 and n_r >= 500
        if inc:
            rule_donors.add(int(s))
        donor_rows.append(dict(dataset=pid, parent=G, donor=int(s), file=names[int(s)] + ".fcs",
                               n_parent_followup=n_f, n_parent_reference=n_r, included=inc,
                               rule="e5b: >= 500 parent cells in follow-up and in reference split"))
    assert rule_donors == frozen_donors, (G, rule_donors, frozen_donors)
DP = pd.DataFrame(donor_rows)
excl = DP[~DP.included]
assert len(excl) == 1 and excl.parent.iloc[0] == "CD8T" and excl.donor.iloc[0] == 6
c4 = DP[DP.parent == "CD4T"]
assert c4.included.all() and len(c4) == 13
assert (int(c4.n_parent_followup.min()), int(c4.n_parent_reference.min())) == (2064, 1978)
assert max(int(c4.n_parent_followup.max()), int(c4.n_parent_reference.max())) == 13309
check[pid] = dict(pre_files=27, pre_used=13, PreWN08=28060, post_files=27,
                  events_raw_used=int(pre[pre.used].tot.sum()), cells_after_dead_filter=int(len(Xt)),
                  initial_all_cells=318526, CD4T_per_donor=dict(min_followup=2064, min_reference=1978, max=13309,
                                                               included="13/13"),
                  followup=nfol, CD8T_donor6=dict(followup=int(excl.n_parent_followup.iloc[0]),
                                                  reference=int(excl.n_parent_reference.iloc[0])),
                  asserted=["27 Pre / 13 used / PreWN08=28060", "followup=159264", "N_d per donor (e5b_per_donor)",
                            "n_ref", "n_parent", "calibration_N/reference_N", "pi (lat06)",
                            "e5b per-donor inclusion rule reproduces donor sets; only CD8T donor 6 excluded"])

# CytoNorm files (S1E caption context: cells per file)
cn = []
for f in sorted(glob.glob(os.path.join(WF, "data", "raw", "CytoNorm", "*.fcs"))):
    cn.append(dict(dataset="cytonorm", file=os.path.basename(f), group="batch file", tot=tot_header(f), used=True,
                   reason="real_batch.py: all events"))
FD = pd.concat([FD, pd.DataFrame(cn)], ignore_index=True)
check["cytonorm_tot"] = {r["file"]: r["tot"] for r in cn}

FD.to_csv(os.path.join(OUT, "figS1_files.tsv"), sep="\t", index=False)
pd.DataFrame(split_rows).to_csv(os.path.join(OUT, "figS1_split.tsv"), sep="\t", index=False)
pd.DataFrame(parent_rows).to_csv(os.path.join(OUT, "figS1_parents.tsv"), sep="\t", index=False)
DP.to_csv(os.path.join(OUT, "figS1_donor_parent.tsv"), sep="\t", index=False)
pd.DataFrame(rare_rows).to_csv(os.path.join(OUT, "figS1_rareB.tsv"), sep="\t", index=False)
json.dump(check, open(os.path.join(OUT, "figS1_ledger_check.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(check, indent=1))
print(pd.DataFrame(split_rows).to_string(index=False))
print(pd.DataFrame(parent_rows).to_string(index=False))
print(pd.DataFrame(rare_rows).to_string(index=False))
