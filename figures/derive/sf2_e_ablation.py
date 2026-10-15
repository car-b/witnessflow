"""SF2e derive: requirement ablation (e5f, all 8 settings) and the allocation ledgers (e5j), deterministic only.

Inputs (frozen): results/{levine32_v2, wingmen_v1}/e5f_attribution.tsv (144 + 144 rows),
                 results/{levine32_v2, wingmen_v1}/ledger/ledger_summary.tsv (lp_true rows: 18 per parent = 54 runs),
                 results/*/ledger/*_targets.tsv (18 per-target ledgers = 3 parents x 2 initial designs x 3 optimizer
                 seeds; saved for allocation 0 only, src/e5j_ledger.py line 101);
                 figures/data/fig4C_attribution_shuffle.tsv, fig4D_ledger.tsv, figures/derived/fig5_C_right_dlam_status.tsv
                 (assertions only).
Top (e5f): baseline = classical stack (LJCR_stack); it does not depend on the initial design, so its 9 stored rows per
setting are 3 allocations x 3 duplicates and are de-duplicated to 3; the uniform ('worst') requirement does not use
the initial data either and is de-duplicated the same way (asserted identical).  Each record is paired with the
baseline at the same allocation; per cell and metric we count records higher / identical / lower than their baseline.
State: 'higher in all', 'lower in all', 'identical in all', 'mixed' (no test, no significance statement).
Bottom: (i) gained / lost counts of the 54 lp_true runs; the classical baseline of a run depends only on its
allocation (3 baselines per parent, each shared by 6 runs; flagged, not counted as 18).  (ii) integer
dlam = lam_opt - lam_base per target in the 18 allocation-0 ledgers; these 18 ledgers are the allocation-0 subset
of the 54 runs (gained / lost recomputed from the targets and asserted equal to the summary rows).
Outputs: figures/derived/figS2_E_cells.tsv, figS2_E_e5f_records.tsv, figS2_E_runs.tsv, figS2_E_ledgers.tsv,
         figS2_E_dlam.tsv
"""
import glob
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
METH = ["LJCR_stack", "hybrid_worst", "hybrid_knn", "hybrid_lp"]
EQ = 1e-12


def rd(*p):
    return pd.read_csv(os.path.join(*p), sep="\t", keep_default_na=False, na_values=[""])


def e5f():
    rec = pd.concat([rd(RES, d, "e5f_attribution.tsv") for d in DS], ignore_index=True)
    assert len(rec) == 288 and set(rec.method) == set(METH)
    key = ["dataset", "parent", "eps", "r"]
    st = rec[key].drop_duplicates()
    assert len(st) == 8
    assert (rec.groupby(key + ["method"]).size() == 9).all()
    # duplicates across initial designs (design does not use the initial data)
    for m in ("LJCR_stack", "hybrid_worst"):
        g = rec[rec.method == m].groupby(key + ["alloc"])
        assert g.half_ok.nunique().max() == 1 and g.err_ok.nunique().max() == 1, m
    rec["unique_record"] = ~(rec.method.isin(["LJCR_stack", "hybrid_worst"]) & (rec.init_rep > 0))
    assert not rec[rec.method.isin(["hybrid_knn", "hybrid_lp"])].duplicated(key + ["method", "init_rep", "alloc"]).any()
    base = rec[(rec.method == "LJCR_stack") & rec.unique_record].set_index(key + ["alloc"])
    for met in ("half_ok", "err_ok"):
        rec[f"{met}_base"] = [base.loc[tuple(x), met] for x in rec[key + ["alloc"]].values]
        rec[f"{met}_diff"] = rec[met] - rec[f"{met}_base"]
    rows = []
    for (d, p, e, r), g in rec.groupby(key, sort=False):
        for m in METH:
            u = g[(g.method == m) & g.unique_record]
            row = dict(dataset=d, parent=p, eps=e, r=r, method=m, n_rows=int((g.method == m).sum()), n_unique=len(u))
            for met in ("half_ok", "err_ok"):
                row[f"{met}_mean"] = u[met].mean()
                row[f"{met}_min"] = u[met].min()
                row[f"{met}_max"] = u[met].max()
                if m == "LJCR_stack":
                    row[f"{met}_n_higher"] = row[f"{met}_n_equal"] = row[f"{met}_n_lower"] = np.nan
                    row[f"{met}_state"] = "baseline"
                    row[f"{met}_mean_diff"] = np.nan
                else:
                    df = u[f"{met}_diff"]
                    hi, eq, lo = int((df > EQ).sum()), int((df.abs() <= EQ).sum()), int((df < -EQ).sum())
                    row[f"{met}_n_higher"], row[f"{met}_n_equal"], row[f"{met}_n_lower"] = hi, eq, lo
                    row[f"{met}_state"] = ("identical in all" if eq == len(df) else "higher in all" if hi == len(df)
                                           else "lower in all" if lo == len(df) else "mixed")
                    row[f"{met}_mean_diff"] = df.mean()
            rows.append(row)
    cells = pd.DataFrame(rows)
    assert len(cells) == 32
    assert set(cells[cells.method == "LJCR_stack"].n_unique) == {3}
    assert set(cells[cells.method == "hybrid_worst"].n_unique) == {3}
    assert set(cells[cells.method.isin(["hybrid_knn", "hybrid_lp"])].n_unique) == {9}
    # the e5f classical stack equals the classical stack of the e5i default settings (same stack, same allocations)
    f4 = rd(DATA, "fig4C_attribution_shuffle.tsv")
    for d, p, r, e in (("levine32_v2", "T", 142, 0.05), ("wingmen_v1", "CD4T", 180, 0.03), ("wingmen_v1", "CD8T", 180, 0.05)):
        a = cells[(cells.dataset == d) & (cells.parent == p) & (cells.r == r) & (cells.eps == e) & (cells.method == "LJCR_stack")]
        b = f4[(f4.dataset == d) & (f4.parent == p) & (f4.req_kind == "LJCR_stack")]
        assert len(a) == len(b) == 1
        assert abs(a.half_ok_mean.iloc[0] - b.half_ok.iloc[0]) < 1e-12 and abs(a.err_ok_mean.iloc[0] - b.err_ok.iloc[0]) < 1e-12
    # WINGMEN CD4T: uniform requirement reproduces the classical stack exactly (as in Fig. 5b)
    w = cells[(cells.parent == "CD4T") & (cells.method == "hybrid_worst")]
    assert (w.half_ok_state == "identical in all").all() and (w.err_ok_state == "identical in all").all()
    return rec, cells


def ledger():
    runs = []
    for d in DS:
        s = rd(RES, d, "ledger", "ledger_summary.tsv")
        runs.append(s[s.req == "lp_true"])
    runs = pd.concat(runs, ignore_index=True)
    assert len(runs) == 54 and (runs.groupby("parent").size() == 18).all()
    assert set(runs.init_rep) == {0, 1} and set(runs.opt_seed) == {4, 5, 6} and set(runs.alloc) == {0, 1, 2}
    # the classical baseline of a run depends only on its allocation: 3 per parent, each shared by 6 runs
    assert runs.groupby(["parent", "alloc"]).half_ok_base.nunique().max() == 1
    runs["baseline_id"] = runs.parent + "_alloc" + runs.alloc.astype(str)
    runs["runs_sharing_baseline"] = runs.groupby("baseline_id").alloc.transform("size")
    assert set(runs.runs_sharing_baseline) == {6}
    runs["has_target_ledger"] = runs.alloc == 0
    f = rd(DATA, "fig4D_ledger.tsv")
    f = f[f.req == "lp_true"].set_index("parent")
    for p, g in runs.groupby("parent"):
        for c in ("half_ok", "half_ok_base", "gained_half", "lost_half", "gained_err", "lost_err"):
            assert abs(g[c].mean() - f.loc[p, c]) < 1e-9, (p, c)
    # per-target ledgers (allocation 0)
    files = sorted(glob.glob(os.path.join(RES, "*", "ledger", "*_targets.tsv")))
    assert len(files) == 18
    led, dl = [], []
    for fp in files:
        d = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        p, rep, opt = re.match(r"(\w+?)_rep(\d)_opt(\d)_targets\.tsv", os.path.basename(fp)).groups()
        rep, opt = int(rep), int(opt)
        t = pd.read_csv(fp, sep="\t")
        sr = runs[(runs.parent == p) & (runs.init_rep == rep) & (runs.opt_seed == opt) & (runs.alloc == 0)]
        assert len(sr) == 1
        sr = sr.iloc[0]
        eps = float(sr.eps)
        ph, bph = t.half_opt <= eps, t.half_base <= eps
        pe, bpe = t.err_opt <= eps, t.err_base <= eps
        gh, lh, ge, le = int((ph & ~bph).sum()), int((~ph & bph).sum()), int((pe & ~bpe).sum()), int((~pe & bpe).sum())
        assert (gh, lh, ge, le) == (sr.gained_half, sr.lost_half, sr.gained_err, sr.lost_err), fp
        assert abs(ph.mean() - sr.half_ok) < 1e-12 and abs(bph.mean() - sr.half_ok_base) < 1e-12
        t["dlam"] = t.lam_opt - t.lam_base
        vc = t.dlam.value_counts().sort_index()
        led.append(dict(dataset=d, parent=p, eps=eps, r=int(sr.r), init_rep=rep, opt_seed=opt, alloc=0,
                        n_targets=len(t), gained_half=gh, lost_half=lh, gained_err=ge, lost_err=le,
                        dlam_min=int(t.dlam.min()), dlam_max=int(t.dlam.max()), dlam_mean=float(t.dlam.mean()),
                        n_dlam_neg=int((t.dlam < 0).sum()), n_dlam_zero=int((t.dlam == 0).sum()),
                        n_dlam_pos=int((t.dlam > 0).sum()), lam_total_base=int(t.lam_base.sum()),
                        lam_total_opt=int(t.lam_opt.sum()), source=os.path.relpath(fp, WF).replace("\\", "/")))
        for k, c in vc.items():
            g = t[t.dlam == k]
            dl.append(dict(dataset=d, parent=p, init_rep=rep, opt_seed=opt, dlam=int(k), n_targets=int(c),
                           share=c / len(t), gained_half=int((ph & ~bph)[g.index].sum()),
                           lost_half=int((~ph & bph)[g.index].sum())))
    led = pd.DataFrame(led)
    dl = pd.DataFrame(dl)
    # Fig. 5c right: CD4T rep 0 opt 4
    f5 = rd(DER, "fig5_C_right_dlam_status.tsv")
    x = dl[(dl.parent == "CD4T") & (dl.init_rep == 0) & (dl.opt_seed == 4)].set_index("dlam")
    for _, q in f5.iterrows():
        assert x.loc[q.dlam, "n_targets"] == q.n and x.loc[q.dlam, "gained_half"] == q.gained and x.loc[q.dlam, "lost_half"] == q.lost
    assert (led.groupby("parent").size() == 6).all()
    # tube budget fixed: every tube holds C(6,3) = 20 candidate triples, so sum(lam) = 20 r in both designs
    assert (led.lam_total_base == 20 * led.r).all() and (led.lam_total_opt == 20 * led.r).all()
    return runs, led, dl


def main():
    rec, cells = e5f()
    runs, led, dl = ledger()
    # link: the e5f LP-requirement records (optimizer seed 4) of the three ledger settings equal the ledger's
    # lp_true runs with optimizer seed 4 (same stack, seeds, allocations)
    a = rec[(rec.method == "hybrid_lp") & (rec.init_rep < 2)]
    b = runs[runs.opt_seed == 4]
    m = a.merge(b, on=["parent", "r", "eps", "init_rep", "alloc"], suffixes=("_e5f", "_led"))
    assert len(m) == 18 and (m.half_ok_e5f - m.half_ok_led).abs().max() < 1e-12
    rec.to_csv(os.path.join(DER, "figS2_E_e5f_records.tsv"), sep="\t", index=False)
    cells.to_csv(os.path.join(DER, "figS2_E_cells.tsv"), sep="\t", index=False)
    runs.to_csv(os.path.join(DER, "figS2_E_runs.tsv"), sep="\t", index=False)
    led.to_csv(os.path.join(DER, "figS2_E_ledgers.tsv"), sep="\t", index=False)
    dl.to_csv(os.path.join(DER, "figS2_E_dlam.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(cells[["dataset", "parent", "eps", "r", "method", "n_unique", "half_ok_mean", "half_ok_state", "err_ok_mean",
                 "err_ok_state", "err_ok_n_higher", "err_ok_n_equal", "err_ok_n_lower"]].to_string(index=False))
    print(runs.groupby("parent")[["gained_half", "lost_half", "gained_err", "lost_err"]].agg(["min", "mean", "max"]).round(1))
    print(led[["parent", "init_rep", "opt_seed", "gained_half", "lost_half", "dlam_min", "dlam_max", "n_dlam_neg", "n_dlam_zero", "n_dlam_pos"]].to_string(index=False))


if __name__ == "__main__":
    main()
