"""SF2a derive: all e5b replay runs (every node x three metrics x min_cells), deterministic reorganisation only.

Inputs (frozen, read-only):
    results/levine32_v2/e5b_steiner_nodes.tsv   (1560 rows)
    results/wingmen_v1/e5b_steiner_nodes.tsv    (1540 rows)
    results/wingmen_v1/e5b_r180.tsv             (140 rows; no dataset / eps columns)
    figures/data/fig3A_sawtooth.tsv             (frozen summary, used only for assertions)
The r = 180 supplement lacks `dataset` and `eps`; they are filled from the frozen e5b settings of the same parents
(CD4T 0.05, CD8T 0.08; asserted equal to the eps column of results/wingmen_v1/e5b_steiner_nodes.tsv), not borrowed
from another experiment.  No run is re-computed, interpolated or re-randomised.
Outputs: figures/derived/figS2_A_runs.tsv (3240 runs), figS2_A_seed.tsv (design-seed level),
         figS2_A_summary.tsv (method x node level), figS2_A_grid.tsv (full grid incl. non-existent combinations).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(WF, "results")
DATA = os.path.join(WF, "figures", "data")
DER = os.path.join(WF, "figures", "derived")

R180_EPS = {"CD4T": 0.05, "CD8T": 0.08}      # frozen e5b setting of these parents (asserted below)
FULL = {"wingmen_v1": [130, 260], "levine32_v2": [71, 142, 213, 284]}
BASE = {"wingmen_v1": 130, "levine32_v2": 71}
METHODS = ["classical_stack", "random", "balanced_random", "PSL_3design"]
COLS = ["dataset", "parent", "eps", "r", "method", "seed", "alloc", "covered", "half_ok", "err_ok", "min_cells",
        "plan_eps", "lam_min", "lam_cv"]


def rd(*p):
    return pd.read_csv(os.path.join(*p), sep="\t")


def main():
    lev = rd(RES, "levine32_v2", "e5b_steiner_nodes.tsv")
    win = rd(RES, "wingmen_v1", "e5b_steiner_nodes.tsv")
    r18 = rd(RES, "wingmen_v1", "e5b_r180.tsv")
    assert (len(lev), len(win), len(r18)) == (1560, 1540, 140)
    assert "dataset" not in r18.columns and "eps" not in r18.columns
    for p, e in R180_EPS.items():
        assert set(win[win.parent == p].eps) == {e}, p
    r18 = r18.assign(dataset="wingmen_v1", eps=r18.parent.map(R180_EPS))
    lev["source_table"] = "results/levine32_v2/e5b_steiner_nodes.tsv"
    win["source_table"] = "results/wingmen_v1/e5b_steiner_nodes.tsv"
    r18["source_table"] = "results/wingmen_v1/e5b_r180.tsv (dataset, eps filled from frozen e5b settings)"
    runs = pd.concat([lev[COLS + ["source_table"]], win[COLS + ["source_table"]], r18[COLS + ["source_table"]]],
                     ignore_index=True)
    assert len(runs) == 3240
    assert not runs.duplicated(["dataset", "parent", "r", "method", "seed", "alloc"]).any()
    runs["rounds"] = runs.r / runs.dataset.map(BASE)
    runs["complete_round"] = [r in FULL[d] for d, r in zip(runs.dataset, runs.r)]
    # structure checks: covered is a design property (constant across allocations of one design)
    assert runs.groupby(["dataset", "parent", "r", "method", "seed"]).covered.nunique().max() == 1
    assert set(runs[runs.method == "classical_stack"].seed) == {0}
    assert set(runs[runs.method == "PSL_3design"].r) == {570}
    assert set(runs[runs.method == "PSL_3design"].dataset) == {"levine32_v2"}
    n = runs.groupby(["dataset", "parent", "r", "method"]).size()
    assert set(n[n.index.get_level_values(3) == "classical_stack"]) == {10}
    assert set(n[n.index.get_level_values(3).isin(["random", "balanced_random"])]) == {30}
    assert set(n[n.index.get_level_values(3) == "PSL_3design"]) == {10}

    # design-seed level
    seed = runs.groupby(["dataset", "parent", "eps", "r", "method", "seed"]).agg(
        n_alloc=("alloc", "size"), covered=("covered", "first"), half_ok=("half_ok", "mean"),
        half_min=("half_ok", "min"), half_max=("half_ok", "max"), err_ok=("err_ok", "mean"),
        err_min=("err_ok", "min"), err_max=("err_ok", "max"), min_cells=("min_cells", "mean"),
        min_cells_min=("min_cells", "min"), min_cells_max=("min_cells", "max")).reset_index()
    # method x node level
    summ = runs.groupby(["dataset", "parent", "eps", "r", "method"]).agg(
        n_runs=("alloc", "size"), n_seeds=("seed", "nunique"), covered=("covered", "mean"),
        covered_min=("covered", "min"), covered_max=("covered", "max"), half_ok=("half_ok", "mean"),
        half_sd=("half_ok", lambda x: x.std(ddof=0)), half_min=("half_ok", "min"), half_max=("half_ok", "max"),
        err_ok=("err_ok", "mean"), err_min=("err_ok", "min"), err_max=("err_ok", "max"),
        min_cells=("min_cells", "mean"), min_cells_min=("min_cells", "min"),
        min_cells_max=("min_cells", "max")).reset_index()
    summ["complete_round"] = [r in FULL[d] for d, r in zip(summ.dataset, summ.r)]

    # assertions against the frozen summary behind Fig. 3b
    saw = rd(DATA, "fig3A_sawtooth.tsv")
    assert len(saw) == 140
    m = saw.merge(summ, on=["dataset", "parent", "r", "method"], suffixes=("_frozen", ""))
    assert len(m) == 140 == len(summ)
    assert np.allclose(m.half_ok_frozen, m.half_ok, atol=1e-12)
    assert np.allclose(m.err_ok_frozen, m.err_ok, atol=1e-12)
    assert (m.n == m.n_runs).all()
    sd_ok0 = np.allclose(m.half_sd_frozen, m.half_sd, atol=1e-12)
    if not sd_ok0:   # frozen half_sd may be the sample SD; report which one matches
        sd1 = runs.groupby(["dataset", "parent", "r", "method"]).half_ok.std(ddof=1).reset_index(name="sd1")
        m2 = m.merge(sd1, on=["dataset", "parent", "r", "method"])
        assert np.allclose(m2.half_sd_frozen, m2.sd1.fillna(0), atol=1e-12)
        summ = summ.drop(columns="half_sd").merge(sd1.rename(columns={"sd1": "half_sd"}),
                                                  on=["dataset", "parent", "r", "method"])
        summ["half_sd_ddof"] = 1
    else:
        summ["half_sd_ddof"] = 0
    # caption numbers of Fig. 3b / claims table C3
    def val(d, p, r, meth, col):
        s = summ[(summ.dataset == d) & (summ.parent == p) & (summ.r == r) & (summ.method == meth)]
        assert len(s) == 1
        return float(s[col].iloc[0])
    assert round(100 * val("wingmen_v1", "CD4T", 180, "classical_stack", "half_ok"), 2) == 63.96
    assert round(100 * val("levine32_v2", "B", 71, "classical_stack", "half_ok"), 1) == 79.3
    assert round(100 * val("levine32_v2", "B", 142, "classical_stack", "half_ok"), 1) == 83.7

    # full grid incl. combinations that were never run (explicit, not interpolated)
    nodes = {d: sorted(runs[runs.dataset == d].r.unique()) for d in BASE}
    parents = {d: sorted(runs[runs.dataset == d].parent.unique()) for d in BASE}
    grid = pd.DataFrame([dict(dataset=d, parent=p, r=r, method=mm) for d in BASE for p in parents[d]
                         for r in nodes[d] for mm in METHODS])
    grid = grid.merge(summ[["dataset", "parent", "r", "method", "n_runs"]], how="left",
                      on=["dataset", "parent", "r", "method"])
    grid["n_runs"] = grid.n_runs.fillna(0).astype(int)
    grid["status"] = np.where(grid.n_runs > 0, "run", "not run (no record; not interpolated)")
    assert (grid.n_runs > 0).sum() == 140 and len(grid) == 4 * (12 + 12 + 11 + 11)

    runs.to_csv(os.path.join(DER, "figS2_A_runs.tsv"), sep="\t", index=False)
    seed.to_csv(os.path.join(DER, "figS2_A_seed.tsv"), sep="\t", index=False)
    summ.to_csv(os.path.join(DER, "figS2_A_summary.tsv"), sep="\t", index=False)
    grid.to_csv(os.path.join(DER, "figS2_A_grid.tsv"), sep="\t", index=False)
    print("runs", len(runs), "seed rows", len(seed), "summary rows", len(summ), "grid", len(grid),
          "half_sd ddof", int(summ.half_sd_ddof.iloc[0]))
    for col in ("covered", "half_ok", "err_ok"):
        print(col, "range of runs", runs[col].min().round(4), runs[col].max().round(4))
    print(runs.groupby(["dataset", "parent"]).min_cells.max())


if __name__ == "__main__":
    main()
