"""SF2b derive: per-donor e5b replay, all nodes (exploratory), deterministic summaries only.

Input  (frozen): results/wingmen_v1/e5b_per_donor.tsv (900 rows = 25 parent-donor combinations x 6 nodes x 2 methods
                 x 3 allocations); figures/data/fig3C_per_donor_shape.tsv (frozen summary, assertions only).
Output: figures/derived/figS2_B_cells.tsv - one row per parent x donor x node x method: mean / min / max over the
        3 allocations of half_ok and err_ok, the three allocation values, N_d and eps_d; the excluded combination
        (CD8T donor 6, e5b rule: < 500 parent cells in the follow-up or reference split) is listed explicitly with
        empty values (not imputed).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(WF, "results")
DATA = os.path.join(WF, "figures", "data")
DER = os.path.join(WF, "figures", "derived")
NODES = [130, 142, 156, 180, 221, 260]


def main():
    d = pd.read_csv(os.path.join(RES, "wingmen_v1", "e5b_per_donor.tsv"), sep="\t")
    assert len(d) == 900
    combos = d[["parent", "donor"]].drop_duplicates()
    assert len(combos) == 25
    assert sorted(d[d.parent == "CD4T"].donor.unique()) == list(range(13))
    assert sorted(d[d.parent == "CD8T"].donor.unique()) == [i for i in range(13) if i != 6]
    assert sorted(d.r.unique()) == NODES and set(d.method) == {"classical_stack", "random"}
    assert set(d.alloc) == {0, 1, 2}
    assert d.groupby(["parent", "donor"])[["N_d", "eps_d"]].nunique().max().max() == 1
    assert not d.duplicated(["parent", "donor", "r", "method", "alloc"]).any()
    g = d.sort_values("alloc").groupby(["parent", "donor", "r", "method"])
    cells = g.agg(N_d=("N_d", "first"), eps_d=("eps_d", "first"), n_alloc=("alloc", "size"),
                  half_mean=("half_ok", "mean"), half_min=("half_ok", "min"), half_max=("half_ok", "max"),
                  err_mean=("err_ok", "mean"), err_min=("err_ok", "min"), err_max=("err_ok", "max")).reset_index()
    for k in range(3):
        a = d[d.alloc == k].set_index(["parent", "donor", "r", "method"])
        cells[f"half_a{k}"] = [a.loc[tuple(x), "half_ok"] for x in cells[["parent", "donor", "r", "method"]].values]
        cells[f"err_a{k}"] = [a.loc[tuple(x), "err_ok"] for x in cells[["parent", "donor", "r", "method"]].values]
    assert (cells.n_alloc == 3).all() and len(cells) == 300
    cells["half_range"] = cells.half_max - cells.half_min
    cells["err_range"] = cells.err_max - cells.err_min
    cells["status"] = "analysed"
    # assertions against the frozen per-donor summary of Fig. 3d
    f = pd.read_csv(os.path.join(DATA, "fig3C_per_donor_shape.tsv"), sep="\t")
    c = cells[cells.method == "classical_stack"].set_index(["parent", "donor", "r"]).half_mean
    for _, row in f.iterrows():
        k = (row.parent, int(row.donor))
        assert abs(min(c[k + (130,)], c[k + (260,)]) - row.endpoint_min) < 1e-12
        assert abs(c[k + (180,)] - row.node180) < 1e-12
    assert len(f) == 25
    # explicit excluded combination
    miss = pd.DataFrame([dict(parent="CD8T", donor=6, r=r, method=m, n_alloc=0,
                              status="excluded by e5b rule (< 500 CD8T cells in follow-up or reference split)")
                         for r in NODES for m in ("classical_stack", "random")])
    out = pd.concat([cells, miss], ignore_index=True).sort_values(["parent", "donor", "method", "r"])
    out.to_csv(os.path.join(DER, "figS2_B_cells.tsv"), sep="\t", index=False)
    print("cells", len(cells), "+ excluded", len(miss))
    print("range pp half: median %.2f max %.2f | err: median %.2f max %.2f" % (
        100 * cells.half_range.median(), 100 * cells.half_range.max(), 100 * cells.err_range.median(),
        100 * cells.err_range.max()))
    print("mean ranges", cells[["half_mean", "err_mean"]].min().round(4).tolist(), cells[["half_mean", "err_mean"]].max().round(4).tolist())
    print(cells.groupby("parent").eps_d.agg(["min", "max"]))


if __name__ == "__main__":
    main()
