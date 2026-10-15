"""Figure source data + draft panels for the frozen computational version (WitnessFlow v1).
Writes witnessflow/figures/data/*.tsv (exact numbers behind every data panel) and figures/png/*.png (drafts).
Schematic panels (Fig 1A/1B, 2A, 4A) are not generated here.  usage: python make_figures.py"""
import itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import budget, design  # noqa: E402

R = os.path.join(ROOT, "results")
DD = os.path.join(ROOT, "figures", "data")
PD = os.path.join(ROOT, "figures", "png")
PLOT = "--png" in sys.argv  # 2026-10-09: figure drafts postponed; PNGs only with --png
os.makedirs(DD, exist_ok=True)
if PLOT:
    os.makedirs(PD, exist_ok=True)


def rd(p, **kw):
    return pd.read_csv(os.path.join(R, p), sep="\t", keep_default_na=False, na_values=[""], **kw)


def save(df, name):
    df.to_csv(os.path.join(DD, name + ".tsv"), sep="\t", index=False)


# ---------------- Fig 3A: sawtooth (e5b + r180 supplement)
rows = []
for ds, G in (("levine32_v2", "T"), ("levine32_v2", "B"), ("wingmen_v1", "CD4T"), ("wingmen_v1", "CD8T")):
    d = rd(f"{ds}/e5b_steiner_nodes.tsv")
    if ds == "wingmen_v1":
        d = pd.concat([d, rd(f"{ds}/e5b_r180.tsv").assign(dataset=ds)], ignore_index=True)
    d = d[d.parent == G]
    g = d.groupby(["r", "method"]).agg(half_ok=("half_ok", "mean"), half_sd=("half_ok", "std"),
                                       err_ok=("err_ok", "mean"), n=("half_ok", "size")).reset_index()
    g.insert(0, "parent", G)
    g.insert(0, "dataset", ds)
    rows.append(g)
f3a = pd.concat(rows, ignore_index=True)
save(f3a, "fig3A_sawtooth")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (ds, G, unit) in zip(axes, (("levine32_v2", "T", 71), ("wingmen_v1", "CD4T", 130))):
    for meth, sty in (("classical_stack", "-o"), ("random", "--s"), ("balanced_random", ":^"), ("PSL_3design", "*")):
        s = f3a[(f3a.dataset == ds) & (f3a.parent == G) & (f3a.method == meth)].sort_values("r")
        if len(s):
            ax.plot(s.r, s.half_ok, sty, label=meth, ms=8 if meth == "PSL_3design" else 4)
    rmax = f3a[(f3a.dataset == ds) & (f3a.parent == G)].r.max()
    for m in range(1, 9):
        if unit * m <= rmax:
            ax.axvline(unit * m, color="grey", lw=0.5, alpha=0.4)
    ax.set_xlim(0, rmax * 1.05)
    ax.set_title(f"{ds} {G}: half-width attainment vs tubes")
    ax.set_xlabel("tubes r (fixed total cells)")
    ax.set_ylabel("fraction of triples")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7)
fig.tight_layout()
PLOT and fig.savefig(os.path.join(PD, "fig3A_sawtooth.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 3B: multiplicity distribution of the WINGMEN classical stack (exact, combinatorial)
blocks = design.ljcr_blocks(26, 6, 3)
cand = list(range(26))
tarr = np.array(list(itertools.combinations(cand, 3)))
rows = []
for r in (104, 130, 142, 180, 221, 260):
    st = budget.ljcr_stack(cand, 6, r, seed=0)
    W = np.zeros((len(st), 26), bool)
    for i, X in enumerate(st):
        W[i, list(X)] = True
    lam = W[:, tarr].all(2).sum(0)
    for k, c in zip(*np.unique(lam, return_counts=True)):
        rows.append(dict(r=r, lam=int(k), n_triples=int(c)))
f3b = pd.DataFrame(rows)
save(f3b, "fig3B_multiplicity_wingmen")

# ---------------- Fig 3C: per-donor shape (min of endpoints 130/260 minus the 180 node), classical stack
d = rd("wingmen_v1/e5b_per_donor.tsv")
d = d[d.method == "classical_stack"].groupby(["parent", "donor", "r"]).half_ok.mean().unstack("r")
f3c = pd.DataFrame({"parent": [i[0] for i in d.index], "donor": [i[1] for i in d.index],
                    "endpoint_min": np.minimum(d[130], d[260]).values, "node180": d[180].values})
f3c["difference"] = f3c.endpoint_min - f3c.node180
save(f3c, "fig3C_per_donor_shape")

# ---------------- Fig 3D: tube split
save(rd("wingmen_v1/e5b_tube_split.tsv"), "fig3D_tube_split")

# ---------------- Fig 2C: orthogonal array vs search vs random (WINGMEN)
d = rd("wingmen_v1/e5c_oa_replay.tsv")
f2c = d.groupby(["parent", "method"]).agg(covered=("covered", "mean"), half_ok=("half_ok", "mean"),
                                          err_ok=("err_ok", "mean"), half_min=("half_ok", "min"),
                                          half_max=("half_ok", "max")).reset_index()
save(f2c, "fig2C_orthogonal_array")
fig, ax = plt.subplots(figsize=(6, 3.5))
order = ["OA_quadratic", "anneal_search", "balanced_random", "legal_random"]
x = np.arange(len(order))
for j, G in enumerate(("CD4T", "CD8T")):
    s = f2c[f2c.parent == G].set_index("method").loc[order]
    ax.bar(x + 0.38 * j, s.half_ok, 0.38, label=f"{G} half-width")
ax.set_xticks(x + 0.19, order, rotation=15, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("125 tubes, 1250 cross-family triples (artificial 5-slot constraint)")
ax.legend(fontsize=8)
fig.tight_layout()
PLOT and fig.savefig(os.path.join(PD, "fig2C_orthogonal_array.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 4B: QC (fixed family)
rows = []
for ds in ("levine32_v2", "wingmen_v1"):
    s = rd(f"{ds}/e5d_fixed_family_summary.tsv")
    s.insert(0, "dataset", ds)
    rows.append(s)
f4b = pd.concat(rows, ignore_index=True)
save(f4b, "fig4B_qc_fixed_family")
fig, ax = plt.subplots(figsize=(9, 3.5))
sc = ["null", "single+0.5", "single-0.5", "single+0.25", "single-0.25", "wholetube+0.5", "joint_only", "composition", "global_drift"]
x = np.arange(len(sc))
for j, ds in enumerate(("levine32_v2", "wingmen_v1")):
    s = f4b[f4b.dataset == ds].set_index("scenario").loc[sc]
    ax.bar(x + 0.2 * (2 * j), s.single_only_alarm.astype(float), 0.2, label=f"{ds} single only")
    ax.bar(x + 0.2 * (2 * j + 1), s.alarm.astype(float), 0.2, label=f"{ds} single+pair")
ax.set_xticks(x + 0.3, sc, rotation=25, fontsize=8)
ax.set_ylabel("alarm rate")
ax.legend(fontsize=7)
ax.set_title("Built-in consistency checks (one Bonferroni family, alpha 0.001)")
fig.tight_layout()
PLOT and fig.savefig(os.path.join(PD, "fig4B_qc.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 4C: attribution with label shuffle (e5i)
rows = []
for ds in ("levine32_v2", "wingmen_v1"):
    d = rd(f"{ds}/e5i_shuffle.tsv")
    # the classical baseline does not depend on the initial design: keep its 3 unique allocations only
    d = d[~((d.req_kind == "LJCR_stack") & d.duplicated(["parent", "alloc", "req_kind"]))]
    rows.append(d.groupby(["dataset", "parent", "req_kind"]).agg(half_ok=("half_ok", "mean"), half_sd=("half_ok", "std"),
                                                                 err_ok=("err_ok", "mean"), n_records=("half_ok", "size"))
                .reset_index())
f4c = pd.concat(rows, ignore_index=True)
save(f4c, "fig4C_attribution_shuffle")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, (ds, G) in zip(axes, (("levine32_v2", "T"), ("wingmen_v1", "CD4T"), ("wingmen_v1", "CD8T"))):
    s = f4c[(f4c.dataset == ds) & (f4c.parent == G)].set_index("req_kind").loc[["LJCR_stack", "worst", "lp_shuffled", "lp_true"]]
    ax.bar(range(4), s.half_ok, yerr=s.half_sd.fillna(0), color=["grey", "tab:blue", "tab:orange", "tab:green"])
    ax.set_xticks(range(4), ["classical", "uniform", "shuffled", "true"], fontsize=8)
    ax.set_title(f"{ds} {G}", fontsize=9)
axes[0].set_ylabel("half-width attainment")
fig.tight_layout()
PLOT and fig.savefig(os.path.join(PD, "fig4C_attribution.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 4D: ledger (gained / lost) and reference-error trade-off
rows = []
for ds in ("levine32_v2", "wingmen_v1"):
    d = pd.read_csv(os.path.join(R, ds, "ledger", "ledger_summary.tsv"), sep="\t")
    rows.append(d.groupby(["dataset", "parent", "req"])[["half_ok", "half_ok_base", "gained_half", "lost_half", "gained_err",
                                                         "lost_err", "gained_half_mean_refvar", "lost_err_mean_refvar",
                                                         "all_mean_refvar"]].mean().reset_index())
save(pd.concat(rows, ignore_index=True), "fig4D_ledger")

# ---------------- Supplementary data tables
save(pd.concat([rd(f"{ds}/e5g_threshold_design.tsv") for ds in ("levine32_v2", "wingmen_v1")])
     .groupby(["dataset", "offset", "r", "method"])[["half_ok", "err_ok"]].mean().reset_index(), "supp_threshold_design")
save(pd.concat([rd(f"{ds}/e5l_act5_threshold.tsv") for ds in ("levine32_v2", "wingmen_v1")])
     .groupby(["dataset", "offset", "req_kind"])[["half_ok", "err_ok"]].agg(["mean", "std"]).reset_index(), "supp_act5_threshold")
bnd = []
for ds in ("levine32_v2", "wingmen_v1"):
    b = rd(f"{ds}/e5k_boundary.tsv")
    b.insert(0, "dataset", ds)
    bnd.append(b)
b = pd.concat(bnd, ignore_index=True)
save(b[b.part == "A_unequal"].groupby(["dataset", "sigma", "method"])[["tube_size_cv", "half_ok", "err_ok"]].mean().reset_index(),
     "supp_boundary_unequal")
save(b[b.part == "B_composition"].groupby(["dataset", "mode"])[["half_ok", "err_ok", "ci_cover", "alarm", "alarm_at_shifted_tube"]]
     .mean().reset_index(), "supp_boundary_composition")
import glob  # noqa: E402
save(pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(R, "calibration", "e5h_*.tsv"))]), "supp_calibration")
print("figure data:", sorted(os.listdir(DD)))
print("figure png:", sorted(os.listdir(PD)) if PLOT else "not generated (use --png)")
