"""Part 4: headline comparison numbers pulled from existing witnessflow/results TSVs (no recomputation
of replays).  Output: out_lat/lat03_numbers.txt"""
import os
from math import comb, sqrt
import numpy as np
import pandas as pd
from lat_common import OUT, WF

R = os.path.join(WF, "results")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def rd(p):
    return pd.read_csv(os.path.join(R, p), sep="\t")


def main():
    pd.set_option("display.width", 250)
    # 1. combinatorial coverage (E2)
    for ds, v in (("levine32_v2", 20), ("wingmen_v1", 26)):
        e2 = rd(f"{ds}/explore_screen_design.tsv")
        t = e2[e2.capacity.isin([6, 10])].pivot_table(index=["capacity", "r"], columns="method", values="frac_triples")
        log(f"\n[1] {ds}/explore_screen_design.tsv  (v={v})")
        log(t.round(3).to_string())
        for k in (6, 10):
            p = comb(k, 3) / comb(v, 3)
            nfull = e2[e2.capacity == k].ljcr_full.dropna().iloc[0]
            log(f"   k={k}: La Jolla size {int(nfull)}; random expected coverage at r={int(nfull)}: "
                f"{1 - (1 - p) ** nfull:.4f}; r needed for random to reach 99% expected: "
                f"{np.log(0.01) / np.log(1 - p):.0f}")
    # 2. fixed cell budget (E3b)
    for ds, cases in (("levine32_v2", [("T", 0.07), ("B", 0.10), ("T", 0.05)]),
                      ("wingmen_v1", [("CD4T", 0.07), ("CD4T", 0.05), ("CD8T", 0.07)])):
        e = rd(f"{ds}/e3b_budget.tsv")
        log(f"\n[2] {ds}/e3b_budget.tsv  (mean over design seeds x 3 allocation seeds)")
        for G, eps in cases:
            s = e[(e.parent == G) & (np.isclose(e.eps, eps))]
            for metric in ("witnessed", "emp_err_ok", "emp_half_ok", "min_cells"):
                t = s.groupby(["r", "method"])[metric].mean().unstack("method")
                log(f"  {G} eps={eps} {metric}:\n" + t.round(3).to_string())
            lr = s.groupby("r").lam_req.first()
            log(f"  lam_req by r: {lr.to_dict()}")
    # 3. E3_budget uniform cells: theoretical optimum cells per triple for a perfectly balanced design
    e3 = rd("levine32_v2/e3_budget.tsv")
    log("\n[3] levine32_v2/e3_budget.tsv: uniform_cells = N*pi*C(6,3)/C(20,3) (cells/triple of a 3-design)")
    for G, g in e3.groupby("parent"):
        uc = g.uniform_cells.iloc[0]
        Npi = uc * comb(20, 3) / comb(6, 3)
        log(f"  {G}: uniform_cells={uc:.1f}  => N*pi={Npi:.0f}; covering-71 min cells = N*pi/71 = {Npi / 71:.1f}")
        for name, n in (("3-design (r/lambda=57)", uc), ("La Jolla 71 / stacks (r/lambda=71)", Npi / 71)):
            log(f"     {name}: worst-case eps for ALL triples = 1.96*0.5/sqrt(n) = {1.96 * 0.5 / sqrt(n):.4f}")
        for eps in (0.03, 0.05, 0.07, 0.10):
            nreq = 1.96 ** 2 * 0.25 / eps ** 2
            log(f"     eps={eps}: n_req={nreq:.1f}; 3-design feasible: {uc >= nreq}; covering-71 feasible: {Npi / 71 >= nreq}")
    # 4. hybrid (E3e) gains
    for ds in ("levine32_v2", "wingmen_v1"):
        e = rd(f"{ds}/e3e_robust.tsv")
        s = e[e.margin == 0.02].groupby(["parent", "eps", "r", "method"])[["half_ok", "err_ok", "covered"]].mean()
        log(f"\n[4] {ds}/e3e_robust.tsv margin=0.02 (mean over 3 init reps x 3 alloc seeds)")
        log(s.unstack("method").round(3).to_string())
    # 5. initial audit, shift stress tests
    log("\n[5] shift stress tests (initial_audit.tsv, claims with n_direct==0)")
    for ds in ("levine32_v2_shift0", "levine32_v2_shiftAllp0p5", "levine32_v2_shiftNonBBp0p5", "levine32_v2_shiftNonBBm0p5"):
        a = rd(f"{ds}/initial_audit.tsv")
        a = a[(a.n_direct == 0) & a.reliable]
        err = (a.knn_block - a.ref).abs()
        inside = ((a.ref >= a.conf_lo - 1e-12) & (a.ref <= a.conf_hi + 1e-12)).mean()
        log(f"  {ds}: n={len(a)}  ref-in-interval={inside:.3f}  blockkNN max err={err.max():.3f}  "
            f">5pp={(err > 0.05).mean():.3f}  parent_frac_range(mean)={a.parent_frac_range.mean():.4f}")
    b = rd("levine32_v2/bridge_test.tsv")
    log("\n[6] levine32_v2/bridge_test.tsv alarm rate")
    log(b.groupby(["design", "scenario"]).agg(alarm=("alarm", "mean"), unmeasured=("n_unmeasured_markers", "mean"),
                                               claims_lost=("claims_with_unmeasured_marker", "mean")).round(3).to_string())
    # 7. per-claim follow-up (ILP vs random)
    for ds in ("levine32_v2", "wingmen_v1"):
        f = rd(f"{ds}/followup.tsv")
        t = f.groupby(["r", "method", "rep", "frep"]).correct.sum().groupby(["r", "method"]).mean().unstack("method")
        log(f"\n[7] {ds}/followup.tsv: mean # claims correctly resolved")
        log(t.round(2).to_string())
    with open(os.path.join(OUT, "lat03_numbers.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
