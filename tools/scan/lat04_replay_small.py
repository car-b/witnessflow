"""Part 5 (empirical, small): evaluate a few structured designs with the same fixed-cell-budget replay
logic as src/e3b.py (same splits, thresholds, parents, allocation seeds 1000*a+r, CP half-width, error vs
independent reference split).  No initial-audit replay is rerun.

Levine (v=20 candidates):  r=570  PSL(2,19)-orbit 3-(20,6,10) design (perfect balance) vs La Jolla stack,
                           random, balanced random, and annealing from the stack towards lam>=10;
                           r=71/142 La Jolla stack for reference.
WINGMEN (v=26):            r=130 La Jolla = Steiner system S(3,6,26) vs random / balanced random;
                           r=142/180/260 stacks for reference (260 = two Steiner copies).
Output: out_lat/lat04_replay_small.tsv (+ stdout summary)."""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd
from lat_common import OUT, WF, design, budget, lam_vec
import replay, audit  # noqa: E402  (path set by lat_common)
from e3b import cp_half  # noqa: E402

Z, CAP, ALLOC = 1.96, 6, [0, 1, 2]


def prep(pfile):
    proto = json.load(open(os.path.join(WF, "protocols", pfile), encoding="utf-8"))
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(WF, "results", proto["protocol_id"], "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    return proto, markers, part, Bin, pm, cand


def parent_data(Bin, pm, part, G, tarr):
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    piG = pm[G][ini].mean()
    Rc = Bin[ref & pm[G]]
    refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
    folidx = np.flatnonzero(fol)
    gpos = np.flatnonzero(pm[G][folidx])
    codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.int64) for S in tarr])
    return piG, refp, len(folidx), gpos, codesF


def alloc_counts(N, gpos, codesF, r, a):
    well_of = np.empty(N, int)
    order = np.random.default_rng(1000 * a + r).permutation(N)
    for i, chunk in enumerate(np.array_split(order, r)):
        well_of[chunk] = i
    wg = well_of[gpos]
    C = np.zeros((len(codesF), r, 8), np.int32)
    for t in range(len(codesF)):
        C[t] = np.bincount(wg * 8 + codesF[t], minlength=r * 8).reshape(r, 8)
    return C


def evaluate(ds, G, eps_list, r, designs, tarr, piG, refp, N, gpos, codesF, rows):
    allocs = {a: alloc_counts(N, gpos, codesF, r, a) for a in ALLOC}
    for name, wells in designs.items():
        L = np.stack([np.isin(tarr, list(X)).all(1) for X in wells], 1).astype(np.int32)  # [T, r]
        lam = L.sum(1)
        for a, C in allocs.items():
            cnt = np.einsum("ti,tix->tx", L, C)
            n = cnt.sum(1)
            p = cnt / np.maximum(n, 1)[:, None]
            half = np.where(n > 0, cp_half(cnt, n).max(1), 1.0)
            err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
            for eps in eps_list:
                nreq = Z * Z * 0.25 / eps ** 2
                rows.append(dict(dataset=ds, parent=G, eps=eps, r=r, design=name, alloc=a,
                                 lam_req=max(1, math.ceil(nreq * r / (N * piG))), lam_min=int(lam.min()),
                                 lam_max=int(lam.max()), witnessed=float((n >= nreq).mean()),
                                 half_ok=float((half <= eps).mean()), err_ok=float((err <= eps).mean()),
                                 covered=float((lam >= 1).mean()), min_cells=int(n.min()), mean_cells=float(n.mean())))


def main():
    t0 = time.time()
    rows = []
    # ------------------------------------------------------------------ Levine
    proto, markers, part, Bin, pm, cand = prep("levine32_v2.json")
    tarr = np.array(list(itertools.combinations(cand, 3)))
    v = len(markers)
    psl = [set(cand[int(x)] for x in line.split()) for line in open(os.path.join(OUT, "psl19_3design_570.txt"))]
    st570 = budget.ljcr_stack(cand, CAP, 570, seed=0)
    ann, _ = budget.anneal_lambda(st570, cand, tarr, np.ones(len(tarr)), 10, v, iters=200_000, seed=1)
    lam_ann = np.stack([np.isin(tarr, list(X)).all(1) for X in ann], 1).sum(1)
    print(f"[{time.time() - t0:.0f}s] Levine loaded; anneal(stack570 -> lam>=10, 200k it): lam {lam_ann.min()}..{lam_ann.max()}, "
          f"frac lam>=10 {np.mean(lam_ann >= 10):.3f}", flush=True)
    designs_by_r = {
        570: {"PSL(2,19) 3-design": psl, "LJCR_stack": st570, "stack+anneal(lam>=10)": ann,
              **{f"random_s{s}": design.random_design(cand, CAP, 570, seed=s) for s in range(3)},
              **{f"balanced_random_s{s}": budget.balanced_random(cand, CAP, 570, seed=s) for s in range(3)}},
        71: {"LJCR_stack": budget.ljcr_stack(cand, CAP, 71, seed=0)},
        142: {"LJCR_stack": budget.ljcr_stack(cand, CAP, 142, seed=0)},
    }
    for G, eps_list in (("T", [0.05, 0.06, 0.07]), ("B", [0.07, 0.08, 0.10])):
        piG, refp, N, gpos, codesF = parent_data(Bin, pm, part, G, tarr)
        print(f"   {G}: pi={piG:.4f} N={N} N*pi={N * piG:.0f}", flush=True)
        for r, ds_ in designs_by_r.items():
            evaluate("levine32_v2", G, eps_list, r, ds_, tarr, piG, refp, N, gpos, codesF, rows)
        print(f"[{time.time() - t0:.0f}s] Levine {G} done", flush=True)
    del Bin, codesF
    # ------------------------------------------------------------------ WINGMEN
    proto, markers, part, Bin, pm, cand = prep("wingmen_v1.json")
    tarr = np.array(list(itertools.combinations(cand, 3)))
    print(f"[{time.time() - t0:.0f}s] WINGMEN loaded: {len(cand)} candidates", flush=True)
    designs_by_r = {
        130: {"LJCR = S(3,6,26)": budget.ljcr_stack(cand, CAP, 130, seed=0),
              **{f"random_s{s}": design.random_design(cand, CAP, 130, seed=s) for s in range(3)},
              **{f"balanced_random_s{s}": budget.balanced_random(cand, CAP, 130, seed=s) for s in range(3)}},
        142: {"LJCR_stack": budget.ljcr_stack(cand, CAP, 142, seed=0)},
        180: {"LJCR_stack": budget.ljcr_stack(cand, CAP, 180, seed=0)},
        260: {"LJCR_stack (2 Steiner copies)": budget.ljcr_stack(cand, CAP, 260, seed=0),
              **{f"random_s{s}": design.random_design(cand, CAP, 260, seed=s) for s in range(3)}},
    }
    for G, eps_list in (("CD4T", [0.03, 0.04, 0.05, 0.07]), ("CD8T", [0.05, 0.06, 0.07])):
        piG, refp, N, gpos, codesF = parent_data(Bin, pm, part, G, tarr)
        print(f"   {G}: pi={piG:.4f} N={N} N*pi={N * piG:.0f}", flush=True)
        for r, ds_ in designs_by_r.items():
            evaluate("wingmen_v1", G, eps_list, r, ds_, tarr, piG, refp, N, gpos, codesF, rows)
        print(f"[{time.time() - t0:.0f}s] WINGMEN {G} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lat04_replay_small.tsv"), sep="\t", index=False)
    df["design_grp"] = df.design.str.replace(r"_s\d$", "", regex=True)
    pd.set_option("display.width", 250)
    s = df.groupby(["dataset", "parent", "eps", "r", "design_grp"])[
        ["lam_req", "lam_min", "witnessed", "half_ok", "err_ok", "covered", "min_cells", "mean_cells"]].mean()
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
