"""Boundary stress: how much of the design gain survives non-ideal allocation, and
do the built-in checks warn?
  A. unequal tube sizes, phenotype-independent: tube weight ~ lognormal(0, sigma), sigma in {0, .25, .5, 1};
     classical vs random (paired cell allocation).
  B. composition shift in 10% of tubes: drop 30% of the parent cells that are positive for marker X,
     X in the tube's panel ('measured') or X not in the panel ('unmeasured').  Report reference-error
     attainment, fraction of triples whose 8 CP intervals all contain the reference ('ci_cover'), and alarms of
     the fixed-family single+pair checks (alpha 0.001 / (r*21)).
usage: python e5k_boundary.py <protocol.json>"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402
from e5b import well_counts, membership, evaluate  # noqa: E402
from e5d import run_checks  # noqa: E402

SET = {"levine32_v2": ("T", 0.06), "wingmen_v1": ("CD4T", 0.05)}
SIGMAS = [0.0, 0.25, 0.5, 1.0]
REPS_A, REPS_B = 10, 20


def ci_cover(L, C, refp):
    cnt = np.einsum("ti,itx->tx", L, C)
    n = cnt.sum(1)
    k = cnt.astype(float)
    nn = np.broadcast_to(n[:, None].astype(float), k.shape)
    lo = np.where(k > 0, beta.ppf(0.025, k, nn - k + 1), 0.0)
    hi = np.where(k < nn, beta.ppf(0.975, k + 1, nn - k), 1.0)
    ok = ((refp >= np.nan_to_num(lo, nan=0) - 1e-12) & (refp <= np.nan_to_num(hi, nan=1) + 1e-12)).all(1)
    return float(ok[n > 0].mean())


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    G, eps = SET[pid]
    out = os.path.join(ROOT, "results", pid)
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    piG = pm[G][ini].mean()
    Rc = Bin[ref & pm[G]]
    refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
    Bf = Bin[folidx].astype(np.int64)
    gmask = pm[G][folidx]
    blocks = design.ljcr_blocks(len(cand), 6, 3)
    wells = [set(cand[x] for x in b) for b in blocks]
    r = len(wells)
    Lc = membership(wells, tarr, v)
    Lr = membership(design.random_design(cand, 6, r, seed=0), tarr, v)
    rows = []
    gpos = np.flatnonzero(gmask)
    codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
    # ---------- A: unequal tube sizes
    for sigma in SIGMAS:
        for rep in range(REPS_A):
            rng = np.random.default_rng(31000 + rep + int(100 * sigma))
            wts = np.exp(rng.normal(0, sigma, r))
            well_of = rng.choice(r, N, p=wts / wts.sum())
            C = well_counts(codesF, well_of[gpos], r)
            for meth, L in (("classical", Lc), ("random", Lr)):
                m = evaluate(L, C, refp, eps, N, piG)
                rows.append(dict(part="A_unequal", sigma=sigma, rep=rep, method=meth, ci_cover=ci_cover(L, C, refp),
                                 tube_size_cv=float(np.bincount(well_of, minlength=r).std() / (N / r)), **m))
        print(f"[{time.time() - t0:.0f}s] A sigma={sigma} done", flush=True)
    # ---------- B: composition shift in 10% of tubes
    gpos_all = np.flatnonzero(gmask)
    MT = r * 21
    for mode in ("none", "measured", "unmeasured"):
        for rep in range(REPS_B):
            rng = np.random.default_rng(42000 + rep + {"none": 0, "measured": 500, "unmeasured": 900}[mode])
            order = rng.permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, r)):
                well_of[chunk] = i
            keep = np.ones(N, bool)
            shifted = rng.choice(r, max(1, r // 10), replace=False)
            for ti in shifted if mode != "none" else []:
                if mode == "measured":
                    X = int(rng.choice(sorted(wells[ti])))
                else:
                    X = int(rng.choice([c for c in cand if c not in wells[ti]]))
                sel = np.flatnonzero((well_of == ti) & gmask & (Bf[:, X] == 1))
                keep[rng.choice(sel, int(0.3 * len(sel)), replace=False)] = False
            kg = np.flatnonzero(keep & gmask)
            codesK = np.stack([audit.codes(Bin[folidx[kg]], S).astype(np.uint8) for S in tarr])
            C = well_counts(codesK, well_of[kg], r)
            m = evaluate(Lc, C, refp, eps, N, piG)
            res = []
            for md in ("single", "pair"):
                res += [x for x in run_checks(Bf[keep], well_of[keep], wells, md) if not np.isnan(x[0])]
            pmin, tmin, items = min(res, key=lambda x: x[0])
            alarm = pmin < 1e-3 / MT
            rows.append(dict(part="B_composition", mode=mode, rep=rep, method="classical", ci_cover=ci_cover(Lc, C, refp),
                             alarm=bool(alarm), alarm_at_shifted_tube=bool(alarm and tmin in set(shifted)), **m))
        print(f"[{time.time() - t0:.0f}s] B {mode} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5k_boundary.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    a = df[df.part == "A_unequal"].groupby(["sigma", "method"])[["tube_size_cv", "half_ok", "err_ok", "ci_cover"]].mean()
    print(a.round(3).to_string())
    b = df[df.part == "B_composition"].groupby("mode")[["half_ok", "err_ok", "ci_cover", "alarm", "alarm_at_shifted_tube"]].mean()
    print(b.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
