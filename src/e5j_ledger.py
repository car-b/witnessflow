"""Per-target allocation ledger: replay the e5i settings with the same seeds, SAVE the
requirement vectors, final panels and per-target outcomes, and tabulate which targets gained / lost the
half-width criterion (and the reference-error criterion) relative to the classical stack and to the shuffled
requirement.  usage: python e5j_ledger.py <protocol.json>"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, audit  # noqa: E402
from e3e import initial_designs  # noqa: E402
from e5b import well_counts, membership  # noqa: E402
from e5i_shuffle import SET, INIT_REPS, OPT_SEEDS, SHUFFLES, ALLOC, ITERS, W_COV, Z  # noqa: E402


def per_target(L, C, refp):
    cnt = np.einsum("ti,itx->tx", L, C)
    n = cnt.sum(1)
    k = cnt.astype(float)
    nn = np.broadcast_to(n[:, None].astype(float), k.shape)
    lo = np.where(k > 0, beta.ppf(0.025, k, nn - k + 1), 0.0)
    hi = np.where(k < nn, beta.ppf(0.975, k + 1, nn - k), 1.0)
    half = np.where(n > 0, ((np.nan_to_num(hi, nan=1.0) - np.nan_to_num(lo, nan=0.0)) / 2).max(1), 1.0)
    p = cnt / np.maximum(n, 1)[:, None]
    err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
    return L.sum(1), n, half, err


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    out = os.path.join(ROOT, "results", pid, "ledger")
    os.makedirs(out, exist_ok=True)
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(ROOT, "results", pid, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    mi = {m: i for i, m in enumerate(markers)}
    bbi = [mi[m] for m in proto["backbone"]]
    nonbb = [i for i in range(len(markers)) if i not in bbi]
    cand = [i for i in nonbb if thr[markers[i]]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    t2 = np.vstack([tarr, tarr])
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    ii = np.flatnonzero(ini)
    inits = initial_designs(proto, nonbb, bbi, len(ii))
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    names = ["/".join(markers[i] for i in S) for S in tarr]
    summary = []
    for G, eps, r in SET[pid]:
        piG = pm[G][ini].mean()
        gI = pm[G][ii]
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        refvar = (refp * (1 - refp)).max(1)  # reference-based 'true' worst-pattern variance (evaluation only)
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        st = budget.ljcr_stack(cand, 6, r, seed=0)
        allocC = {}
        for a in ALLOC:
            order = np.random.default_rng(1000 * a + r).permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, r)):
                well_of[chunk] = i
            allocC[a] = well_counts(codesF, well_of[gpos], r)
        Lst = membership(st, tarr, v)
        base = {a: per_target(Lst, C, refp) for a, C in allocC.items()}
        for rep in INIT_REPS:
            panels, tubes = inits[rep]
            zero = np.zeros(len(ii), int)
            v_lp = np.zeros(len(tarr))
            for t, S in enumerate(tarr):
                lo, hi = audit.conf_bounds_all(audit.evidence(Bin[ii], tuple(S), gI, zero, 1, tubes, panels), 1)
                v_lp[t] = audit.max_binom_var(lo, hi).max()
            lam_lp = np.maximum(1, np.ceil(Z * Z * v_lp / eps ** 2 * r / (N * piG))).astype(int)
            reqs = {"lp_true": lam_lp}
            for s in SHUFFLES:
                reqs[f"lp_shuffled{s}"] = np.random.default_rng(500 + s + 10 * rep).permutation(lam_lp)
            np.savez(os.path.join(out, f"{G}_rep{rep}_requirements.npz"), v_lp=v_lp, **reqs)
            for name, lam in reqs.items():
                w2 = np.concatenate([1.0 / lam, np.full(len(tarr), W_COV)])
                l2 = np.concatenate([lam, np.ones(len(tarr), int)])
                for os_ in OPT_SEEDS:
                    wells, _ = budget.anneal_lambda(st, cand, t2, w2, l2, v, iters=ITERS, seed=os_,
                                                    t0=float(np.mean(1.0 / lam)))
                    json.dump([sorted(int(x) for x in X) for X in wells],
                              open(os.path.join(out, f"{G}_rep{rep}_{name}_opt{os_}_panels.json"), "w"))
                    L = membership(wells, tarr, v)
                    for a, C in allocC.items():
                        lam_T, n_T, half_T, err_T = per_target(L, C, refp)
                        bl, bn, bh, be = base[a]
                        ph, pe = half_T <= eps, err_T <= eps
                        bph, bpe = bh <= eps, be <= eps
                        if name == "lp_true" and a == 0:
                            pd.DataFrame(dict(triple=names, req_lam=lam, v_lp=v_lp, ref_var=refvar,
                                              lam_base=bl, lam_opt=lam_T, n_base=bn, n_opt=n_T,
                                              half_base=bh, half_opt=half_T, err_base=be, err_opt=err_T)).to_csv(
                                os.path.join(out, f"{G}_rep{rep}_opt{os_}_targets.tsv"), sep="\t", index=False)
                        gain_h, lose_h = ph & ~bph, ~ph & bph
                        gain_e, lose_e = pe & ~bpe, ~pe & bpe
                        summary.append(dict(dataset=pid, parent=G, eps=eps, r=r, init_rep=rep, req=name.rstrip("012"),
                                            shuffle=name[-1] if "shuffled" in name else "", opt_seed=os_, alloc=a,
                                            half_ok=ph.mean(), half_ok_base=bph.mean(),
                                            gained_half=int(gain_h.sum()), lost_half=int(lose_h.sum()),
                                            gained_err=int(gain_e.sum()), lost_err=int(lose_e.sum()),
                                            gained_half_mean_reqlam=float(lam[gain_h].mean()) if gain_h.any() else np.nan,
                                            gained_half_mean_refvar=float(refvar[gain_h].mean()) if gain_h.any() else np.nan,
                                            lost_err_mean_refvar=float(refvar[lose_e].mean()) if lose_e.any() else np.nan,
                                            lost_err_mean_lamopt=float(lam_T[lose_e].mean()) if lose_e.any() else np.nan,
                                            lost_err_mean_lambase=float(bl[lose_e].mean()) if lose_e.any() else np.nan,
                                            all_mean_refvar=float(refvar.mean())))
            print(f"[{time.time() - t0:.0f}s] {G} rep {rep} done", flush=True)
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(out, "ledger_summary.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["parent", "req"])[["half_ok", "half_ok_base", "gained_half", "lost_half", "gained_err", "lost_err",
                                         "gained_half_mean_refvar", "lost_err_mean_refvar", "all_mean_refvar"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
