"""End-to-end replay for a frozen protocol (Levine-32 first).

usage: python replay.py <protocol.json> [--quick]
outputs -> results/<protocol_id>/
"""
import json, os, sys, time, itertools
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fcs, audit, design, q1  # noqa: E402


# ------------------------------------------------------------------ data + protocol
def load(proto):
    ds = proto["dataset"]
    if ds.get("type") == "fcs_dir":
        import datasets
        Xt, markers, sample, _ = datasets.load_fcs_dir(ds)
        return Xt, markers, sample
    X, ch, mk, meta = fcs.read_fcs(os.path.join(ROOT, ds["file"]))
    names = [m for m in mk]
    sample = X[:, names.index(ds["sample_column"])].astype(int)
    keep = [i for i, m in enumerate(names) if m not in ds["exclude_columns"]]
    markers = [names[i] for i in keep]
    Xt = np.arcsinh(X[:, keep] / 5.0)
    return Xt, markers, sample


def split(n, sample, fr, seed):
    rng = np.random.default_rng(seed)
    part = np.empty(n, dtype="<U12")
    keys = ["calibration", "initial", "followup", "reference"]
    for s in np.unique(sample):
        idx = np.flatnonzero(sample == s)
        rng.shuffle(idx)
        cuts = np.cumsum([int(round(fr[k] * len(idx))) for k in keys])[:-1]
        for k, chunk in zip(keys, np.split(idx, cuts)):
            part[chunk] = k
    return part


def thresholds_v1(Xt_cal, markers, seed=0, max_n=100_000):
    """Original v1 rule (kept for reproducibility): 2-GMM on all values, reliable iff dBIC>=10 and
    both weights >= 0.05.  Known artefact: splits the CyTOF zero spike from non-zero values."""
    rng = np.random.default_rng(seed)
    out = {}
    for j, m in enumerate(markers):
        x = Xt_cal[:, j]
        x = (x if len(x) <= max_n else rng.choice(x, max_n, replace=False)).reshape(-1, 1)
        g1 = GaussianMixture(1, random_state=seed).fit(x)
        g2 = GaussianMixture(2, random_state=seed, n_init=3).fit(x)
        dbic = g1.bic(x) - g2.bic(x)
        mu = g2.means_.ravel()
        lo, hi = np.argsort(mu)
        grid = np.linspace(mu[lo], mu[hi], 4001).reshape(-1, 1)
        cross = np.flatnonzero(g2.predict_proba(grid)[:, hi] >= 0.5)
        thr = float(grid[cross[0], 0]) if len(cross) else float(np.mean(mu))
        w = g2.weights_[[lo, hi]]
        out[m] = {"thr": thr, "dBIC": float(dbic), "w_lo": float(w[0]), "w_hi": float(w[1]),
                  "reliable": bool(dbic >= 10 and w.min() >= 0.05 and len(cross) > 0)}
    return out


def thresholds(Xt_cal, markers, rule=None, seed=0, max_n=100_000):
    if rule and "zero_cut" not in rule:
        return thresholds_v1(Xt_cal, markers, seed, max_n)
    return thresholds_v2(Xt_cal, markers, rule, seed, max_n)


def thresholds_v2(Xt_cal, markers, rule=None, seed=0, max_n=100_000):
    """v2 rule: values <= zero_cut (CyTOF zero spike) are negative and excluded from the fit;
    2-GMM on the remaining values; cut = posterior-0.5 crossing between the component means.
    reliable iff Ashman's D >= D_min and the upper component weight >= w_min."""
    rule = rule or {}
    zc = rule.get("zero_cut", 0.05)
    dmin = rule.get("D_min", 1.5)
    wmin = rule.get("w_min", 0.03)
    rng = np.random.default_rng(seed)
    out = {}
    for j, m in enumerate(markers):
        x = Xt_cal[:, j]
        nz = x[x > zc]
        xs = nz if len(nz) <= max_n else rng.choice(nz, max_n, replace=False)
        xs = xs.reshape(-1, 1)
        g2 = GaussianMixture(2, random_state=seed, n_init=3).fit(xs)
        mu = g2.means_.ravel()
        sd = np.sqrt(g2.covariances_.ravel())
        lo, hi = np.argsort(mu)
        grid = np.linspace(mu[lo], mu[hi], 4001).reshape(-1, 1)
        post = g2.predict_proba(grid)[:, hi]
        cross = np.flatnonzero(post >= 0.5)
        thr = float(grid[cross[0], 0]) if len(cross) else float(np.mean(mu))
        w = g2.weights_[[lo, hi]]
        D = float(np.sqrt(2) * (mu[hi] - mu[lo]) / np.sqrt(sd[hi] ** 2 + sd[lo] ** 2))
        ok = bool(D >= dmin and w[1] >= wmin and len(cross) > 0)
        out[m] = {"thr": thr, "ashman_D": D, "zero_frac": float((x <= zc).mean()),
                  "w_lo": float(w[0]), "w_hi": float(w[1]), "mu_lo": float(mu[lo]), "mu_hi": float(mu[hi]),
                  "pos_frac": float((x > thr).mean()), "reliable": ok}
    return out


def parent_masks(Bin, markers, parents):
    idx = {m: i for i, m in enumerate(markers)}
    out = {}
    for name, rule in parents.items():
        m = np.ones(len(Bin), bool)
        for p in rule.get("pos", []):
            m &= Bin[:, idx[p]] == 1
        for q in rule.get("neg", []):
            m &= Bin[:, idx[q]] == 0
        out[name] = m
    return out


def parent_strata(Xbb, inG, cal, K, seed=0):
    """K-means micro-strata on backbone, fitted on calibration cells of the parent only."""
    from sklearn.cluster import KMeans
    st = np.zeros(len(Xbb), int)
    fit = np.flatnonzero(inG & cal)
    if len(fit) < 10 * K:
        return st
    km = KMeans(K, n_init=4, random_state=seed).fit(Xbb[fit])
    g = np.flatnonzero(inG)
    st[g] = km.predict(Xbb[g])
    return st


# ------------------------------------------------------------------ audit of one experiment
def audit_experiment(Bin, Xbb, tube, panels, claims, pmask, cellsel, K=1, strat=None, knn=True, seed=0,
                     groups=None):
    """cellsel: boolean mask of the cells belonging to this experiment.  groups: per-cell sample id; when
    there is more than one sample, a within-sample block-kNN baseline is added."""
    idx = np.flatnonzero(cellsel)
    B = Bin[idx]
    tb = tube[idx]
    st = np.zeros(len(idx), int) if strat is None else strat[idx]
    meas = audit.measured_matrix(len(idx), Bin.shape[1], tb, panels)
    imps = {}
    if knn:
        imps["knn_major"] = audit.knn_impute(B, Xbb[idx], tb, panels, k=10, mode="majority", seed=seed)
        imps["knn_block"] = audit.knn_impute(B, Xbb[idx], tb, panels, k=10, mode="block", seed=seed)
        if groups is not None and len(np.unique(groups[idx])) > 1:
            imps["knn_block_within"] = audit.knn_impute_within(B, Xbb[idx], tb, panels, groups[idx], k=10,
                                                              mode="block", seed=seed)
    rows = []
    for c in claims:
        inG = pmask[c["parent"]][idx]
        ev = audit.evidence(B, c["S"], inG, st, K, tb, panels)
        r = audit.bound_claim(ev, K, c["sigma"])
        r.update(audit.direct_estimate(ev, c["sigma"]))
        r["cond_indep"] = audit.cond_indep_estimate(B, c["S"], inG, st, K, meas, c["sigma"])
        for k_, Bi in imps.items():
            r[k_] = audit.imputed_frequency(Bi, c["S"], inG, c["sigma"])
        r["observed_pairs"] = sum(1 for U, _ in ev if len(U) >= 2)
        r.update(audit.tube_consistency(B, c["S"], inG, tb, panels))
        r["claim"] = c["id"]
        r["n_parent"] = float(inG.sum())
        rows.append(r)
    return rows


# ------------------------------------------------------------------ main
def main(proto_path, quick=False):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    os.makedirs(out, exist_ok=True)
    Xt, markers, sample = load(proto)
    n = len(Xt)
    sp = proto["split"]
    part = split(n, sample, sp, sp["seed"])
    cal = part == "calibration"
    thr = thresholds(Xt[cal], markers, proto.get("threshold_rule"))
    json.dump(thr, open(os.path.join(out, "thresholds.json"), "w"), indent=1)
    Bin = np.zeros(Xt.shape, np.uint8)
    for j, m in enumerate(markers):
        Bin[:, j] = Xt[:, j] > thr[m]["thr"]
    mi = {m: i for i, m in enumerate(markers)}
    bb = [mi[m] for m in proto["backbone"]]
    nonbb = [i for i in range(len(markers)) if i not in bb]
    Xbb = Xt[:, bb]
    pmask = parent_masks(Bin, markers, proto["parents"])

    # claims (sorted triple + permuted pattern)
    claims, excluded = [], []
    for c in proto["claims"]:
        bad = [m for m in c["markers"] if not thr[m]["reliable"]]
        trip = sorted(zip([mi[m] for m in c["markers"]], c["pattern"]))
        S = tuple(t for t, _ in trip)
        sigma = int(sum(b << (2 - q) for q, (_, b) in enumerate(trip)))
        cc = dict(c, S=S, sigma=sigma, unreliable_markers=bad)
        (excluded if bad else claims).append(cc)
    claims_all = claims + excluded  # audit all, but flag
    # reference truth (R split, full measurement)
    ref = part == "reference"
    refrows = []
    for c in claims_all:
        inG = pmask[c["parent"]] & ref
        code = audit.codes(Bin[inG], c["S"])
        k, nn = int((code == c["sigma"]).sum()), int(inG.sum())
        lo, hi = audit.cp_interval(k, nn)
        tab = np.bincount(code, minlength=8).astype(float)
        st = q1.q1_stats(tab, c["sigma"]) if nn > 0 else {}
        refrows.append(dict(claim=c["id"], parent=c["parent"], markers="/".join(c["markers"]),
                            ref=k / max(nn, 1), ref_lo=float(lo), ref_hi=float(hi), n_ref=nn,
                            reliable=not c["unreliable_markers"], **{f"q1_{a}": b for a, b in st.items()}))
    refdf = pd.DataFrame(refrows)
    refdf.to_csv(os.path.join(out, "reference_claims.tsv"), sep="\t", index=False)
    print(f"[{time.time() - t0:.0f}s] thresholds, reference done; {len(claims)} primary claims, "
          f"{len(excluded)} excluded (unreliable cut)", flush=True)

    # optional rare-parent stress: thin one parent in the initial+followup splits (reference untouched)
    keep = np.ones(n, bool)
    ds = proto.get("downsample")
    if ds:
        rngd = np.random.default_rng(ds.get("seed", 0))
        for sp_name in ("initial", "followup"):
            insp = part == sp_name
            g = np.flatnonzero(insp & pmask[ds["parent"]])
            n_non = int((insp & ~pmask[ds["parent"]]).sum())
            f = ds["target_frac"]
            n_keep = min(len(g), int(round(f * n_non / (1 - f))))
            drop = rngd.choice(g, len(g) - n_keep, replace=False)
            keep[drop] = False
        print(f"downsample {ds['parent']} -> {ds['target_frac']}: kept "
              f"{int((keep & pmask[ds['parent']] & (part == 'initial')).sum())} initial parent cells", flush=True)
    idg = proto["initial_design"]
    fu = proto["followup"]
    eps = proto["claim_goal"]["epsilon"]
    reps = 2 if quick else idg["replicates"]
    fureps = 1 if quick else fu["replicates"]
    rng = np.random.default_rng(idg["seed"])           # panels only
    rng_tube = np.random.default_rng(idg["seed"] + 1)  # cell->tube assignment (independent of panels)
    init_rows, fu_rows = [], []
    ini = (part == "initial") & keep
    fol = (part == "followup") & keep
    tube = np.full(n, -1)
    strata_cache = {}
    for rep in range(reps):
        perm = rng.permutation(nonbb)
        panels0 = [set(bb) | set(perm[i * idg["capacity"]:(i + 1) * idg["capacity"]].tolist())
                   for i in range(idg["n_tubes"])]
        tube[:] = -1
        ii = np.flatnonzero(ini)
        tube[ii] = rng_tube.integers(0, idg["n_tubes"], len(ii))
        # optional cross-tube stress: shift (asinh units) the measured markers of one initial tube, re-gate
        Bin_r, Xbb_r, pmask_r = Bin, Xbb, pmask
        tsft = proto.get("tube_shift")
        if tsft:
            t_s = tsft["tube"]
            sel = np.flatnonzero(ini & (tube == t_s))
            pool = sorted(panels0[t_s]) if tsft["markers"] == "all" else \
                sorted(set(panels0[t_s]) - set(bb)) if tsft["markers"] == "nonbackbone" else \
                [mi[m] for m in tsft["markers"] if mi[m] in panels0[t_s]]
            Bin_r, Xbb_r = Bin.copy(), Xbb.copy()
            for m in pool:
                x = Xt[sel, m] + tsft["delta"]
                Bin_r[sel, m] = x > thr[markers[m]]["thr"]
                if m in bb:
                    Xbb_r[sel, bb.index(m)] = x
            pmask_r = parent_masks(Bin_r, markers, proto["parents"])
        rows = audit_experiment(Bin_r, Xbb_r, tube, panels0, claims_all, pmask_r, ini, seed=rep, groups=sample)
        for r in rows:
            r["rep"] = rep
        # micro-strata sensitivity (bounds only): K-means on calibration backbone within each parent
        for K in proto["micro_strata"].get("sensitivity_K", []):
            idx = np.flatnonzero(ini)
            for r in rows:
                c = next(cc for cc in claims_all if cc["id"] == r["claim"])
                st = strata_cache.setdefault((c["parent"], K), parent_strata(Xbb, pmask[c["parent"]], cal, K))
                inG = pmask_r[c["parent"]][idx]
                ev = audit.evidence(Bin_r[idx], c["S"], inG, st[idx], K, tube[idx], panels0)
                rb = audit.bound_claim(ev, K, c["sigma"])
                r[f"conf_w_K{K}"] = rb["conf_hi"] - rb["conf_lo"]
                r[f"ident_w_K{K}"] = rb["ident_hi"] - rb["ident_lo"]
                r[f"ref_in_K{K}"] = bool(rb["conf_lo"] - 1e-9 <= refdf.set_index("claim").loc[c["id"], "ref"] <= rb["conf_hi"] + 1e-9)
        init_rows += rows
        # ---- follow-up design (weights from initial data only)
        ir = {r["claim"]: r for r in rows}
        tw = {}
        for c in claims:
            r = ir[c["id"]]
            if r["n_direct"] > 0 and (r["direct_hi"] - r["direct_lo"]) / 2 <= eps:
                continue  # already resolved
            w = min((r["conf_hi"] - r["conf_lo"]) / (2 * eps), 1.0)
            tw[c["S"]] = min(tw.get(c["S"], 0) + w, 1.0)
        targets = list(tw)
        w = [tw[S] for S in targets]
        for r_ in fu["r_grid"]:
            for meth in fu["methods"]:
                for frep in range(fureps):
                    seed = 1000 * rep + 10 * r_ + frep
                    X = None
                    if not targets:
                        X = [set() for _ in range(r_)]
                    elif meth == "random":
                        X = design.random_design(nonbb, fu["capacity"], r_, seed)
                    elif meth == "naive_packing":
                        X = design.naive_packing(nonbb, fu["capacity"], r_, targets, w)
                    elif meth == "greedy":
                        X = design.greedy_design(nonbb, fu["capacity"], r_, targets, w)
                    elif meth == "greedy+anneal":
                        g = design.greedy_design(nonbb, fu["capacity"], r_, targets, w)
                        X, _ = design.anneal(g, nonbb, fu["capacity"], targets, w, iters=5000, seed=seed)
                    elif meth == "uniform_LJCR":
                        X, _ = design.uniform_design(nonbb, fu["capacity"], r_, seed)
                    elif meth == "ilp":
                        X, _, _ = design.ilp_design(nonbb, fu["capacity"], r_, targets, w, time_limit=20)
                    if X is None:
                        continue
                    if meth not in ("random", "uniform_LJCR") and frep > 0:
                        continue  # deterministic methods: one replicate
                    design.check_design(X, nonbb, fu["capacity"])
                    panels_fu = [set(bb) | set(T) for T in X]
                    tube_all = tube.copy()
                    fi = np.flatnonzero(fol)
                    order = np.random.default_rng(seed).permutation(fi)
                    for k, chunk in enumerate(np.array_split(order, r_)):
                        tube_all[chunk] = idg["n_tubes"] + k
                    sel = ini | fol
                    rows2 = audit_experiment(Bin_r, Xbb_r, tube_all, panels0 + panels_fu, claims, pmask_r, sel,
                                             knn=False, seed=seed)
                    cov = design.covered_mask(X, targets) if targets else np.array([])
                    for r2 in rows2:
                        rf = refdf.set_index("claim").loc[r2["claim"]]
                        res = r2["n_direct"] > 0 and (r2["direct_hi"] - r2["direct_lo"]) / 2 <= eps
                        corr = res and not (r2["direct_hi"] < rf["ref_lo"] or r2["direct_lo"] > rf["ref_hi"])
                        fu_rows.append(dict(rep=rep, r=r_, method=meth, frep=frep, claim=r2["claim"],
                                            n_direct=r2["n_direct"], direct=r2["direct"],
                                            half=(r2["direct_hi"] - r2["direct_lo"]) / 2 if r2["n_direct"] else np.nan,
                                            resolved=bool(res), correct=bool(corr),
                                            conf_w=r2["conf_hi"] - r2["conf_lo"],
                                            weighted_cov=float(np.dot(w, cov)) if len(cov) else 0.0,
                                            n_targets=len(targets)))
        print(f"[{time.time() - t0:.0f}s] rep {rep} done ({len(targets)} targets)", flush=True)
    idf = pd.DataFrame(init_rows).merge(refdf, on="claim", how="left")
    idf.to_csv(os.path.join(out, "initial_audit.tsv"), sep="\t", index=False)
    fdf = pd.DataFrame(fu_rows)
    fdf.to_csv(os.path.join(out, "followup.tsv"), sep="\t", index=False)
    print(f"[{time.time() - t0:.0f}s] done -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv)
