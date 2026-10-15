"""Per-claim evidence audit for 3-marker binary phenotype claims.

Claim k = (parent G, ordered marker triple S=(a,b,c), target pattern sigma in {0..7}).
Value f_k = P(X_S = sigma | G).  Pattern code = x_a*4 + x_b*2 + x_c.

An experiment = cells, each assigned to one tube; tube j measures panel_j (a set of marker
indices; always contains the backbone, which defines G and the micro-strata s=0..K-1).

Bounds:  variables z[s,x] = P(s, X_S=x | G), sum z = 1.
Every tube group g (tubes sharing U_g = S ∩ panel) yields events (s, tau) over the measured
sub-pattern tau of U_g; each event's probability is constrained to its Clopper-Pearson interval
(Bonferroni over the claim's events).  min/max sum_s z[s,sigma] -> confidence interval [L, U].
identification width: same LP with *point* constraints taken from the max-entropy fit (which is
consistent by construction) -> what stays undetermined even with infinite cells.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta
from sklearn.neighbors import NearestNeighbors

PAT = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8)  # code -> bits


def codes(B, S):
    a, b, c = S
    return B[:, a].astype(np.int64) * 4 + B[:, b].astype(np.int64) * 2 + B[:, c]


def cp_interval(k, n, alpha=0.05):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


def sub_matrix(U):
    """A[tau, x] = 1 if pattern x restricted to positions U (in order) equals tau."""
    A = np.zeros((2 ** len(U), 8))
    for x in range(8):
        t = 0
        for q in U:
            t = t * 2 + int(PAT[x][q])
        A[t, x] = 1
    return A


# ---------------------------------------------------------------- evidence tables
def evidence(B, S, in_parent, strat, K, tube, panels):
    """Pooled counts per tube group.  Returns list of (U, counts[K, 2^|U|]) with U possibly ().
    Only measured entries are used: U = positions of S measured by the tube."""
    groups = {}
    code = codes(B, S)
    for j, P in enumerate(panels):
        U = tuple(q for q, m in enumerate(S) if m in P)
        sel = (tube == j) & in_parent
        if not sel.any():
            continue
        bits = PAT[code[sel]][:, list(U)] if U else np.zeros((sel.sum(), 0), np.uint8)
        sub = np.zeros(sel.sum(), np.int64)
        for q in range(len(U)):
            sub = sub * 2 + bits[:, q]
        cnt = np.zeros((K, 2 ** len(U)))
        np.add.at(cnt, (strat[sel], sub), 1)
        groups[U] = groups.get(U, 0) + cnt
    return list(groups.items())


def _constraint_rows(ev, K):
    """For each event (group, s, tau) -> (row over z[K*8], count, n_group)."""
    rows = []
    for U, cnt in ev:
        n = cnt.sum()
        if n <= 0:
            continue
        A = sub_matrix(U)
        for s in range(K):
            for t in range(A.shape[0]):
                r = np.zeros(K * 8)
                r[s * 8:(s + 1) * 8] = A[t]
                rows.append((r, cnt[s, t], n))
    return rows


def _lp_minmax(A_ub, b_ub, A_eq, b_eq, obj):
    out = []
    for sgn in (1, -1):
        res = linprog(sgn * obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, 1), method="highs")
        if res.status != 0:
            return None
        out.append(sgn * res.fun)
    return out


def maxent_fit(ev, K, iters=2000, tol=1e-11):
    """EM maximum-likelihood z[K*8] for coarsened multinomial data: every group observes counts of
    (s, tau) = a coarsening of (s, x).  Started from uniform (so among likelihood-equivalent joints it
    lands near the max-entropy one).  Unlike sequential IPF, the result does not depend on the order
    of groups and is well defined when the groups' empirical margins are mutually inconsistent."""
    z = np.full(K * 8, 1.0 / (K * 8))
    mats = []
    N = 0.0
    for U, cnt in ev:
        n = cnt.sum()
        if n <= 0:
            continue
        A = sub_matrix(U)
        M = np.zeros((K * A.shape[0], K * 8))
        for s in range(K):
            M[s * A.shape[0]:(s + 1) * A.shape[0], s * 8:(s + 1) * 8] = A
        mats.append((M, cnt.reshape(-1)))
        N += n
    if not mats:
        return z
    for _ in range(iters):
        exp = np.zeros(K * 8)
        for M, c in mats:
            cur = M @ z
            exp += z * (M.T @ np.where(cur > 0, c / np.maximum(cur, 1e-300), 0.0))
        znew = exp / N
        if np.abs(znew - z).max() < tol:
            z = znew
            break
        z = znew
    return z


def bound_claim(ev, K, sigma, alpha=0.05):
    """Returns dict with conf_lo/hi, ident_lo/hi, maxent, lp_status."""
    rows = _constraint_rows(ev, K)
    obj = np.zeros(K * 8)
    obj[[s * 8 + sigma for s in range(K)]] = 1
    A_eq, b_eq = np.ones((1, K * 8)), np.array([1.0])
    zme = maxent_fit(ev, K)
    out = {"maxent": float(obj @ zme), "n_events": len(rows)}
    if not rows:
        out.update(conf_lo=0.0, conf_hi=1.0, ident_lo=0.0, ident_hi=1.0, lp_status="no_evidence")
        return out
    R = np.vstack([r for r, _, _ in rows])
    k = np.array([c for _, c, _ in rows])
    n = np.array([nn for _, _, nn in rows])
    lo, hi = cp_interval(k, n, alpha / len(rows))
    res = _lp_minmax(np.vstack([R, -R]), np.concatenate([hi, -lo]), A_eq, b_eq, obj)
    if res is None:
        out.update(conf_lo=np.nan, conf_hi=np.nan, lp_status="incompatible_margins")
    else:
        out.update(conf_lo=float(res[0]), conf_hi=float(res[1]), lp_status="ok")
    pt = R @ zme
    res2 = _lp_minmax(None, None, np.vstack([A_eq, R]), np.concatenate([b_eq, pt]), obj)
    if res2 is None:
        out.update(ident_lo=np.nan, ident_hi=np.nan)
    else:
        out.update(ident_lo=float(res2[0]), ident_hi=float(res2[1]))
    return out


def conf_bounds_all(ev, K, alpha=0.05):
    """Confidence LP bounds [lo, hi] for all 8 patterns (16 LPs, no identification LPs)."""
    rows = _constraint_rows(ev, K)
    lo, hi = np.zeros(8), np.ones(8)
    if not rows:
        return lo, hi
    R = np.vstack([r for r, _, _ in rows])
    k = np.array([c for _, c, _ in rows])
    n = np.array([nn for _, _, nn in rows])
    clo, chi = cp_interval(k, n, alpha / len(rows))
    A_ub, b_ub = np.vstack([R, -R]), np.concatenate([chi, -clo])
    A_eq, b_eq = np.ones((1, K * 8)), np.array([1.0])
    for s in range(8):
        obj = np.zeros(K * 8)
        obj[[t * 8 + s for t in range(K)]] = 1
        res = _lp_minmax(A_ub, b_ub, A_eq, b_eq, obj)
        if res is None:
            return np.zeros(8), np.ones(8)
        lo[s], hi[s] = res
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def max_binom_var(lo, hi):
    """max p(1-p) over p in [lo, hi] (elementwise)."""
    p = np.clip(0.5, lo, hi)
    return p * (1 - p)


def direct_estimate(ev, sigma, alpha=0.05):
    for U, cnt in ev:
        if U == (0, 1, 2):
            c = cnt.sum(0)
            n = c.sum()
            lo, hi = cp_interval(c[sigma], n, alpha)
            return {"n_direct": float(n), "direct": float(c[sigma] / n) if n else np.nan,
                    "direct_lo": float(lo), "direct_hi": float(hi)}
    return {"n_direct": 0.0, "direct": np.nan, "direct_lo": np.nan, "direct_hi": np.nan}


def cond_indep_estimate(B, S, in_parent, strat, K, measured, sigma):
    """P(X_S=sigma|G) = sum_s P(s|G) prod_q P(x_q|s) using every cell that measured marker q."""
    bits = PAT[sigma]
    ps = np.bincount(strat[in_parent], minlength=K).astype(float)
    ps /= ps.sum()
    est = 0.0
    for s in range(K):
        sel_s = in_parent & (strat == s)
        prod = ps[s]
        for q, m in enumerate(S):
            sel = sel_s & measured[:, m]
            if not sel.any():
                prod *= 0.5
                continue
            p1 = B[sel, m].mean()
            prod *= p1 if bits[q] else (1 - p1)
        est += prod
    return float(est)


# ---------------------------------------------------------------- per-tube consistency (before pooling)
def tube_consistency(B, S, in_parent, tube, panels):
    """For every pair of tubes that both measure a common nonempty subset U of S, compare the
    within-parent distribution of the U sub-pattern (chi-square test of homogeneity) and report the
    smallest p-value and the largest total-variation distance.  Also compares parent fractions.
    A conflict means 'pooling premise questionable' -- it is a flag, not a correction."""
    from scipy.stats import chi2_contingency
    code = codes(B, S)
    minp, maxtv, n_checks = 1.0, 0.0, 0
    tubes = [j for j in range(len(panels)) if (tube == j).any()]
    for a in range(len(tubes)):
        for b in range(a + 1, len(tubes)):
            ja, jb = tubes[a], tubes[b]
            U = [q for q, m in enumerate(S) if m in panels[ja] and m in panels[jb]]
            if not U:
                continue
            tab = []
            for j in (ja, jb):
                sel = (tube == j) & in_parent
                bits = PAT[code[sel]][:, U]
                sub = np.zeros(sel.sum(), np.int64)
                for q in range(len(U)):
                    sub = sub * 2 + bits[:, q]
                tab.append(np.bincount(sub, minlength=2 ** len(U)))
            tab = np.array(tab, float)
            keep = tab.sum(0) > 0
            if keep.sum() < 2 or (tab.sum(1) == 0).any():
                continue
            p = chi2_contingency(tab[:, keep])[1]
            pr = tab / tab.sum(1, keepdims=True)
            minp = min(minp, float(p))
            maxtv = max(maxtv, float(0.5 * np.abs(pr[0] - pr[1]).sum()))
            n_checks += 1
    # parent fraction per tube
    fr = [in_parent[tube == j].mean() for j in tubes]
    if n_checks == 0:  # nothing comparable: 'not checkable', never a pass
        minp, maxtv = np.nan, np.nan
    return {"tube_minp": minp, "tube_maxtv": maxtv, "tube_checks": n_checks,
            "parent_frac_range": float(max(fr) - min(fr)) if fr else 0.0}


# ---------------------------------------------------------------- kNN imputation baselines
def measured_matrix(n, v, tube, panels):
    M = np.zeros((n, v), bool)
    for j, P in enumerate(panels):
        sel = tube == j
        M[np.ix_(sel, sorted(P))] = True
    return M


def knn_impute(B, Xbb, tube, panels, k=10, mode="majority", seed=0):
    """mode='majority': per missing marker, majority of k backbone neighbours (from tubes that
    measured it).  mode='block': per (recipient tube, donor tube), copy the whole block of the
    donor tube's markers from ONE randomly chosen neighbour among the k nearest."""
    rng = np.random.default_rng(seed)
    n, v = B.shape
    have = measured_matrix(n, v, tube, panels)
    Bimp = np.where(have, B, 0).astype(B.dtype)  # hidden truth is masked before imputation
    used = set().union(*panels)
    for j in range(len(panels)):
        rec = np.flatnonzero(tube == j)
        if len(rec) == 0:
            continue
        for jd, Pd in enumerate(panels):
            if jd == j:
                continue
            block = sorted(set(Pd) - set(panels[j]))
            if not block:
                continue
            don = np.flatnonzero(tube == jd)
            if len(don) == 0:
                continue
            nn = NearestNeighbors(n_neighbors=min(k, len(don))).fit(Xbb[don])
            _, idx = nn.kneighbors(Xbb[rec])
            # only fill markers not already filled from an earlier donor tube
            todo = [m for m in block if not have[rec[0], m]]
            if not todo:
                continue
            if mode == "majority":
                for m in todo:
                    Bimp[rec, m] = B[don[idx], m].mean(1) >= 0.5
            else:
                pick = don[idx[np.arange(len(rec)), rng.integers(0, idx.shape[1], len(rec))]]
                for m in todo:
                    Bimp[rec, m] = B[pick, m]
            have[np.ix_(rec, todo)] = True
    missing = [m for m in sorted(used) if not have[:, m].all()]
    if missing:
        raise ValueError(f"knn_impute: markers without any donor tube for some cells: {missing}")
    return Bimp


def knn_impute_within(B, Xbb, tube, panels, groups, k=10, mode="block", seed=0):
    """Same as knn_impute but donors restricted to the recipient's own group (e.g. the same donor/sample)."""
    Bimp = np.zeros_like(B)
    for g in np.unique(groups):
        s = groups == g
        Bimp[s] = knn_impute(B[s], Xbb[s], tube[s], panels, k=k, mode=mode, seed=seed)
    return Bimp


def imputed_frequency(Bimp, S, in_parent, sigma):
    c = codes(Bimp[in_parent], S)
    return float((c == sigma).mean()) if len(c) else np.nan
