"""E3: witness design under a fixed follow-up cell budget.

N cells are split evenly over r wells; well i measures backbone + X_i (|X_i| = c).  For a parent G
with fraction pi_G, triple T receives  n_T = sum_{i: T ⊆ X_i} (N/r) pi_G  parent cells.
A triple is 'witnessed to precision eps' if n_T >= n_req = z^2 * 0.25 / eps^2 (worst-case p(1-p)).
So the requirement is lambda_T >= lam_req(r) = ceil(n_req * r / (N * pi_G)): what matters is the
*fraction of wells* containing each triple -> balance, not mere coverage.
"""
import itertools
import numpy as np
import design


def lam_counts(wells, tarr, v):
    """tarr: int array [T,3] of marker ids; wells: list of sets."""
    lam = np.zeros(len(tarr), int)
    for X in wells:
        m = np.zeros(v, bool)
        m[list(X)] = True
        lam += m[tarr].all(1)
    return lam


def balanced_random(cand, c, r, seed=0):
    """Every marker appears (almost) equally often: concatenate random permutations, cut into wells
    (re-drawing a well if it would repeat a marker)."""
    rng = np.random.default_rng(seed)
    wells, stream = [], []
    while len(wells) < r:
        if len(stream) < c:
            stream += list(rng.permutation(cand))
        X = []
        i = 0
        while len(X) < c and i < len(stream):
            if stream[i] not in X:
                X.append(stream.pop(i))
            else:
                i += 1
        wells.append(set(int(x) for x in X))
    return wells


def ljcr_stack(cand, c, r, seed=0):
    """Concatenate independently relabelled copies of the best known covering C(v,c,3), truncated
    to r wells -> lambda >= floor(r / |cover|) for every triple."""
    blocks = design.ljcr_blocks(len(cand), c, 3)
    if blocks is None:
        return None
    rng = np.random.default_rng(seed)
    wells = []
    while len(wells) < r:
        perm = rng.permutation(len(cand))
        lab = [cand[i] for i in perm]
        for b in blocks:
            if len(wells) < r:
                wells.append(set(lab[x] for x in b))
    return wells


def packing_deficit(cand, c, r, tarr, w, need_cells, cells_per_well):
    """'One question at a time' with repeated coverage: each new well packs the markers of the
    targets with the largest remaining cell deficit (ties: weight) while capacity allows; every
    target fully inside the well then has its deficit reduced by the expected parent cells/well."""
    w = np.asarray(w, float)
    deficit = np.broadcast_to(np.asarray(need_cells, float), (len(tarr),)).copy()  # scalar or per-target
    wells = []
    for _ in range(r):
        order = np.lexsort((-w, -deficit))
        X = set()
        for t in order:
            if deficit[t] <= 0:
                break
            new = set(tarr[t].tolist()) - X
            if len(X) + len(new) <= c:
                X |= new
            if len(X) == c:
                break
        if len(X) < c:  # fill leftover capacity with the most frequent markers among open targets
            open_t = tarr[deficit > 0]
            cnt = np.bincount(open_t.ravel(), minlength=max(cand) + 1) if len(open_t) else np.zeros(max(cand) + 1)
            for m in np.argsort(-cnt):
                if len(X) == c:
                    break
                if m in cand and m not in X:
                    X.add(int(m))
            for m in cand:
                if len(X) == c:
                    break
                X.add(m)
        m = np.zeros(max(cand) + 1, bool)
        m[list(X)] = True
        inside = m[tarr].all(1)
        deficit[inside] -= cells_per_well
        wells.append(X)
    return wells


def greedy_lambda(cand, c, r, tarr, w, lam_req, v):
    """Add wells one at a time; build each well marker by marker maximizing the weighted deficit
    reduction  sum_T w_T [lam_T < lam_req] * 1[T ⊆ X]  (ties: pairs-in-X potential)."""
    w = np.asarray(w, float)
    lam = np.zeros(len(tarr), int)
    wells = []
    for _ in range(r):
        need = w * (lam < lam_req)
        X = np.zeros(v, bool)
        for _k in range(c):
            best, bm = None, None
            for m in cand:
                if X[m]:
                    continue
                X[m] = True
                inX = X[tarr]
                full = need[inX.all(1)].sum()
                part = need[(inX.sum(1) == 2) & (tarr == m).any(1)].sum()
                X[m] = False
                key = (full, part)
                if best is None or key > best:
                    best, bm = key, m
            X[bm] = True
        wells.append(set(np.flatnonzero(X).tolist()))
        lam += X[tarr].all(1)
    return wells


def anneal_lambda(wells, cand, tarr, w, lam_req, v, iters=30000, seed=0, t0=None):
    """Swap search maximizing  sum_T w_T * min(lam_T, lam_req)  (= weighted progress to the
    requirement; its maximum lam_req * sum w  means every triple is witnessed to precision)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    lam_req = np.broadcast_to(np.asarray(lam_req), (len(tarr),))  # scalar or per-target requirement
    by_m = {m: np.flatnonzero((tarr == m).any(1)) for m in cand}
    W = [np.zeros(v, bool) for _ in wells]
    for X, S in zip(W, wells):
        X[list(S)] = True
    lam = np.zeros(len(tarr), int)
    for X in W:
        lam += X[tarr].all(1)
    score = float((w * np.minimum(lam, lam_req)).sum())
    best, best_s = [X.copy() for X in W], score
    t0 = max(w.mean(), 1e-9) if t0 is None else t0  # pass t0 explicitly when w mixes scales
    cand = list(cand)
    for it in range(iters):
        temp = t0 * (1 - it / iters) + 1e-9
        i = int(rng.integers(len(W)))
        X = W[i]
        ins = np.flatnonzero(X)
        a = int(ins[rng.integers(len(ins))])
        b = cand[int(rng.integers(len(cand)))]
        if X[b]:
            continue
        ta = by_m[a]
        lose = ta[X[tarr[ta]].all(1)]
        X[a] = False
        X[b] = True
        tb = by_m[b]
        gain = tb[X[tarr[tb]].all(1)]
        lam2 = lam.copy()
        lam2[lose] -= 1
        lam2[gain] += 1
        idx = np.union1d(lose, gain)
        d = float((w[idx] * (np.minimum(lam2[idx], lam_req[idx]) - np.minimum(lam[idx], lam_req[idx]))).sum())
        if d >= 0 or rng.random() < np.exp(d / temp):
            lam = lam2
            score += d
            if score > best_s + 1e-12:
                best_s, best = score, [Y.copy() for Y in W]
        else:
            X[b] = False
            X[a] = True
    return [set(np.flatnonzero(X).tolist()) for X in best], best_s
