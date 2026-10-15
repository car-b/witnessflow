"""Follow-up tube design: weighted maximum coverage of target marker triples.

A design is a list of r tubes; each tube is a set of <= c candidate (non-backbone) markers
(the backbone is implicitly in every tube).  Target S (a triple of marker indices) is covered
iff S ⊆ tube for some tube.  Objective: total weight of covered targets.
"""
import itertools
import os
import sqlite3
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

LJCR_DB = os.environ.get("WITNESSFLOW_LJCR_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ljcr_snapshot_subset.sqlite"))


# ------------------------------------------------------------------ evaluation / verifier
def covered_mask(tubes, targets):
    """Independent verifier via bitmasks."""
    tm = [sum(1 << m for m in T) for T in tubes]
    out = np.zeros(len(targets), bool)
    for i, S in enumerate(targets):
        sm = (1 << S[0]) | (1 << S[1]) | (1 << S[2])
        out[i] = any((t & sm) == sm for t in tm)
    return out


def score(tubes, targets, w):
    return float(np.asarray(w)[covered_mask(tubes, targets)].sum())


def check_design(tubes, cand, c):
    cs = set(cand)
    for T in tubes:
        assert len(T) <= c, f"tube over capacity: {sorted(T)}"
        assert set(T) <= cs, f"non-candidate marker in tube: {sorted(set(T) - cs)}"


# ------------------------------------------------------------------ baselines
def random_design(cand, c, r, seed=0):
    rng = np.random.default_rng(seed)
    return [set(rng.choice(cand, c, replace=False).tolist()) for _ in range(r)]


def naive_packing(cand, c, r, targets, w):
    """'one question at a time': fill each tube with the markers of the heaviest still-uncovered
    targets while capacity allows (no look-ahead for overlaps)."""
    order = np.argsort(-np.asarray(w))
    covered = np.zeros(len(targets), bool)
    tubes = []
    for _ in range(r):
        T = set()
        for i in order:
            if covered[i]:
                continue
            need = set(targets[i]) - T
            if len(T) + len(need) <= c:
                T |= need
        tubes.append(T)
        covered |= covered_mask([T], targets)
    return tubes


def ljcr_blocks(v, k, t=3, db=LJCR_DB):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute("select size, blocks from covers where v=? and k=? and t=?", (v, k, t)).fetchone()
    con.close()
    if row is None:
        return None
    size, blob = row
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(size, k)
    return [set(map(int, b)) for b in arr]


def uniform_design(cand, c, r, seed=0):
    """r blocks drawn from the best known uniform covering C(|cand|, c, 3) (La Jolla table),
    relabelled onto the candidate markers at random.  Ignores claim weights by construction."""
    blocks = ljcr_blocks(len(cand), c, 3)
    if blocks is None:
        return None, None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cand))
    lab = [cand[i] for i in perm]
    pick = rng.choice(len(blocks), min(r, len(blocks)), replace=False)
    return [set(lab[x] for x in blocks[i]) for i in pick], len(blocks)


# ------------------------------------------------------------------ greedy + local search
class Cov:
    """Incremental coverage state for swap moves."""

    def __init__(self, tubes, cand, targets, w):
        self.tubes = [set(T) for T in tubes]
        self.targets = [tuple(S) for S in targets]
        self.w = np.asarray(w, float)
        self.by_m = {m: [] for m in cand}
        for i, S in enumerate(self.targets):
            for m in S:
                self.by_m.setdefault(m, []).append(i)
        self.cnt = np.zeros(len(targets), int)  # number of tubes covering each target
        for T in self.tubes:
            for i in self._inside(T):
                self.cnt[i] += 1
        self.total = float(self.w[self.cnt > 0].sum())

    def _inside(self, T):
        seen = set()
        for m in T:
            for i in self.by_m.get(m, ()):
                if i not in seen and all(x in T for x in self.targets[i]):
                    seen.add(i)
        return seen

    def delta_swap(self, ti, a, b):
        T = self.tubes[ti]
        d = 0.0
        for i in self.by_m.get(a, ()):
            S = self.targets[i]
            if all(x in T for x in S) and self.cnt[i] == 1:
                d -= self.w[i]
        for i in self.by_m.get(b, ()):
            S = self.targets[i]
            if a in S:
                continue
            if all((x in T) or x == b for x in S) and self.cnt[i] == 0:
                d += self.w[i]
        return d

    def apply_swap(self, ti, a, b):
        T = self.tubes[ti]
        for i in self.by_m.get(a, ()):
            if all(x in T for x in self.targets[i]):
                self.cnt[i] -= 1
        T.discard(a)
        T.add(b)
        for i in self.by_m.get(b, ()):
            if all(x in T for x in self.targets[i]):
                self.cnt[i] += 1
        self.total = float(self.w[self.cnt > 0].sum())


def greedy_design(cand, c, r, targets, w):
    w = np.asarray(w, float)
    covered = np.zeros(len(targets), bool)
    tubes = []
    tset = [set(S) for S in targets]
    for _ in range(r):
        T = set()
        while len(T) < c:
            best, bm = None, None
            for m in cand:
                if m in T:
                    continue
                T2 = T | {m}
                full = part = 0.0
                for i, S in enumerate(tset):
                    if covered[i]:
                        continue
                    k = len(S & T2)
                    if k == 3:
                        full += w[i]
                    elif k == 2 and m in S:
                        part += w[i]
                key = (full, part)
                if best is None or key > best:
                    best, bm = key, m
            T.add(bm)
        tubes.append(T)
        covered |= covered_mask([T], targets)
    return tubes


def anneal(tubes, cand, c, targets, w, iters=20000, t0=None, seed=0):
    rng = np.random.default_rng(seed)
    st = Cov(tubes, cand, targets, w)
    best_total, best = st.total, [set(T) for T in st.tubes]
    wpos = np.asarray(w, float)
    t0 = t0 if t0 is not None else (wpos[wpos > 0].mean() if (wpos > 0).any() else 1.0)
    cand = list(cand)
    for it in range(iters):
        temp = t0 * (1 - it / iters) + 1e-9
        ti = int(rng.integers(len(st.tubes)))
        T = st.tubes[ti]
        if not T:
            continue
        a = list(T)[int(rng.integers(len(T)))]
        b = cand[int(rng.integers(len(cand)))]
        if b in T:
            continue
        d = st.delta_swap(ti, a, b)
        if d >= 0 or rng.random() < np.exp(d / temp):
            st.apply_swap(ti, a, b)
            if st.total > best_total + 1e-12:
                best_total, best = st.total, [set(X) for X in st.tubes]
    return best, best_total


def ilp_design(cand, c, r, targets, w, time_limit=60):
    """Exact weighted max coverage (small instances).  Vars: y[i,m], u[i,S], z[S]."""
    cand = list(cand)
    M, T = len(cand), len(targets)
    pos = {m: j for j, m in enumerate(cand)}
    ny, nu, nz = r * M, r * T, T
    N = ny + nu + nz
    Y = lambda i, j: i * M + j
    U = lambda i, s: ny + i * T + s
    Z = lambda s: ny + nu + s
    rows, lb, ub = [], [], []

    def add(coefs, lo, hi):
        rows.append(coefs)
        lb.append(lo)
        ub.append(hi)

    for i in range(r):
        add({Y(i, j): 1 for j in range(M)}, -np.inf, c)
        for s, S in enumerate(targets):
            for m in S:
                add({U(i, s): 1, Y(i, pos[m]): -1}, -np.inf, 0)
    for s in range(T):
        d = {Z(s): 1}
        for i in range(r):
            d[U(i, s)] = -1
        add(d, -np.inf, 0)
    A = np.zeros((len(rows), N))
    for k, d in enumerate(rows):
        for j, v in d.items():
            A[k, j] = v
    cvec = np.zeros(N)
    for s in range(T):
        cvec[Z(s)] = -w[s]
    res = milp(cvec, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(N),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, None, res
    x = np.round(res.x).astype(int)
    tubes = [set(cand[j] for j in range(M) if x[Y(i, j)]) for i in range(r)]
    return tubes, -res.fun, res
