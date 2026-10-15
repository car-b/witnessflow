"""P2 / e5e: 2-fold covering C_2(20,6,3) with r blocks (spec v2).  Feasible baseline: 142 = two stacked
La Jolla C(20,6,3) copies; counting lower bound 117.  Goal: any certificate with r <= 141.

Methods (same wall-clock budget, single thread):
  anneal   : start = La Jolla stack truncated to r blocks; budget.anneal_lambda on sum min(lam,2) with
             restarts until time is up.
  breakout : weighted local search (breakout style; our own implementation of the breakout weighting idea used in search round R4):
             focus on a random deficient triple, try all 'swap one point of a block that already holds 2 of
             the triple's points for the missing point', take the best weighted move; at local minima add
             +1 weight to every deficient triple.
Certificate: every block has 6 distinct points in 0..19 and every one of the 1140 triples has lam >= 2,
verified independently (itertools enumeration).
usage: python e5e.py <method> <r> <seed> <seconds> <out_dir>
"""
import itertools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget, design  # noqa: E402

V, K, LREQ = 20, 6, 2
TRI = list(itertools.combinations(range(V), 3))
TIDX = {t: i for i, t in enumerate(TRI)}
BY_POINT = {p: [i for i, t in enumerate(TRI) if p in t] for p in range(V)}


def lam_of(blocks):
    lam = np.zeros(len(TRI), int)
    for b in blocks:
        for t in itertools.combinations(sorted(b), 3):
            lam[TIDX[t]] += 1
    return lam


def verify(blocks):
    if any(len(set(b)) != K or not all(0 <= x < V for x in b) for b in blocks):
        return False, -1
    lam = lam_of(blocks)
    return bool(lam.min() >= LREQ), int(lam.min())


def ljcr_stack_start(r, seed):
    base = design.ljcr_blocks(V, K, 3)
    rng = np.random.default_rng(seed)
    blocks = []
    while len(blocks) < r:
        perm = rng.permutation(V)
        for b in base:
            if len(blocks) < r:
                blocks.append(set(int(perm[x]) for x in b))
    return blocks


def run_anneal(r, seed, seconds):
    t0 = time.time()
    tarr = np.array(TRI)
    best_def, best = None, None
    s = seed
    while time.time() - t0 < seconds:
        start = ljcr_stack_start(r, s)
        wells, _ = budget.anneal_lambda(start, list(range(V)), tarr, np.ones(len(TRI)), LREQ, V,
                                        iters=40000, seed=s, t0=0.5)
        d = int(np.maximum(0, LREQ - lam_of(wells)).sum())
        if best_def is None or d < best_def:
            best_def, best = d, wells
        if d == 0:
            break
        s += 1000
    return best, best_def


def run_breakout(r, seed, seconds):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    blocks = [set(b) for b in ljcr_stack_start(r, seed)]
    lam = lam_of(blocks)
    w = np.ones(len(TRI))
    best_def = int(np.maximum(0, LREQ - lam).sum())
    best = [set(b) for b in blocks]
    while time.time() - t0 < seconds and best_def > 0:
        deficient = np.flatnonzero(lam < LREQ)
        if len(deficient) == 0:
            break
        T = TRI[int(rng.choice(deficient))]
        best_move, best_gain = None, 0.0
        for bi, B in enumerate(blocks):
            inside = [p for p in T if p in B]
            if len(inside) != 2:
                continue
            miss = [p for p in T if p not in B][0]
            for out in B:
                if out in T:
                    continue
                # gain = weighted deficiency reduction of swapping out -> miss in block bi
                B2 = (B - {out}) | {miss}
                lose = [TIDX[tuple(sorted(t))] for t in itertools.combinations(sorted(B), 3) if out in t]
                gain_ = [TIDX[tuple(sorted(t))] for t in itertools.combinations(sorted(B2), 3) if miss in t]
                g = 0.0
                for i in lose:
                    if lam[i] <= LREQ:
                        g -= w[i]
                for i in gain_:
                    if lam[i] < LREQ:
                        g += w[i]
                if g > best_gain + 1e-12 or (best_move is None and g >= 0 and rng.random() < 0.05):
                    best_gain, best_move = g, (bi, out, miss)
        if best_move is not None and best_gain > 0:
            bi, out, miss = best_move
            B = blocks[bi]
            for t in itertools.combinations(sorted(B), 3):
                if out in t:
                    lam[TIDX[t]] -= 1
            B.discard(out)
            B.add(miss)
            for t in itertools.combinations(sorted(B), 3):
                if miss in t:
                    lam[TIDX[t]] += 1
            d = int(np.maximum(0, LREQ - lam).sum())
            if d < best_def:
                best_def, best = d, [set(b) for b in blocks]
        else:
            w[lam < LREQ] += 1.0  # breakout: penalize currently deficient triples
    return best, best_def


if __name__ == "__main__":
    method, r, seed, seconds, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    blocks, d = (run_anneal if method == "anneal" else run_breakout)(r, seed, seconds)
    ok, lmin = verify([sorted(b) for b in blocks])
    res = dict(method=method, r=r, seed=seed, seconds=seconds, wall=time.time() - t0, deficiency=d,
               certificate=ok, lam_min=lmin, blocks=[sorted(int(x) for x in b) for b in blocks])
    json.dump(res, open(os.path.join(out, f"{method}_r{r}_s{seed}.json"), "w"))
    print(method, r, seed, "deficiency", d, "certificate", ok, "lam_min", lmin, flush=True)
