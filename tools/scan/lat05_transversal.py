"""Part 5 (combinatorial): channel families -> transversal design / orthogonal array from finite geometry.

Setting: 24 antibodies = 6 channel families (e.g. 6 fluorochromes / metal groups) x 4 antibodies each.
A tube takes exactly one antibody per family (6 markers per tube, as in WitnessFlow).  Targets = all
cross-family triples (C(6,3)*4^3 = 1280).  Same-family triples can never be co-measured.

Construction: hyperoval in PG(2,4) (6 points, no 3 collinear) -> OA(64; 3, 6, 4) of index 1:
row (a,b,c) in GF(4)^3, column P=(x,y,z): entry a x + b y + c z.  Every cross-family triple lies in
exactly one of 64 tubes (= the counting lower bound 1280/20).
Baselines: random transversal tubes; simulated annealing search on the same tube format.
Outputs: out_lat/lat05_transversal.tsv, out_lat/lat05_stdout.txt, out_lat/oa64_tubes.txt"""
import itertools, os, time
from math import comb
import numpy as np
import pandas as pd
from lat_common import OUT, design

# ---------------------------------------------------------------- GF(4)
EXP = [1, 2, 3]
LOG = {1: 0, 2: 1, 3: 2}


def mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[(LOG[a] + LOG[b]) % 3]


def det3(P, Q, R):
    (a, b, c), (d, e, f), (g, h, i) = P, Q, R
    t1 = mul(a, mul(e, i) ^ mul(f, h))
    t2 = mul(b, mul(d, i) ^ mul(f, g))
    t3 = mul(c, mul(d, h) ^ mul(e, g))
    return t1 ^ t2 ^ t3


def hyperoval():
    pts = [(1, t, mul(t, t)) for t in range(4)] + [(0, 0, 1), (0, 1, 0)]
    assert all(det3(*T) != 0 for T in itertools.combinations(pts, 3)), "three collinear points"
    return pts


def oa64():
    pts = hyperoval()
    rows = []
    for a, b, c in itertools.product(range(4), repeat=3):
        rows.append([mul(a, x) ^ mul(b, y) ^ mul(c, z) for (x, y, z) in pts])
    return np.array(rows)          # [64, 6] values in 0..3


COLT = list(itertools.combinations(range(6), 3))           # 20 column triples


def lam_cross(A):
    """A: [r, 6] array of per-family choices -> lam over all 1280 cross-family triples."""
    lam = np.zeros((20, 64), int)
    for ci, (i, j, k) in enumerate(COLT):
        idx = A[:, i] * 16 + A[:, j] * 4 + A[:, k]
        lam[ci] += np.bincount(idx, minlength=64)
    return lam.ravel()


def anneal_transversal(r, iters, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 4, (r, 6))
    lam = lam_cross(A).reshape(20, 64)
    cov = int((lam > 0).sum())
    best = cov
    bestA = A.copy()
    cols_of = {j: [ci for ci, T in enumerate(COLT) if j in T] for j in range(6)}
    t0 = 1.0
    for it in range(iters):
        temp = t0 * (1 - it / iters) + 1e-9
        i, j = int(rng.integers(r)), int(rng.integers(6))
        old, new = A[i, j], int(rng.integers(4))
        if new == old:
            continue
        d = 0
        ch = []
        for ci in cols_of[j]:
            a, b, c = COLT[ci]
            vals = [A[i, a], A[i, b], A[i, c]]
            pos = (a, b, c).index(j)
            vo = vals[0] * 16 + vals[1] * 4 + vals[2]
            vals[pos] = new
            vn = vals[0] * 16 + vals[1] * 4 + vals[2]
            if lam[ci, vo] == 1:
                d -= 1
            if lam[ci, vn] == 0:
                d += 1
            ch.append((ci, vo, vn))
        if d >= 0 or rng.random() < np.exp(d / temp):
            for ci, vo, vn in ch:
                lam[ci, vo] -= 1
                lam[ci, vn] += 1
            A[i, j] = new
            cov += d
            if cov > best:
                best, bestA = cov, A.copy()
                if best == 1280:
                    break
    return bestA, best


def stats(lam, r, name):
    return dict(design=name, r=r, covered=float((lam > 0).mean()), lam_min=int(lam.min()),
                lam_max=int(lam.max()), cv=float(lam.std() / lam.mean()), frac0=float((lam == 0).mean()),
                min_share=float(lam.min() / r), share_bound=20 / 1280)


def main():
    t0 = time.time()
    rows = []
    A = oa64()
    lam = lam_cross(A)
    print(f"OA(64;3,6,4) from the PG(2,4) hyperoval: lambda over 1280 cross-family triples: "
          f"min {lam.min()} max {lam.max()}")
    rows.append(stats(lam, 64, "OA64 (hyperoval, GF(4))"))
    rows.append(stats(lam_cross(np.vstack([A, A])), 128, "OA64 x2"))
    with open(os.path.join(OUT, "oa64_tubes.txt"), "w") as f:
        for row in A:
            f.write(" ".join(f"F{j}:{v}" for j, v in enumerate(row)) + "\n")
    for r in (64, 128, 192, 256, 320):
        st = []
        for s in range(200):
            Ar = np.random.default_rng(s).integers(0, 4, (r, 6))
            st.append(stats(lam_cross(Ar), r, "random transversal"))
        d = pd.DataFrame(st).mean(numeric_only=True).to_dict()
        d.update(design="random transversal (200 seeds)", r=r)
        rows.append(d)
    p = 20 / 1280
    print(f"random transversal expected coverage at r=64: {1 - (1 - p) ** 64:.4f}; "
          f"r for 99% expected: {np.log(0.01) / np.log(1 - p):.0f}")
    for iters in (200_000, 1_000_000):
        res = []
        for s in range(3):
            Ab, best = anneal_transversal(64, iters, seed=s)
            res.append(best / 1280)
            print(f"  anneal r=64 iters={iters} seed={s}: covered {best}/1280  [{time.time() - t0:.0f}s]", flush=True)
        rows.append(dict(design=f"anneal search, {iters} iters (3 seeds; mean / best)", r=64,
                         covered=float(np.mean(res)), covered_best=float(np.max(res))))
    # smallest r at which annealing reaches full cross-family coverage (quick scan)
    for r in (66, 68, 70, 72, 76, 80):
        Ab, best = anneal_transversal(r, 300_000, seed=0)
        rows.append(dict(design="anneal search, 300k iters, seed 0", r=r, covered=best / 1280))
        print(f"  anneal r={r}: covered {best}/1280  [{time.time() - t0:.0f}s]", flush=True)
        if best == 1280:
            break
    # unconstrained reference: covering all C(24,3) triples (ignores the family constraint)
    L = design.ljcr_blocks(24, 6, 3)
    rows.append(dict(design="(reference) La Jolla C(24,6,3), all 2024 triples, ignores families",
                     r=len(L) if L else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lat05_transversal.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
