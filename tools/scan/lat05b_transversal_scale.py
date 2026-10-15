"""Part 5 scale test: does annealing still find a perfect transversal design when the instance grows?
OA(q^3; 3, k, q) from a conic in PG(2,q), q prime (q=5: k=6, 30 antibodies, 125 tubes;
q=7: k=8, 56 antibodies, 343 tubes).  Search baseline = same annealer as lat05 (one-entry moves).
Output: out_lat/lat05b_scale.tsv"""
import itertools, os, time
import numpy as np
import pandas as pd
from lat_common import OUT


def oa_conic(q, k):
    pts = [(1, t, (t * t) % q) for t in range(q)] + [(0, 0, 1)]
    pts = pts[:k]
    for P, Q, R in itertools.combinations(pts, 3):
        assert round(np.linalg.det(np.array([P, Q, R], float))) % q != 0
    return np.array([[(a * x + b * y + c * z) % q for (x, y, z) in pts]
                     for a, b, c in itertools.product(range(q), repeat=3)])


def lam_cross(A, q, colt):
    lam = np.zeros((len(colt), q ** 3), int)
    for ci, (i, j, k) in enumerate(colt):
        lam[ci] += np.bincount(A[:, i] * q * q + A[:, j] * q + A[:, k], minlength=q ** 3)
    return lam


def anneal(r, q, k, iters, seed, A0=None):
    colt = list(itertools.combinations(range(k), 3))
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, (r, k)) if A0 is None else A0.copy()
    lam = lam_cross(A, q, colt)
    cov = int((lam > 0).sum())
    best, total = cov, lam.size
    cols_of = {j: [(ci, T.index(j)) for ci, T in enumerate(colt) if j in T] for j in range(k)}
    for it in range(iters):
        temp = (1 - it / iters) + 1e-9
        i, j = int(rng.integers(r)), int(rng.integers(k))
        old, new = A[i, j], int(rng.integers(q))
        if new == old:
            continue
        d, ch = 0, []
        for ci, pos in cols_of[j]:
            a, b, c = colt[ci]
            vals = [A[i, a], A[i, b], A[i, c]]
            vo = vals[0] * q * q + vals[1] * q + vals[2]
            vals[pos] = new
            vn = vals[0] * q * q + vals[1] * q + vals[2]
            d += int(lam[ci, vn] == 0) - int(lam[ci, vo] == 1)
            ch.append((ci, vo, vn))
        if d >= 0 or rng.random() < np.exp(d / temp):
            for ci, vo, vn in ch:
                lam[ci, vo] -= 1
                lam[ci, vn] += 1
            A[i, j] = new
            cov += d
            if cov > best:
                best = cov
                if best == total:
                    return best, total, it
    return best, total, iters


def main():
    t0 = time.time()
    rows = []
    for q, k, iters_list in ((5, 6, (300_000, 2_000_000)), (7, 8, (1_000_000, 4_000_000))):
        A = oa_conic(q, k)
        colt = list(itertools.combinations(range(k), 3))
        lam = lam_cross(A, q, colt)
        r = q ** 3
        p = len(colt) / lam.size
        rows.append(dict(q=q, k=k, markers=q * k, targets=lam.size, r=r, design="OA (conic)",
                         covered=float((lam > 0).mean()), lam_min=int(lam.min()), lam_max=int(lam.max())))
        rows.append(dict(q=q, k=k, markers=q * k, targets=lam.size, r=r, design="random transversal (expected)",
                         covered=1 - (1 - 1 / q ** 3) ** r))
        print(f"q={q} k={k}: OA lam {lam.min()}..{lam.max()} over {lam.size} cross-family triples, r={r}", flush=True)
        for iters in iters_list:
            for s in range(2):
                best, total, it = anneal(r, q, k, iters, s)
                rows.append(dict(q=q, k=k, markers=q * k, targets=total, r=r, design=f"anneal {iters} iters seed {s}",
                                 covered=best / total, uncovered=total - best, iters_used=it,
                                 seconds=round(time.time() - t0)))
                print(f"   anneal iters={iters} seed={s}: {best}/{total} (stopped at it={it})  [{time.time() - t0:.0f}s]",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lat05b_scale.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
