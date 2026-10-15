"""Q1: third-order structure of a 2x2x2 table beyond its pairwise margins.

H3 = KL(p || q2), q2 = max-entropy table with the same pairwise margins (connected information,
Schneidman et al. 2003).  Exact conditional test of 'no three-way interaction': with all pairwise
margins fixed, n111 determines the table; P(table) ∝ 1/prod(n_x!).
"""
import numpy as np
from scipy.special import gammaln
import audit


def table(B, S, mask):
    c = audit.codes(B[mask], S)
    return np.bincount(c, minlength=8).astype(float)


def pairwise_maxent(n):
    p = n / n.sum()
    ev = [((0, 1), audit.sub_matrix((0, 1)) @ p), ((0, 2), audit.sub_matrix((0, 2)) @ p),
          ((1, 2), audit.sub_matrix((1, 2)) @ p)]
    q = np.full(8, 1 / 8)
    mats = [(audit.sub_matrix(U), m) for U, m in ev]
    for _ in range(2000):
        old = q.copy()
        for A, m in mats:
            cur = A @ q
            q = q * (A.T @ np.where(cur > 0, m / np.maximum(cur, 1e-300), 0.0))
            q /= q.sum()
        if np.abs(q - old).max() < 1e-13:
            break
    return q


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))


def exact_test(n):
    """Two-sided exact conditional p-value for no 3-way interaction. n: counts by code."""
    n = np.asarray(n, int)
    N = n.sum()
    # margins (code = a*4 + b*2 + c)
    A1 = n[4:].sum(); B1 = n[[2, 3, 6, 7]].sum(); C1 = n[1::2].sum()
    AB = n[6] + n[7]; AC = n[5] + n[7]; BC = n[3] + n[7]
    a = np.arange(0, N + 1)
    cells = np.stack([
        N - A1 - B1 - C1 + AB + AC + BC - a,  # 000
        C1 - AC - BC + a,                     # 001
        B1 - AB - BC + a,                     # 010
        BC - a,                               # 011
        A1 - AB - AC + a,                     # 100
        AC - a,                               # 101
        AB - a,                               # 110
        a])                                   # 111
    ok = (cells >= 0).all(0)
    cells = cells[:, ok]
    logw = -gammaln(cells + 1).sum(0)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    obs = np.flatnonzero(cells[7] == n[7])[0]
    return float(w[w <= w[obs] * (1 + 1e-9)].sum())


def q1_stats(n, sigma):
    p = n / n.sum()
    q2 = pairwise_maxent(n)
    return {"N": float(n.sum()), "H3": kl(p, q2), "p_sigma": float(p[sigma]), "q2_sigma": float(q2[sigma]),
            "delta_pp": float(100 * (p[sigma] - q2[sigma])), "p_exact": exact_test(n)}
