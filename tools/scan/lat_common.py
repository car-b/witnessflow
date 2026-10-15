"""Shared helpers for the lat_* scans (read-only analysis; outputs go to tools/scan/out_lat/)."""
import itertools, os, sys
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(os.path.dirname(HERE))          # .../witnessflow
SRC = os.path.join(WF, "src")
OUT = os.path.join(HERE, "out_lat")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)
import design, budget  # noqa: E402,F401


def triples(v):
    return np.array(list(itertools.combinations(range(v), 3)))


def lam_vec(blocks, v, tarr=None):
    tarr = triples(v) if tarr is None else tarr
    lam = np.zeros(len(tarr), int)
    for X in blocks:
        m = np.zeros(v, bool)
        m[list(X)] = True
        lam += m[tarr].all(1)
    return lam


def pair_lam(blocks, v):
    P = np.zeros((v, v), int)
    for X in blocks:
        X = sorted(X)
        for a, b in itertools.combinations(X, 2):
            P[a, b] += 1
            P[b, a] += 1
    return P[np.triu_indices(v, 1)]


def degrees(blocks, v):
    d = np.zeros(v, int)
    for X in blocks:
        for x in X:
            d[x] += 1
    return d


def intersections(blocks):
    B = [set(X) for X in blocks]
    out = []
    for i, j in itertools.combinations(range(len(B)), 2):
        out.append(len(B[i] & B[j]))
    return np.array(out)


def summarize_lam(lam, r, v, k):
    bound = comb(k, 3) / comb(v, 3)
    return dict(r=r, min=int(lam.min()), max=int(lam.max()), mean=float(lam.mean()),
                cv=float(lam.std() / lam.mean()) if lam.mean() > 0 else np.nan,
                frac0=float((lam == 0).mean()), min_share=float(lam.min() / r),
                bound=bound, eff=float(lam.min() / r / bound),
                best_min_lam_possible=int(np.floor(r * bound + 1e-9)))
