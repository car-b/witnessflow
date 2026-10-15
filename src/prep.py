"""Preprocessing: transform, binarize, backbone strata."""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def asinh(X, cofactor=5.0):
    return np.arcsinh(np.asarray(X, float) / cofactor)


def gmm_threshold(x, seed=0, max_n=200_000):
    """Threshold between the two components of a 1-D 2-GMM (posterior = 0.5 between the means).
    Returns (thr, info). info['bimodal'] is a crude separation flag (Ashman's D > 2)."""
    rng = np.random.default_rng(seed)
    xs = x if len(x) <= max_n else rng.choice(x, max_n, replace=False)
    g = GaussianMixture(2, random_state=seed).fit(xs.reshape(-1, 1))
    mu = g.means_.ravel()
    sd = np.sqrt(g.covariances_.ravel())
    lo, hi = np.argsort(mu)
    grid = np.linspace(mu[lo], mu[hi], 2001).reshape(-1, 1)
    post = g.predict_proba(grid)[:, hi]
    i = int(np.argmax(post >= 0.5))
    thr = float(grid[i, 0])
    D = np.sqrt(2) * abs(mu[hi] - mu[lo]) / np.sqrt(sd[hi] ** 2 + sd[lo] ** 2)
    return thr, {"mu": mu[[lo, hi]].tolist(), "sd": sd[[lo, hi]].tolist(),
                 "w": g.weights_[[lo, hi]].tolist(), "ashman_D": float(D), "bimodal": bool(D > 2)}


def binarize(Xt, names, thresholds=None, seed=0):
    """Xt: transformed matrix. thresholds: dict name->thr (overrides GMM)."""
    thresholds = dict(thresholds or {})
    info = {}
    B = np.zeros(Xt.shape, dtype=np.uint8)
    for j, n in enumerate(names):
        if n in thresholds:
            thr = thresholds[n]
            info[n] = {"thr": thr, "source": "given"}
        else:
            thr, inf = gmm_threshold(Xt[:, j], seed=seed)
            inf["thr"] = thr
            inf["source"] = "gmm"
            info[n] = inf
        B[:, j] = Xt[:, j] > thr
        info[n]["pos_frac"] = float(B[:, j].mean())
    return B, info


def strata(Xbb, n_strata=12, seed=0, max_fit=200_000):
    """KMeans strata on (transformed) backbone markers. Returns labels, model."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Xbb)) if len(Xbb) <= max_fit else rng.choice(len(Xbb), max_fit, replace=False)
    km = KMeans(n_strata, n_init=4, random_state=seed).fit(Xbb[idx])
    return km.predict(Xbb), km
