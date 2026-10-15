"""P0 / e5b: Steiner nodes and the precision frontier (frozen spec: design/witnessflow/exp_protocols_v1.md).

usage: python e5b.py <protocol.json>
"""
import itertools, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, budget, design, audit  # noqa: E402

CONFIG = {
    "wingmen_v1": {"parents": {"CD4T": 0.05, "CD8T": 0.08},
                   "r": [104, 117, 130, 136, 142, 156, 169, 195, 221, 247, 260], "psl": False,
                   "per_donor_r": [130, 142, 156, 180, 221, 260]},
    "levine32_v2": {"parents": {"T": 0.06, "B": 0.08},
                    "r": [57, 64, 71, 85, 100, 120, 142, 170, 213, 284, 570], "psl": True, "per_donor_r": None},
}
ALLOC_SEEDS = list(range(10))
CAP, Z = 6, 1.96
PSL_FILE = os.path.join(ROOT, "tools", "scan", "out_lat", "psl19_3design_570.txt")


def cp_halfmax(cnt):
    n = cnt.sum(1, keepdims=True).astype(float)
    k = cnt.astype(float)
    nn = np.broadcast_to(n, k.shape)
    lo = np.where(k > 0, beta.ppf(0.025, k, nn - k + 1), 0.0)
    hi = np.where(k < nn, beta.ppf(0.975, k + 1, nn - k), 1.0)
    half = (np.nan_to_num(hi, nan=1.0) - np.nan_to_num(lo, nan=0.0)) / 2
    return np.where(n[:, 0] > 0, half.max(1), 1.0)


def well_counts(codesF, wg, r, chunk=150):
    """C[r, T, 8] = pattern counts of each triple within each well."""
    T, nG = codesF.shape
    C = np.zeros((r, T, 8), np.int32)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        idx = ((np.arange(a, b)[:, None] - a) * r + wg[None, :]) * 8 + codesF[a:b].astype(np.int64)
        cnt = np.bincount(idx.ravel(), minlength=(b - a) * r * 8).reshape(b - a, r, 8)
        C[:, a:b, :] = cnt.transpose(1, 0, 2)
    return C


def membership(wells, tarr, v):
    W = np.zeros((len(wells), v), bool)
    for i, X in enumerate(wells):
        W[i, list(X)] = True
    return W[:, tarr].all(2).T.astype(np.int32)  # [T, r]


def evaluate(L, C, refp, eps, N, piG):
    cnt = np.einsum("ti,itx->tx", L, C)
    n = cnt.sum(1)
    p = cnt / np.maximum(n, 1)[:, None]
    half = cp_halfmax(cnt)
    err = np.where(n > 0, np.abs(p - refp).max(1), 1.0)
    lam = L.sum(1)
    r = L.shape[1]
    exp_min = N * piG * lam.min() / r
    return dict(covered=float((lam >= 1).mean()), half_ok=float((half <= eps).mean()),
                err_ok=float((err <= eps).mean()), min_cells=float(n.min()),
                plan_eps=float(0.98 / math.sqrt(exp_min)) if exp_min > 0 else np.inf,
                lam_min=int(lam.min()), lam_cv=float(lam.std() / lam.mean()))


def load_psl(cand):
    blocks = [list(map(int, l.split())) for l in open(PSL_FILE) if l.strip()]
    return [set(cand[x] for x in b) for b in blocks]


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    pid = proto["protocol_id"]
    cfg = CONFIG[pid]
    out = os.path.join(ROOT, "results", pid)
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    v = len(markers)
    tarr = np.array(list(itertools.combinations(cand, 3)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    up = math.comb(CAP, 3) / len(tarr)
    rows, drows = [], []
    for G, eps in cfg["parents"].items():
        piG = pm[G][ini].mean()
        eps_star = 0.98 / math.sqrt(N * piG * up)
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        print(f"[{time.time() - t0:.0f}s] {pid} {G}: N={N} pi={piG:.3f} eps*={eps_star:.4f} eps={eps}", flush=True)
        for r in cfg["r"]:
            designs = {("classical_stack", 0): budget.ljcr_stack(cand, CAP, r, seed=0)}
            for s in range(3):
                designs[("random", s)] = design.random_design(cand, CAP, r, seed=s)
                designs[("balanced_random", s)] = budget.balanced_random(cand, CAP, r, seed=s)
            if cfg["psl"] and r == 570:
                designs[("PSL_3design", 0)] = load_psl(cand)
            Ls = {k: membership(w, tarr, v) for k, w in designs.items()}
            for a in ALLOC_SEEDS:
                order = np.random.default_rng(1000 * a + r).permutation(N)
                well_of = np.empty(N, int)
                for i, chunk in enumerate(np.array_split(order, r)):
                    well_of[chunk] = i
                C = well_counts(codesF, well_of[gpos], r)
                for (meth, s), L in Ls.items():
                    m = evaluate(L, C, refp, eps, N, piG)
                    rows.append(dict(dataset=pid, parent=G, eps=eps, eps_star=eps_star, r=r, method=meth, seed=s, alloc=a, **m))
            print(f"[{time.time() - t0:.0f}s] {G} r={r} done", flush=True)
        # ---- per-donor replication (each donor's own follow-up cells and reference)
        if cfg["per_donor_r"]:
            for d in np.unique(sample):
                fd = np.flatnonzero(fol & (sample == d))
                gd = np.flatnonzero(pm[G][fd])
                Rd = Bin[ref & pm[G] & (sample == d)]
                pid_d = pm[G][ini & (sample == d)].mean()
                if len(gd) < 500 or len(Rd) < 500:
                    continue
                Nd = len(fd)
                eps_d = 1.1 * 0.98 / math.sqrt(Nd * pid_d * up)
                refd = np.stack([np.bincount(audit.codes(Rd, S), minlength=8) / len(Rd) for S in tarr])
                cF = np.stack([audit.codes(Bin[fd[gd]], S).astype(np.uint8) for S in tarr])
                for r in cfg["per_donor_r"]:
                    Lc = membership(budget.ljcr_stack(cand, CAP, r, seed=0), tarr, v)
                    Lr = membership(design.random_design(cand, CAP, r, seed=0), tarr, v)
                    for a in range(3):
                        order = np.random.default_rng(7000 + 1000 * a + r + int(d)).permutation(Nd)
                        well_of = np.empty(Nd, int)
                        for i, chunk in enumerate(np.array_split(order, r)):
                            well_of[chunk] = i
                        C = well_counts(cF, well_of[gd], r)
                        for meth, L in (("classical_stack", Lc), ("random", Lr)):
                            m = evaluate(L, C, refd, eps_d, Nd, pid_d)
                            drows.append(dict(dataset=pid, parent=G, donor=int(d), N_d=Nd, eps_d=eps_d, r=r, method=meth, alloc=a, **m))
                print(f"[{time.time() - t0:.0f}s] {G} donor {d} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5b_steiner_nodes.tsv"), sep="\t", index=False)
    if drows:
        pd.DataFrame(drows).to_csv(os.path.join(out, "e5b_per_donor.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    for metric in ["half_ok", "err_ok", "covered"]:
        print(f"\n== {metric}")
        print(df.groupby(["parent", "r", "method"])[metric].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
