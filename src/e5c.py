"""P1 / e5c: artificial 5-slot (family) constraint on WINGMEN = strength-3 orthogonal array by quadratic
interpolation over F_5 (frozen spec v2: design/witnessflow/exp_protocols_v1.md).

Families: the 26 reliable candidates minus the one with the largest metal mass, sorted by mass, cut into
5 consecutive groups of 5 (a deterministic ARTIFICIAL slot constraint, no physical claim).
Each tube takes exactly one marker per family (5 markers + backbone).  Targets = 1250 cross-family triples.
usage: python e5c.py <wingmen protocol.json>
"""
import glob, itertools, json, math, os, re, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, audit, fcs, datasets  # noqa: E402
from e5b import well_counts, evaluate  # noqa: E402

PARENTS = {"CD4T": 0.05, "CD8T": 0.08}
ALLOC_SEEDS = list(range(10))
DESIGN_SEEDS = [0, 1, 2]
ANNEAL_ITERS = 200_000
Q = 5


def oa_design(fam):
    tubes = []
    for a, b, c in itertools.product(range(Q), repeat=3):
        tubes.append([fam[x][(a + b * x + c * x * x) % Q] for x in range(Q)])
    return tubes


def legal_random(fam, seed):
    rng = np.random.default_rng(seed)
    return [[fam[x][rng.integers(Q)] for x in range(Q)] for _ in range(Q ** 3)]


def balanced_random(fam, seed):
    rng = np.random.default_rng(seed)
    cols = []
    for x in range(Q):
        col = np.repeat(np.arange(Q), Q ** 2)  # each antibody 25 times
        rng.shuffle(col)
        cols.append(col)
    return [[fam[x][cols[x][i]] for x in range(Q)] for i in range(Q ** 3)]


def anneal_transversal(fam, seed, iters=ANNEAL_ITERS):
    """state: choice[i, x] in 0..4; objective = #cross-family triples covered (count>0)."""
    rng = np.random.default_rng(seed)
    R = Q ** 3
    ch = np.stack([rng.permutation(np.repeat(np.arange(Q), Q ** 2)) for _ in range(Q)], 1)  # balanced start
    fams3 = list(itertools.combinations(range(Q), 3))
    cnt = {}
    for i in range(R):
        for f in fams3:
            key = (f, tuple(ch[i, list(f)]))
            cnt[key] = cnt.get(key, 0) + 1
    covered = len(cnt)
    best, best_cov = ch.copy(), covered
    others = {x: [f for f in fams3 if x in f] for x in range(Q)}
    for it in range(iters):
        temp = 1.0 * (1 - it / iters) + 1e-3
        i, x = int(rng.integers(R)), int(rng.integers(Q))
        new = int(rng.integers(Q))
        old = ch[i, x]
        if new == old:
            continue
        d = 0
        lose, gain = [], []
        for f in others[x]:
            k_old = (f, tuple(ch[i, list(f)]))
            lose.append(k_old)
            if cnt[k_old] == 1:
                d -= 1
        ch[i, x] = new
        for f in others[x]:
            k_new = (f, tuple(ch[i, list(f)]))
            gain.append(k_new)
            if cnt.get(k_new, 0) == 0:
                d += 1
        if d >= 0 or rng.random() < math.exp(d / temp):
            for k in lose:
                cnt[k] -= 1
                if cnt[k] == 0:
                    del cnt[k]
            for k in gain:
                cnt[k] = cnt.get(k, 0) + 1
            covered += d
            if covered > best_cov:
                best_cov, best = covered, ch.copy()
        else:
            ch[i, x] = old
    return [[fam[x][best[i, x]] for x in range(Q)] for i in range(R)], best_cov


def multiplicities(tubes, fam, tarr):
    single = np.bincount(np.array(tubes).ravel(), minlength=max(max(f) for f in fam) + 1)
    sing = [single[m] for f in fam for m in f]
    pairs = {}
    for t in tubes:
        for a, b in itertools.combinations(sorted(t), 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    famof = {m: x for x, f in enumerate(fam) for m in f}
    allpairs = [(a, b) for a, b in itertools.combinations(sorted(m for f in fam for m in f), 2) if famof[a] != famof[b]]
    pc = [pairs.get(p, 0) for p in allpairs]
    return dict(single_min=int(min(sing)), single_max=int(max(sing)), pair_min=int(min(pc)), pair_max=int(max(pc)))


def main(proto_path):
    t0 = time.time()
    proto = json.load(open(proto_path, encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    # metal mass of each marker from one FCS header
    f0 = sorted(glob.glob(os.path.join(ROOT, proto["dataset"]["dir"], proto["dataset"]["pattern"])))[0]
    _, ch_, mk_, _ = fcs.read_fcs(f0)
    mass = {datasets.strip_metal(s): int(re.match(r"^(\d+)", s).group(1)) for s in mk_ if re.match(r"^\d+[A-Za-z]{1,2}_", s)}
    cand_sorted = sorted(cand, key=lambda i: mass[markers[i]])
    dropped = cand_sorted[-1]
    use = cand_sorted[:-1]
    fam = [use[Q * x:Q * (x + 1)] for x in range(Q)]
    print("families:", [[f"{markers[i]}({mass[markers[i]]})" for i in f] for f in fam], "dropped:", markers[dropped], flush=True)
    famof = {m: x for x, f in enumerate(fam) for m in f}
    tarr = np.array([t for t in itertools.combinations(sorted(use), 3) if len({famof[m] for m in t}) == 3])
    print("cross-family triples:", len(tarr), flush=True)
    v = len(markers)
    designs = {("OA_quadratic", 0): oa_design(fam)}
    for s in DESIGN_SEEDS:
        designs[("legal_random", s)] = legal_random(fam, 100 + s)
        designs[("balanced_random", s)] = balanced_random(fam, 200 + s)
        an, cov = anneal_transversal(fam, 300 + s)
        designs[("anneal_search", s)] = an
        print(f"[{time.time() - t0:.0f}s] anneal seed {s}: covered {cov}/{len(tarr)}", flush=True)
    W = {}
    mult_rows = []
    for k, tubes in designs.items():
        Wm = np.zeros((len(tubes), v), bool)
        for i, t in enumerate(tubes):
            Wm[i, t] = True
        L = Wm[:, tarr].all(2).T.astype(np.int32)
        W[k] = L
        lam = L.sum(1)
        mult_rows.append(dict(method=k[0], seed=k[1], triple_min=int(lam.min()), triple_max=int(lam.max()),
                              triple_covered=float((lam >= 1).mean()), **multiplicities(tubes, fam, tarr)))
    ini, fol, ref = part == "initial", part == "followup", part == "reference"
    folidx = np.flatnonzero(fol)
    N = len(folidx)
    R = Q ** 3
    rows = []
    for G, eps in PARENTS.items():
        piG = pm[G][ini].mean()
        Rc = Bin[ref & pm[G]]
        refp = np.stack([np.bincount(audit.codes(Rc, S), minlength=8) / len(Rc) for S in tarr])
        gpos = np.flatnonzero(pm[G][folidx])
        codesF = np.stack([audit.codes(Bin[folidx[gpos]], S).astype(np.uint8) for S in tarr])
        eps_star = 0.98 / math.sqrt(N * piG / R)
        for a in ALLOC_SEEDS:
            order = np.random.default_rng(5000 + a).permutation(N)
            well_of = np.empty(N, int)
            for i, chunk in enumerate(np.array_split(order, R)):
                well_of[chunk] = i
            C = well_counts(codesF, well_of[gpos], R)
            for (meth, s), L in W.items():
                m = evaluate(L, C, refp, eps, N, piG)
                rows.append(dict(parent=G, eps=eps, eps_star_OA=eps_star, method=meth, seed=s, alloc=a, **m))
        print(f"[{time.time() - t0:.0f}s] {G} done (eps*_OA={eps_star:.4f})", flush=True)
    pd.DataFrame(mult_rows).to_csv(os.path.join(out, "e5c_multiplicities.tsv"), sep="\t", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5c_oa_replay.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(mult_rows).groupby("method").mean(numeric_only=True).round(3).to_string())
    print(df.groupby(["parent", "method"])[["covered", "half_ok", "err_ok"]].agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
