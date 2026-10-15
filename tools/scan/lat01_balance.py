"""Part 1: balance of triple multiplicities lambda_T for La Jolla coverings, stacked copies, random,
balanced-random, and (v=20) an algebraic 3-design from PSL(2,19) orbits.
Outputs: out_lat/lat01_summary.tsv, out_lat/lat01_hist.tsv, out_lat/lat01_psl19_orbits.tsv,
         out_lat/psl19_3design_570.txt (the 570 blocks)."""
import itertools, os
from math import comb
import numpy as np
import pandas as pd
from lat_common import OUT, design, budget, triples, lam_vec, summarize_lam

K = 6
NSEED = 200


# ---------------------------------------------------------------- PSL(2,q) / PGL(2,q) on PG(1,q)
def mobius_group(q, special):
    INF = q
    sq = {(x * x) % q for x in range(1, q)}
    inv = {x: pow(x, q - 2, q) for x in range(1, q)}
    perms = set()
    for a, b, c, d in itertools.product(range(q), repeat=4):
        det = (a * d - b * c) % q
        if det == 0 or (special and det not in sq):
            continue
        img = []
        for x in range(q + 1):
            if x == INF:
                img.append(INF if c == 0 else (a * inv[c]) % q)
            else:
                den = (c * x + d) % q
                img.append(INF if den == 0 else ((a * x + b) * inv[den]) % q)
        perms.add(tuple(img))
    return np.array(sorted(perms))


def orbits_ksets(G, v, k):
    seen = set()
    orbs = []
    pw = 1 << np.arange(v, dtype=np.int64)
    for S in itertools.combinations(range(v), k):
        m = sum(1 << x for x in S)
        if m in seen:
            continue
        imgs = set((pw[G[:, list(S)]].sum(1)).tolist())
        seen |= imgs
        orbs.append((S, len(imgs), imgs))
    return orbs


def mask_to_block(m, v):
    return {i for i in range(v) if (m >> i) & 1}


def main():
    rows, hist = [], []

    def record(name, v, r, blocks_or_list, seeds=None):
        tarr = triples(v)
        if seeds is None:
            lam = lam_vec(blocks_or_list, v, tarr)
            s = summarize_lam(lam, r, v, K)
            s.update(design=name, v=v, n_designs=1)
            rows.append(s)
            for l, c in zip(*np.unique(lam, return_counts=True)):
                hist.append(dict(design=name, v=v, r=r, lam=int(l), frac=c / len(lam)))
            return
        stats, hs = [], {}
        for sd in seeds:
            lam = lam_vec(blocks_or_list(sd), v, tarr)
            stats.append(summarize_lam(lam, r, v, K))
            for l, c in zip(*np.unique(lam, return_counts=True)):
                hs[int(l)] = hs.get(int(l), 0) + c / len(lam) / len(seeds)
        df = pd.DataFrame(stats)
        s = df.mean(numeric_only=True).to_dict()
        s.update(design=name, v=v, n_designs=len(seeds), r=r,
                 min_p05=float(df["min"].quantile(0.05)), min_p95=float(df["min"].quantile(0.95)),
                 frac0_p05=float(df["frac0"].quantile(0.05)), frac0_p95=float(df["frac0"].quantile(0.95)),
                 min_max=int(df["min"].max()))
        rows.append(s)
        for l, f in sorted(hs.items()):
            hist.append(dict(design=name, v=v, r=r, lam=l, frac=f))

    for v, grid in ((20, [71, 120, 142, 240, 570]), (26, [96, 130, 142, 240, 260])):
        cand = list(range(v))
        L = design.ljcr_blocks(v, K, 3)
        print(f"v={v}: La Jolla C({v},{K},3) has {len(L)} blocks")
        record(f"LJCR_{len(L)}", v, len(L), L)
        record(f"LJCR_x2_same", v, 2 * len(L), L + L)
        for r in grid:
            record("LJCR_stack(seed0)", v, r, budget.ljcr_stack(cand, K, r, seed=0))
            record("LJCR_stack(50 relabel seeds)", v, r, lambda s, r=r: budget.ljcr_stack(cand, K, r, seed=s), range(50))
            record("random", v, r, lambda s, r=r: design.random_design(cand, K, r, seed=s), range(NSEED))
            record("balanced_random", v, r, lambda s, r=r: budget.balanced_random(cand, K, r, seed=s), range(NSEED))

    # ------------------------------------------------ algebraic 3-designs on 20 points: PSL(2,19), PGL(2,19)
    orbrows = []
    best570 = None
    for gname, special in (("PSL(2,19)", True), ("PGL(2,19)", False)):
        G = mobius_group(19, special)
        orbs = orbits_ksets(G, 20, K)
        print(f"{gname}: order {len(G)}, {len(orbs)} orbits on 6-sets, sizes {sorted(o[1] for o in orbs)}")
        tarr = triples(20)
        for S, size, imgs in orbs:
            blocks = [mask_to_block(m, 20) for m in imgs]
            lam = lam_vec(blocks, 20, tarr)
            orbrows.append(dict(group=gname, order=len(G), rep=str(S), size=size, stab=len(G) // size,
                                lam_min=int(lam.min()), lam_max=int(lam.max()), is_3design=bool(lam.min() == lam.max())))
            if gname == "PSL(2,19)" and size == 570 and lam.min() == lam.max() and best570 is None:
                best570 = blocks
    pd.DataFrame(orbrows).to_csv(os.path.join(OUT, "lat01_psl19_orbits.tsv"), sep="\t", index=False)
    if best570 is not None:
        record("PSL(2,19)-orbit 3-(20,6,10) design", 20, 570, best570)
        with open(os.path.join(OUT, "psl19_3design_570.txt"), "w") as f:
            for b in best570:
                f.write(" ".join(map(str, sorted(b))) + "\n")

    df = pd.DataFrame(rows)
    cols = ["v", "design", "r", "n_designs", "min", "min_p05", "min_p95", "max", "mean", "cv", "frac0",
            "frac0_p05", "frac0_p95", "min_share", "bound", "eff", "best_min_lam_possible"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(OUT, "lat01_summary.tsv"), sep="\t", index=False)
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "lat01_hist.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
