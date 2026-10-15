"""Exploratory analyses (reported separately from the frozen primary claims).

E1  Q1 scan: every triple of reliable non-backbone markers in every parent; third-order structure
    beyond pairwise margins on the reference split, replicated on the calibration split and per donor.
E2  Screen-scale witness design: fraction of all marker triples co-measured in >=1 well as a function
    of the number of wells r and well capacity c; random vs greedy vs La Jolla covering vs anneal.

usage: python explore.py <protocol.json> [e1] [e2]
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import replay, audit, design, q1  # noqa: E402


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def e1(proto, out):
    Xt, markers, sample = replay.load(proto)
    part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    Bin = np.stack([Xt[:, j] > thr[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
    mi = {m: i for i, m in enumerate(markers)}
    bb = set(mi[m] for m in proto["backbone"])
    cand = [i for i, m in enumerate(markers) if i not in bb and thr[m]["reliable"]]
    pm = replay.parent_masks(Bin, markers, proto["parents"])
    rows = []
    for pname, inG in pm.items():
        for S in itertools.combinations(cand, 3):
            rec = {"parent": pname, "markers": "/".join(markers[i] for i in S)}
            k = None
            for split_name in ("reference", "calibration"):
                msk = inG & (part == split_name)
                n = q1.table(Bin, S, msk)
                if n.sum() < 200:
                    continue
                p = n / n.sum()
                q2 = q1.pairwise_maxent(n)
                d = p - q2
                if k is None:  # pattern chosen on the reference split, re-checked at the SAME cell
                    k = int(np.argmax(np.abs(d)))
                rec[f"{split_name}_N"] = n.sum()
                rec[f"{split_name}_H3"] = q1.kl(p, q2)
                rec[f"{split_name}_maxdelta_pp"] = 100 * d[k]
                rec[f"{split_name}_pattern"] = k
                rec[f"{split_name}_p"] = q1.exact_test(n)
            # per donor on reference+calibration pooled
            for s in np.unique(sample):
                msk = inG & np.isin(part, ["reference", "calibration"]) & (sample == s)
                n = q1.table(Bin, S, msk)
                if n.sum() >= 200 and "reference_pattern" in rec:
                    p = n / n.sum()
                    q2 = q1.pairwise_maxent(n)
                    rec[f"donor{s}_delta_pp"] = 100 * (p - q2)[rec["reference_pattern"]]
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["reference_q"] = bh(df["reference_p"].fillna(1).values)
    df.to_csv(os.path.join(out, "explore_q1_all_triples.tsv"), sep="\t", index=False)
    sig = (df.reference_q < 0.01) & (df.reference_maxdelta_pp.abs() >= 1.0)
    rep = sig & (np.sign(df.calibration_maxdelta_pp) == np.sign(df.reference_maxdelta_pp)) & \
        (df.calibration_maxdelta_pp.abs() >= 0.5) & (df.calibration_p < 0.01)
    donors = [c for c in df.columns if c.startswith("donor")]
    both = rep.copy()
    for c in donors:
        both &= np.sign(df[c]) == np.sign(df.reference_maxdelta_pp)
    print(f"E1: {len(df)} (parent,triple) tables; q<0.01 & |delta|>=1pp: {sig.sum()}; "
          f"replicated on calibration split: {rep.sum()}; same sign in all donors: {both.sum()}")
    print(df[both].sort_values("reference_maxdelta_pp", key=np.abs, ascending=False)
          .head(15)[["parent", "markers", "reference_N", "reference_pattern", "reference_maxdelta_pp",
                     "calibration_maxdelta_pp"] + donors].round(2).to_string(index=False))
    return df


def e2(proto, out, caps=(3, 6, 10), seeds=5):
    thr = json.load(open(os.path.join(out, "thresholds.json")))
    markers = list(thr)
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    v = len(cand)
    allT = list(itertools.combinations(cand, 3))
    w = np.ones(len(allT))
    rows = []
    for c in caps:
        full = design.ljcr_blocks(v, c, 3)
        nfull = len(full) if full else None
        rgrid = sorted(set([1, 2, 4, 8, 16, 32, 48, 64, 96, 128] + ([nfull] if nfull else [])))
        for r in rgrid:
            if nfull and r > nfull * 1.5:
                continue
            res = {}
            res["random"] = np.mean([design.score(design.random_design(cand, c, r, s), allT, w) for s in range(seeds)])
            if full:
                lab = cand
                blocks = [set(lab[x] for x in b) for b in full]
                # best r-subset of the optimal full covering (greedy pick) and random r-subset
                rng = np.random.default_rng(0)
                res["LJCR_random_subset"] = np.mean([design.score([blocks[i] for i in rng.choice(len(blocks), min(r, len(blocks)), replace=False)], allT, w) for _ in range(seeds)])
                picked, cov = [], np.zeros(len(allT), bool)
                masks = [design.covered_mask([b], allT) for b in blocks]
                for _ in range(min(r, len(blocks))):
                    gains = [(~cov & m).sum() for m in masks]
                    k = int(np.argmax(gains))
                    picked.append(blocks[k])
                    cov |= masks[k]
                res["LJCR_greedy_subset"] = design.score(picked, allT, w)
            if r <= 64:
                g = design.greedy_design(cand, c, r, allT, w)
                res["greedy"] = design.score(g, allT, w)
                a, _ = design.anneal(g, cand, c, allT, w, iters=20000, seed=1)
                res["greedy+anneal"] = design.score(a, allT, w)
            for k_, val in res.items():
                rows.append(dict(capacity=c, r=r, method=k_, frac_triples=val / len(allT), ljcr_full=nfull))
            print(f"  c={c} r={r}: " + "  ".join(f"{k_}={val / len(allT):.3f}" for k_, val in res.items()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "explore_screen_design.tsv"), sep="\t", index=False)
    print(f"E2: v={v} candidate markers, {len(allT)} triples")
    return df


if __name__ == "__main__":
    proto = json.load(open(sys.argv[1], encoding="utf-8"))
    out = os.path.join(ROOT, "results", proto["protocol_id"])
    t0 = time.time()
    if "e1" in sys.argv:
        e1(proto, out)
        print(f"[{time.time() - t0:.0f}s] e1 done", flush=True)
    if "e2" in sys.argv:
        e2(proto, out)
        print(f"[{time.time() - t0:.0f}s] e2 done", flush=True)
