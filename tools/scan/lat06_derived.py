"""Part 3: 'backbone = derived design' checks.
(1) Levine: tubes = backbone(8) U La Jolla block.  For every gating set B' (subset of backbone) and every
    candidate triple T: lambda(T U B') == lambda_X(T)  (the full-panel problem for parent-gated targets is
    exactly the derived covering on the candidates).  Also multiplicities of mixed triples.
(2) WINGMEN S(3,6,26): if a NON-backbone marker x defines a sub-parent, the usable tubes are those through
    x, i.e. the derived design at x (an AG(2,5)): pairs inside x+ are witnessed exactly once, triples only
    in 300/2300.  Promoting x to the backbone instead makes the problem C(25,6,3) (La Jolla 130).
(3) Parent -> multiplicity: lam_req(G) = ceil(n_req r / (N pi_G)); the best precision any design can give
    to ALL triples of parent G is eps*_G = z*0.5/sqrt(N pi_G C(k,3)/C(v,3)) (3-design bound), vs the
    La Jolla covering's z*0.5/sqrt(N pi_G lam_min/r).
Output: out_lat/lat06_derived.txt, out_lat/lat06_parent_frontier.tsv"""
import itertools, json, os
from math import comb, sqrt, ceil
from collections import Counter
import numpy as np
import pandas as pd
from lat_common import OUT, WF, design
import replay  # noqa: E402

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def main():
    # (1) backbone augmentation on Levine geometry: candidates 0..19, backbone 20..27
    X = design.ljcr_blocks(20, 6, 3)
    Bk = list(range(20, 28))
    tubes = [set(b) | set(Bk) for b in X]
    allp = list(range(28))
    lam = Counter()
    for t in tubes:
        for T in itertools.combinations(sorted(t), 3):
            lam[T] += 1
    types = {}
    for T in itertools.combinations(allp, 3):
        nb = sum(x >= 20 for x in T)
        types.setdefault(nb, []).append(lam.get(T, 0))
    for nb in sorted(types):
        log(f"(1) triples with {nb} backbone + {3 - nb} candidate markers: n={len(types[nb])}, "
            f"lambda distribution {dict(sorted(Counter(types[nb]).items()))}")
    ok = True
    for gsize in (1, 2, 4):
        for Bp in itertools.combinations(Bk, gsize):
            for T in itertools.combinations(range(20), 3):
                S = set(T) | set(Bp)
                l1 = sum(S <= t for t in tubes)
                l2 = sum(set(T) <= set(b) for b in X)
                ok &= (l1 == l2)
            break  # one gating set per size suffices (all backbone points are in every tube)
    log(f"(1) lambda(T U B') == lambda_X(T) for all 1140 T and gating sets of size 1,2,4: {ok}")

    # (2) S(3,6,26): sub-parent gated by a non-backbone marker x
    S = [set(b) for b in design.ljcr_blocks(26, 6, 3)]
    x = 0
    thr = [b - {x} for b in S if x in b]
    pairs = Counter(p for b in thr for p in itertools.combinations(sorted(b), 2))
    trip = set(T for b in thr for T in itertools.combinations(sorted(b), 3))
    log(f"(2) S(3,6,26): {len(thr)} of {len(S)} tubes contain marker x; inside x+ cells the witnessed pairs "
        f"{len(pairs)}/{comb(25, 2)} (each exactly {set(pairs.values())} time), witnessed triples "
        f"{len(trip)}/{comb(25, 3)} = {len(trip) / comb(25, 3):.3f}")
    L25 = design.ljcr_blocks(25, 6, 3)
    L25b = design.ljcr_blocks(25, 5, 3)
    log(f"(2) if x is promoted to the backbone (x + 6 other markers per tube): La Jolla C(25,6,3) = "
        f"{len(L25) if L25 else 'n/a'} tubes witness all {comb(25, 3)} triples inside x+; if x must share the "
        f"6 slots (x + 5): C(25,5,3) = {len(L25b) if L25b else 'n/a'} tubes")

    # (3) parent frontier
    rows = []
    for pfile, k in (("levine32_v2.json", 6), ("wingmen_v1.json", 6)):
        proto = json.load(open(os.path.join(WF, "protocols", pfile), encoding="utf-8"))
        Xt, markers, sample = replay.load(proto)
        part = replay.split(len(Xt), sample, proto["split"], proto["split"]["seed"])
        thrj = json.load(open(os.path.join(WF, "results", proto["protocol_id"], "thresholds.json")))
        Bin = np.stack([Xt[:, j] > thrj[m]["thr"] for j, m in enumerate(markers)], 1).astype(np.uint8)
        pm = replay.parent_masks(Bin, markers, proto["parents"])
        bb = set(proto["backbone"])
        v = sum(1 for m in markers if m not in bb and thrj[m]["reliable"])
        N = int((part == "followup").sum())
        Lc = design.ljcr_blocks(v, k, 3)
        b = len(Lc)
        lam_min = 1
        for G in proto["parents"]:
            pi = float(pm[G][part == "initial"].mean())
            npi = N * pi
            share_opt = comb(k, 3) / comb(v, 3)
            eps_opt = 1.96 * 0.5 / sqrt(npi * share_opt)
            eps_cov = 1.96 * 0.5 / sqrt(npi * lam_min / b)
            gate = proto["parents"][G]
            rows.append(dict(dataset=proto["protocol_id"], parent=G, gate="+".join(gate.get("pos", [])) +
                             ("/-" + "-".join(gate.get("neg", [])) if gate.get("neg") else ""),
                             n_gating_markers=len(gate.get("pos", [])) + len(gate.get("neg", [])),
                             pi=round(pi, 4), N_followup=N, v=v, cover_size=b,
                             eps_best_any_design=round(eps_opt, 4), eps_la_jolla_cover=round(eps_cov, 4),
                             lam_req_eps05_at_cover=ceil(1.96 ** 2 * 0.25 / 0.05 ** 2 * b / npi),
                             cells_needed_for_eps05_all_triples=int(ceil(1.96 ** 2 * 0.25 / 0.05 ** 2 / share_opt / pi))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lat06_parent_frontier.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    log("(3) per-parent precision frontier (follow-up budget N fixed, worst-case p=0.5):")
    log(df.to_string(index=False))
    open(os.path.join(OUT, "lat06_derived.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
