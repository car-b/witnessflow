"""Part 2/3: structure of the La Jolla coverings C(20,6,3)=71 and C(26,6,3)=130.
- point degrees, pair multiplicities, block-intersection sizes, triple multiplicities
- automorphism group order + point/block orbits (stabilizer chain with VF2++ individualisation)
- v=26: test isomorphism with the Miquelian inversive plane of order 5 (circles = PGL(2,25)-images of PG(1,5));
  derived design at every point: 2-(25,5,1)? resolvable (affine plane AG(2,5))?
- v=20: look for relations to S(3,6,22) (Witt) by checking derived structures / point deletions is NOT
  attempted blindly; we report what the invariants show.
Outputs: out_lat/lat02_structure.txt, out_lat/lat02_invariants.tsv"""
import itertools, os, time
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from lat_common import OUT, design, triples, lam_vec, pair_lam, degrees, intersections

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def inc_graph(v, blocks, labels=None):
    G = nx.Graph()
    for x in range(v):
        G.add_node(("p", x), lab=("p", labels.get(x, "") if labels else ""))
    for i, B in enumerate(blocks):
        G.add_node(("b", i), lab=("b", ""))
        for x in B:
            G.add_edge(("p", x), ("b", i))
    return G


def maps_exist(v, blocks, fixed, src, dst):
    """Is there an automorphism fixing every point in `fixed` and sending point src -> dst?"""
    l1 = {x: f"f{x}" for x in fixed}
    l2 = dict(l1)
    l1[src] = "X"
    l2[dst] = "X"
    return nx.vf2pp_is_isomorphic(inc_graph(v, blocks, l1), inc_graph(v, blocks, l2), node_label="lab")


def aut_order(v, blocks, deg):
    """Stabiliser chain: |Aut| = prod over base points of orbit sizes (points determine blocks here)."""
    fixed, order, orbit_sizes, first_orbits = [], 1, [], None
    remaining = list(range(v))
    while remaining:
        p0 = remaining[0]
        orb = [q for q in remaining if deg[q] == deg[p0] and (q == p0 or maps_exist(v, blocks, fixed, p0, q))]
        if not fixed:  # full point orbits under Aut (only for the first level)
            first_orbits = []
            left = set(range(v))
            while left:
                a = min(left)
                o = {a} | {q for q in left if q != a and deg[q] == deg[a] and maps_exist(v, blocks, [], a, q)}
                first_orbits.append(sorted(o))
                left -= o
        order *= len(orb)
        orbit_sizes.append(len(orb))
        fixed.append(p0)
        remaining = [q for q in remaining if q != p0]
    return order, orbit_sizes, first_orbits


# ------------------------------------------------------------------ GF(25) inversive plane
def gf25():
    # elements a + b s, s^2 = 2 (2 is a non-square mod 5)
    els = [(a, b) for a in range(5) for b in range(5)]
    idx = {e: i for i, e in enumerate(els)}
    add = lambda x, y: ((x[0] + y[0]) % 5, (x[1] + y[1]) % 5)
    mul = lambda x, y: ((x[0] * y[0] + 2 * x[1] * y[1]) % 5, (x[0] * y[1] + x[1] * y[0]) % 5)
    inv = {}
    for x in els:
        for y in els:
            if mul(x, y) == (1, 0):
                inv[x] = y
    return els, idx, add, mul, inv


def inversive_plane_5():
    els, idx, add, mul, inv = gf25()
    INF = 25
    neg = lambda x: ((-x[0]) % 5, (-x[1]) % 5)
    base = [idx[(a, 0)] for a in range(5)] + [INF]  # PG(1,5) inside PG(1,25)
    circles = set()
    zero = (0, 0)
    for a, b, c, d in itertools.product(els, repeat=4):
        det = add(mul(a, d), neg(mul(b, c)))
        if det == zero:
            continue
        img = []
        for xi in base:
            if xi == INF:
                img.append(INF if c == zero else idx[mul(a, inv[c])])
            else:
                x = els[xi]
                den = add(mul(c, x), d)
                img.append(INF if den == zero else idx[mul(add(mul(a, x), b), inv[den])])
        circles.add(frozenset(img))
    return [set(c) for c in circles]


def is_2design(blocks, v):
    P = pair_lam(blocks, v)
    return P.min() == P.max(), int(P.min()), int(P.max())


def parallel_classes(blocks, v):
    """Group lines of a 2-(n^2, n, 1) design into classes of mutually disjoint lines (affine plane test)."""
    B = [frozenset(b) for b in blocks]
    G = nx.Graph()
    G.add_nodes_from(range(len(B)))
    for i, j in itertools.combinations(range(len(B)), 2):
        if not (B[i] & B[j]):
            G.add_edge(i, j)
    comps = [sorted(c) for c in nx.connected_components(G)]
    ok = all(nx.density(G.subgraph(c)) == 1.0 for c in comps if len(c) > 1)
    return comps, ok


def describe(v, blocks, name):
    k = len(next(iter(blocks)))
    lam = lam_vec(blocks, v)
    P = pair_lam(blocks, v)
    deg = degrees(blocks, v)
    I = intersections(blocks)
    log(f"\n=== {name}: v={v}, k={k}, b={len(blocks)}")
    log(f"point degrees: {dict(sorted(Counter(deg.tolist()).items()))}  (mean {deg.mean():.3f}; "
        f"perfectly balanced would be {len(blocks) * k / v:.3f})")
    log(f"pair multiplicities: {dict(sorted(Counter(P.tolist()).items()))}  (mean {P.mean():.3f})")
    log(f"triple multiplicities: {dict(sorted(Counter(lam.tolist()).items()))}")
    log(f"block intersection sizes: {dict(sorted(Counter(I.tolist()).items()))}")
    reps = Counter(frozenset(b) for b in blocks)
    log(f"repeated blocks: {sum(c - 1 for c in reps.values())}")
    # per-point: how many of the triples through the point are over-covered
    t = triples(v)
    over = [int(((lam > 1) & (t == x).any(1)).sum()) for x in range(v)]
    log(f"per-point # over-covered triples (lam>1) through the point: {over}")
    return dict(name=name, v=v, k=k, b=len(blocks), deg_min=int(deg.min()), deg_max=int(deg.max()),
                pair_min=int(P.min()), pair_max=int(P.max()), lam_min=int(lam.min()), lam_max=int(lam.max()),
                lam_cv=float(lam.std() / lam.mean()), inter_sizes=str(dict(sorted(Counter(I.tolist()).items())))), deg


def point_invariants(v, blocks):
    """Isomorphism-invariant label per point: degree, multiset of pair multiplicities, multiset of the
    multiplicities of triples through the point."""
    P = np.zeros((v, v), int)
    for X in blocks:
        for a, b in itertools.combinations(sorted(X), 2):
            P[a, b] += 1
            P[b, a] += 1
    lam = lam_vec(blocks, v)
    t = triples(v)
    inv = {}
    for x in range(v):
        through = lam[(t == x).any(1)]
        inv[x] = (int(P[x].sum() // 5), tuple(sorted(P[x][np.arange(v) != x].tolist())),
                  tuple(sorted(Counter(through.tolist()).items())))
    return inv


def steiner3_iso(v, B1, B2, seed_map):
    """Backtracking isomorphism between two S(3,k,v) Steiner systems (3 points determine a block).
    seed_map: initial point map (WLOG for a 3-transitive target).  Returns a point map or None."""
    def index(B):
        blk = {}
        for i, b in enumerate(B):
            for T in itertools.combinations(sorted(b), 3):
                blk[T] = i
        return blk
    i1, i2 = index(B1), index(B2)
    S1, S2 = [frozenset(b) for b in B1], [frozenset(b) for b in B2]
    blk1 = lambda a, b, c: S1[i1[tuple(sorted((a, b, c)))]]
    blk2 = lambda a, b, c: S2[i2[tuple(sorted((a, b, c)))]]

    def consistent(f, x, y):
        dom = list(f)
        for a, b in itertools.combinations(dom, 2):
            b1, b2 = blk1(a, b, x), blk2(f[a], f[b], y)
            for z in dom:
                if (z in b1) != (f[z] in b2):
                    return False
        return True

    order = [x for x in range(v) if x not in seed_map]

    def bt(f, used, depth):
        if depth == len(order):
            return dict(f)
        x = order[depth]
        for y in range(v):
            if y in used or not consistent(f, x, y):
                continue
            f[x] = y
            used.add(y)
            res = bt(f, used, depth + 1)
            if res:
                return res
            del f[x]
            used.discard(y)
        return None

    f = dict(seed_map)
    m = bt(f, set(f.values()), 0)
    if m is None:
        return None
    ok = {frozenset(m[x] for x in b) for b in S1} == set(S2)
    return m if ok else None


def frobenius_preserves(ip):
    els, idx, add, mul, inv = gf25()
    def frob(i):
        if i == 25:
            return 25
        x = els[i]
        y = x
        for _ in range(4):
            y = mul(y, x)          # x^5
        return idx[y]
    S = {frozenset(c) for c in ip}
    return all(frozenset(frob(i) for i in c) in S for c in ip)


def main():
    t0 = time.time()
    rows = []
    for v in (26, 20):
        blocks = design.ljcr_blocks(v, 6, 3)
        row, deg = describe(v, blocks, f"LJCR C({v},6,3)")
        rows.append(row)
        if v == 20:
            inv = point_invariants(v, blocks)
            classes = Counter(inv.values())
            log(f"point-invariant classes (degree, pair-multiset, triple-multiset): sizes "
                f"{sorted(classes.values(), reverse=True)}")
            for key, n in sorted(classes.items(), key=lambda kv: -kv[1])[:6]:
                log(f"   class size {n}: degree {key[0]}, pair-mult multiset {dict(Counter(key[1]))}, "
                    f"triple-mult through point {dict(key[2])}")
            if max(classes.values()) == 1:
                log("all points have distinct invariants -> every automorphism fixes every point -> |Aut| = 1 "
                    "(no repeated blocks)")
                row["aut_order"] = 1
            else:
                lab = {x: str(inv[x]) for x in range(v)}
                # stabiliser chain, using invariant labels to prune
                def ok(fixed, s, d):
                    l1 = {x: lab[x] + f"|f{x}" if x in fixed else lab[x] for x in range(v)}
                    l2 = dict(l1)
                    l1[s] = lab[s] + "|X"
                    l2[d] = lab[d] + "|X"
                    return nx.vf2pp_is_isomorphic(inc_graph(v, blocks, l1), inc_graph(v, blocks, l2), node_label="lab")
                fixed, order, osz = [], 1, []
                for p0 in range(v):
                    orb = [q for q in range(v) if q not in fixed and inv[q] == inv[p0] and (q == p0 or ok(fixed, p0, q))]
                    order *= len(orb)
                    osz.append(len(orb))
                    fixed.append(p0)
                log(f"|Aut| = {order} (stabiliser-chain orbit sizes {osz})  [{time.time() - t0:.0f}s]")
                row["aut_order"] = order
        if v == 26:
            ip = inversive_plane_5()
            log(f"Miquelian inversive plane of order 5 built from GF(25): {len(ip)} circles")
            # PGL(2,25) is sharply 3-transitive on PG(1,25): WLOG LJCR points 0,1,2 -> plane points 0,1,2
            m = steiner3_iso(26, blocks, ip, {0: 0, 1: 1, 2: 2})
            iso = m is not None
            if iso:
                log(f"explicit point map LJCR -> GF(25) model (0..24 = a+5b <-> a+b*s, s^2=2; 25 = infinity): {m}")
            log(f"LJCR C(26,6,3) isomorphic to the inversive plane of order 5: {iso}   [{time.time() - t0:.0f}s]")
            row["iso_inversive_plane_5"] = iso
            fp = frobenius_preserves(ip)
            log(f"constructed plane is invariant under PGL(2,25) (by construction, order 15600) and under the "
                f"Frobenius x->x^5: {fp}  => |Aut| >= 31200 (= |PGammaL(2,25)|, the known full group)")
            row["aut_order"] = ">=31200 (PGammaL(2,25))" if (iso and fp) else "n/a"
            # derived designs
            ok_all, res_all = True, True
            for x in range(26):
                der = [set(b) - {x} for b in blocks if x in b]
                relab = {y: i for i, y in enumerate(sorted(set(range(26)) - {x}))}
                der = [{relab[y] for y in b} for b in der]
                d2, lo, hi = is_2design(der, 25)
                comps, par = parallel_classes(der, 25)
                ok_all &= d2 and lo == 1
                res_all &= par and len(comps) == 6 and all(len(c) == 5 for c in comps)
            log(f"derived design at every point is a 2-(25,5,1) design: {ok_all}; "
                f"resolvable into 6 parallel classes of 5 lines (affine plane AG(2,5)): {res_all}")
            row["derived_all_AG25"] = bool(ok_all and res_all)
            # residual-type check: blocks avoiding a point
            res = [b for b in blocks if 0 not in b]
            lam_res = lam_vec(res, 26)
            t = triples(26)
            lr = lam_res[~(t == 0).any(1)]
            log(f"residual at point 0: {len(res)} blocks on 25 points; triple multiplicities among the 25 points: "
                f"{dict(Counter(lr.tolist()))}")
        if v == 20:
            # look for any point whose deletion / derivation is regular
            for x in range(20):
                der = [set(b) - {x} for b in blocks if x in b]
                P = pair_lam(der + [set()], 20)
                mask = np.array([a != x and b != x for a, b in itertools.combinations(range(20), 2)])
                P = P[mask]
                if x < 3 or P.min() == P.max():
                    log(f"derived structure at point {x}: {len(der)} blocks of size 5 on 19 points; "
                        f"pair multiplicities {dict(sorted(Counter(P.tolist()).items()))}")
            # is there a sub-collection forming a smaller Steiner-like structure? report blocks through the 6-covered triple
            lam = lam_vec(blocks, 20)
            t = triples(20)
            top = t[lam == lam.max()]
            log(f"triples with max multiplicity {lam.max()}: {top.tolist()}")
            log("blocks (sorted):")
            for b in sorted(sorted(b) for b in blocks):
                log("   " + " ".join(f"{x:2d}" for x in b))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "lat02_invariants.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, "lat02_structure.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
