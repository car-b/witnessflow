"""Explicit automorphism group of La Jolla C(20,6,3)=71: generator, point cycles, block orbits, and the
substructure around the special pair (the only pair with multiplicity 12).  Output: out_lat/lat02b_aut20.txt"""
import itertools, os
from collections import Counter
import networkx as nx
from lat_common import OUT, design, lam_vec, triples
from lat02_structure import inc_graph, point_invariants

v = 20
B = [frozenset(b) for b in design.ljcr_blocks(20, 6, 3)]
inv = point_invariants(v, B)
lab = {x: str(inv[x]) for x in range(v)}
out = []
gens = []
for q in range(1, v):
    if inv[q] != inv[0]:
        continue
    l1, l2 = dict(lab), dict(lab)
    l1[0] = lab[0] + "|X"
    l2[q] = lab[q] + "|X"
    G1, G2 = inc_graph(v, B, l1), inc_graph(v, B, l2)
    m = nx.vf2pp_isomorphism(G1, G2, node_label="lab")
    if m:
        perm = {a[1]: b[1] for a, b in m.items() if a[0] == "p"}
        gens.append(perm)
        break
g = gens[0]
seen, cycles = set(), []
for x in range(v):
    if x in seen:
        continue
    c, y = [], x
    while y not in seen:
        seen.add(y)
        c.append(y)
        y = g[y]
    cycles.append(tuple(c))
out.append(f"automorphism g (0 -> {g[0]}): point cycles {cycles}")
img = lambda b: frozenset(g[x] for x in b)
assert {img(b) for b in B} == set(B)
seenb, borb = set(), []
for b in B:
    if b in seenb:
        continue
    o = {b, img(b), img(img(b))}
    seenb |= o
    borb.append(len(o))
out.append(f"block orbit sizes under <g>: {dict(Counter(borb))}  (71 = 3*{borb.count(3)} + {borb.count(1)})")
fixed_blocks = [sorted(b) for b in B if img(b) == b]
out.append(f"fixed blocks: {fixed_blocks}")
P = Counter()
for b in B:
    for a, c in itertools.combinations(sorted(b), 2):
        P[(a, c)] += 1
special = [p for p, n in P.items() if n == 12]
out.append(f"pair(s) with multiplicity 12: {special}")
a, c = special[0]
thr = [sorted(b - {a, c}) for b in B if a in b and c in b]
out.append(f"the {len(thr)} blocks through that pair, residues (4-sets on the other 18 points): {thr}")
cnt = Counter(x for r in thr for x in r)
out.append(f"   point coverage of these residues: {dict(sorted(cnt.items()))}")
# remove the two special points: what is left on 18 points?
rest = [b for b in B if a not in b and c not in b]
out.append(f"blocks avoiding both special points: {len(rest)}; blocks containing exactly one: "
           f"{sum((a in b) != (c in b) for b in B)}; containing both: {len(thr)}")
txt = "\n".join(out)
print(txt)
open(os.path.join(OUT, "lat02b_aut20.txt"), "w", encoding="utf-8").write(txt + "\n")
