"""Independent check: LJCR C(26,6,3) blocks form a Steiner system S(3,6,26) (every triple exactly once),
point degree 30, pair multiplicity 6, block intersections in {0,1,2}."""
import itertools, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import design

B = [tuple(sorted(b)) for b in design.ljcr_blocks(26, 6, 3)]
print("blocks", len(B))
tri = Counter(t for b in B for t in itertools.combinations(b, 3))
print("triples covered", len(tri), "of", 26 * 25 * 24 // 6, "multiplicities", Counter(tri.values()))
pair = Counter(p for b in B for p in itertools.combinations(b, 2))
print("pair multiplicities", Counter(pair.values()))
deg = Counter(x for b in B for x in b)
print("point degrees", Counter(deg.values()))
inter = Counter(len(set(a) & set(b)) for a, b in itertools.combinations(B, 2))
print("block intersection sizes", dict(sorted(inter.items())))
