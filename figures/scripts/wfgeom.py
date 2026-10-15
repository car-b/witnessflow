"""Shared, deterministic geometry for the WitnessFlow figures (figure plan v2, items D1/D2).

- LJCR C(26,6,3) blocks (= S(3,6,26)), in LJCR order (block index 0..129 = e5b tube index for r <= 130).
- WINGMEN marker label of every design point under budget.ljcr_stack(seed=0)  (e5b / ledger mapping).
- Rose-window structure: each block = base_k + s (mod 26), k = 0..4 orbit, s = shift.
- GF(25) = F5[s]/(s^2-2) coordinates (lat02 isomorphism; model index i = 5a+b <-> a + b*s, 25 = infinity),
  optionally re-centred by a Moebius map so that a chosen point goes to infinity.
- AG(2,5) view: lines through infinity with their slope class; norm circles N(z-c)=rho for the other blocks.
Everything is asserted, nothing is random beyond reproducing ljcr_stack(seed=0)."""
import itertools, json, os, sqlite3
from collections import Counter
import numpy as np

WF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LJCR_DB = os.environ.get("WITNESSFLOW_LJCR_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ljcr_snapshot_subset.sqlite"))
ROSE_BASE = [(0, 2, 4, 12, 15, 18), (0, 1, 3, 7, 9, 10), (0, 2, 7, 11, 17, 21), (0, 1, 6, 11, 12, 19), (0, 1, 2, 5, 14, 23)]
# lat02_structure.txt: explicit point map LJCR -> GF(25) model index (5a+b <-> a+b*s; 25 = infinity)
LAT02_ISO = {0: 0, 1: 1, 2: 2, 3: 10, 4: 14, 5: 4, 6: 15, 7: 9, 8: 6, 9: 11, 10: 7, 11: 24, 12: 22, 13: 19, 14: 25, 15: 13,
             16: 5, 17: 8, 18: 20, 19: 16, 20: 21, 21: 23, 22: 12, 23: 3, 24: 18, 25: 17}
INF = "inf"


def ljcr26():
    con = sqlite3.connect(f"file:{LJCR_DB}?mode=ro", uri=True)
    size, blob = con.execute("select size, blocks from covers where v=26 and k=6 and t=3").fetchone()
    con.close()
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(size, 6)
    return [tuple(sorted(map(int, b))) for b in arr]


BLOCKS = ljcr26()
assert len(BLOCKS) == 130


def rose_orbits():
    """block index -> (orbit k, shift s) with block == base_k + s (mod 26)."""
    lookup = {}
    for k, b in enumerate(ROSE_BASE):
        for s in range(26):
            lookup[frozenset((x + s) % 26 for x in b)] = (k, s)
    out = [lookup[frozenset(b)] for b in BLOCKS]  # KeyError if the sets differ
    assert len(set(out)) == 130
    return out


ORBIT = rose_orbits()


def wingmen_labels():
    """marker name of every design point 0..25 under ljcr_stack(seed=0) (first copy)."""
    proto = json.load(open(os.path.join(WF, "protocols", "wingmen_v1.json"), encoding="utf-8"))
    thr = json.load(open(os.path.join(WF, "results", "wingmen_v1", "thresholds.json"), encoding="utf-8"))
    markers = list(thr.keys())
    bb = set(proto["backbone"])
    cand = [i for i, m in enumerate(markers) if m not in bb and thr[m]["reliable"]]
    assert len(cand) == 26
    perm = np.random.default_rng(0).permutation(len(cand))
    return [markers[cand[perm[x]]] for x in range(26)]


LABELS = wingmen_labels()
POINT = {m: i for i, m in enumerate(LABELS)}
TSTAR = ("CCR7", "CD27", "CD45RA")
TSTAR_TUBE = [i for i, b in enumerate(BLOCKS) if set(POINT[m] for m in TSTAR) <= set(b)]
assert TSTAR_TUBE == [72], TSTAR_TUBE
assert sorted(LABELS[x] for x in BLOCKS[72]) == sorted(["CD127", "CCR4", "CCR7", "CD27", "CD45RA", "PD1"])


# ------------------------------------------------------------------ GF(25)
def gadd(x, y):
    return ((x[0] + y[0]) % 5, (x[1] + y[1]) % 5)


def gsub(x, y):
    return ((x[0] - y[0]) % 5, (x[1] - y[1]) % 5)


def gmul(x, y):
    return ((x[0] * y[0] + 2 * x[1] * y[1]) % 5, (x[0] * y[1] + x[1] * y[0]) % 5)


ELS = [(a, b) for a in range(5) for b in range(5)]
GINV = {x: y for x in ELS for y in ELS if gmul(x, y) == (1, 0)}


def norm(x):
    return (x[0] * x[0] - 2 * x[1] * x[1]) % 5


def coords(inf_point=None):
    """design point -> GF(25) element (a, b) meaning a + b*s, or INF.  If inf_point is given, apply
    w = 1/(z - z0) (z0 = that point's element), which sends z0 -> INF and INF -> 0; circles are preserved."""
    base = {p: (INF if i == 25 else (i // 5, i % 5)) for p, i in LAT02_ISO.items()}
    if inf_point is None or base[inf_point] == INF:
        return base
    z0 = base[inf_point]
    out = {}
    for p, z in base.items():
        if z == INF:
            out[p] = (0, 0)
        elif z == z0:
            out[p] = INF
        else:
            out[p] = GINV[gsub(z, z0)]
    return out


def ag_view(inf_point):
    """Return dict with per-point coords, lines (blocks through inf_point) with slope class, and norm circles."""
    C = coords(inf_point)
    lines, circles = [], []
    for bi, b in enumerate(BLOCKS):
        if inf_point in b:
            pts = [C[p] for p in b if p != inf_point]
            d = gsub(pts[1], pts[0])
            # direction class: slope m for (1, m) after scaling, 'inf' for (0, 1)
            if d[0] == 0:
                slope = INF
            else:
                slope = gmul((d[1], 0), (pow(d[0], 3, 5), 0))[0]  # d1 / d0 in F5
            for q in pts[2:]:  # collinearity check
                e = gsub(q, pts[0])
                assert (d[0] * e[1] - d[1] * e[0]) % 5 == 0
            lines.append(dict(block=bi, slope=slope, points=[p for p in b if p != inf_point]))
        else:
            zs = [C[p] for p in b]
            found = None
            for c in ELS:
                ns = {norm(gsub(z, c)) for z in zs}
                if len(ns) == 1 and 0 not in ns:
                    found = (c, ns.pop())
                    break
            assert found is not None, bi
            circles.append(dict(block=bi, centre=found[0], rho=found[1], points=list(b)))
    assert len(lines) == 30 and len(circles) == 100
    assert Counter(l["slope"] for l in lines) == Counter({s: 5 for s in [0, 1, 2, 3, 4, INF]})
    assert Counter((c["centre"], c["rho"]) for c in circles).most_common(1)[0][1] == 1
    return dict(coords=C, lines=lines, circles=circles)


if __name__ == "__main__":
    print("labels:", LABELS)
    print("orbit sizes:", Counter(k for k, s in ORBIT))
    print("T* tube 72 orbit/shift:", ORBIT[72], "markers:", [LABELS[x] for x in BLOCKS[72]])
    v = ag_view(POINT["CCR7"])
    q, r = POINT["CD27"], POINT["CD45RA"]
    print("CCR7 at infinity; CD27 at", v["coords"][q], " CD45RA at", v["coords"][r])
    thru = [l for l in v["lines"] if q in l["points"]]
    print("lines through CD27:", [(l["block"], l["slope"]) for l in thru])
    print("line through CD27 & CD45RA:", [l["block"] for l in thru if r in l["points"]])
