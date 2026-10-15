"""Independent check of the four WitnessFlow designs in Supplementary Table S3 (sheet Blocks).

usage:  python verify_designs.py [path/to/S3_Blocks.tsv]
        (default: tsv/S3_Blocks.tsv next to this script)

Standalone: Python standard library only; reads nothing but the Blocks TSV and imports no project code.
The design specifications below are written into this script; they are NOT inferred from the points that
happen to occur in the file, so a missing point, block or design is reported as a failure.

  design  blocks x size  target set and pass condition
  S26     130 x 6        all C(26,3) = 2600 triples of the points 0..25;  lambda = 1 for every triple
  C20      71 x 6        all C(20,3) = 1140 triples of the points 0..19;  lambda >= 1 for every triple
                         (the recorded certificate range is lambda 1-6; a different range is reported as a failure)
  OA      125 x 5        points 0..24 in 5 families: point p has family p // 5 and slot p % 5
                         (point = 5 * family + slot); every block holds exactly one point of each family;
                         targets = the 1250 cross-family triples only (three points from three different
                         families, C(5,3) * 5^3); lambda = 1 for every cross-family triple. Triples with two
                         points of one family are not targets (they are reported, not judged).
  PSL     570 x 6        all C(20,3) = 1140 triples of the points 0..19;  lambda = 10 for every triple

Checks per design: present in the file; block count; block size; every point an integer in 0..v-1; no
repeated point inside a block; no repeated block; block_index unique; lambda over the target set (with the
multiplicity histogram including lambda = 0). Point-degree and pair-multiplicity histograms (including zero)
are printed for information. Rows with a design_id outside the four specifications are a failure.

Scope: this verifies combinatorial properties of these four designs only. It does not cover the 44
improved-bound certificates of sheet Improved_bounds and does not verify automorphism groups, isomorphism
identity or optimality.

Output: one PASS / FAIL line per design (with the reasons for a FAIL) and an overall line; the exit code is
0 only if every design passes."""
import csv
import hashlib
import itertools
import os
import sys
from collections import Counter

SPECS = {
    "S26": dict(v=26, b=130, k=6, target="all", rule="eq", lam=1),
    "C20": dict(v=20, b=71, k=6, target="all", rule="ge", lam=1, recorded_range=(1, 6)),
    "OA": dict(v=25, b=125, k=5, target="cross_family", rule="eq", lam=1, families=5, family_size=5),
    "PSL": dict(v=20, b=570, k=6, target="all", rule="eq", lam=10),
}
N_TARGETS = {"S26": 2600, "C20": 1140, "OA": 1250, "PSL": 1140}
POINT_COLS = [f"point_{j}" for j in range(1, 7)]


def family(p):
    return p // 5


def is_target(design, t):
    if SPECS[design]["target"] == "all":
        return True
    return len({family(x) for x in t}) == 3


def parse_point(s):
    s = s.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        x = float(s)                      # tolerate "12.0"; anything non-integral is an error
        if not x.is_integer():
            raise ValueError(f"non-integer point label {s!r}")
        return int(x)


def hist(counter_values):
    return dict(sorted(Counter(counter_values).items()))


def check_design(name, rows):
    sp = SPECS[name]
    v, k, b = sp["v"], sp["k"], sp["b"]
    reasons, info = [], []
    blocks, idx = [], []
    for r in rows:
        try:
            pts = [parse_point(r.get(c, "") or "") for c in POINT_COLS]
        except ValueError as e:
            reasons.append(f"block_index {r.get('block_index')}: {e}")
            continue
        pts = [p for p in pts if p is not None]
        bi = r.get("block_index", "")
        idx.append(bi)
        if len(pts) != k:
            reasons.append(f"block_index {bi}: block size {len(pts)} != {k}")
        bad = [p for p in pts if not 0 <= p < v]
        if bad:
            reasons.append(f"block_index {bi}: point label(s) {bad} outside 0..{v - 1}")
        if len(set(pts)) != len(pts):
            reasons.append(f"block_index {bi}: repeated point in block {sorted(pts)}")
        if r.get("block_size", "").strip() not in ("", str(len(pts))):
            reasons.append(f"block_index {bi}: block_size column {r['block_size']!r} != {len(pts)} listed points")
        if name == "OA" and len(pts) == k and len({family(p) for p in pts}) != sp["families"]:
            reasons.append(f"block_index {bi}: block {sorted(pts)} does not hold one point of each of the 5 families")
        blocks.append(tuple(sorted(pts)))
    if len(blocks) != b:
        reasons.append(f"block count {len(blocks)} != {b}")
    if len(set(idx)) != len(idx):
        reasons.append("repeated block_index")
    dup = [blk for blk, n in Counter(blocks).items() if n > 1]
    if dup:
        reasons.append(f"{len(dup)} repeated block(s), e.g. {list(dup[0])}")
    # multiplicities over the full point / pair / triple universe (labels outside 0..v-1 are ignored here)
    good = [blk for blk in blocks if all(0 <= p < v for p in blk) and len(set(blk)) == len(blk)]
    deg = Counter(p for blk in good for p in blk)
    pair = Counter(q for blk in good for q in itertools.combinations(blk, 2))
    trip = Counter(q for blk in good for q in itertools.combinations(blk, 3))
    deg_h = hist(deg.get(p, 0) for p in range(v))
    if name == "OA":
        cross_pairs = [q for q in itertools.combinations(range(v), 2) if family(q[0]) != family(q[1])]
        within_pairs = [q for q in itertools.combinations(range(v), 2) if family(q[0]) == family(q[1])]
        info.append(f"point degree histogram {{degree: points}} = {deg_h}")
        info.append(f"cross-family pair multiplicity {{lambda: pairs}} = {hist(pair.get(q, 0) for q in cross_pairs)} "
                    f"({len(cross_pairs)} pairs)")
        info.append(f"within-family pair multiplicity = {hist(pair.get(q, 0) for q in within_pairs)} ({len(within_pairs)} pairs)")
    else:
        allp = list(itertools.combinations(range(v), 2))
        info.append(f"point degree histogram {{degree: points}} = {deg_h}")
        info.append(f"pair multiplicity {{lambda: pairs}} = {hist(pair.get(q, 0) for q in allp)} ({len(allp)} pairs)")
    targets = [t for t in itertools.combinations(range(v), 3) if is_target(name, t)]
    if len(targets) != N_TARGETS[name]:
        reasons.append(f"target count {len(targets)} != {N_TARGETS[name]} (specification error)")
    lam = [trip.get(t, 0) for t in targets]
    lam_h = hist(lam)
    info.append(f"target triple multiplicity {{lambda: triples}} = {lam_h} ({len(targets)} targets)")
    if name == "OA":
        others = [t for t in itertools.combinations(range(v), 3) if not is_target(name, t)]
        info.append(f"non-target (within-family) triple multiplicity = {hist(trip.get(t, 0) for t in others)} "
                    f"({len(others)} triples, not judged)")
    if lam:
        lo, hi = min(lam), max(lam)
        if sp["rule"] == "eq":
            nbad = sum(x != sp["lam"] for x in lam)
            if nbad:
                reasons.append(f"{nbad} target triple(s) with lambda != {sp['lam']} (observed range {lo}-{hi})")
        else:
            nbad = sum(x < sp["lam"] for x in lam)
            if nbad:
                reasons.append(f"{nbad} target triple(s) with lambda < {sp['lam']} (lambda = 0: {lam_h.get(0, 0)})")
            if "recorded_range" in sp and (lo, hi) != sp["recorded_range"]:
                reasons.append(f"lambda range {lo}-{hi} differs from the recorded certificate range "
                               f"{sp['recorded_range'][0]}-{sp['recorded_range'][1]}")
        crit = f"lambda = {sp['lam']}" if sp["rule"] == "eq" else f"lambda >= {sp['lam']}"
        summary = f"{len(blocks)} blocks x {k}; {len(targets)} targets; {crit}; observed lambda {lo}-{hi}"
    else:
        summary = f"{len(blocks)} blocks; no target evaluated"
    return reasons, info, summary


def main(argv):
    path = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsv", "S3_Blocks.tsv")
    if not os.path.isfile(path):
        print(f"FAIL  input file not found: {path}")
        return 2
    with open(path, "rb") as f:
        raw = f.read()
    print(f"input  {path}")
    print(f"sha256 {hashlib.sha256(raw).hexdigest()}  ({len(raw)} bytes)")
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    need = {"design_id", "block_index", *POINT_COLS}
    missing_cols = sorted(need - set(reader.fieldnames or []))
    if missing_cols:
        print(f"FAIL  missing column(s): {missing_cols}")
        return 1
    by = {}
    for r in reader:
        by.setdefault(r["design_id"], []).append(r)
    failed = False
    for name in SPECS:
        if name not in by:
            print(f"FAIL  {name}: design missing from the file")
            failed = True
            continue
        reasons, info, summary = check_design(name, by[name])
        print(f"{'FAIL' if reasons else 'PASS'}  {name}: {summary}")
        for x in info:
            print(f"        {x}")
        for x in reasons[:20]:
            print(f"        reason: {x}")
        if len(reasons) > 20:
            print(f"        reason: ... {len(reasons) - 20} more")
        failed = failed or bool(reasons)
    extra = sorted(set(by) - set(SPECS))
    if extra:
        print(f"FAIL  unexpected design_id value(s) in the file: {extra}")
        failed = True
    print(f"OVERALL {'FAIL' if failed else 'PASS'}: {len(SPECS)} designs specified, "
          f"{sum(len(v) for v in by.values())} block rows read")
    print("Scope: combinatorial properties of these 4 designs only; not the 44 improved-bound certificates; "
          "automorphism groups, isomorphism identity and optimality are not verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
