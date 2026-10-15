"""Verify the covering certificates in this folder (see README.md).

usage:  python verify_certificates.py [cell ...]      e.g.  python verify_certificates.py "C(39,8,4)"
A covering C(v,k,t) of size b is a family of b k-subsets (blocks) of {0..v-1} that together contain every t-subset.
Exit code 0 if every selected certificate passes."""
import csv
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def check(v, k, t, size, blocks):
    """Return (passed, message) for one design given as a list of 0-based blocks."""
    if len(blocks) != size:
        return False, f"{len(blocks)} blocks, expected {size}"
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False, f"block {b} does not have {k} distinct points"
        if min(b) < 0 or max(b) >= v:
            return False, f"block {b} has a point outside 0..{v - 1}"
    n = math.comb(v, t)
    seen = np.zeros(n, dtype=bool)
    # rank of a sorted t-subset a_1 < ... < a_t in the combinatorial number system: sum_i C(a_i, i)
    table = np.array([[math.comb(a, i + 1) for a in range(v)] for i in range(t)], dtype=np.int64)
    positions = np.array(list(itertools.combinations(range(k), t)), dtype=np.int64)
    for b in blocks:
        sub = np.array(sorted(b), dtype=np.int64)[positions]
        rank = np.zeros(len(sub), dtype=np.int64)
        for i in range(t):
            rank += table[i][sub[:, i]]
        seen[rank] = True
    uncovered = int(n - seen.sum())
    return uncovered == 0, f"{uncovered} of {n} {t}-subsets uncovered"


def main(argv):
    with open(os.path.join(HERE, "index.tsv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    wanted = {a.replace(" ", "") for a in argv[1:]}
    if wanted:
        rows = [r for r in rows if r["cell"] in wanted]
        missing = wanted - {r["cell"] for r in rows}
        if missing:
            print("not in index.tsv:", ", ".join(sorted(missing)))
            return 2
    failures = 0
    for r in rows:
        v, k, t, size = (int(r[c]) for c in ("v", "k", "t", "size"))
        with open(os.path.join(HERE, *r["json_file"].split("/")), encoding="utf-8") as f:
            d = json.load(f)
        with open(os.path.join(HERE, *r["txt_file"].split("/")), encoding="utf-8") as f:
            txt = [[int(x) - 1 for x in line.split()] for line in f if line.strip()]
        blocks = [list(b) for b in d["blocks"]]
        same = (d["v"], d["k"], d["t"], d["size"]) == (v, k, t, size) and \
            sorted(map(sorted, blocks)) == sorted(map(sorted, txt))
        ok, msg = check(v, k, t, size, blocks)
        ok = ok and same
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {r['cell']:<14} size {size:>5}  {msg}" + ("" if same else "  (JSON and text differ)"))
    print(f"{len(rows) - failures}/{len(rows)} certificates pass")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
