# Covering certificates

This folder holds the 44 coverings C(v,k,t) of S3 Table (sheet Improved_bounds): for cells with 2 <= t <= 4, v <= 120 and
k <= 40, coverings with fewer blocks than the snapshot of the online covering table taken on 2026-10-04. A covering
C(v,k,t) of size b is a family of b k-subsets (blocks) of a v-set such that every t-subset lies in at least one block. The
values are feasible upper bounds, not proved minima. None of them had been submitted to a covering repository when the
article was written.

## Layout

- `R1/` to `R4/`: one folder per search round (S3 Table, column `trace_certificate_file`; a cell found in two rounds is
  stored under the round of the saved file).
- `C(v,k,t)_b.json`: keys `cell`, `v`, `k`, `t`, `size`, `points`, `blocks`; points are 0-based (0 to v-1), one block per line.
- `C(v,k,t)_b.txt`: the same blocks with 1-based points (1 to v), one block per line, as in the La Jolla repository.
- `index.tsv`: one row per certificate with the online-table and La Jolla v1.2 values, the recorded lower bound, the saving,
  and the SHA256 of both files.

## Verification

    python verify_certificates.py               # all 44
    python verify_certificates.py "C(39,8,4)"   # selected cells

For each certificate the verifier checks that the JSON and text files hold the same blocks, that the number of blocks equals
the stated size, that every block has k distinct points in range, and that every t-subset of the v points lies in at least
one block (exhaustive check with NumPy). It does not test minimality.

Licence: CC BY 4.0.
