# Release notes (public copy of 2026-10-14)

This repository is a public copy of the WitnessFlow computational version frozen on 2026-10-09 (`FREEZE_v1/`) and of the
post-freeze addendum of 2026-10-10/11 (`FREEZE_v1_ADDENDUM/`). `release_map.tsv` lists every file of the freeze manifest
and of the addendum manifest with the SHA256 recorded there, the path of the released copy and its SHA256.

## Unchanged files

520 released files are byte-identical to the recorded versions (`sha256_matches_record` = True in
`release_map.tsv`).

## Edited code files

Only comments, docstrings and machine-specific path constants were edited. With docstrings removed, the abstract syntax
trees of the released and recorded versions are identical except for these statements:

- `src/design.py`, `figures/scripts/wfgeom.py`: the La Jolla database path `LJCR_DB` is read from the environment variable
  `WITNESSFLOW_LJCR_DB` and defaults to `data/ljcr_snapshot_subset.sqlite` (`src/design.py` gained `import os`).
- `tools/make_v2.py`, `make_rare.py`, `make_shift.py`, `make_shift0.py`, `make_stress.py`, `add_e3_section.py`: the protocol
  folder is located relative to the script instead of by an absolute path (`import os` added). These one-off generators
  still join file names with Windows separators; their outputs are the frozen protocols in `protocols/`.

In the other edited code files, internal references in comments and docstrings were removed or reworded. Edited code files:

- `figures/derive/d3_tstar_tube.py`
- `figures/derive/sf2_c_forest.py`
- `figures/derive/sf2_d_selfcheck.py`
- `figures/scripts/fig2.py`
- `figures/scripts/fig4.py`
- `figures/scripts/figS2.py`
- `figures/scripts/wfgeom.py`
- `figures/scripts/wfstyle.py`
- `src/audit.py`
- `src/bridge.py`
- `src/compare_stress.py`
- `src/design.py`
- `src/e3b.py`
- `src/e3e.py`
- `src/e5e.py`
- `src/e5h_calib.py`
- `src/e5i_shuffle.py`
- `src/e5j_ledger.py`
- `src/e5k_boundary.py`
- `src/e5l_act5_threshold.py`
- `src/make_figures.py`
- `src/replay.py`
- `src/sum_e5d_fixed.py`
- `src/sum_e5d_unified.py`
- `tests/test_toy.py`
- `tools/add_e3_section.py`
- `tools/make_rare.py`
- `tools/make_shift.py`
- `tools/make_shift0.py`
- `tools/make_stress.py`
- `tools/make_v2.py`

## Rewritten tables and records

- `FREEZE_v1/MANIFEST.tsv`
- `FREEZE_v1_ADDENDUM/MANIFEST_addendum.tsv`
- `FREEZE_v1_ADDENDUM/PROVENANCE_edges.tsv`
- `tools/scan/crosscheck.tsv`
- `tools/scan/main_table.tsv`
- `FREEZE_v1_ADDENDUM/traces/ (42 trace files)`

- `tools/scan/main_table.tsv`: column `src` gives the search round (R1-R4, as in S3 Table) instead of internal run labels.
- `tools/scan/crosscheck.tsv`: column `file` gives the released certificate (`certificates/R*/...`); in column `sources` the
  run-folder labels were replaced by search rounds (R1-R4; R4-best is the best-of list of R4), while the origin tags
  (for example `baseline_B`, `prod`, `abl`) were kept. The row for C(33,20,5) (t = 5, outside the screen) has no released
  certificate.
- `FREEZE_v1/MANIFEST.tsv` and `FREEZE_v1_ADDENDUM/`: some files that are not released are listed under neutral names or
  identifiers (for example `ljcr_snapshot_db` for the La Jolla snapshot); rows, byte counts and SHA256 values are unchanged.
  Machine-specific path prefixes were removed from the traces, and references to internal review rounds were removed from
  free-text fields.
- Edited notes: `FREEZE_v1/README.md`, `FREEZE_v1/REPRODUCE.md`, `FREEZE_v1/FIGURE_MAP.md` and
  `FREEZE_v1_ADDENDUM/test_toy_rerun_2026-10-10.log` (machine-specific paths and internal references).

`results/e5c.stdout.txt`, the run log that is the source of the Fig 2d marker families, is not in the freeze manifest; it is
recorded in the post-freeze provenance edges (`FREEZE_v1_ADDENDUM/PROVENANCE_edges.tsv`, role `unlisted_result`, with its
SHA256) and is released unchanged.

Because of these edits, the SHA256 values recorded in the manifests, in the addendum and in S7 Table (sheet Sources) refer
to the recorded versions; `release_map.tsv` gives both values.

## New files

`README.md`, `RELEASE_NOTES.md`, `release_map.tsv`, `LICENSE`, `LICENSE-data.md`, `requirements.txt`, `.gitignore`,
`data/README.md`, `data/raw/README.md`, `figures/README.md`, `data/ljcr_snapshot_subset.sqlite`, `certificates/` (exported from the saved covering files listed in
S3 Table; 0-based points in JSON, 1-based points in text), `figures/source_data/fig1A_cells.tsv` and `fig1A_paths.tsv`
(public form of the Fig 1a display tables: round labels as in S3 Table; node ids, run ids and evidence as in the S3 Fig a
Source Data) and `figures/source_data/fig3A_illustrative_cp.tsv` (the illustrative Clopper-Pearson half-widths of Fig 3a,
recomputed with the formula of `figures/scripts/fig3.py`).

## Not released

- Raw data and the full La Jolla snapshot (see `data/README.md`).
- Internal notes (including the post hoc reviews A, B and G cited in S3 Fig a), internal scans of the search records, and
  internal helper and maintenance scripts, including `tools/freeze_manifest.py`, which needs the full internal project tree
  (the manifest it wrote is `FREEZE_v1/MANIFEST.tsv`).
- Scripts that read these notes: `figures/scripts/fig1.py`, `figS3.py`, `composeFig1.py`, `composeFigS3.py`, the
  Supplementary Table builders (`table_S1.py`-`table_S7.py`, `wftables.py`, `_verify_tables.py`), `figures/derive/sf3_*.py`,
  and the tools that produced the addendum. Their outputs are in S1 Data, `figures/source_data/` and `figures/supp_tables/`.
- Display tables with internal provenance columns (`figures/derived/fig1_*_1A.tsv`, `figures/derived/figS3_A_*.tsv`); their
  public versions are in `figures/source_data/`.
- Figure files (provided with the article).
