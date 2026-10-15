# WitnessFlow: code, protocols, results and covering certificates

Code and data release for the article

> WitnessFlow: classical designs for auditing and planning joint-phenotype evidence in multi-tube cytometry
> (journal, year and DOI to be added on publication)

WitnessFlow uses classical combinatorial designs (covering designs, Steiner systems, orthogonal arrays) to decide which
markers are measured together in multi-tube cytometry, to audit for each claim whether its evidence is direct or only
constrained by margins, and to allocate a fixed cell budget to the evidence each claim needs. This repository holds the
computational version used for the article (frozen on 2026-10-09), the post-freeze provenance addendum, the data of every
figure panel and the 44 covering certificates listed in S3 Table.

## Contents

| Path | Contents |
|---|---|
| `src/` | Analysis code: evidence audit, follow-up and budgeted designs, replays, supplementary experiments (e3, e5b-e5l) |
| `tests/test_toy.py` | 26 engineering checks |
| `protocols/` | The 12 frozen analysis protocols (JSON; S7 Table, sheet Protocols) |
| `tools/` | Download and checksum helpers, generators of the protocol variants, the S(3,6,26) check, and the design-structure analyses `tools/scan/lat*.py` with their frozen outputs in `tools/scan/out_lat/` |
| `tools/scan/main_table.tsv`, `history.tsv`, `crosscheck.tsv` | Frozen tables behind S3 Table: improved bounds, record history, independent re-check |
| `results/` | Frozen result tables of all protocols and experiments (225 files) and the run log `e5c.stdout.txt` |
| `figures/` | Figure scripts (`scripts/`, `derive/`), frozen figure tables (`data/`), display tables (`derived/`), Source Data (`source_data/`) and Supplementary Tables S1-S7 with UTF-8 TSV mirrors (`supp_tables/`) |
| `certificates/` | The 44 coverings with fewer blocks than the 2026-10-04 snapshot of the online covering table, as JSON and text, with an index and a standalone verifier |
| `data/` | Download instructions and SHA256 values for the public raw data; `ljcr_snapshot_subset.sqlite` with the 216 La Jolla coverings the code can read |
| `FREEZE_v1/` | Freeze manifest (SHA256 of 448 files) and freeze notes (in Chinese) |
| `FREEZE_v1_ADDENDUM/` | Post-freeze addendum: runtime traces of the figure and table scripts, provenance edges, test rerun log |
| `RELEASE_NOTES.md`, `release_map.tsv` | How this public copy differs from the recorded files, file by file |

## Requirements

Python 3.13 with the package versions in `requirements.txt` (the frozen environment, Windows 11). The figure scripts use the
Arial font.

    pip install -r requirements.txt

## Quick checks (no raw data needed)

Run from the repository root:

    python certificates/verify_certificates.py      # 44 certificates: block size, point range, every t-subset covered
    python tests/test_toy.py                        # 26 PASS, FAILS: 0
    python figures/supp_tables/verify_designs.py    # the four replay designs of S3 Table
    python tools/verify_s3626.py                    # La Jolla C(26,6,3) is a Steiner system S(3,6,26)

## Reproducing the analyses

Download the raw data into `data/raw/` and check the SHA256 values (`data/README.md`). Then run, from the repository root
(`<protocol>` is a file in `protocols/`; seeds are fixed in the scripts or their arguments):

1. Main replays. They write `thresholds.json`, which every later step reads:
   `python src/replay.py protocols/levine32_v2.json`, `python src/replay.py protocols/wingmen_v1.json`, and the stress
   protocols `levine32_v2_rareB_{1,0p3,0p1}.json` and `levine32_v2_shift{0,Allp0p5,NonBBp0p5,NonBBm0p5}.json`
   (`levine32_v1.json` keeps the superseded threshold rule; `levine32_v2_bb4/bb6.json` are early stress runs).
2. Summaries: `python src/summarize.py results/<protocol_id>`, `python src/compare_rare.py`, `python src/compare_shift.py`,
   `python src/compare_stress.py`.
3. Exploration: `python src/explore.py <protocol> e1 e2`; bridge markers: `python src/bridge.py protocols/levine32_v2.json`.
4. Budgeted designs: `python src/e3e.py <protocol>` and `python src/e5f_attrib.py <protocol>` (`e3.py`, `e3b.py`, `e3c.py`
   and `e3d.py` are earlier exploratory versions, superseded by `e3e.py`, `e5f_attrib.py` and `e5i_shuffle.py`).
5. Supplementary experiments:
   - `python src/e5b.py <protocol>` (both datasets); `python src/e5b_supp.py protocols/wingmen_v1.json`;
   - `python src/e5c.py protocols/wingmen_v1.json > results/e5c.stdout.txt` (its log lists the Fig 2d marker families);
   - `python src/e5d.py <protocol> 1000 200 <k> 8` for k = 0..7, then `python src/sum_e5d.py results/<protocol_id>` and
     `python src/sum_e5d_fixed.py results/<protocol_id> <r>` with r = 71 (Levine-32) or 130 (WINGMEN);
   - `python src/e5e.py <anneal|breakout> 141 <seed> <seconds> <out_dir>` (seeds 0-4 for 600 s, seeds 10-14 for 1,800 s;
     the runs stop on wall-clock time, so their block tables can differ between machines; the frozen ones are in
     `results/e5e_lambda2*`);
   - `python src/e5h_calib.py <pairs|singles|pairs+direct|mixed|direct_only> 1000`; `python src/e5g_threshold.py <protocol>`;
   - `python src/e5i_shuffle.py <protocol>`, `python src/e5j_ledger.py <protocol>`, `python src/e5k_boundary.py <protocol>`,
     `python src/e5l_act5_threshold.py <protocol>`;
   - CytoNorm magnitude reference: `python src/real_batch.py`.
6. Design checks: `python tools/verify_s3626.py`; from `tools/scan/`: `python lat01_balance.py`, `python lat02_structure.py`
   (needs networkx), `python lat05_transversal.py` and the other `lat*.py`; `lat04_replay_small.py`, `lat06_derived.py` and
   `lat07_bridge_free.py` read the raw data.
7. Frozen figure tables: `python src/make_figures.py` (writes `figures/data/`).
8. Display tables and panels of Figs 2-5 and S1-S2 Figs, in this order (each from its own folder, `figures/derive/` or
   `figures/scripts/`): `d3_tstar_tube.py`, `d3b_pair_tubes.py`; `fig2_data.py`, `fig2.py`, `fig2_check.py`, `fig3.py`,
   `fig4.py`, `fig5.py`; `sf1_ledger.py`, `sf1_tables.py`, `sf2_a_nodes.py` ... `sf2_e_ablation.py`; `figS1.py`, `figS2.py`;
   `composeFig2.py` ... `composeFigS2.py`. `d3_tstar_tube.py`, `d3b_pair_tubes.py` and `sf1_ledger.py` read the raw data.
   Panels are written to `figures/panels/`, pages to `figures/composite/`. The scripts require the Arial font
   (`figures/scripts/compose.py` reads the bold face from `C:\Windows\Fonts`; change `LETTER_FONT` there on other systems).

Re-running a script overwrites the corresponding frozen file; compare with the SHA256 in `release_map.tsv`. In a test run
from a fresh copy of this repository, `python src/replay.py protocols/levine32_v2.json` and `python src/real_batch.py`
rewrote five frozen files. Four were byte-identical (`thresholds.json`, `reference_claims.tsv`, `followup.tsv`,
`batch_shift.tsv`). The fifth, `results/levine32_v2/initial_audit.tsv`, was written by an earlier version of `src/replay.py`:
all its 37 columns and 120 rows were reproduced exactly, and the current code adds five diagnostic columns
(`knn_block_within`, `tube_minp`, `tube_maxtv`, `tube_checks`, `parent_frac_range`); S4 Table marks the within-sample kNN
baseline as not computed for Levine-32. The figure scripts of step 8 reproduced the released display tables and the
article's panel and page PNG images byte for byte (PDF files embed a creation date and were not compared).

## La Jolla covering tables

`src/design.py` and `figures/scripts/wfgeom.py` read coverings from `data/ljcr_snapshot_subset.sqlite`: 216 coverings with
t = 3, v <= 32 and k <= 12, copied without change from the La Jolla Coverings Repository snapshot (version 1.2) used in the
study (D. Gordon; archived at Zenodo, doi:10.5281/zenodo.19735294; CC BY 4.0). To use a complete snapshot with the same
table layout, set the environment variable `WITNESSFLOW_LJCR_DB`.

## Not included

- Raw data (public; see `data/README.md`) and the full La Jolla snapshot (1.3 GB).
- Internal notes and internal scans of the search records. The scripts for Fig 1 and S3 Fig and the builders of the
  Supplementary Tables also read these notes and are therefore not included; the data of every panel are in S1 Data and
  `figures/source_data/`, and the finished tables are in `figures/supp_tables/`.
- `release_map.tsv` gives the status of every recorded file.

## Licence

Code: MIT (`LICENSE`). Data, certificates, tables and documentation: CC BY 4.0 (`LICENSE-data.md`). Raw data keep their
original licences.

## Citation

Cite the article (reference to be added on publication) and the archived release of this repository (DOI to be added).
