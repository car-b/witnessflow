# Figures

| Path | Contents |
|---|---|
| `scripts/` | Drawing scripts for Figs 2-5 (`fig2.py`-`fig5.py`, `fig2_data.py`, `fig2_check.py`) and S1-S2 Figs (`figS1.py`, `figS2.py`); shared style (`wfstyle.py`; Arial 7 pt, PDF audit in `save()`), design geometry (`wfgeom.py`), page composition (`compose.py`, `composeFig*.py`) and the PDF audit of all panels (`_audit_all.py`) |
| `derive/` | Deterministic derivations from frozen results: `d3_tstar_tube.py` (Fig 1d), `d3b_pair_tubes.py` (Fig 4c), `sf1_*.py` (S1 Fig), `sf2_*.py` (S2 Fig); `d3_tstar_tube.py`, `d3b_pair_tubes.py` and `sf1_ledger.py` also read the raw data (`data/README.md`) |
| `data/` | The 13 frozen figure tables (`src/make_figures.py`, FREEZE_v1) |
| `derived/` | Display tables: the values drawn in each panel |
| `source_data/` | Source Data of S1-S3 Figs, and public tables for Fig 1a (`fig1A_cells.tsv`, `fig1A_paths.tsv`) and Fig 3a (`fig3A_illustrative_cp.tsv`) |
| `supp_tables/` | Supplementary Tables S1-S7 (xlsx with README and data dictionary sheets), UTF-8 TSV mirrors in `tsv/`, table legends, and `verify_designs.py` |

Run the derivations and drawing scripts from their own folders in the order given in the main README (step 8), for example
`cd figures/scripts` then `python fig3.py`. Panels are written to `figures/panels/` (PDF with editable text, and PNG). S1 Data
(supplementary file of the article) collects every table used for the main and supplementary figures in one workbook.

The file names in `data/` and in `FREEZE_v1/FIGURE_MAP.md` follow the panel numbering at the freeze, before the figures were
rearranged (for example `fig2C_orthogonal_array.tsv` belongs to Fig 2e and `fig4D_ledger.tsv` to Fig 5c); `S1_Data_index.tsv`
of S1 Data maps every table to its final figure and panel. The scripts for Fig 1 and S3 Fig are not included (see the main
README); the display tables they wrote that are released here are listed in S1 Data.
