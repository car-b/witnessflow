"""Supplementary Figure 2 composite (1:1 panel PDFs, no scaling) -> figures/composite/FigS2.pdf/.png.

Layout (suppplan_v2 SF2): a | b ; c (full width) ; d | e.  Panel sizes are set in figS2.py so that every row is
184 mm wide (2 mm gap) and the panels of a row have equal height.  Run from figures/scripts:  python composeFigS2.py
"""
from compose import Row, Col, P, compose, report

W = 184                  # content width (mm); compose() tolerates the PDF media-box rounding (1e-3 mm)

lay = Col([Row([P("S2A", "a"), P("S2B", "b")]),
           P("S2C", "c"),
           Row([P("S2D", "d"), P("S2E", "e")])])
print("layout natural width", round(lay.w, 1), "height", round(lay.h, 1))
rep = compose("FigS2", lay, width=W)
report(rep)
assert rep["audit_ok"], rep
assert rep["page_mm"][1] <= 297.0
