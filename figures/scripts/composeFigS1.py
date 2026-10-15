"""Compose Supplementary Figure 1 (suppplan_v2 SF1): S1A across the top; S1B left with S1C over S1D on the right;
S1E across the bottom.  1:1 panel PDFs, no scaling (compose.py).

Panel sizes are set in figS1.py so that every row fills the 184 mm content width with the 2 mm minimum gap and the
left atlas is exactly as tall as the right stack (S1B = S1C + S1D + one letter band + one vertical gap = 143.5 mm):
S1A 184 x 52;  S1B 112 x 143.5;  S1C 70 x 70;  S1D 70 x 69.5;  S1E 184 x 49.
Output: figures/composite/FigS1.pdf/.png.   usage: python composeFigS1.py
"""
from compose import Row, Col, P, compose, report

WIDTH = 184.0
layout = Col([
    P("S1A", "a"),
    Row([P("S1B", "b"), Col([P("S1C", "c"), P("S1D", "d")])]),
    P("S1E", "e"),
])

if __name__ == "__main__":
    # PDF page boxes are stored as float32, so a 184 mm panel reads back as 184.000009 mm; compose() checks with a
    # 1e-6 tolerance. Absorb only that read-back noise (a real overflow of > 0.001 mm still fails here).
    assert layout.w <= WIDTH + 1e-3, f"layout {layout.w:.4f} mm wider than {WIDTH} mm"
    mid = layout.c[1]
    left, right = mid.c[0].h, mid.c[1].h
    assert abs(left - right) < 0.05, f"S1B column {left:.2f} mm vs S1C+S1D column {right:.2f} mm"
    rep = compose("FigS1", layout, width=max(WIDTH, layout.w))
    report(rep)
    assert rep["audit_ok"], rep
    assert rep["page_mm"][1] <= 297.0
