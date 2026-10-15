"""Compose Figure 5 (figplan_v2): 5A across the top; 5B | 5C; 5D | 5E.  1:1 panel PDFs, no scaling (compose.py).

Panel sizes are set in fig5.py so that every row fills the 184 mm content width with the 2 mm minimum gap and the
panels of a row are equally tall:  5A 184 x 52;  5B, 5C 91 x 72;  5D, 5E 91 x 72.
Output: figures/composite/Fig5.pdf/.png.   usage: python composeFig5.py
"""
from compose import Row, Col, P, compose, report

WIDTH = 184.0
layout = Col([
    P("5A", "a"),
    Row([P("5B", "b"), P("5C", "c")]),
    Row([P("5D", "d"), P("5E", "e")]),
])

if __name__ == "__main__":
    # PDF page boxes are stored as float32, so a 184 mm panel reads back as 184.000009 mm; compose() checks with a
    # 1e-6 tolerance. Absorb only that read-back noise (a real overflow of > 0.001 mm still fails here).
    assert layout.w <= WIDTH + 1e-3, f"layout {layout.w:.4f} mm wider than {WIDTH} mm"
    rep = compose("Fig5", layout, width=max(WIDTH, layout.w))
    report(rep)
    assert rep["audit_ok"], rep
