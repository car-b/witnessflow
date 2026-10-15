"""Figure 3 composite (1:1, no scaling) -> figures/composite/Fig3.pdf / .png

Layout (figure plan v2): top = 3A (narrow strip, left) | [3B over 3C] (right; shared WINGMEN x axis);
bottom = 3D | 3E. Content width 184 mm. Panel sizes are set in fig3.py (re-render there, never scale here):
    3A 50 x 120 | 3B 132 x 80 over 3C 132 x 36    (50 + 2 + 132 = 184; 120 + 3 = 80 + 3 + 1 + 36 + 3)
    3D 100 x 76 | 3E 82 x 76                       (100 + 2 + 82 = 184)
Run from figures/scripts:  python composeFig3.py
"""
from compose import Row, Col, P, compose, report, LETTER, V_GAP

W = 184

a, b, c, d, e = P("3A", "a"), P("3B", "b"), P("3C", "c"), P("3D", "d"), P("3E", "e")
stack = Col([b, c])
top = Row([a, stack])
bottom = Row([d, e])
lay = Col([top, bottom])

# layout checks: rows filled to W, equal heights within rows, 3B / 3C left edges identical (shared x axis)
print(f"top row {top.w:.1f} mm, bottom row {bottom.w:.1f} mm (target {W})")
print(f"3A {a.h:.1f} mm vs 3B+3C stack {stack.h:.1f} mm (incl. letter bands, gap {V_GAP} mm)")
print(f"3D {d.h0:.1f} mm vs 3E {e.h0:.1f} mm")
# PDF media boxes round panel sizes to ~1e-6 mm, so a row of exactly 184 mm can read 184.000002 mm, which
# compose()'s 1e-6 width tolerance rejects; offer max(W, natural width) instead (difference < 1e-5 mm).
assert lay.w <= W + 1e-3, f"layout {lay.w:.3f} mm wider than {W} mm"
rep = compose("Fig3", lay, width=max(W, lay.w))
report(rep)
pos = {pid: (x, y, w, h) for pid, _, x, y, w, h in rep["panels"]}
assert pos["3B"][0] == pos["3C"][0] and pos["3B"][2] == pos["3C"][2], "3B / 3C not aligned"
slack = {"top row": W - top.w, "bottom row": W - bottom.w, "3A vs stack": abs(a.h - stack.h),
         "3D vs 3E": abs(d.h0 - e.h0)}
print("slack (mm):", {k: round(v, 2) for k, v in slack.items()})
