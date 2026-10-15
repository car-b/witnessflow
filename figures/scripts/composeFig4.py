"""Figure 4 composite (1:1, no scaling) -> figures/composite/Fig4.pdf / .png

Layout (figure plan v2): 4A across the top; middle 4B | 4C (4C = visual centre); bottom 4D | 4E.
Content width 184 mm. Panel sizes are set in fig4.py (re-render there, never scale here):
    4A 184 x 60
    4B 82 x 70 | 4C 100 x 70     (82 + 2 + 100 = 184)
    4D 100 x 60 | 4E 82 x 60     (100 + 2 + 82 = 184)
Run from figures/scripts:  python composeFig4.py
"""
from compose import Row, Col, P, compose, report

W = 184

a, b, c, d, e = P("4A", "a"), P("4B", "b"), P("4C", "c"), P("4D", "d"), P("4E", "e")
mid = Row([b, c])
bot = Row([d, e])
lay = Col([a, mid, bot])

print(f"4A {a.w:.1f} mm, middle row {mid.w:.1f} mm, bottom row {bot.w:.1f} mm (target {W})")
print(f"middle 4B {b.h0:.1f} / 4C {c.h0:.1f} mm; bottom 4D {d.h0:.1f} / 4E {e.h0:.1f} mm")
# PDF media boxes round panel sizes to ~1e-6 mm, so a row of exactly 184 mm can read 184.000002 mm, which
# compose()'s 1e-6 width tolerance rejects; offer max(W, natural width) instead (difference < 1e-5 mm).
assert lay.w <= W + 1e-3, f"layout {lay.w:.3f} mm wider than {W} mm"
rep = compose("Fig4", lay, width=max(W, lay.w))
report(rep)
slack = {"4A": W - a.w, "middle row": W - mid.w, "bottom row": W - bot.w,
         "4B vs 4C": abs(b.h0 - c.h0), "4D vs 4E": abs(d.h0 - e.h0)}
print("slack (mm):", {k: round(v, 2) for k, v in slack.items()})
