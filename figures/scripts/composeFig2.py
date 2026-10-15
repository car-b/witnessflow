"""Figure 2 composite: 2A (left block) | 2B over 2C (right column); bottom row 2D | 2E (1:1 by compose.py).

Panel sizes are set in fig2.py so that both rows are exactly 184 mm wide with 2 mm gaps, 2A is as tall as the
2B + 2C stack (2A = 2B + 2C + letter band + gap) and 2D is as tall as 2E:
2A 90 x 114, 2B 92 x 56, 2C 92 x 54;  2D 102 x 86, 2E 80 x 86.
usage (from figures/scripts):  python composeFig2.py   -> figures/composite/Fig2.pdf/.png
"""
from compose import Row, Col, P, compose, report, H_GAP

# PDF media boxes carry the panel size to ~1e-5 mm (e.g. 2B = 92.000005), which can trip compose's 1e-6 width check
# for rows that fill 184 mm exactly; the 1e-4 mm slack is given back by Row justification (gaps print as 2.0 mm).
GAP = H_GAP - 1e-4

right = Col([P("2B", "b"), P("2C", "c")])
lay = Col([Row([P("2A", "a"), right], gap=GAP),
           Row([P("2D", "d"), P("2E", "e")], gap=GAP)])

if __name__ == "__main__":
    rep = compose("Fig2", lay, width=184)
    report(rep)
    top, bot = lay.c
    nat = lambda r: sum(c.w for c in r.c) + H_GAP * (len(r.c) - 1)      # row width at 2 mm gaps
    print(f"   top row width at 2 mm gaps {nat(top):.1f} mm; 2A block {top.c[0].h:.1f} mm vs 2B+2C stack {right.h:.1f} mm")
    print(f"   bottom row width at 2 mm gaps {nat(bot):.1f} mm; heights {[round(c.h0, 1) for c in bot.c]}")
    assert rep["audit_ok"], "composite audit failed"
