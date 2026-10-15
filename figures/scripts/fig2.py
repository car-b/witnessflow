"""Figure 2 (classical geometry arranges witnesses): single panels 2A-2E.
usage: python fig2.py [2A 2B 2C 2D 2E]      (default: all)
All geometry from wfgeom; all numbers from fig2_data (asserted against frozen sources)."""
import sys
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

import wfstyle as S
S.setup()
import wfgeom as G
import fig2_data as D

FP = FontProperties(family="Arial", size=S.FS)
FPB = FontProperties(family="Arial", size=S.FS, weight="bold")
PT = 25.4 / 72  # mm per pt
GREY_BB = S.BACKBONE     # backbone slot fill (shared M1 card constant)
GREY_BB_EDGE = S.MUTED


# ------------------------------------------------------------------ helpers
def canvas(panel, h=None, w=None):
    """w: optional width override (composite Fig 2 fills each 184 mm row with 2 mm gaps: 2A 90, 2D 102)."""
    W0, H0 = S.SIZES[panel]
    W, H = w or W0, h or H0
    assert 0.8 * H0 <= H <= 1.2 * H0 and W0 <= W <= W0 + 3
    fig = S.new(panel, w=W, h=H, layout=None)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax, W, H


def axmm(fig, W, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def tw(s, bold=False):
    """text width in mm at 7 pt Arial"""
    return TextPath((0, 0), s, prop=FPB if bold else FP).get_extents().width * PT


def T(ax, x, y, s, **kw):
    kw.setdefault("fontsize", S.FS)
    kw.setdefault("color", S.INK)
    return ax.text(x, y, s, **kw)


def tier(fig, text, W, H):
    return S.tier_tag(fig, text)          # wfstyle default: inside the top-right corner


def tstar_text(ax, x, y, rest="", ms=5.5, bold=False, **kw):
    """'T' + gold vector star (wfstyle.tstar_label) followed by `rest`; returns x of the end of the text."""
    kw.setdefault("color", S.INK)
    if bold:
        kw["fontweight"] = "bold"
    S.tstar_label(ax, x, y, text="T", ms=ms, **kw)
    dx = tw("T", bold) + (0.6 + 0.95 * ms + 2.2) * PT   # text advance > ink width: keep a visible gap
    if rest:
        T(ax, x + dx, y, rest, va=kw.pop("va", "center"), **kw)
    return x + dx + (tw(rest, bold) if rest else 0)


def arrow(ax, p, q, lw=S.LW, color=S.INK, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=1.5,head_width=0.8", mutation_scale=1,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, **kw))


def slot_row(ax, x, y, fills, w=2.3, h=2.3, gap=0.35, edges=None, lw=S.THIN):
    """M1 tube card: a row of square slots; returns right edge x."""
    for i, fc in enumerate(fills):
        ec = edges[i] if edges else S.INK
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=fc, ec=ec, lw=lw))
    return x + len(fills) * (w + gap) - gap


# ------------------------------------------------------------------ 2A rose window
def ring_xy(i, cx, cy, R, n=26):
    th = np.deg2rad(90 - i * 360 / n)
    return cx + R * np.cos(th), cy + R * np.sin(th)


def hexagon(ax, block, cx, cy, R, **kw):
    pts = [ring_xy(p, cx, cy, R) for p in sorted(block)]   # ring order = ascending point number
    ax.add_patch(Polygon(pts, closed=True, **kw))


def panel_2A():
    D.rose()
    hist, frac0, ndes = D.lambda_hist()
    # composite Fig 2: 2A (90 x 114) is as tall as the 2B (56) + 2C (54) stack plus one letter band and gap;
    # drawn for 88 x 112: the window moves up by half the extra height, the histogram + key stay right-aligned
    fig, ax, W, H = canvas("2A", h=114, w=90)
    tstar_pts = {G.POINT[m] for m in G.TSTAR}
    tube = G.TSTAR_TUBE[0]
    K = 4                                          # highlighted orbit = orbit of the T* tube
    assert G.ORBIT[tube][0] == K
    base = G.ROSE_BASE[K]
    oc = S.ORBIT
    tier(fig, "Math", W, H)

    # ---------- top band: (1) base block -> (2) one orbit -> (3) 5 orbits
    r0 = 4.6
    yb = H - 1.0 - r0
    c1, c2 = (8.6, yb), (31.0, yb)
    for (cx, cy) in (c1, c2):
        ax.add_patch(Circle((cx, cy), r0, fc="none", ec=S.GRID, lw=S.THIN))
    for s in range(26):
        hexagon(ax, [(x + s) % 26 for x in base], *c2, r0, fc="none", ec=oc[K], lw=S.THIN)
    hexagon(ax, base, *c1, r0, fc="none", ec=oc[K], lw=S.LW, joinstyle="round")
    for (cx, cy) in (c1, c2):
        for i in range(26):
            x, y = ring_xy(i, cx, cy, r0)
            on = (cx, cy) == c1 and i in base
            ax.add_patch(Circle((x, y), 0.38 if on else 0.24, fc=S.INK if on else S.MUTED, ec="none", zorder=4))
    yc1, yc2 = yb - r0 - 2.0, yb - r0 - 5.0
    T(ax, c1[0], yc1, "base block B5", ha="center", va="center")
    T(ax, c1[0], yc2, "{0,1,2,5,14,23}", ha="center", va="center", color=S.MUTED)
    T(ax, c2[0], yc1, "26 shifts = 1 orbit", ha="center", va="center")
    T(ax, c2[0], yc2, "B5 + s (mod 26)", ha="center", va="center", color=S.MUTED)
    arrow(ax, (c1[0] + r0 + 1.2, yb), (c2[0] - r0 - 1.2, yb))
    T(ax, (c1[0] + c2[0]) / 2, yb + 0.8, "+s", ha="center", va="bottom")
    xa = c2[0] + r0 + 1.2
    arrow(ax, (xa, yb), (xa + 4.8, yb))
    T(ax, xa + 2.4, yb + 0.8, "× 5", ha="center", va="bottom")
    # formula block (= stage 3, drawn below as the full window)
    xf = xa + 6.0
    T(ax, xf, yb + 4.0, "5 orbits = 130 tubes", va="center", fontweight="bold")
    T(ax, xf, yb + 0.8, "130 × C(6,3) = C(26,3) = 2600", va="center")
    T(ax, xf, yb - 2.4, "every triple in exactly 1 tube", va="center")
    # T* tube card (M1): 6 candidate slots + 8 backbone slots
    cand = sorted(G.BLOCKS[tube])
    fills = [S.TSTAR if p in tstar_pts else "white" for p in cand] + [GREY_BB] * 8
    edges = [S.INK if p in tstar_pts else oc[K] for p in cand] + [GREY_BB_EDGE] * 8
    yc = yc1 - 0.7
    xr = slot_row(ax, xf, yc, fills, w=1.4, h=1.4, gap=0.22, edges=edges)
    tstar_text(ax, xr + 1.0, yc + 0.7, f"tube {tube + 1}")
    T(ax, xf, yc2, f"B5 + {G.ORBIT[tube][1]}; 6 candidates + 8 backbone", va="center", color=S.MUTED)

    # ---------- main window (stage 3)
    # x: right-hand labels (GranzymeB, CD66b) are longer than left-hand ones; -1.5 mm balances the two label margins
    cx, cy, R = W / 2 - 1.5, 58.6 + (H - 112) / 2, 24.0
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=S.GRID, lw=S.THIN))
    for bi, b in enumerate(G.BLOCKS):
        k = G.ORBIT[bi][0]
        if k != K:
            hexagon(ax, b, cx, cy, R, fc="none", ec=oc[k], lw=S.THIN, alpha=0.17)
    for bi, b in enumerate(G.BLOCKS):
        if G.ORBIT[bi][0] == K and bi != tube:
            hexagon(ax, b, cx, cy, R, fc="none", ec=oc[K], lw=S.THIN, alpha=0.85)
    hexagon(ax, G.BLOCKS[tube], cx, cy, R, fc="none", ec=S.INK, lw=S.LW, joinstyle="round")
    for i in range(26):
        x, y = ring_xy(i, cx, cy, R)
        if i in tstar_pts:
            ax.add_patch(Circle((x, y), 0.95, fc=S.TSTAR, ec=S.INK, lw=S.THIN, zorder=5))
        else:
            ax.add_patch(Circle((x, y), 0.55, fc=S.INK, ec="none", zorder=5))
        th = 90 - i * 360 / 26
        thn = ((th + 180) % 360) - 180
        right = -90 < thn <= 90
        rot = thn if right else thn - 180
        xn, yn = ring_xy(i, cx, cy, R + 2.7)
        T(ax, xn, yn, str(i), ha="center", va="center", rotation=rot, rotation_mode="anchor", color=S.MUTED)
        xl, yl = ring_xy(i, cx, cy, R + 4.6)
        T(ax, xl, yl, G.LABELS[i], ha="left" if right else "right", va="center", rotation=rot, rotation_mode="anchor",
          fontweight="bold" if i in tstar_pts else "normal")

    # ---------- incidence strip: 26 points x 130 tubes = 5 circulant 26 x 26 blocks
    cw = 0.36
    x0, y0 = 6.2, 9.0
    cols = sorted(range(130), key=lambda bi: G.ORBIT[bi])          # orbit, then shift
    img = np.ones((26, 130, 3))
    for j, bi in enumerate(cols):
        rgb = mpl.colors.to_rgb(oc[G.ORBIT[bi][0]])
        for p in G.BLOCKS[bi]:
            img[p, j] = rgb
    ax.imshow(img, extent=(x0, x0 + 130 * cw, y0, y0 + 26 * cw), origin="upper", interpolation="nearest",
              aspect="auto", zorder=2)
    for k in range(1, 5):
        ax.plot([x0 + 26 * k * cw] * 2, [y0, y0 + 26 * cw], color=S.INK, lw=S.THIN, zorder=3)
    ax.add_patch(Rectangle((x0, y0), 130 * cw, 26 * cw, fc="none", ec=S.INK, lw=S.THIN, zorder=3))
    for k in range(5):
        T(ax, x0 + (26 * k + 13) * cw, y0 - 1.4, f"B{k + 1}", ha="center", va="center")
    jt = cols.index(tube)
    ax.annotate("", xy=(x0 + (jt + 0.5) * cw, y0 + 26 * cw + 0.2), xytext=(x0 + (jt + 0.5) * cw, y0 + 26 * cw + 2.6),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=S.THIN, color=S.INK,
                                shrinkA=0, shrinkB=0))
    T(ax, x0 + (jt + 0.5) * cw - 0.8, y0 + 26 * cw + 1.4, f"tube {tube + 1}", ha="right", va="center")
    for p in tstar_pts:
        yy = y0 + (25.5 - p) * cw
        ax.add_patch(Circle((x0 - 0.9, yy), 0.42, fc=S.TSTAR, ec=S.INK, lw=S.THIN))
    T(ax, x0 - 2.3, y0 + 13 * cw, "point", rotation=90, ha="center", va="center")
    T(ax, x0, y0 + 26 * cw + 1.4, "26 points × 130 tubes", va="center")
    # geometry carries no biological meaning (kept in the panel)
    T(ax, 0.8, 4.4, "ring order is a labelling; hexagon = 6-set;", va="center", color=S.MUTED)
    T(ax, 0.8, 1.4, "edges, areas and crossings carry no meaning", va="center", color=S.MUTED)

    # ---------- lambda(T) histogram + key (bottom-right corner)
    hx, hy, hw, hh = W - 22.6, 10.6, 21.4, 10.6
    hax = axmm(fig, W, H, hx, hy, hw, hh)
    xs = np.arange(6)
    hax.bar(xs - 0.2, hist.random_mean_frac * 100, width=0.4, color=S.METHOD["random"], lw=0)
    hax.bar(xs + 0.2, hist.steiner_frac * 100, width=0.4, color=S.METHOD["classical"], lw=0)
    hax.set_xticks(xs, ["0", "1", "2", "3", "4", "5+"])
    hax.tick_params(axis="x", length=0, pad=1.2)
    hax.set_yticks([0, 50, 100])
    hax.set_ylim(0, 105); hax.set_xlim(-1.45, 5.55)
    hax.set_xlabel("λ(T), tubes per triple", labelpad=1)
    hax.set_ylabel("% of triples", labelpad=1)
    hax.text(-0.2, frac0 * 100 + 2.5, f"{frac0 * 100:.1f}%", ha="center", va="bottom", fontsize=S.FS)
    kx = W - 0.5 - tw(f"mean of {ndes} designs")
    ax.add_patch(Rectangle((kx, 29.6), 1.6, 1.6, fc=S.METHOD["classical"], ec="none"))
    T(ax, kx + 2.3, 30.4, "S(3,6,26)", va="center")
    ax.add_patch(Rectangle((kx, 26.4), 1.6, 1.6, fc=S.METHOD["random"], ec="none"))
    T(ax, kx + 2.3, 27.2, "random 130 tubes", va="center")
    T(ax, kx, 24.0, f"mean of {ndes} designs", va="center", color=S.MUTED)
    return fig


# ------------------------------------------------------------------ AG(2,5) torus (2B, 2C)
def slope_key(s):
    return "inf" if s in ("inf", G.INF) else int(s)


class Torus:
    """5 x 5 torus window: display x = a (0..4), y = b shifted so Q = CD27 sits in the middle row (-2..2).
    Lines are drawn point to point; a step that wraps mod 5 is drawn dashed to the window edge and dashed again
    from the opposite edge (no Euclidean arcs)."""

    def __init__(self, ax, A, X0, Y0, u):
        self.ax, self.A, self.X0, self.Y0, self.u = ax, A, X0, Y0, u
        self.P = A["P"].set_index("point")

    def X(self, x):
        return self.X0 + x * self.u

    def Y(self, y):
        return self.Y0 + (y + 2) * self.u

    def xy(self, p):
        r = self.P.loc[p]
        return self.X(float(r.x)), self.Y(float(r.y))

    def frame(self, chevrons=True, ticks=True, chev_x=2.8, chev_y=1.5):
        ax, u = self.ax, self.u
        x0, y0, x1, y1 = self.X(-0.5), self.Y(-2.5), self.X(4.5), self.Y(2.5)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=S.MUTED, lw=S.THIN, zorder=1))
        if chevrons:   # edge identification marks: ^ on left/right edges, >> on top/bottom edges
            c = 0.5
            for xx, yc in ((x0, self.Y(-chev_y)), (x1, self.Y(chev_y))):
                ax.plot([xx - c, xx, xx + c], [yc - c / 2, yc + c / 2, yc - c / 2], color=S.MUTED, lw=S.THIN,
                        solid_joinstyle="miter", zorder=1)
            xm = self.X(chev_x)
            for yy in (y0, y1):
                for d in (-0.45, 0.45):
                    ax.plot([xm + d - c / 2, xm + d + c / 2, xm + d - c / 2], [yy - c, yy, yy + c], color=S.MUTED,
                            lw=S.THIN, solid_joinstyle="miter", zorder=1)
        if ticks:
            for a in range(5):
                T(ax, self.X(a), y0 - 0.7, str(a), ha="center", va="top", color=S.MUTED)
            qb = 0
            for y in range(-2, 3):
                T(ax, x0 - 0.8, self.Y(y), str((y + qb) % 5), ha="right", va="center", color=S.MUTED)
        return x0, y0, x1, y1

    def segments(self, points, slope):
        pts = sorted(((float(self.P.loc[p].x), float(self.P.loc[p].y)) for p in points),
                     key=(lambda t: t[1]) if slope == "inf" else (lambda t: t[0]))
        segs = []
        if slope == "inf":
            for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
                segs.append(([(xa, ya), (xb, yb)], False))
            return segs
        m = int(slope)
        ms = m if m <= 2 else m - 5
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            assert xb == xa + 1
            lift = ya + ms
            if lift == yb:
                segs.append(([(xa, ya), (xb, yb)], False))
            else:
                assert abs(lift - yb) == 5
                edge = 2.5 if lift > 2 else -2.5
                t = (edge - ya) / ms
                segs.append(([(xa, ya), (xa + t, edge)], True))
                segs.append(([(xa + t, -edge), (xb, yb)], True))
        return segs

    def draw_line(self, points, slope, color, lw, alpha=1.0, z=2):
        for (p, q), wrapped in self.segments(points, slope):
            self.ax.plot([self.X(p[0]), self.X(q[0])], [self.Y(p[1]), self.Y(q[1])], color=color, lw=lw, alpha=alpha,
                         ls=(0, (2.2, 1.4)) if wrapped else "-", solid_capstyle="round", dash_capstyle="butt",
                         zorder=z)

    def node(self, p, r=0.75, fc="white", ec=S.INK, lw=S.THIN, z=6, marker=None):
        x, y = self.xy(p)
        if marker is None:
            self.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
        else:
            self.ax.plot([x], [y], marker=marker, ms=r, mfc=fc, mec=ec, mew=lw, ls="none", zorder=z)


def slope_legend(ax, x, y, step=6.2, seg=2.6, label=True):
    if label:
        T(ax, x, y, "slope", va="center")
        x += tw("slope") + 1.4
    for k in [0, 1, 2, 3, 4, "inf"]:
        ax.plot([x, x + seg], [y, y], color=S.SLOPE[k], lw=S.LW, solid_capstyle="butt")
        lab = "vert." if k == "inf" else str(k)
        T(ax, x + seg + 0.6, y, lab, va="center")
        x += seg + 1.2 + tw(lab) + 1.2
    return x


def panel_2B():
    A = D.ag()
    lad = D.levine_ladder()
    # composite Fig 2: 2B 92 x 56 (drawn for 60; the torus unit shrinks 7.1 -> 6.9 mm to keep the header clear)
    fig, ax, W, H = canvas("2B", h=56)
    tier(fig, "Math", W, H)
    q, r, inf = A["q"], A["r"], A["inf"]
    L = A["L"]
    tst = {G.POINT[m] for m in G.TSTAR}

    # ---------- header: the designated point at infinity
    yh = H - 2.6
    ax.add_patch(Circle((2.0, yh), 1.0, fc=S.TSTAR, ec=S.INK, lw=S.THIN, zorder=6))
    T(ax, 3.6, yh, "∞ = CCR7", va="center", fontweight="bold")
    T(ax, 3.6 + tw("∞ = CCR7", True) + 2.0, yh, "∞ is one designated point of the inversive plane", va="center",
      color=S.MUTED)
    T(ax, 1.0, yh - 3.4, "Q = CD27,  R = CD45RA;  line + ∞ = 1 tube", va="center")
    T(ax, 1.0, yh - 6.6, "dashed = wraps mod 5", va="center", color=S.MUTED)

    # ---------- main torus
    u = 6.9
    tor = Torus(ax, A, X0=6.6 + u / 2, Y0=4.8 + u / 2, u=u)
    x0, y0, x1, y1 = tor.frame(chev_x=2.75, chev_y=1.5)
    for _, l in L.iterrows():
        if not l.through_Q:
            k = slope_key(l.slope)
            tor.draw_line(list(map(int, l.points.split())), k, S.SLOPE[k], S.THIN, alpha=0.5, z=2)
    for _, l in L.iterrows():
        if l.through_Q:
            k = slope_key(l.slope)
            tor.draw_line(list(map(int, l.points.split())), k, S.SLOPE[k], S.LW, z=3)
    for p in range(26):
        if p == inf:
            continue
        tor.node(p, r=1.0 if p in tst else 0.6, fc=S.TSTAR if p in tst else "white")
    # slope labels at the ends of the 6 lines through Q (direct labels, ink text)
    for xx, yy, s, ha, va in [(0.75, 2.62, "3", "center", "bottom"), (2.0, 2.62, "vert.", "center", "bottom"),
                              (3.25, 2.62, "2", "center", "bottom"), (4.58, 2.58, "1", "left", "bottom"),
                              (4.62, 0.0, "0", "left", "center"), (4.62, -2.0, "4", "left", "center")]:
        T(ax, tor.X(xx), tor.Y(yy), s, ha=ha, va=va)
    T(ax, tor.X(-0.5), tor.Y(2.62), "slope", ha="left", va="bottom", color=S.MUTED)
    # Q, R and the other three markers of tube 73 on line QR
    l72 = L[L.through_Q_and_R == 1].iloc[0]
    bb = dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85)
    qx, qy = tor.xy(q)
    T(ax, qx + 1.5, qy + 0.75, "Q", ha="left", va="center", fontweight="bold", bbox=bb, zorder=7)
    rx, ry = tor.xy(r)
    T(ax, rx - 1.5, ry + 0.75, "R", ha="right", va="center", fontweight="bold", bbox=bb, zorder=7)
    for p in map(int, l72.points.split()):
        if p in (q, r):
            continue
        x, y = tor.xy(p)
        if y > tor.Y(1.5):        # top-right corner point: label outside the window
            T(ax, x1 + 0.6, y - 0.2, G.LABELS[p], ha="left", va="top")
        elif y < tor.Y(-1.5):     # bottom-left corner point: label between node and frame
            T(ax, x + 1.0, y - 0.25, G.LABELS[p], ha="left", va="top", bbox=bb, zorder=7)
        else:
            T(ax, x + 1.0, y - 0.25, G.LABELS[p], ha="left", va="top", bbox=bb, zorder=7)

    # ---------- side view: 4 norm circles about Q (same window)
    us = 3.8
    sxl = 54.5
    side = Torus(ax, A, X0=sxl + us / 2, Y0=27.6 + us / 2, u=us)
    sx0, sy0, sx1, sy1 = side.frame(chevrons=False, ticks=False)
    mk = {1: "o", 2: "s", 3: "^", 4: "D"}
    msz = {1: 3.5, 2: 3.2, 3: 3.7, 4: 3.0}
    DARK = "#5A5A5A"
    for _, row in A["P"].iterrows():
        p = int(row.point)
        if p == inf:
            continue
        if p == q:
            side.node(p, r=0.9, fc=S.TSTAR)
            continue
        rho = int(row.rho_about_Q)
        side.node(p, r=msz[rho], fc=S.TSTAR if p in tst else DARK, ec=S.INK, marker=mk[rho])
    T(ax, sx0, sy1 + 1.2, "4 norm circles about Q", va="bottom", fontweight="bold")
    lx = sx1 + 2.0
    for j, rho in enumerate([1, 2, 3, 4]):
        yy = sy1 - 1.4 - j * 3.0
        ax.plot([lx + 0.9], [yy], marker=mk[rho], ms=msz[rho], mfc=DARK, mec=S.INK, mew=S.THIN, ls="none")
        T(ax, lx + 2.4, yy, f"ρ = {rho}", va="center")
    T(ax, lx, sy0 + 3.6, "N(z − Q) = ρ", va="center", color=S.MUTED)
    T(ax, lx, sy0 + 0.6, "4 × 6 partition", va="center", color=S.MUTED)
    T(ax, sx0, sy0 - 1.9, "ρ has no size order;", va="center", color=S.MUTED)
    T(ax, sx0, sy0 - 4.9, "25 centres generate 100 tubes", va="center", color=S.MUTED)

    # ---------- multiplicity ladder: classical 30 / 6 / 1 vs Levine 71-tube base covering
    ty = 17.2
    c1, c2, c3 = 52.5, 63.8, 74.6
    T(ax, c2, ty, "classical", va="center")
    T(ax, c3, ty, "Levine-32", va="center")
    T(ax, c1, ty - 3.0, "tubes per", va="center", color=S.MUTED)
    T(ax, c2, ty - 3.0, "(torus)", va="center", color=S.MUTED)
    T(ax, c3, ty - 3.0, "71-tube base", va="center", color=S.MUTED)
    ax.plot([c1, W - 1.0], [ty - 4.7, ty - 4.7], color=S.MUTED, lw=S.THIN)
    rows = [("marker", "30 lines", lad["point"]), ("pair", "6 via Q", lad["pair"]), ("triple", "1 = QR", lad["triple"])]
    for j, (lab, cl, (lo, hi)) in enumerate(rows):
        yy = ty - 6.6 - j * 3.1
        T(ax, c1, yy, lab, va="center")
        T(ax, c2, yy, cl, va="center")
        T(ax, c3, yy, f"{lo}–{hi}", va="center")
    tstar_text(ax, c1, 1.8, f"tube {int(l72.block0) + 1} = line QR + ∞")
    return fig


def panel_2C():
    A = D.ag()
    Gt = D.gating(A)
    # composite Fig 2: 2C 92 x 54 (drawn for 58; legend and repair block move closer to their neighbours)
    fig, ax, W, H = canvas("2C", h=54)
    tier(fig, "Math", W, H)
    S.limits_tag(fig)                     # wfstyle default: inside the top-left corner
    q, r, inf = A["q"], A["r"], A["inf"]
    L, P = A["L"], A["P"]
    tst = {G.POINT[m] for m in G.TSTAR}
    at = {(int(row.x), int(row.y)): int(row.point) for _, row in P.iterrows() if row.point != inf}

    # example triples (display coordinates); verified against the gated tube set
    wit = [at[(1, 2)], at[(2, 2)], at[(3, 2)]]
    tri1 = [at[(0, 0)], at[(1, 1)], at[(0, 1)]]
    tri2 = [at[(3, -1)], at[(4, -1)], at[(4, -2)]]
    cov = Gt["trip_cov"]
    assert tuple(sorted(wit)) in cov and tuple(sorted(tri1)) not in cov and tuple(sorted(tri2)) not in cov
    wline = L[[set(wit) <= set(map(int, l.split())) for l in L.points]].iloc[0]

    # ---------- header (left)
    xh = 0.01 * W + tw("Limits") + 2.8
    T(ax, xh, H - 2.3, "gate on x = CCR7 (∞)", va="center", fontweight="bold")
    T(ax, 0.8, H - 5.6, "only the 30 tubes with x remain (30 lines)", va="center", color=S.MUTED)
    # legend for the examples
    ly = H - 10.0
    ax.plot([1.2, 4.6], [ly, ly], color=S.SLOPE[slope_key(wline.slope)], lw=S.LW)
    for xx in (1.2, 2.9, 4.6):
        ax.add_patch(Circle((xx, ly), 0.6, fc=S.METHOD["classical"], ec=S.INK, lw=S.THIN, zorder=5))
    T(ax, 6.0, ly, "witnessed (collinear)", va="center")
    ax.add_patch(Polygon([(1.4, ly - 4.6), (4.4, ly - 4.6), (2.9, ly - 2.2)], closed=True, fc="white", ec=S.MUTED,
                         lw=S.LW, hatch=S.LIMITS_HATCH))
    T(ax, 6.0, ly - 3.4, "no co-measurement", va="center")

    # ---------- torus
    u = 6.4
    tor = Torus(ax, A, X0=5.0 + u / 2, Y0=4.6 + u / 2, u=u)
    x0, y0, x1, y1 = tor.frame(chev_x=2.5, chev_y=1.5)
    for _, l in L.iterrows():
        k = slope_key(l.slope)
        if l.block0 == wline.block0:
            continue
        tor.draw_line(list(map(int, l.points.split())), k, S.SLOPE[k], S.THIN, alpha=0.4, z=2)
    k = slope_key(wline.slope)
    tor.draw_line(list(map(int, wline.points.split())), k, S.SLOPE[k], S.LW, z=3)
    for tri in (tri1, tri2):
        ax.add_patch(Polygon([tor.xy(p) for p in tri], closed=True, fc="white", ec=S.MUTED, lw=S.LW,
                             hatch=S.LIMITS_HATCH, zorder=4))
    for p in range(26):
        if p == inf:
            continue
        if p in wit:
            tor.node(p, r=0.95, fc=S.METHOD["classical"])
        elif p in tri1 or p in tri2:
            tor.node(p, r=0.95, fc="white")
        elif p in tst:
            tor.node(p, r=0.9, fc=S.TSTAR)
        else:
            tor.node(p, r=0.5, fc="white")

    # ---------- counts: pairs 300/300, triples 300/2300
    bx0, bx1 = 49.0, W - 1.2
    T(ax, 40.5, H - 9.2, "other 25 markers, inside x+ cells", va="center", color=S.MUTED)
    yb1, yb2, bh = H - 14.6, H - 20.4, 3.2
    T(ax, 40.5, yb1 + bh / 2, "pairs", va="center")
    T(ax, 40.5, yb2 + bh / 2, "triples", va="center")
    ax.add_patch(Rectangle((bx0, yb1), bx1 - bx0, bh, fc=S.METHOD["classical"], ec="none"))
    T(ax, bx0 + 1.0, yb1 + bh / 2, "300 / 300, each once", va="center", color="white")
    fw = (bx1 - bx0) * 300 / 2300
    ax.add_patch(Rectangle((bx0, yb2), fw, bh, fc=S.METHOD["classical"], ec="none"))
    ax.add_patch(Rectangle((bx0 + fw, yb2), bx1 - bx0 - fw, bh, fc="white", ec=S.MUTED, lw=S.LW,
                           hatch=S.LIMITS_HATCH))
    T(ax, bx1 - 1.0, yb2 + bh / 2, "2000 not co-measured", ha="right", va="center",
      bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    T(ax, bx0, yb2 - 1.7, "300 / 2300 = 13% witnessed", va="center")
    T(ax, 40.5, yb2 - 5.0, "coverage fraction, not detection sensitivity", va="center", color=S.MUTED)

    # ---------- repair: x into the backbone + redesign (capacity changes)
    yr = yb2 - 5.0 - 4.6                  # 4.6 mm below the 'coverage fraction' line (26.2 at H = 58 was 6.4)
    T(ax, 40.5, yr, "repair: x into the backbone, then redesign", va="center", fontweight="bold")
    sw, gp = 1.6, 0.25
    rows = [(6, "130 tubes", "x + 6 = 7 slots: C(25,6,3)", yr - 5.2), (5, "256 tubes", "capacity 6, x + 5: C(25,5,3)",
                                                                              yr - 13.0)]
    for ncand, big, small, yy in rows:
        fills = ["white"] * ncand + [S.TSTAR] + [GREY_BB] * 8
        edges = [S.INK] * ncand + [S.INK] + [GREY_BB_EDGE] * 8
        xr = slot_row(ax, 40.5, yy, fills, w=sw, h=sw, gap=gp, edges=edges)
        T(ax, 40.5 + ncand * (sw + gp) + sw / 2, yy + sw + 0.3, "x", ha="center", va="bottom")
        T(ax, xr + 1.6, yy + sw / 2, big, va="center", fontweight="bold")
        T(ax, 40.5, yy - 1.9, small, va="center", color=S.MUTED)
    T(ax, 40.5, 4.6, "existing covering certificates,", va="center", color=S.MUTED)
    T(ax, 40.5, 1.6, "not proven minima", va="center", color=S.MUTED)
    return fig


def panel_2D():
    O = D.oa()
    names, two, spec, demo = O["names"], O["two"], O["spec"], O["demo"]
    # composite Fig 2: 2D 102 x 86 = as tall as 2E, row 2D + 2 mm + 2E = 184 mm.  Drawn for 100 x 80: the extra
    # width widens the column pitch, the extra height ex goes into the slot pitch, the lower blocks and the codebook.
    fig, ax, W, H = canvas("2D", h=86, w=102)
    ex = H - 80.0
    tier(fig, "Math (artificial constraint)", W, H)
    oa = S.METHOD["oa"]

    # ---------- cylinder grid: x = family (column), y = slot within family (row, mod 5)
    bw = [max(tw(m) for m in names[x]) + 1.6 for x in range(5)]
    bh, py, pitch = 3.5, 6.0 + 0.1 * ex, 15.0 + (W - 100) / 4
    XC = [8.6 + bw[0] / 2 + i * pitch for i in range(5)]
    YB = H - 13.5 - 4.5 * py                   # centre of slot 0 row (39.5 at 80 mm)

    def YC(y):
        return YB + y * py

    top, bot = YC(4.5), YC(-0.5)
    gx0, gx1 = XC[0] - bw[0] / 2 - 0.8, XC[4] + bw[4] / 2 + 0.8
    for yy in (top, bot):       # identified edges of the cylinder
        ax.plot([gx0, gx1], [yy, yy], color=S.MUTED, lw=S.THIN)
        xm = (XC[1] + XC[2]) / 2
        for d in (-0.45, 0.45):
            ax.plot([xm + d - 0.25, xm + d + 0.25, xm + d - 0.25], [yy - 0.5, yy, yy + 0.5], color=S.MUTED, lw=S.THIN,
                    solid_joinstyle="miter")
    for y in range(5):
        T(ax, gx0 - 0.8, YC(y), str(y), ha="right", va="center", color=S.MUTED)
    T(ax, gx0 - 4.2, YC(2), "slot y", rotation=90, ha="center", va="center", color=S.MUTED)
    for x in range(5):
        T(ax, XC[x], top + 0.6, f"x = {x}", ha="center", va="bottom", fontweight="bold")
    T(ax, gx0, top + 4.0, "family", va="bottom", color=S.MUTED)

    def path(vals, color, lw, alpha, z):
        for x in range(4):
            y1, y2 = vals[x], vals[x + 1]
            d = ((y2 - y1 + 2) % 5) - 2
            xa, xb = XC[x] + bw[x] / 2, XC[x + 1] - bw[x + 1] / 2
            if 0 <= y1 + d <= 4:
                ax.plot([xa, xb], [YC(y1), YC(y2)], color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle="round")
            else:
                edge = 4.5 if y1 + d > 4 else -0.5
                t = (edge - y1) / d
                xe = xa + t * (xb - xa)
                kw = dict(color=color, lw=lw, alpha=alpha, zorder=z, ls=(0, (2.2, 1.4)), dash_capstyle="butt")
                ax.plot([xa, xe], [YC(y1), YC(edge)], **kw)
                ax.plot([xe, xb], [YC(9 - edge - 5) if False else YC(-0.5 if edge > 4 else 4.5), YC(y2)], **kw)

    vals = lambda a, b, c: [(a + b * x + c * x * x) % 5 for x in range(5)]
    for abc in two:
        if abc != demo["abc"]:
            path(vals(*abc), oa, S.LW, 0.32, 2)
    path(demo["values"], oa, S.LW, 1.0, 3)
    for x in range(5):
        for y in range(5):
            on = demo["values"][x] == y
            specified = on and spec.get(x) == y
            fc = oa if specified else ("white" if on else S.FAMILY[x])
            ec = oa if on else "none"
            ax.add_patch(Rectangle((XC[x] - bw[x] / 2, YC(y) - bh / 2), bw[x], bh, fc=fc, ec=ec,
                                   lw=S.LW if on else 0, zorder=4))
            T(ax, XC[x], YC(y), names[x][y], ha="center", va="center", zorder=5,
              color="white" if specified else S.INK, fontweight="bold" if on else "normal")

    # ---------- statement + legend
    T(ax, 1.0, H - 2.4, "three specified values in three columns determine", va="center")
    T(ax, 1.0, H - 5.6, "a unique polynomial of degree ≤ 2 over GF(5):  f(x) = a + bx + cx²", va="center")
    ly = bot - 3.4 - 0.05 * ex
    ax.add_patch(Rectangle((gx0, ly - 1.1), 3.0, 2.2, fc=oa, ec=oa, lw=S.LW))
    T(ax, gx0 + 3.8, ly, "specified", va="center")
    xl = gx0 + 3.8 + tw("specified") + 3.0
    ax.add_patch(Rectangle((xl, ly - 1.1), 3.0, 2.2, fc="white", ec=oa, lw=S.LW))
    T(ax, xl + 3.8, ly, "determined", va="center")
    xl2 = xl + 3.8 + tw("determined") + 3.0
    ax.plot([xl2, xl2 + 3.4], [ly, ly], color=oa, lw=S.LW, alpha=0.32)
    T(ax, xl2 + 4.2, ly, f"5 tubes through {names[0][spec[0]]} + {names[2][spec[2]]}", va="center")
    T(ax, gx1 + 0.6, (top + bot) / 2, "dashed = wraps mod 5", rotation=-90, ha="center", va="center", color=S.MUTED)
    T(ax, gx0, ly - 3.4, "tubes through 1 / 2 / 3 markers: 25 / 5 / 1  (different families)", va="center",
      color=S.MUTED)

    # ---------- strength comparison: lines (c = 0) vs quadratics
    by = ly - 9.4 - 0.15 * ex
    bx0, bwid = 33.0, 16.0
    rows = [("lines a + bx (c = 0): 25 tubes", O["n_lin"], "strength 2"),
            ("a + bx + cx²: 125 tubes", O["n_quad"], "strength 3")]
    for j, (lab, n, st) in enumerate(rows):
        yy = by - j * (4.4 + 0.03 * ex)
        T(ax, bx0 - 1.2, yy, lab, ha="right", va="center")
        frac = n / 1250
        ax.add_patch(Rectangle((bx0, yy - 1.4), bwid * frac, 2.8, fc=oa, ec="none"))
        if frac < 1:
            ax.add_patch(Rectangle((bx0 + bwid * frac, yy - 1.4), bwid * (1 - frac), 2.8, fc="white", ec=S.MUTED,
                                   lw=S.LW, hatch=S.LIMITS_HATCH))
        T(ax, bx0 + bwid + 1.2, yy, f"{n} / 1250 · {st}", va="center")
    T(ax, bx0 - 1.2 - tw(rows[0][0]), by - 8.2 - 0.06 * ex,
      "cross-family pairs: each once (lines) vs each 5 times (quadratics)", va="center", color=S.MUTED)

    # ---------- M1 card: 5 candidate slots (one per family) + 8 backbone
    cy = 7.0 + 0.1 * ex
    x = 1.0
    for f, m in enumerate(demo["markers"]):
        w = tw(m) + 1.6
        ax.add_patch(Rectangle((x, cy), w, 3.2, fc=S.FAMILY[f], ec=S.INK, lw=S.THIN))
        T(ax, x + w / 2, cy + 1.6, m, ha="center", va="center")
        x += w + 0.35
    xr = slot_row(ax, x + 0.6, cy + 0.45, [GREY_BB] * 8, w=2.3, h=2.3, gap=0.3, edges=[GREY_BB_EDGE] * 8)
    T(ax, 1.0, cy + 5.0 + 0.05 * ex, f"tube {demo['tube_display']}: f(x) = 1 + x + x²;  5 candidates + 8 backbone",
      va="center")
    T(ax, 1.0, 1.8 + 0.03 * ex, "families = artificial slot constraint, not real spillover", va="center", color=S.MUTED)

    # ---------- codebook: 125 tubes x 5 families, symbol f(x) in F5 (RS[5,3])
    code = O["code"]
    cw, rh = 2.0, 0.42 + 0.008 * ex
    kx0, ky1 = W - 1.6 - 5 * cw, H - 15.0
    cmap = mpl.colors.ListedColormap([S.SEQ(v) for v in np.linspace(0.08, 0.9, 5)])
    ax.imshow(code, extent=(kx0, kx0 + 5 * cw, ky1 - 125 * rh, ky1), cmap=cmap, vmin=-0.5, vmax=4.5,
              interpolation="nearest", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((kx0, ky1 - 125 * rh), 5 * cw, 125 * rh, fc="none", ec=S.INK, lw=S.THIN, zorder=3))
    ax.plot([kx0, kx0 + 5 * cw], [ky1 - 25 * rh] * 2, color=S.INK, lw=S.THIN, zorder=3)
    for xx in range(5):
        T(ax, kx0 + (xx + 0.5) * cw, ky1 + 0.5, str(xx), ha="center", va="bottom")
    T(ax, W - 0.8, ky1 + 6.6, "codebook", ha="right", va="bottom", fontweight="bold")
    T(ax, W - 0.8, ky1 + 3.4, "RS[5,3]", ha="right", va="bottom", color=S.MUTED)
    T(ax, kx0 - 1.0, ky1 - 12.5 * rh, "c = 0", rotation=90, ha="right", va="center")
    ax.plot([kx0 - 0.5, kx0 - 0.5], [ky1 - 0.2, ky1 - 25 * rh + 0.2], color=S.INK, lw=S.THIN)
    row = O["order"].index(demo["index0"])
    yr = ky1 - (row + 0.5) * rh
    ax.plot([kx0 - 0.5], [yr], marker=">", ms=3.0, mfc=S.INK, mec="none", ls="none")
    T(ax, kx0 - 1.6, yr, str(demo["tube_display"]), ha="right", va="center")
    yk = ky1 - 125 * rh - 1.0
    T(ax, kx0 + 2.5 * cw, yk, "125 × 5", ha="center", va="top")
    for v in range(5):
        ax.add_patch(Rectangle((kx0 + v * cw, yk - 5.6), cw, 1.6, fc=cmap(v), ec="none"))
        T(ax, kx0 + (v + 0.5) * cw, yk - 6.2, str(v), ha="center", va="top")
    T(ax, kx0 - 1.0, yk - 4.8, "f(x)", ha="right", va="center", color=S.MUTED)
    return fig


def panel_2E():
    agg, mr = D.replay()
    fig, ax, W, H = canvas("2E", h=86)
    tier(fig, "Replay", W, H)
    meths = D.METHODS_2E
    key = {m: S.ALIAS[m] for m in meths}
    metrics = [("covered", "Triples covered"), ("half_ok", "Half-width ≤ ε"), ("err_ok", "Error ≤ ε")]
    VMIN, VMAX = 0.5, 1.05
    c0, cw, cg = 23.0, 16.0, 3.0
    CX = [c0 + i * (cw + cg) for i in range(3)]

    def X(i, v):
        return CX[i] + (v - VMIN) / (VMAX - VMIN) * cw

    T(ax, 0.8, H - 2.2, "WINGMEN · r = 125 · 1250 cross-family triples", va="center")
    ytitle = H - 6.4
    for i, (_, lab) in enumerate(metrics):
        T(ax, CX[i] + cw / 2, ytitle, lab, ha="center", va="center", fontweight="bold")
    pitch = 4.0
    yf1 = ytitle - 3.5
    yf2 = yf1 - 4 * pitch - 3.6
    ybottom = yf2 - 4 * pitch - 1.4
    for i in range(3):
        for v in (0.6, 0.8, 1.0):
            ax.plot([X(i, v)] * 2, [ybottom, ytitle - 2.0], color=S.GRID, lw=S.THIN, zorder=0)
            ax.plot([X(i, v)] * 2, [ybottom, ybottom - 0.9], color=S.INK, lw=S.LW)
            T(ax, X(i, v), ybottom - 1.4, f"{v * 100:.0f}", ha="center", va="top")
        ax.plot([CX[i], CX[i] + cw], [ybottom, ybottom], color=S.INK, lw=S.LW, solid_capstyle="butt")
    for parent, yf in (("CD4T", yf1), ("CD8T", yf2)):
        eps = float(agg[agg.parent == parent].eps.iloc[0])
        T(ax, 0.8, yf, f"{parent.replace('T', ' T')}  (ε = {eps:g})", va="center", fontweight="bold")
        for j, m in enumerate(meths):
            yy = yf - (j + 1) * pitch + 0.4
            k = key[m]
            T(ax, c0 - 1.6, yy, S.LABEL[k], ha="right", va="center")
            row = agg[(agg.parent == parent) & (agg.method == m)].iloc[0]
            for i, (col, _) in enumerate(metrics):
                lo, hi, mu = row[f"{col}_min"], row[f"{col}_max"], row[f"{col}_mean"]
                ax.plot([X(i, lo), X(i, hi)], [yy, yy], color=S.METHOD[k], lw=S.LW, solid_capstyle="butt", zorder=2)
                for v in (lo, hi):
                    ax.plot([X(i, v)] * 2, [yy - 0.65, yy + 0.65], color=S.METHOD[k], lw=S.LW, zorder=2)
                ax.plot([X(i, mu)], [yy], marker=S.MARKER[k], ms=3.6, mfc=S.METHOD[k], mec=S.INK, mew=S.THIN,
                        ls="none", zorder=3)
                s = f"{mu * 100:.1f}%" if mu < 0.9995 else "100%"
                T(ax, X(i, mu), yy + 0.8, s, ha="center", va="bottom", zorder=4)
    yn = ybottom - 5.2
    T(ax, CX[1] + cw / 2, yn, "% of targets", ha="center", va="center")
    notes = ["half-width / error judged on the max over 8 patterns",
             "per-pattern 95% CP marginal; not joint",
             "symbol = mean; bar = min–max over replicates",
             "OA: 1 design × 10 allocations; others: 3 designs × 10 allocations"]
    for j, s in enumerate(notes):
        T(ax, 0.8, yn - 3.2 - 2.9 * j, s, va="center", color=S.MUTED)

    # ---------- secondary layer: multiplicity ranges over designs (e5c_multiplicities)
    ys = yn - 3.2 - 2.9 * 4 - 0.5
    T(ax, 0.8, ys, "multiplicity range over designs", va="center", fontweight="bold")
    cols = [("single", "marker", 25), ("pair", "pair", 5), ("triple", "triple", 1)]
    sx0, sw, sh, sg = c0, 11.0, 3.0, 0.4
    for i, (_, lab, _) in enumerate(cols):
        T(ax, sx0 + i * (sw + sg) + sw / 2, ys - 3.0, lab, ha="center", va="center", color=S.MUTED)
    T(ax, c0 - 1.6, ys - 3.0, "tubes per", ha="right", va="center", color=S.MUTED)
    norm = mpl.colors.Normalize(0, 7)
    lev = lambda s: norm(s) * 0.85
    for j, m in enumerate(meths):
        r = mr.set_index("method").loc[m]
        yy = ys - 6.2 - j * (sh + sg)
        T(ax, c0 - 1.6, yy, S.LABEL[key[m]], ha="right", va="center")
        for i, (col, _, ideal) in enumerate(cols):
            lo, hi = int(r[f"{col}_lo"]), int(r[f"{col}_hi"])
            spread = (hi - lo) / ideal
            x = sx0 + i * (sw + sg)
            ax.add_patch(Rectangle((x, yy - sh / 2), sw, sh, fc=S.SEQ(lev(spread)), ec="none"))
            T(ax, x + sw / 2, yy, f"{lo}–{hi}", ha="center", va="center", color="white" if lev(spread) > 0.45 else S.INK)
    xk = sx0 + 3 * (sw + sg) + 1.6
    yk = ys - 6.2
    T(ax, xk, yk, "shade =", va="center", color=S.MUTED)
    T(ax, xk, yk - 3.3, "(max − min) / OA", va="center", color=S.MUTED)
    for v in range(8):
        ax.add_patch(Rectangle((xk + v * 1.5, yk - 7.5), 1.5, 1.6, fc=S.SEQ(lev(v)), ec="none"))
    T(ax, xk + 0.75, yk - 8.2, "0", ha="center", va="top", color=S.MUTED)
    T(ax, xk + 7 * 1.5 + 0.75, yk - 8.2, "7", ha="center", va="top", color=S.MUTED)
    return fig


PANELS = {"2A": panel_2A, "2B": panel_2B, "2C": panel_2C, "2D": panel_2D, "2E": panel_2E}

if __name__ == "__main__":
    which = sys.argv[1:] or list(PANELS)
    for p in which:
        fig = PANELS[p]()
        S.save(fig, p)
