"""Compose single-panel PDFs into figure pages, 1:1 (no scaling, so Arial 7 pt and 1/0.5 pt strokes are preserved).

Layout = nested Row / Col / P(anel). Every panel keeps its exported physical size; a panel that does not fit must be
re-rendered at a new size by its own script (never scaled here). Panel letters (Arial Bold 7 pt, lowercase) sit in a
LETTER band above the top-left corner of each panel. Output: figures/composite/<name>.pdf (vector, text editable) and
<name>.png (600 dpi), then the same strict audit as single panels plus an A4 bound check.

    from compose import Row, Col, P, compose
    compose("Fig4", Col([P("4A", "a"), Row([P("4B", "b"), P("4C", "c")]), Row([P("4D", "d"), P("4E", "e")])]),
            width=184)

Row(children, gap=H_GAP, justify=True): children side by side, tops aligned; if the row is narrower than the width
offered by its parent and justify=True, the extra width goes into the gaps (gap never shrinks below `gap`).
Col(children, gap=V_GAP): children stacked, left aligned.  P(panel, letter): one panel PDF figures/panels/fig<panel>.pdf.
"""
import os
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
PANELS = os.path.join(FIG, "panels")
OUT = os.path.join(FIG, "composite")
MM = 72 / 25.4
A4 = (210.0, 297.0)
H_GAP = 2.0          # mm between panels in a row (minimum)
V_GAP = 1.0          # mm between a panel bottom and the next letter band
LETTER = 3.0         # mm letter band above each panel
MARGIN = 1.0         # mm page margin around the content
LETTER_FONT = r"C:\Windows\Fonts\arialbd.ttf"
LETTER_PT = 7.0


def _open_pdf(path):
    """Read through Python's open() so the provenance tracer (audit hook) sees the read; PyMuPDF's own C-level
    file I/O is invisible to sys.addaudithook."""
    with open(path, "rb") as f:
        return fitz.open(stream=f.read(), filetype="pdf")


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def panel_size(pid):
    with _open_pdf(os.path.join(PANELS, f"fig{pid}.pdf")) as d:
        r = d[0].rect
    return r.width / MM, r.height / MM


class P:
    def __init__(self, pid, letter):
        self.pid, self.letter = pid, letter
        self.w, self.h0 = panel_size(pid)
        self.h = self.h0 + LETTER

    def place(self, x, y, avail_w, out):
        out.append((self, x, y))


class Row:
    def __init__(self, children, gap=H_GAP, justify=True):
        self.c, self.gap, self.justify = children, gap, justify
        self.w = sum(c.w for c in children) + gap * (len(children) - 1)
        self.h = max(c.h for c in children)

    def place(self, x, y, avail_w, out):
        extra = max(0.0, avail_w - self.w) if (self.justify and len(self.c) > 1) else 0.0
        g = self.gap + extra / max(1, len(self.c) - 1)
        for c in self.c:
            c.place(x, y, c.w, out)
            x += c.w + g


class Col:
    def __init__(self, children, gap=V_GAP):
        self.c, self.gap = children, gap
        self.w = max(c.w for c in children)
        self.h = sum(c.h for c in children) + gap * (len(children) - 1)

    def place(self, x, y, avail_w, out):
        for c in self.c:
            c.place(x, y, max(avail_w, c.w) if isinstance(c, Row) else c.w, out)
            y += c.h + self.gap


def compose(name, layout, width=None, check_only=False):
    """width: content width in mm offered to the top layout (rows justify to it). Returns a report dict."""
    W = width or layout.w
    assert layout.w <= W + 1e-3, f"{name}: layout {layout.w:.1f} mm wider than requested {W:.1f} mm"
    W = max(W, layout.w)   # absorb PDF page-size rounding (~1e-6 mm)
    placed = []
    layout.place(MARGIN, MARGIN, W, placed)
    pw = MARGIN * 2 + W
    ph = MARGIN * 2 + layout.h
    assert pw <= A4[0] + 1e-6 and ph <= A4[1] + 1e-6, f"{name}: page {pw:.1f} x {ph:.1f} mm exceeds A4"
    # overlap check (panel boxes incl. letter band)
    boxes = [(p.pid, x, y, x + p.w, y + p.h) for p, x, y in placed]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6 and a[2] < b[4] - 1e-6 and b[2] < a[4] - 1e-6:
                raise AssertionError(f"{name}: panels {a[0]} and {b[0]} overlap")
    rep = dict(name=name, page_mm=(round(pw, 1), round(ph, 1)),
               panels=[(p.pid, p.letter, round(x, 1), round(y, 1), round(p.w, 1), round(p.h0, 1)) for p, x, y in placed])
    area = sum(p.w * p.h0 for p, _, _ in placed)
    rep["panel_area_frac"] = round(area / (pw * ph), 3)
    if check_only:
        return rep
    os.makedirs(OUT, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=pw * MM, height=ph * MM)
    page.insert_font(fontname="ArialBold", fontfile=LETTER_FONT)
    for p, x, y in placed:
        src = _open_pdf(os.path.join(PANELS, f"fig{p.pid}.pdf"))
        top = y + LETTER
        page.show_pdf_page(fitz.Rect(x * MM, top * MM, (x + p.w) * MM, (top + p.h0) * MM), src, 0)
        src.close()
        page.insert_text(((x + 0.3) * MM, (y + LETTER - 0.75) * MM), p.letter, fontname="ArialBold",
                         fontsize=LETTER_PT, color=(0.133, 0.133, 0.133))
    pdf = os.path.join(OUT, f"{name}.pdf")
    _write_bytes(pdf, doc.tobytes(garbage=4, deflate=True))       # Python-level write: visible to the tracer
    page = doc[0]
    _write_bytes(os.path.join(OUT, f"{name}.png"), page.get_pixmap(dpi=600).tobytes("png"))
    doc.close()
    rep.update(audit(pdf))
    return rep


def audit(pdf, tol=0.05):
    """Same rules as wfstyle.save: every text span 7 pt and inside the page; strokes 1 or 0.5 pt; Arial fonts only."""
    doc = fitz.open(pdf)
    page = doc[0]
    R = page.rect
    small, outside, widths = [], [], set()
    for b in page.get_text("dict", clip=fitz.INFINITE_RECT())["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                if abs(s["size"] - LETTER_PT) > tol:
                    small.append((s["text"], round(s["size"], 2)))
                x0, y0, x1, y1 = s["bbox"]
                if x0 < R.x0 or y0 < R.y0 or x1 > R.x1 or y1 > R.y1:
                    outside.append(s["text"])
    for d in page.get_drawings():
        if d.get("color") is not None and d.get("width"):
            widths.add(round(float(d["width"]), 3))
    fonts = sorted({f[3] for f in page.get_fonts(full=True)})
    doc.close()
    badw = sorted(w for w in widths if min(abs(w - 1.0), abs(w - 0.5)) > 0.011)
    badf = [f for f in fonts if "Arial" not in f]
    ok = not (small or outside or badw or badf)
    return dict(audit_ok=ok, n_small=len(small), n_outside=len(outside), bad_widths=badw, fonts=fonts, bad_fonts=badf,
                small_eg=small[:5], outside_eg=outside[:5])


def report(rep):
    print(f"{rep['name']}: page {rep['page_mm'][0]} x {rep['page_mm'][1]} mm; panel area {rep['panel_area_frac']:.0%}; "
          f"audit {'OK' if rep.get('audit_ok', True) else 'BAD'} "
          + ("" if rep.get("audit_ok", True) else str({k: rep[k] for k in ('n_small', 'n_outside', 'bad_widths', 'bad_fonts', 'small_eg', 'outside_eg')})))
    for pid, L, x, y, w, h in rep["panels"]:
        print(f"   {L}  {pid:4s} at ({x:6.1f},{y:6.1f})  {w:5.1f} x {h:5.1f} mm")
