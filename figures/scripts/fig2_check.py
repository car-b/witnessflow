"""dev check: font sizes / fonts / stroke widths in the Fig. 2 panel PDFs."""
import os
from collections import Counter
import fitz
import wfstyle as S

for p in ["2A", "2B", "2C", "2D", "2E"]:
    path = os.path.join(S.PANELS, f"fig{p}.pdf")
    doc = fitz.open(path)
    page = doc[0]
    sizes, fonts, small = Counter(), Counter(), []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                sizes[round(s["size"], 2)] += 1
                fonts[s["font"]] += 1
                if round(s["size"], 2) != 7.0:
                    small.append((round(s["size"], 2), s["text"]))
    widths = Counter(round(d["width"], 2) for d in page.get_drawings() if d.get("width") is not None)
    w, h = page.rect.width / 72 * 25.4, page.rect.height / 72 * 25.4
    print(f"fig{p}: {w:.1f} x {h:.1f} mm | sizes {dict(sizes)} | fonts {dict(fonts)}")
    print(f"   stroke widths (pt): {dict(sorted(widths.items()))}")
    if small:
        print("   non-7pt spans:", small[:12])
