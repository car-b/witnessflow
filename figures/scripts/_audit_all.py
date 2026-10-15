"""Audit every panel PDF with wfstyle.pdf_audit (7 pt only, inside page, 1/0.5 pt strokes) without re-rendering."""
import glob, os
import wfstyle as S

rows = []
for f in sorted(glob.glob(os.path.join(S.PANELS, "fig*.pdf"))):
    small, outside, badw, widths = S.pdf_audit(f)
    fonts = [x[0] for x in S.pdf_fonts(f)]
    ok = not (small or outside or badw) and all("Arial" in x for x in fonts)
    rows.append((os.path.basename(f), ok, len(small), len(outside), badw))
    print(f"{os.path.basename(f):10s} {'OK ' if ok else 'BAD'} small={len(small):2d} outside={len(outside):2d} badw={badw} widths={widths}"
          + (f"  e.g. small={small[:3]} outside={outside[:2]}" if not ok else ""))
print("panels:", len(rows), "compliant:", sum(r[1] for r in rows))
