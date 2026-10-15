"""QC channels of downloaded FCS files: quantiles of DNA / dead / beads / a few lineage channels (asinh/5)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fcs

files = sorted(glob.glob(sys.argv[1]))[: int(sys.argv[2]) if len(sys.argv) > 2 else 4]
want = ["191Ir_cells1", "193Ir_cells2", "103Rh_dead", "140Ce_beads1", "154Sm_CD3", "160Gd_CD14", "152Sm_CD66b", "Event_length"]
for f in files:
    X, ch, mk, meta = fcs.read_fcs(f)
    print(f"{os.path.basename(f)}  events={len(X)}  datatype={meta.get('$DATATYPE')}")
    for w in want:
        if w in mk:
            j = mk.index(w)
            x = np.arcsinh(X[:, j] / 5) if w != "Event_length" else X[:, j]
            q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            print(f"   {w:14s} " + " ".join(f"{v:7.2f}" for v in q) + f"   zero%={np.mean(X[:, j] <= 0) * 100:5.1f}")
