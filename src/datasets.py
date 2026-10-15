"""Dataset loaders beyond the single-file Levine-32 case.

fcs_dir: one FCS per sample (e.g. WINGMEN PBMC); marker name = $PnS with the '<mass><element>_' prefix
stripped; samples = files matching `pattern` with >= min_events; optional light dead-cell filter.
"""
import glob, os, re
import numpy as np
import fcs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def strip_metal(name):
    return re.sub(r"^\d+[A-Za-z]{1,2}_", "", name)


def load_fcs_dir(ds):
    files = sorted(glob.glob(os.path.join(ROOT, ds["dir"], ds.get("pattern", "*.fcs"))))
    Xs, samples, names, markers_ref = [], [], [], None
    for f in files:
        X, ch, mk, meta = fcs.read_fcs(f)
        if len(X) < ds.get("min_events", 0):
            continue
        mk = [strip_metal(m) for m in mk]
        keep = [i for i, m in enumerate(mk) if m in ds["markers"]]
        mk_keep = [mk[i] for i in keep]
        if markers_ref is None:
            markers_ref = [m for m in ds["markers"] if m in mk_keep]
        idx = [keep[mk_keep.index(m)] for m in markers_ref]
        Xt = np.arcsinh(X[:, idx] / ds.get("cofactor", 5.0))
        dead = ds.get("dead_channel")
        if dead:
            j = mk.index(dead) if dead in mk else None
            if j is not None:
                ok = np.arcsinh(X[:, j] / 5.0) <= ds.get("dead_max_asinh", 99)
                Xt = Xt[ok]
        Xs.append(Xt)
        names.append(os.path.splitext(os.path.basename(f))[0])
        samples.append(np.full(len(Xt), len(names) - 1))
    return np.vstack(Xs), markers_ref, np.concatenate(samples), names
