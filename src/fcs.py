"""Minimal FCS 2.0/3.0/3.1 list-mode reader (no external deps).

read_fcs(path) -> (data float64 [n_events, n_par], channels [$PnN], markers [$PnS or $PnN], meta dict)
"""
import numpy as np


def _parse_text(raw: bytes) -> dict:
    s = raw.decode("latin-1")
    delim = s[0]
    # split on single delimiters; doubled delimiters are escaped literals
    parts, cur, i = [], [], 1
    while i < len(s):
        c = s[i]
        if c == delim:
            if i + 1 < len(s) and s[i + 1] == delim:
                cur.append(delim)
                i += 2
                continue
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur))
    meta = {}
    for k, v in zip(parts[0::2], parts[1::2]):
        meta[k.strip().upper()] = v.strip()
    return meta


def read_fcs(path):
    with open(path, "rb") as f:
        buf = f.read()
    ver = buf[:6].decode("ascii", "replace")
    if not ver.startswith("FCS"):
        raise ValueError(f"not an FCS file: {path} (starts with {buf[:16]!r})")
    off = [int(buf[a:b].decode("ascii").strip() or 0) for a, b in ((10, 18), (18, 26), (26, 34), (34, 42))]
    t0, t1, d0, d1 = off
    meta = _parse_text(buf[t0:t1 + 1])
    if d0 == 0 or d1 == 0:
        d0 = int(meta.get("$BEGINDATA", 0))
        d1 = int(meta.get("$ENDDATA", 0))
    npar = int(meta["$PAR"])
    ntot = int(meta["$TOT"])
    dtype = meta.get("$DATATYPE", "F").upper()
    order = meta.get("$BYTEORD", "1,2,3,4").replace(" ", "")
    endian = "<" if order.startswith("1") else ">"
    if meta.get("$MODE", "L").upper() != "L":
        raise ValueError("only list mode supported")
    raw = buf[d0:d1 + 1]
    if dtype in ("F", "D"):
        dt = np.dtype(endian + ("f4" if dtype == "F" else "f8"))
        arr = np.frombuffer(raw[: ntot * npar * dt.itemsize], dtype=dt).reshape(ntot, npar)
        data = arr.astype(np.float64)
    elif dtype == "I":
        bits = [int(meta[f"$P{j}B"]) for j in range(1, npar + 1)]
        if len(set(bits)) != 1 or bits[0] not in (8, 16, 32, 64):
            raise ValueError(f"unsupported integer layout {set(bits)}")
        dt = np.dtype(endian + {8: "u1", 16: "u2", 32: "u4", 64: "u8"}[bits[0]])
        arr = np.frombuffer(raw[: ntot * npar * dt.itemsize], dtype=dt).reshape(ntot, npar)
        data = arr.astype(np.float64)
        for j in range(npar):
            rng = int(float(meta.get(f"$P{j + 1}R", 0)))
            if rng > 0 and (rng & (rng - 1)) == 0:
                data[:, j] = arr[:, j] & (rng - 1)
    else:
        raise ValueError(f"unsupported $DATATYPE {dtype}")
    channels = [meta.get(f"$P{j}N", f"P{j}") for j in range(1, npar + 1)]
    markers = [meta.get(f"$P{j}S", "") or channels[j - 1] for j in range(1, npar + 1)]
    meta["_version"] = ver
    return data, channels, markers, meta


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        d, ch, mk, m = read_fcs(p)
        print(p, m["_version"], d.shape)
        for c, k in zip(ch, mk):
            print(f"  {c:>16s}  {k}")
