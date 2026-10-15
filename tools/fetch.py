"""Download public files with size/hash logging.  usage: python fetch.py <url> <out_path>"""
import hashlib, os, sys, urllib.request

url, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
h = hashlib.sha256()
n = 0
with urllib.request.urlopen(req, timeout=120) as r, open(out + ".part", "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b:
            break
        f.write(b)
        h.update(b)
        n += len(b)
os.replace(out + ".part", out)
with open(out, "rb") as f:
    head = f.read(16)
print(f"{out}\t{n}\tsha256={h.hexdigest()}\thead={head!r}")
