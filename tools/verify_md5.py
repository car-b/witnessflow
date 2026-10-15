import hashlib, os, re, sys
d = sys.argv[1]
bad = ok = 0
for line in open(os.path.join(d, "MANIFEST.tsv"), encoding="utf-8"):
    if line.startswith("#"):
        continue
    parts = line.rstrip("\n").split("\t")
    name = parts[0]
    md5 = re.search(r"supplied_md5=(\w+)", line).group(1)
    h = hashlib.md5(open(os.path.join(d, name), "rb").read()).hexdigest()
    if h == md5:
        ok += 1
    else:
        bad += 1
        print("MISMATCH", name)
print(f"md5 ok={ok} bad={bad}")
