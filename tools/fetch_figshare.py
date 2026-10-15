"""Download all files of a figshare article (public, CC BY), logging size + sha256 to MANIFEST.tsv.
usage: fetch_figshare.py <article_id> <out_dir>"""
import hashlib, json, os, sys, urllib.request

aid, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
req = urllib.request.Request(f"https://api.figshare.com/v2/articles/{aid}", headers={"User-Agent": "Mozilla/5.0"})
art = json.load(urllib.request.urlopen(req, timeout=60))
man = open(os.path.join(out, "MANIFEST.tsv"), "a", encoding="utf-8")
man.write(f"# figshare {aid} doi={art.get('doi')} license={art.get('license', {}).get('name')} title={art['title']}\n")
for f in art["files"]:
    path = os.path.join(out, f["name"])
    if os.path.exists(path) and os.path.getsize(path) == f["size"]:
        print("skip", f["name"]); continue
    h = hashlib.sha256()
    r = urllib.request.urlopen(urllib.request.Request(f["download_url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=300)
    with open(path + ".part", "wb") as fo:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fo.write(b)
            h.update(b)
    os.replace(path + ".part", path)
    ok = os.path.getsize(path) == f["size"]
    man.write(f"{f['name']}\t{f['size']}\t{h.hexdigest()}\tsupplied_md5={f.get('computed_md5')}\tsize_ok={ok}\n")
    man.flush()
    print(f["name"], f["size"], "ok" if ok else "SIZE MISMATCH", flush=True)
print("done")
