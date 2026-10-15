import json
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols", "wingmen_v1.json")
pr = json.load(open(p, encoding="utf-8"))
pr["e3"] = {"eps": {"CD4T": [0.03, 0.05], "CD8T": [0.05, 0.07]},
            "note": "added 2026-10-09 after the quick replay; mirrors the Levine e3e choices (larger parent tighter eps)"}
json.dump(pr, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("ok")
