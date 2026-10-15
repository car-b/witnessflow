"""Rare-parent stress protocols (B cells thinned in initial+followup splits), derived from v2."""
import json
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")
v2 = json.load(open(p + r"\levine32_v2.json", encoding="utf-8"))
for f, tag in ((0.01, "1"), (0.003, "0p3"), (0.001, "0p1")):
    pr = json.loads(json.dumps(v2))
    pr["protocol_id"] = f"levine32_v2_rareB_{tag}"
    pr["amends"] = "levine32_v2"
    pr["note"] = (f"Stress test: B cells thinned to {f:.1%} of all cells in the initial and follow-up splits "
                  "(within-B distribution preserved by uniform subsampling); reference split untouched. "
                  "Everything else as v2.")
    pr["downsample"] = {"parent": "B", "target_frac": f, "seed": 11}
    json.dump(pr, open(p + rf"\levine32_v2_rareB_{tag}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(pr["protocol_id"])
