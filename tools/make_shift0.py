import json
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")
v2 = json.load(open(p + r"\levine32_v2.json", encoding="utf-8"))
pr = json.loads(json.dumps(v2))
pr["protocol_id"] = "levine32_v2_shift0"
pr["amends"] = "levine32_v2"
pr["note"] = "Matched control for the tube_shift stress tests: same code path, delta = 0."
pr["tube_shift"] = {"tube": 0, "markers": "all", "delta": 0.0}
json.dump(pr, open(p + r"\levine32_v2_shift0.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(pr["protocol_id"])
