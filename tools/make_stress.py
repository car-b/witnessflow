"""Backbone-reduction stress protocols derived from levine32_v2 (written before running them)."""
import json, math, os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")
v2 = json.load(open(p + r"\levine32_v2.json", encoding="utf-8"))
variants = {
    "bb6": {"backbone": ["CD45", "CD3", "CD19", "CD33", "CD34", "CD14"],
            "parents": {"HSPC": {"pos": ["CD34"], "neg": ["CD3", "CD19"]},
                        "T": {"pos": ["CD3"], "neg": ["CD19"]},
                        "B": {"pos": ["CD19"], "neg": ["CD3"]},
                        "Mono": {"pos": ["CD33", "CD14"], "neg": ["CD3", "CD19"]}}},
    "bb4": {"backbone": ["CD45", "CD3", "CD19", "CD33"],
            "parents": {"T": {"pos": ["CD3"], "neg": ["CD19"]},
                        "B": {"pos": ["CD19"], "neg": ["CD3"]}}},
}
n_markers = 32
for name, var in variants.items():
    pr = json.loads(json.dumps(v2))
    pr["protocol_id"] = f"levine32_v2_{name}"
    pr["amends"] = "levine32_v2"
    pr["note"] = (f"Stress test: backbone reduced to {len(var['backbone'])} markers; removed backbone markers become "
                  "ordinary markers distributed over the initial tubes; parents restricted to those gateable on "
                  "the reduced backbone; claims restricted accordingly.  Same splits, seeds, budgets as v2.")
    pr["backbone"] = var["backbone"]
    pr["parents"] = var["parents"]
    pr["claims"] = [c for c in v2["claims"] if c["parent"] in var["parents"]]
    n_non = n_markers - len(var["backbone"])
    pr["initial_design"]["capacity"] = math.ceil(n_non / pr["initial_design"]["n_tubes"])
    pr["followup"]["capacity"] = 6
    json.dump(pr, open(p + rf"\levine32_v2_{name}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(name, len(pr["claims"]), "claims; initial capacity", pr["initial_design"]["capacity"])
