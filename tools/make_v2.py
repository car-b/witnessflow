import json
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")
v1 = json.load(open(p + r"\levine32_v1.json", encoding="utf-8"))
v2 = dict(v1)
v2["protocol_id"] = "levine32_v2"
v2["amends"] = "levine32_v1"
v2["frozen_before_unblinding"] = False
v2["note"] = "v2 was written AFTER the v1 quick run (2 replicates) had been inspected."
v2["amendment"] = (
    "Threshold rule changed after the v1 quick run (2 replicates) showed that the 2-GMM split the CyTOF "
    "zero spike from non-zero values (cuts ~0.02 for CD4/CD133/CD14/CD15/CD117), and that dBIC>=10 is "
    "uninformative at n~50k.  v2: values <= 0.05 are negative and excluded from the fit; 2-GMM on the rest; "
    "reliable iff Ashman's D >= 1.5 and upper weight >= 0.03 (D_min chosen after seeing the per-marker D "
    "diagnostic; textbook value 2 would drop 13/32 markers in mixed bone marrow).  Claims, splits, designs, "
    "budgets unchanged.")
v2["threshold_rule"] = {"method": "zero-spike-excluded 2-GMM", "zero_cut": 0.05, "D_min": 1.5, "w_min": 0.03,
                        "sensitivity_offsets": [-0.25, -0.125, 0.0, 0.125, 0.25]}
json.dump(v2, open(p + r"\levine32_v2.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("ok")
