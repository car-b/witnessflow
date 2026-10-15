"""Cross-tube intensity-shift stress protocols (simulated batch effect in initial tube 0), from levine32_v2.
Written before running.  'all' = every marker measured in tube 0 incl. backbone (staining/acquisition
brightness); 'nonbackbone' = only tube-specific markers (invisible to backbone-based checks)."""
import json
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")
v2 = json.load(open(p + r"\levine32_v2.json", encoding="utf-8"))
for tag, markers, delta in (("shiftAll+0p5", "all", 0.5), ("shiftNonBB+0p5", "nonbackbone", 0.5),
                            ("shiftNonBB-0p5", "nonbackbone", -0.5)):
    pr = json.loads(json.dumps(v2))
    pr["protocol_id"] = f"levine32_v2_{tag.replace('+', 'p').replace('-', 'm')}"
    pr["amends"] = "levine32_v2"
    pr["note"] = (f"Stress test (simulated, not real batch data): in every replicate, cells of initial tube 0 get "
                  f"{delta:+.1f} asinh units on {markers} markers measured in that tube, then are re-gated with the "
                  "global thresholds.  Reference and follow-up cells unshifted.")
    pr["tube_shift"] = {"tube": 0, "markers": markers, "delta": delta}
    json.dump(pr, open(p + rf"\{pr['protocol_id']}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(pr["protocol_id"])
