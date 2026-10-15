"""Engineering checks for the audit core (light version)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import audit
import design

rng = np.random.default_rng(0)
fails = 0


def check(name, cond, info=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not cond:
        fails += 1


def parity_world(n, even=True):
    bits = audit.PAT[[0, 3, 5, 6]] if even else audit.PAT[[1, 2, 4, 7]]
    return bits[rng.integers(0, 4, n)]


# --- 1. parity: pairwise tubes cannot decide P(111); identification range must be [0, 0.25]
for even in (True, False):
    n = 60000
    B = parity_world(n, even)
    tube = rng.integers(0, 3, n)
    panels = [{0, 1}, {0, 2}, {1, 2}]
    inG = np.ones(n, bool)
    st = np.zeros(n, int)
    ev = audit.evidence(B, (0, 1, 2), inG, st, 1, tube, panels)
    r = audit.bound_claim(ev, 1, 7)
    truth = 0.0 if even else 0.25
    check(f"parity({'even' if even else 'odd'}) ident range ~[0,0.25]",
          abs(r["ident_lo"]) < 0.01 and abs(r["ident_hi"] - 0.25) < 0.01, f"{r['ident_lo']:.4f},{r['ident_hi']:.4f}")
    check(f"parity({'even' if even else 'odd'}) conf interval contains truth",
          r["conf_lo"] - 1e-9 <= truth <= r["conf_hi"] + 1e-9, f"{r['conf_lo']:.4f},{r['conf_hi']:.4f} truth={truth}")
    check(f"parity({'even' if even else 'odd'}) maxent = 0.125", abs(r["maxent"] - 0.125) < 0.01, f"{r['maxent']:.4f}")
    # add a tube measuring all three -> direct estimate decides
    tube2 = rng.integers(0, 4, n)
    panels2 = panels + [{0, 1, 2}]
    ev2 = audit.evidence(B, (0, 1, 2), inG, st, 1, tube2, panels2)
    r2 = audit.bound_claim(ev2, 1, 7)
    d = audit.direct_estimate(ev2, 7)
    check(f"parity({'even' if even else 'odd'}) + ABC tube: ident collapses", r2["ident_hi"] - r2["ident_lo"] < 0.01,
          f"{r2['ident_lo']:.4f},{r2['ident_hi']:.4f}")
    check(f"parity({'even' if even else 'odd'}) direct estimate correct", abs(d["direct"] - truth) < 0.01, f"{d['direct']:.4f}")

# --- 2. no evidence at all -> [0,1]
B = rng.integers(0, 2, (1000, 3)).astype(np.uint8)
ev = audit.evidence(B, (0, 1, 2), np.ones(1000, bool), np.zeros(1000, int), 1, np.zeros(1000, int), [set()])
r = audit.bound_claim(ev, 1, 0)
check("no evidence -> [0,1]", abs(r["conf_lo"]) < 1e-9 and abs(r["conf_hi"] - 1) < 1e-9,
      f"{r['conf_lo']},{r['conf_hi']},{r['lp_status']}")

# --- 3. zero counts are not probability zero: tiny direct sample with 0 hits gives hi > 0
B = np.zeros((30, 3), np.uint8)
ev = audit.evidence(B, (0, 1, 2), np.ones(30, bool), np.zeros(30, int), 1, np.zeros(30, int), [{0, 1, 2}])
r = audit.bound_claim(ev, 1, 7)
check("zero count keeps positive upper bound", r["conf_hi"] > 0.05, f"hi={r['conf_hi']:.3f}")

# --- 4. confidence coverage on a random 3-way table with pairwise tubes (simulation)
cover, reps = 0, 200
p = rng.dirichlet(np.ones(8))
for i in range(reps):
    n = 3000
    x = rng.choice(8, n, p=p)
    B = audit.PAT[x]
    tube = rng.integers(0, 3, n)
    ev = audit.evidence(B, (0, 1, 2), np.ones(n, bool), np.zeros(n, int), 1, tube, [{0, 1}, {0, 2}, {1, 2}])
    r = audit.bound_claim(ev, 1, 7)
    cover += r["conf_lo"] - 1e-9 <= p[7] <= r["conf_hi"] + 1e-9
check("confidence bounds cover truth >= 93% (200 sims)", cover / reps >= 0.93, f"{cover}/{reps}")

# --- 5. strata K=2 with backbone-dependent marker: bounds still valid
n = 40000
s = rng.integers(0, 2, n)
B = np.zeros((n, 3), np.uint8)
B[:, 0] = rng.random(n) < np.where(s == 1, 0.9, 0.1)
B[:, 1] = rng.random(n) < np.where(s == 1, 0.8, 0.2)
B[:, 2] = rng.random(n) < 0.5
tube = rng.integers(0, 2, n)
ev = audit.evidence(B, (0, 1, 2), np.ones(n, bool), s, 2, tube, [{0}, {1, 2}])
r = audit.bound_claim(ev, 2, 7)
truth = (audit.codes(B, (0, 1, 2)) == 7).mean()
check("K=2 strata bounds contain truth", r["conf_lo"] <= truth <= r["conf_hi"], f"{r['conf_lo']:.3f}<={truth:.3f}<={r['conf_hi']:.3f}")
ev1 = audit.evidence(B, (0, 1, 2), np.ones(n, bool), np.zeros(n, int), 1, tube, [{0}, {1, 2}])
r1 = audit.bound_claim(ev1, 1, 7)
check("strata (K=2) tighten identification vs K=1",
      (r["ident_hi"] - r["ident_lo"]) < (r1["ident_hi"] - r1["ident_lo"]),
      f"K2={r['ident_hi'] - r['ident_lo']:.3f} K1={r1['ident_hi'] - r1['ident_lo']:.3f}")

# --- 6. kNN block vs majority on parity: both blind (estimate ~0.125)
n = 20000
B = parity_world(n, True)
Xbb = rng.normal(size=(n, 2))
tube = rng.integers(0, 3, n)
panels = [{0, 1}, {0, 2}, {1, 2}]
for mode in ("majority", "block"):
    Bi = audit.knn_impute(B, Xbb, tube, panels, k=10, mode=mode)
    f = audit.imputed_frequency(Bi, (0, 1, 2), np.ones(n, bool), 7)
    check(f"knn-{mode} cannot see parity (est far from 0)", f > 0.05, f"est={f:.3f} truth=0")

# --- 6b. EM fit is order independent with inconsistent margins (counterexample)
evA = [((0,), np.array([[20.0, 80.0]])), ((0, 1), np.array([[45.0, 45.0, 5.0, 5.0]]))]
z1 = audit.maxent_fit(evA, 1)
z2 = audit.maxent_fit(evA[::-1], 1)
pA1 = z1[4:].sum()
check("EM fit order-independent", np.allclose(z1, z2, atol=1e-6), f"P(A=1)={pA1:.3f} vs {z2[4:].sum():.3f}")
check("EM fit = pooled MLE for A ((80+10)/200)", abs(pA1 - 90 / 200) < 1e-3, f"{pA1:.4f}")

# --- 6c. kNN refuses to run when a used marker has no donor (no hidden-truth leak)
B = rng.integers(0, 2, (300, 3)).astype(np.uint8)
try:
    audit.knn_impute(B, rng.normal(size=(300, 2)), np.zeros(300, int), [{0, 1, 2}], k=5)
    check("kNN with single full tube runs", True)
except ValueError:
    check("kNN with single full tube runs", False)
try:
    audit.knn_impute(B, rng.normal(size=(300, 2)), rng.integers(0, 2, 300), [{0, 1}, {0, 1, 2}], k=5)
    check("kNN fills marker 2 for tube 0 from tube 1", True)
except ValueError as e:
    check("kNN fills marker 2 for tube 0 from tube 1", False, str(e))

# --- 6d. tube consistency flags a shifted tube, not a homogeneous split
n = 20000
B = rng.integers(0, 2, (n, 3)).astype(np.uint8)
tube = rng.integers(0, 2, n)
panels = [{0, 1}, {0, 2}]
inG = np.ones(n, bool)
r0 = audit.tube_consistency(B, (0, 1, 2), inG, tube, panels)
B2 = B.copy()
shift = (tube == 1) & (rng.random(n) < 0.15)
B2[shift, 0] = 1  # tube 1 reads marker 0 'brighter'
r1 = audit.tube_consistency(B2, (0, 1, 2), inG, tube, panels)
check("tube consistency: homogeneous not flagged", r0["tube_minp"] > 1e-3, f"p={r0['tube_minp']:.3g}")
check("tube consistency: shifted tube flagged", r1["tube_minp"] < 1e-6 and r1["tube_maxtv"] > 0.05,
      f"p={r1['tube_minp']:.3g} tv={r1['tube_maxtv']:.3f}")

# --- 7. design verifier + ILP on a tiny instance
cand = list(range(8))
targets = [(0, 1, 2), (3, 4, 5), (0, 6, 7), (1, 3, 6)]
w = [1, 1, 1, 1]
tubes, val, res = design.ilp_design(cand, 4, 2, targets, w, time_limit=30)
check("ILP tiny: covers 2 targets with 2 tubes of 4? (>=2)", val is not None and val >= 2, f"val={val}")
check("verifier agrees with ILP value", design.score(tubes, targets, w) == val)
g = design.greedy_design(cand, 4, 2, targets, w)
design.check_design(g, cand, 4)
check("greedy design legal", True)

print("FAILS:", fails)
sys.exit(1 if fails else 0)
