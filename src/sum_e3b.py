import os, sys
import pandas as pd
d = sys.argv[1]
df = pd.read_csv(os.path.join(d, "e3b_budget.tsv"), sep="\t")
pd.set_option("display.width", 250)
order = ["random", "balanced_random", "packing_deficit", "LJCR_stack", "LJCR_stack+anneal", "greedy+anneal"]
for (G, eps) in [("T", 0.07), ("T", 0.05), ("B", 0.10), ("B", 0.07)]:
    sub = df[(df.parent == G) & (df.eps == eps)]
    print(f"\n=== parent {G}, eps {eps}: witnessed / empirical half-width<=eps / |err vs ref|<=eps  (mean over seeds)")
    for metric in ["witnessed", "emp_half_ok", "emp_err_ok"]:
        t = sub.groupby(["r", "method"])[metric].mean().unstack("method")[[m for m in order if m in sub.method.unique()]]
        print(f"-- {metric}")
        print(t.round(3).to_string())
