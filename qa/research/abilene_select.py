"""Researcher, 16 Oct 2026: choose the Abilene trace scenario's load and budget.

Abilene topology (SNDlib), one-week SNDlib trace with random episode starts
(data/abilene_trace_20040301_7d.csv.gz), built exactly as training builds it
(examples/train_agent.make_env). VALIDATION seeds 1000-1029; test seeds
(2000+) unused.

Rules, fixed before running (the corrected definitions of
qa/research/headline_settle30.py):
  C1 best heuristic delivers 65-88%
  C2 bandwidth helps: max(queue_prop - max_bw, congestion - load_aware) >= 3 pts (signed)
  C3 routing helps:   max(load_aware - max_bw, congestion - queue_prop) >= 3 pts (signed)
  C4 Kendall tau between reward and delivered ranking of the 4 heuristics >= 2/3
  C5 95% CI half-width of the best heuristic's reward <= 0.03
Selection: among cells meeting C1-C5, the one whose best-heuristic delivered
share is closest to 76.5% (the middle of C1), then the lower load. If none
meets all five, C3 is dropped (the scenario is then labelled
bandwidth-dominated) and the same tie-break applies.

    PYTHONPATH=. python qa/research/abilene_select.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.baselines import POLICIES, policy_rng            # noqa: E402
from carla_gym.utils.stats import ci95                          # noqa: E402
from examples.train_agent import parse_args                     # noqa: E402
from scripts.evaluate_saved import episode                      # noqa: E402

TRACE = os.path.join("data", "abilene_trace_20040301_7d.csv.gz")
LOADS, BUDGETS = [40, 60, 80, 100, 120], [0.5, 0.6, 0.7, 0.8]
SEEDS = range(1000, 1030)
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]


def flags(load, budget):
    return ["--topology", "abilene", "--nodes", "12", "--trace", TRACE, "--load", str(load),
            "--budget", str(budget), "--trace-random-start"]


def job(args):
    load, budget, pol, seed = args
    rng = policy_rng(seed)
    r = episode(parse_args(flags(load, budget)), lambda e, o: POLICIES[pol](e, o, rng), seed, False, 20)
    return {"load": load, "budget": budget, "policy": pol, "seed": seed,
            "reward": r["reward"], "delivered": r["delivered_fraction"]}


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def main():
    jobs = [(l, b, p, s) for l, b in itertools.product(LOADS, BUDGETS) for p in HEUR for s in SEEDS]
    with Pool(10) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=8))
    here = os.path.dirname(__file__)
    df.to_csv(os.path.join(here, "abilene_select_episodes.csv"), index=False)
    rows = []
    for (l, b), g in df.groupby(["load", "budget"]):
        m = g.groupby("policy")[["reward", "delivered"]].mean()
        piv = g.pivot(index="seed", columns="policy", values="delivered") * 100
        best = m.loc[HEUR, "reward"].idxmax()
        ci = ci95(g[g.policy == best].reward.to_numpy())
        best_d = 100 * m.loc[HEUR, "delivered"].max()
        bw = max((piv.queue_proportional - piv.max_bandwidth).mean(), (piv.congestion_aware - piv.load_aware_routing).mean())
        rt = max((piv.load_aware_routing - piv.max_bandwidth).mean(), (piv.congestion_aware - piv.queue_proportional).mean())
        tau = kendall(m.loc[HEUR, "reward"].to_numpy(), m.loc[HEUR, "delivered"].to_numpy())
        crit = [65 <= best_d <= 88, bw >= 3, rt >= 3, tau >= 2 / 3 - 1e-9, ci <= 0.03]
        rows.append({"load": l, "budget": b, "best_heuristic": best, "best_reward": round(m.loc[best, "reward"], 3),
                     "ci95": round(ci, 4), "best_delivered_%": round(best_d, 1), "bandwidth_pts": round(bw, 2),
                     "routing_pts": round(rt, 2), "tau": round(tau, 2),
                     "criteria_met": "".join(str(i + 1) if c else "." for i, c in enumerate(crit)),
                     "all5": all(crit), "without_C3": all(c for i, c in enumerate(crit) if i != 2)})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(here, "abilene_select_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"Abilene, one-week SNDlib trace, random starts; validation seeds {SEEDS.start}-{SEEDS.stop - 1}\n")
    print(s.drop(columns=["all5", "without_C3"]).to_string(index=False))
    for label, col in (("all of C1-C5", "all5"), ("C1, C2, C4, C5 (C3 dropped)", "without_C3")):
        q = s[s[col]].copy()
        if len(q):
            q["dist"] = (q["best_delivered_%"] - 76.5).abs()
            q = q.sort_values(["dist", "load"])
            pick = q.iloc[0]
            print(f"\nqualifying under {label}: {len(q)} cell(s)")
            print(f"SELECTED: load {pick.load:g}, budget {pick.budget:g}"
                  + ("" if col == "all5" else "  (labelled bandwidth-dominated: C3 dropped by the fallback rule)"))
            break
        print(f"\nno cell meets {label}")


if __name__ == "__main__":
    main()
