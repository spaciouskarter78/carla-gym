"""Researcher, 16 Oct 2026. POST HOC: written after qa/research/abilene_select.py
found no cell meeting its pre-set rules (C5 failed in every congested cell).

Paired stability for the cells meeting C1-C4: 95% CI half-width, over the 30
validation seeds, of (best heuristic - second-best heuristic) reward, and of
(best heuristic - max_bandwidth). Validation seeds only; reads
qa/research/abilene_select_episodes.csv.

Revised rule C5' (approved by the user, recorded in docs/ABILENE_SCENARIO.md):
paired CI of (best - second-best) reward <= 0.03. Selection otherwise as
pre-set: closest best-heuristic delivered share to 76.5%, then lower load.

    PYTHONPATH=. python qa/research/abilene_select_paired.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.utils.stats import ci95  # noqa: E402

here = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(here, "abilene_select_episodes.csv"))
s = pd.read_csv(os.path.join(here, "abilene_select_summary.csv"))
cand = s[s.criteria_met.str.startswith("1234")]
rows = []
for r in cand.itertuples():
    g = df[(df.load == r.load) & (df.budget == r.budget)]
    piv = g.pivot(index="seed", columns="policy", values="reward")
    order = piv.mean().sort_values(ascending=False).index
    d1 = (piv[order[0]] - piv[order[1]]).to_numpy()
    d2 = (piv[order[0]] - piv["max_bandwidth"]).to_numpy()
    rows.append({"load": r.load, "budget": r.budget, "best": order[0], "delivered_%": r._6,
                 "unpaired_ci": r.ci95, "best_minus_2nd": round(d1.mean(), 3), "paired_ci": round(ci95(d1), 3),
                 "best_minus_max_bw": round(d2.mean(), 3), "paired_ci_vs_max_bw": round(ci95(d2), 3)})
t = pd.DataFrame(rows)
t["C5_paired"] = t.paired_ci <= 0.03
pd.set_option("display.width", 200)
print("cells meeting C1-C4; paired 95% CI half-width over 30 validation seeds\n")
print(t.to_string(index=False))
q = t[t.C5_paired].assign(dist=lambda x: (x["delivered_%"] - 76.5).abs()).sort_values(["dist", "load"])
print(f"\nqualifying under C1-C4 + C5': {len(q)} cell(s)")
print(f"SELECTED: load {q.iloc[0].load:g}, budget {q.iloc[0].budget:g} "
      f"(best heuristic {q.iloc[0].best}, {q.iloc[0]['delivered_%']}% delivered)")
