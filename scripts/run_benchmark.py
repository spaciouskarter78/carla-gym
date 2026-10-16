"""
scripts/run_benchmark.py

Scenario matrix for the rule-based baselines: every reference policy on
topologies x offered loads x bandwidth budgets (synthetic Poisson traffic),
on a node-bottleneck scenario, and on the dataset-driven traces.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/run_benchmark.py                  # full matrix, ~10-15 min
    python scripts/run_benchmark.py --quick          # small subset, < 1 min

Writes to --out (default results/benchmark/):
    episodes.csv   one row per (scenario, policy, seed): post-warmup means
    summary.csv    mean and 95% confidence interval over seeds
    summary.md     reward table, one row per scenario; the best policy is bold
                   only when its interval overlaps no other, ties are marked ~
    qos.md         delivered and dropped fraction of traffic per scenario

Every policy sees the same seeds within a scenario. The 95% CI is the
Student-t interval over the per-seed episode means.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from carla_gym.baselines import POLICIES, policy_rng
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv
from carla_gym.utils.qos import EpisodeQoS
from carla_gym.utils.stats import ci95

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
# Per-step reward terms, averaged over the episode ...
METRICS = ["reward", "throughput", "loss", "latency_steps", "deadline_miss"]
# ... and episode-aggregate QoS from the raw counters (see utils/qos.py).
QOS = ["delivered_fraction", "dropped_fraction", "latency_delivered", "deadline_miss_per_delivered"]


def scenarios(quick: bool) -> list[dict]:
    out = []
    topologies = ["grid", "ring"] if quick else ["grid", "ring", "star", "mesh", "random"]
    loads = [1.0] if quick else [0.5, 1.0, 2.0]
    budgets = [0.8] if quick else [0.8, 0.5]
    for topo in topologies:
        for load in loads:
            for budget in budgets:
                out.append({"name": f"{topo}/load x{load:g}/budget {budget:g}",
                            "family": "synthetic", "topology": topo, "num_nodes": 9,
                            "load_scale": load, "budget": budget})
    # Degraded centre node of the 3x3 grid: half its buffer and link capacity.
    out.append({"name": "grid/bottleneck node 4 x0.5/budget 0.8", "family": "bottleneck",
                "topology": "grid", "num_nodes": 9, "load_scale": 1.0, "budget": 0.8,
                "bottlenecks": {4: 0.5}})
    # Real topology and traffic: SNDlib's Abilene network and one week of its
    # measured 5-minute traffic matrices (data/ABILENE_TRACE.md), each episode
    # starting at a random point of the week. Load 120 / budget 0.8 is the
    # training scenario (docs/ABILENE_SCENARIO.md); load 80 is less congested.
    trace = "abilene_trace_20040301_7d.csv.gz"
    for load in ([120.0] if quick else [80.0, 120.0]):
        for budget in ([0.8] if quick else [0.8, 0.5]):
            out.append({"name": f"abilene trace/load {load:g}/budget {budget:g}",
                        "family": "trace", "topology": "abilene", "num_nodes": 12,
                        "load_scale": 1.0, "budget": budget, "trace": trace,
                        "target_load": load, "random_start": True})
    return out


def run_episode(sc: dict, policy, seed: int, steps: int, warmup: int) -> dict:
    traffic = None
    if "trace" in sc:
        traffic = TraceTrafficGenerator(list(range(sc["num_nodes"])),
                                        os.path.join(DATA, sc["trace"]),
                                        target_load=sc["target_load"], seed=seed,
                                        random_start=sc.get("random_start", False))
    env = CarlaGymEnv(topology=sc["topology"], num_nodes=sc["num_nodes"],
                      load_scale=sc["load_scale"], bandwidth_budget=sc["budget"],
                      bottlenecks=sc.get("bottlenecks"), max_steps=steps,
                      traffic=traffic, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = policy_rng(seed)
    rows = []
    qos = EpisodeQoS()
    for t in range(steps):
        obs, reward, _, _, info = env.step(policy(env, obs, rng))
        rows.append((reward, info["reward_throughput"], info["reward_loss"],
                     info["mean_latency_steps"], info["reward_deadline_miss"]))
        if t >= warmup:
            qos.add(info)
    return {**dict(zip(METRICS, np.mean(rows[warmup:], axis=0))), **qos.summary()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, default=10, help="episodes (seeds) per cell")
    p.add_argument("--seed0", type=int, default=2000, help="first seed")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=os.path.join("results", "benchmark"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for sc in scenarios(a.quick):
        for name, fn in POLICIES.items():
            for seed in range(a.seed0, a.seed0 + a.seeds):
                rows.append({"scenario": sc["name"], "family": sc["family"],
                             "policy": name, "seed": seed,
                             **run_episode(sc, fn, seed, a.steps, a.warmup)})
        print(f"done: {sc['name']}", flush=True)
    episodes = pd.DataFrame(rows)
    episodes.to_csv(os.path.join(a.out, "episodes.csv"), index=False)

    grouped = episodes.groupby(["scenario", "family", "policy"], sort=False)
    summary = grouped[METRICS + QOS].mean()
    summary["reward_ci95"] = grouped["reward"].apply(ci95)
    summary["n_seeds"] = grouped["reward"].size()
    summary = summary.reset_index()
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    policies = list(POLICIES)
    lines = ["| Scenario | " + " | ".join(policies) + " |",
             "|---|" + "---:|" * len(policies)]
    for sc_name, block in summary.groupby("scenario", sort=False):
        block = block.set_index("policy")
        best = block["reward"].idxmax()
        best_low = block.loc[best, "reward"] - block.loc[best, "reward_ci95"]
        # every policy whose interval overlaps the best one's is a statistical tie
        tied = [pol for pol in policies
                if block.loc[pol, "reward"] + block.loc[pol, "reward_ci95"] >= best_low]
        cells = []
        for pol in policies:
            cell = f"{block.loc[pol, 'reward']:+.3f} ± {block.loc[pol, 'reward_ci95']:.3f}"
            if pol == best and len(tied) == 1:
                cell = f"**{cell}**"
            elif pol in tied:
                cell = f"{cell} ~"
            cells.append(cell)
        lines.append(f"| {sc_name} | " + " | ".join(cells) + " |")
    header = (f"Mean post-warmup reward ± 95% CI over {a.seeds} seeds "
              f"({a.seed0}..{a.seed0 + a.seeds - 1}), {a.steps} steps, warmup {a.warmup}.\n"
              f"**Bold**: best policy, its interval overlaps no other. "
              f"`~`: best group, intervals overlap (a statistical tie).\n\n")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")

    qlines = ["| Scenario | " + " | ".join(policies) + " |", "|---|" + "---:|" * len(policies)]
    for sc_name, block in summary.groupby("scenario", sort=False):
        block = block.set_index("policy")
        qlines.append(f"| {sc_name} | " + " | ".join(
            f"{100 * block.loc[p, 'delivered_fraction']:.1f} / {100 * block.loc[p, 'dropped_fraction']:.1f}"
            for p in policies) + " |")
    with open(os.path.join(a.out, "qos.md"), "w", encoding="utf-8") as f:
        f.write("Traffic delivered / dropped, as % of traffic offered after warm-up "
                "(episode aggregate; the rest is backlog growth).\n\n" + "\n".join(qlines) + "\n")
    print(f"\nWrote {len(episodes)} episodes to {a.out}")


if __name__ == "__main__":
    main()
