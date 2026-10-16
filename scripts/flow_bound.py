"""
scripts/flow_bound.py

Fluid flow upper bound (carla_gym/bounds.py) on the delivered fraction, next
to what the policies achieved. Needs SciPy (`pip install carla-gym[bounds]`).

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/flow_bound.py

Writes results/bounds/:
    training.md / .csv   the four training scenarios: bound with free and with
                         shortest-path routing, best heuristic, PPO, SAC
                         (delivered fraction on test seeds, from qos.csv)
    benchmark.md / .csv  every synthetic and bottleneck scenario of the
                         baseline matrix, against its best heuristic

Trace scenarios are listed with their bound, but it uses the trace's mean
traffic matrix and ignores bursts, so it is loose for them.
"""
from __future__ import annotations

import os

import pandas as pd

from carla_gym.bounds import scenario_bound
from carla_gym.env import CarlaGymEnv
from examples.train_agent import make_env, parse_args
from scripts.run_benchmark import scenarios as benchmark_scenarios
from scripts.run_training_matrix import SCENARIOS

OUT = os.path.join("results", "bounds")
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]


def pct(x):
    return f"{100 * x:.1f}"


def training():
    rows = []
    for name, flags in SCENARIOS.items():
        env = make_env(parse_args(flags), 0, rescale=False)()
        free, short = scenario_bound(env, "free"), scenario_bound(env, "shortest")
        q = pd.read_csv(os.path.join("results", "rl", name, "qos.csv"))
        mean = q.groupby("policy").delivered_fraction.mean()
        best = mean[HEUR].idxmax()
        rows.append({"scenario": name, "flags": " ".join(flags),
                     "bound_free": free["delivered_fraction"], "bound_shortest": short["delivered_fraction"],
                     "best_heuristic": best, "best_heuristic_delivered": mean[best],
                     "ppo_delivered": mean.get("PPO"), "sac_delivered": mean.get("SAC")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "training.csv"), index=False)
    lines = ["| Scenario | Bound, free routing | Bound, shortest paths | Best heuristic | PPO | SAC |",
             "|---|---:|---:|---|---:|---:|"]
    for r in df.itertuples():
        lines.append(f"| {r.scenario} | {pct(r.bound_free)} | {pct(r.bound_shortest)} | "
                     f"{r.best_heuristic} {pct(r.best_heuristic_delivered)} | {pct(r.ppo_delivered)} | "
                     f"{pct(r.sac_delivered)} |")
    return lines


def benchmark():
    summ = pd.read_csv(os.path.join("results", "benchmark", "summary.csv"))
    rows = []
    for sc in benchmark_scenarios(False):
        if sc["family"] == "trace":
            continue
        env = CarlaGymEnv(topology=sc["topology"], num_nodes=sc["num_nodes"], load_scale=sc["load_scale"],
                          bandwidth_budget=sc["budget"], bottlenecks=sc.get("bottlenecks"), seed=0)
        free, short = scenario_bound(env, "free"), scenario_bound(env, "shortest")
        s = summ[(summ.scenario == sc["name"]) & summ.policy.isin(HEUR)]
        b = s.loc[s.delivered_fraction.idxmax()]
        rows.append({"scenario": sc["name"], "bound_free": free["delivered_fraction"],
                     "bound_shortest": short["delivered_fraction"], "best_heuristic": b.policy,
                     "best_heuristic_delivered": b.delivered_fraction})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    lines = ["| Scenario | Bound, free routing | Bound, shortest paths | Best heuristic | Best heuristic delivers | Room below the bound (points, upper limit) |",
             "|---|---:|---:|---|---:|---:|"]
    for r in df.itertuples():
        lines.append(f"| {r.scenario} | {pct(r.bound_free)} | {pct(r.bound_shortest)} | {r.best_heuristic} | "
                     f"{pct(r.best_heuristic_delivered)} | {100 * (r.bound_free - r.best_heuristic_delivered):+.1f} |")
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    head = ("Fluid flow upper bound on the delivered fraction (%), from carla_gym/bounds.py: the "
            "expected traffic matrix, link capacities and the bandwidth budget, with the allocation "
            "chosen optimally. No policy can exceed it in the long run; a finite window can exceed it "
            "slightly through backlog carried in. Policy figures: delivered % on test seeds 2000-2009, "
            "steps 20..199. The bound is loose (docs/RL_RESULTS_0.4.0.md, section 2.6): the room "
            "below it is an upper limit on what any policy could gain, not a measure of it.")
    t = training()
    with open(os.path.join(OUT, "training.md"), "w", encoding="utf-8") as f:
        f.write(head + " The trace row's bound uses the trace's mean matrix over the week and ignores its daily cycle, so it is looser still.\n\n" + "\n".join(t) + "\n")
    print("\n".join(t), "\n", flush=True)
    b = benchmark()
    with open(os.path.join(OUT, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write(head + "\n\n" + "\n".join(b) + "\n")
    print("\n".join(b))


if __name__ == "__main__":
    main()
