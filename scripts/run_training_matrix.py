"""
scripts/run_training_matrix.py

Queue the RL training runs behind the results under results/rl/: PPO and SAC
on four scenarios, five training seeds each, plus three PPO ablations. Runs N
jobs at a time, skips finished ones, and can be re-started safely.

Scenarios (0.4.0 simulator; chosen on validation seeds, see
docs/HEADLINE_SCENARIO.md):
    grid_l125_b06        headline: 3x3 grid, load x1.25, budget 0.6
    grid_bottleneck_b08  3x3 grid, centre node at half capacity, budget 0.8
    abilene_l120_b08     SNDlib Abilene topology and one-week traffic trace,
                         mean load 120, budget 0.8 (replaces pcap40_grid16_b05,
                         whose trace had no established provenance)
    random9_l15_b06      fixed random graph (topology_seed 0), load x1.5,
                         budget 0.6; bandwidth-dominated, not a replication
                         of the headline
Ablations (PPO only): bandwidth-only and routing-only on the headline, and
routing-only on random9_l15_b06.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/run_training_matrix.py --workers 10
    python scripts/run_training_matrix.py --list          # show the queue only

Step budgets: PPO 1,000,000 and SAC 200,000 per run. They are not equal on
purpose: SAC is about ten times slower per step and levels off much earlier.
Each algorithm is run until its held-out curve is flat; compare final values
and sample efficiency, not rewards at equal step counts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HEADLINE = ["--load-scale", "1.25", "--budget", "0.6"]
RANDOM9 = ["--topology", "random", "--topology-seed", "0", "--load-scale", "1.5", "--budget", "0.6"]
# Real topology and traffic: SNDlib Abilene, one week of 5-minute matrices (docs/ABILENE_SCENARIO.md).
ABILENE = ["--topology", "abilene", "--nodes", "12", "--trace", "data/abilene_trace_20040301_7d.csv.gz",
           "--load", "120", "--budget", "0.8", "--trace-random-start"]
SCENARIOS = {
    # name: extra train_agent.py arguments
    "grid_l125_b06": HEADLINE,
    "grid_bottleneck_b08": ["--bottleneck", "4:0.5"],
    "abilene_l120_b08": ABILENE,
    "random9_l15_b06": RANDOM9,
}
ABLATIONS = {
    "grid_l125_b06_bandwidth_only": HEADLINE + ["--control", "bandwidth"],
    "grid_l125_b06_routing_only": HEADLINE + ["--control", "routing"],
    "random9_l15_b06_routing_only": RANDOM9 + ["--control", "routing"],
}
BOTTLENECK = SCENARIOS["grid_bottleneck_b08"]
# docs/NEXT_EXPERIMENTS.md, E3: the remaining one-half ablations (PPO only).
ABLATIONS_E3 = {
    "grid_bottleneck_b08_bandwidth_only": BOTTLENECK + ["--control", "bandwidth"],
    "grid_bottleneck_b08_routing_only": BOTTLENECK + ["--control", "routing"],
    "random9_l15_b06_bandwidth_only": RANDOM9 + ["--control", "bandwidth"],
}
# docs/NEXT_EXPERIMENTS.md, E2: PPO hyperparameter grid on the headline.
HP_SWEEP = {
    f"g{g}_lr{lr}_ent{ent}": HEADLINE + ["--gamma", g, "--lr", lr, "--ent-coef", ent]
    for g in ("0.95", "0.99") for lr in ("3e-4", "1e-4") for ent in ("0", "0.01")
}
QUEUES = {
    # name: (configurations, results folder, algorithms)
    "matrix": ({**SCENARIOS, **ABLATIONS}, "rl", None),
    "ablations-e3": (ABLATIONS_E3, "rl", ("ppo",)),
    "hp-sweep": (HP_SWEEP, "rl_hp", ("ppo",)),
}


def known_configurations() -> dict:
    """Every named configuration in any queue, for tools that rebuild a run's env."""
    return {**SCENARIOS, **ABLATIONS, **ABLATIONS_E3, **HP_SWEEP}
BUDGET = {"ppo": ("1000000", "50000"), "sac": ("200000", "20000")}   # (timesteps, eval-every)
SEEDS = range(5)


def jobs(seeds=SEEDS, budget=BUDGET, algos_all=("ppo", "sac"), queue="matrix", only=None):
    configs, folder, algos_fixed = QUEUES[queue]
    if only:
        unknown = set(only) - set(configs)
        if unknown:
            raise SystemExit(f"--only: not in queue {queue!r}: {sorted(unknown)}")
        configs = {k: v for k, v in configs.items() if k in only}
    out = []
    for seed in seeds:                       # seed-major order: every scenario gets early results
        for name, extra in configs.items():
            if algos_fixed is not None:
                algos = list(algos_fixed)
            else:
                algos = ["ppo"] if name in ABLATIONS else list(algos_all)
            for algo in algos:
                steps, every = budget[algo]
                d = os.path.join("results", folder, name, f"{algo}_seed{seed}")
                out.append((d, ["--algo", algo, "--seed", str(seed), "--timesteps", steps,
                                "--eval-every", every, "--out", d] + extra))
    return out


def run(job):
    d, args = job
    if os.path.exists(os.path.join(ROOT, d, "final_eval.csv")):
        return d, "skipped (already finished)"
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    env = {**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(os.path.join(ROOT, d, "train.log"), "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, os.path.join("examples", "train_agent.py")] + args,
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return d, f"{'done' if code == 0 else f'FAILED (exit {code})'} in {(time.time() - t0) / 60:.1f} min"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--list", action="store_true")
    p.add_argument("--seeds", type=int, default=5, help="training seeds per configuration")
    p.add_argument("--ppo-steps", type=int, default=int(BUDGET["ppo"][0]))
    p.add_argument("--sac-steps", type=int, default=int(BUDGET["sac"][0]))
    p.add_argument("--no-sac", action="store_true", help="train PPO only")
    p.add_argument("--queue", choices=sorted(QUEUES), default="matrix",
                   help="matrix: the 0.4.0 matrix; ablations-e3 and hp-sweep: docs/NEXT_EXPERIMENTS.md")
    p.add_argument("--only", nargs="+", default=None, help="run only these configurations of the queue")
    a = p.parse_args()
    budget = {"ppo": (str(a.ppo_steps), str(max(a.ppo_steps // 20, 1000))),
              "sac": (str(a.sac_steps), str(max(a.sac_steps // 10, 1000)))}
    queue = jobs(range(a.seeds), budget, ("ppo",) if a.no_sac else ("ppo", "sac"), a.queue, a.only)
    if a.list:
        for d, args in queue:
            print(d, " ".join(args))
        print(f"{len(queue)} runs")
        return
    print(f"{len(queue)} runs, {a.workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, (d, status) in enumerate(pool.map(run, queue), 1):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(queue)} {d}: {status}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
