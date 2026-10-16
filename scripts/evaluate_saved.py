"""
scripts/evaluate_saved.py

Re-evaluate the saved agents of a results directory, and every baseline, on
the test seeds, and report episode-aggregate QoS next to the reward: the
fraction of offered traffic delivered and dropped, the mean delay of delivered
packets, and deadline misses per delivered packet.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/evaluate_saved.py results/rl/grid_l125_b06
    python scripts/evaluate_saved.py results/rl/grid_bottleneck_b08
    python scripts/evaluate_saved.py some/dir --scenario abilene_l120_b08
    python scripts/evaluate_saved.py some/dir --load-scale 1.25 --budget 0.6

Expects <dir>/<algo>_seed<N>/model.zip from examples/train_agent.py. The env
is rebuilt with train_agent.make_env from the same flags the run was trained
with, so bottlenecks, traces, the random topology and the --control
ablations are reproduced exactly. The flags come from, in order:
  * --scenario NAME, looked up in scripts/run_training_matrix.py;
  * otherwise the directory name, if it is a scenario or ablation there;
  * plus any further train_agent.py flags given on this command line.
With none of these, train_agent.py's defaults apply (3x3 grid, budget 0.8).
Trains nothing. Writes qos.csv and qos.md.

Self-check: the reward of every agent and deterministic baseline must equal
the one recorded in that run's final_eval.csv (same env, same test seeds).
A mismatch means the env was not rebuilt as trained, and the script exits
with an error unless --no-check is given. The `random` baseline is not
checked: training drew its actions from a different generator.

The per-step reward terms `throughput` and `loss` are ratios within one step;
their episode mean is not the fraction of traffic delivered. This table is.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from carla_gym.baselines import POLICIES, policy_rng
from carla_gym.utils.qos import EpisodeQoS
from carla_gym.utils.stats import ci95
from examples.train_agent import make_env, parse_args as train_args
from scripts.run_training_matrix import known_configurations

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
COLS = ["reward", "delivered_fraction", "dropped_fraction", "latency_delivered",
        "deadline_miss_per_delivered"]


def episode(targs, act_fn, seed, rescale, warmup):
    """One test episode, built exactly as examples/train_agent.py builds its
    final-evaluation env: constructed with seed 0, then reset with `seed`."""
    env = make_env(targs, 0, rescale=rescale)()
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    rewards, qos = [], EpisodeQoS()
    for t in range(targs.steps):
        obs, r, _, _, info = env.step(act_fn(base, obs))
        if t >= warmup:
            rewards.append(r)
            qos.add(info)
    return {"reward": float(np.mean(rewards)), **qos.summary()}


def scenario_flags(root, name, extra):
    """(scenario name or None, train_agent.py flags) for this directory."""
    known = known_configurations()
    name = name or os.path.basename(os.path.normpath(root))
    if name not in known:
        return None, list(extra)
    return name, list(known[name]) + list(extra)


def self_check(root, df):
    """Largest difference between re-evaluated rewards and final_eval.csv."""
    worst, n = 0.0, 0
    for d in sorted(glob.glob(os.path.join(root, "*_seed*"))):
        m = re.fullmatch(r"(\w+?)_seed(\d+)", os.path.basename(d))
        path = os.path.join(d, "final_eval.csv")
        if not m or not os.path.exists(path):
            continue
        recorded = pd.read_csv(path)
        for r in recorded[recorded.policy != "random"].itertuples():
            mine = df[(df.policy == r.policy) & (df.test_seed == r.test_seed)]
            if r.policy == m.group(1).upper():
                mine = mine[mine.train_seed == int(m.group(2))]
            if mine.empty:
                continue
            worst = max(worst, abs(float(mine.reward.iloc[0]) - r.reward))
            n += 1
    return n, worst


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root")
    p.add_argument("--scenario", default=None, help="name from scripts/run_training_matrix.py")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--no-check", action="store_true", help="do not compare with final_eval.csv")
    a, extra = p.parse_known_args()
    name, flags = scenario_flags(a.root, a.scenario, extra)
    targs = train_args(flags)
    seeds = range(a.seed0, a.seed0 + a.episodes)
    print(f"scenario: {name or '(train_agent defaults)'}  flags: {' '.join(flags) or '(none)'}", flush=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_seed*"))):
        m = re.fullmatch(r"(\w+?)_seed(\d+)", os.path.basename(d))
        path = os.path.join(d, "model.zip")
        if not m or m.group(1) not in ALGOS or not os.path.exists(path):
            continue
        model = ALGOS[m.group(1)].load(path, device="cpu")
        fn = lambda e, o, model=model: model.predict(o, deterministic=True)[0]
        for s in seeds:
            rows.append({"policy": m.group(1).upper(), "kind": "RL agent",
                         "train_seed": int(m.group(2)), "test_seed": s,
                         **episode(targs, fn, s, True, a.warmup)})
    for pname, fn in POLICIES.items():
        for s in seeds:
            rng = policy_rng(s)
            rows.append({"policy": pname, "kind": "baseline", "train_seed": -1, "test_seed": s,
                         **episode(targs, lambda e, o, fn=fn, rng=rng: fn(e, o, rng), s, False, a.warmup)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.root, "qos.csv"), index=False)

    n, worst = self_check(a.root, df)
    print(f"self-check against final_eval.csv: {n} rewards compared, max abs difference {worst:.2e}", flush=True)
    if not a.no_check and (n == 0 or worst > 1e-9):
        raise SystemExit("self-check failed: the env was not rebuilt as trained "
                         "(wrong --scenario or flags?); qos.csv written, qos.md not")

    out = []
    for (policy, kind), g in df.groupby(["policy", "kind"]):
        # agents: one value per training seed; baselines: one per test episode
        unit = g.groupby("train_seed")[COLS].mean() if kind == "RL agent" else g.set_index("test_seed")[COLS]
        row = {"policy": policy, "kind": kind, "n": len(unit)}
        for c in COLS:
            row[c], row[c + "_ci95"] = unit[c].mean(), ci95(unit[c])
        out.append(row)
    table = pd.DataFrame(out).sort_values("reward", ascending=False)

    def cell(r, c, scale=1.0, fmt="{:.1f}"):
        return f"{fmt.format(scale * r[c])} ± {fmt.format(scale * r[c + '_ci95'])}"

    lines = ["| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['policy']} | {r['reward']:+.3f} | {cell(r, 'delivered_fraction', 100)} | "
                     f"{cell(r, 'dropped_fraction', 100)} | {cell(r, 'latency_delivered', fmt='{:.2f}')} | "
                     f"{cell(r, 'deadline_miss_per_delivered', fmt='{:.3f}')} |")
    scen = f"`{name}` ({' '.join(flags)})" if name else f"train_agent defaults{' + ' + ' '.join(flags) if flags else ''}"
    note = (f"Scenario {scen}. "
            f"Episode aggregates over steps {a.warmup}..{targs.steps - 1} of test seeds "
            f"{a.seed0}..{a.seed0 + a.episodes - 1}, mean ± 95% CI (agents: over training seeds; "
            f"baselines: over test episodes). Delivered % and Dropped % are of traffic offered "
            f"in the same window; the remainder is backlog growth. The `random` row draws its "
            f"own actions here, so its reward differs slightly from the reward table. "
            f"Agent and deterministic-baseline rewards match final_eval.csv "
            f"({n} compared, max difference {worst:.1e}).\n\n")
    with open(os.path.join(a.root, "qos.md"), "w", encoding="utf-8") as f:
        f.write(note + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
