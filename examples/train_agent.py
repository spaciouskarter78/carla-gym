"""
examples/train_agent.py  --  train an RL agent on CarlaGym and benchmark it
against the rule-based baselines on held-out seeds.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python examples/train_agent.py --algo ppo --timesteps 300000 --seed 0
    python examples/train_agent.py --algo sac --timesteps 100000 --seed 0
    python examples/train_agent.py --algo ppo --topology abilene --nodes 12 --trace data/abilene_trace_20040301_7d.csv.gz --load 120 --budget 0.8 --trace-random-start

Writes to --out (default results/<algo>_seed<seed>/):
    train_monitor.csv   per-episode training return (stable-baselines3 Monitor)
    eval_curve.csv      held-out reward every --eval-every steps (validation seeds)
    final_eval.csv      agent and every baseline on the test seeds
    model.zip           trained policy

Seed protocol: training envs use seeds seed..seed+n_envs-1 and then draw fresh
traffic every episode; the learning curve uses validation seeds 1000+; the
final table uses test seeds 2000+, which neither training nor the curve saw.

Train on the CONSTRAINED setting (--budget < 1). Unconstrained, "allocate 1.0
everywhere" is near-optimal and there is little to learn.
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from carla_gym.baselines import POLICIES
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv
from carla_gym.wrappers import ControlSubsetWrapper

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
VALIDATION_SEED0 = 1000
TEST_SEED0 = 2000


def make_env(args, seed, rescale=True, monitor_file=None):
    """`rescale=True` exposes the action space as [-1, 1] (mapped linearly to
    the env's native [0, 1]). An SB3 Gaussian policy starts with mean 0; on
    the native space that is the zero-bandwidth corner and half of all
    exploration noise is clipped away. Heuristics use the native space."""
    def _f():
        traffic = None
        if args.trace:
            traffic = TraceTrafficGenerator(list(range(args.nodes)), args.trace,
                                            target_load=args.load, seed=seed,
                                            random_start=getattr(args, "trace_random_start", False))
        env = CarlaGymEnv(topology=args.topology, num_nodes=args.nodes,
                          max_steps=args.steps, bandwidth_budget=args.budget,
                          load_scale=getattr(args, "load_scale", 1.0),
                          bottlenecks=parse_bottlenecks(getattr(args, "bottleneck", None)),
                          topology_seed=getattr(args, "topology_seed", 0),
                          traffic=traffic, seed=seed)
        # Ablations: the agent controls one half of the action. Baselines are
        # evaluated on the native env (rescale=False) and keep full control.
        if rescale and getattr(args, "control", "both") != "both":
            env = ControlSubsetWrapper(env, args.control)
        if rescale:
            env = gym.wrappers.RescaleAction(env, np.float32(-1.0), np.float32(1.0))
        return Monitor(env, filename=monitor_file)
    return _f


def parse_bottlenecks(spec):
    """'4:0.5,7:0.3' -> {4: 0.5, 7: 0.3}; None or '' -> None."""
    if not spec:
        return None
    return {int(n): float(s) for n, s in (item.split(":") for item in spec.split(","))}


def rollout_rewards(env, act_fn, seeds, steps, warmup=20):
    """Mean post-warmup reward of one episode per seed."""
    # `env` is Monitor-wrapped (see make_env); Monitor does not forward
    # custom attributes like num_edges/num_nodes that the reference policies
    # need, so hand those policies the unwrapped env while still stepping
    # the wrapped one (to keep Monitor's episode logging intact).
    base_env = env.unwrapped
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        r = []
        for _ in range(steps):
            obs, rew, term, trunc, _ = env.step(act_fn(base_env, obs))
            r.append(rew)
            if term or trunc:
                break
        out.append(float(np.mean(r[warmup:])))
    return out


def rollout(env, act_fn, seeds, steps, warmup=20):
    r = rollout_rewards(env, act_fn, seeds, steps, warmup)
    return float(np.mean(r)), float(np.std(r))


class HeldOutEval(BaseCallback):
    """Evaluates the deterministic policy on validation seeds during training."""

    def __init__(self, args, every, seeds):
        super().__init__()
        self.args, self.every, self.seeds = args, every, list(seeds)
        self.env = make_env(args, 0)()
        self.rows = []
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            fn = lambda e, o: self.model.predict(o, deterministic=True)[0]
            m, sd = rollout(self.env, fn, self.seeds, self.args.steps)
            self.rows.append({"timesteps": self.num_timesteps,
                              "reward_mean": m, "reward_std": sd})
            print(f"[eval] timesteps={self.num_timesteps:>8d}  "
                  f"held-out reward={m:+.3f} +/- {sd:.3f}", flush=True)
        return True


def build_model(algo, venv, args):
    if algo == "ppo":
        return PPO("MlpPolicy", venv, n_steps=512, batch_size=256, gamma=args.gamma,
                   learning_rate=args.lr, ent_coef=args.ent_coef, seed=args.seed, verbose=0,
                   policy_kwargs=dict(log_std_init=args.log_std_init))
    # Off-policy: one gradient step per env step, squashed (tanh) actions.
    common = dict(gamma=args.gamma, learning_rate=args.lr, batch_size=256,
                  buffer_size=200_000, learning_starts=2_000,
                  seed=args.seed, verbose=0)
    if algo == "sac":
        return SAC("MlpPolicy", venv, **common)
    n_actions = venv.action_space.shape[0]
    noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))
    return TD3("MlpPolicy", venv, action_noise=noise, **common)


def parse_args(argv=None, default_algo="ppo"):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--algo", choices=sorted(ALGOS), default=default_algo)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--steps", type=int, default=200, help="episode length")
    p.add_argument("--topology", default="grid")
    p.add_argument("--topology-seed", type=int, default=0,
                   help="graph seed for --topology random; fixed so every run sees one graph")
    p.add_argument("--nodes", type=int, default=9)
    p.add_argument("--budget", type=float, default=0.8)
    p.add_argument("--load-scale", type=float, default=1.0,
                   help="multiplier on synthetic Poisson arrival rates")
    p.add_argument("--bottleneck", default=None,
                   help="degraded nodes as node:scale pairs, e.g. 4:0.5 or 4:0.5,7:0.3")
    p.add_argument("--control", choices=["both", "bandwidth", "routing"], default="both",
                   help="ablation: let the agent control only one half of the action")
    p.add_argument("--trace", default=None, help="CSV: step,src,dst,volume")
    p.add_argument("--load", type=float, default=104.0, help="trace target load")
    p.add_argument("--trace-random-start", action="store_true",
                   help="start each episode at a random time bin of the trace (long traces)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="parallel envs (default 4 for PPO, 1 for SAC/TD3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--test-episodes", type=int, default=10)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy coefficient")
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="PPO: initial log std of the Gaussian policy (SB3 default 0.0)")
    a = p.parse_args(argv)
    if a.n_envs is None:
        a.n_envs = 4 if a.algo == "ppo" else 1
    if a.out is None:
        a.out = os.path.join("results", f"{a.algo}_seed{a.seed}")
    return a


def main(argv=None, default_algo="ppo"):
    a = parse_args(argv, default_algo)
    os.makedirs(a.out, exist_ok=True)

    # Distinct seed per env rank, so parallel envs do not share one traffic stream.
    venv = DummyVecEnv([
        make_env(a, a.seed + rank,
                 monitor_file=os.path.join(a.out, f"train_{rank}"))
        for rank in range(a.n_envs)])
    venv.seed(a.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)   # reward scale ~[-2,1]

    model = build_model(a.algo, venv, a)
    evaluator = HeldOutEval(a, a.eval_every,
                            range(VALIDATION_SEED0, VALIDATION_SEED0 + a.eval_episodes))
    model.learn(total_timesteps=a.timesteps, callback=evaluator)
    model.save(os.path.join(a.out, "model"))
    venv.close()
    pd.DataFrame(evaluator.rows).to_csv(os.path.join(a.out, "eval_curve.csv"), index=False)

    # One training-return file: merge the per-env Monitor logs by wall-clock time.
    frames = []
    for rank in range(a.n_envs):
        path = os.path.join(a.out, f"train_{rank}.monitor.csv")
        frames.append(pd.read_csv(path, skiprows=1).assign(env=rank))
        os.remove(path)
    train = pd.concat(frames).sort_values("t").reset_index(drop=True)
    train["timesteps"] = train["l"].cumsum()
    train.rename(columns={"r": "episode_return", "l": "episode_length"})[
        ["timesteps", "episode_return", "episode_length", "env"]
    ].to_csv(os.path.join(a.out, "train_monitor.csv"), index=False)

    # ---- final evaluation: test seeds neither training nor the curve used ----
    test_seeds = list(range(TEST_SEED0, TEST_SEED0 + a.test_episodes))
    agent_env = make_env(a, 0)()                  # [-1, 1] actions, as trained
    native_env = make_env(a, 0, rescale=False)()  # [0, 1] actions, for heuristics
    random_rng = np.random.default_rng(a.seed)
    entries = [(a.algo.upper(), lambda e, o: model.predict(o, deterministic=True)[0], agent_env)]
    for name, fn in POLICIES.items():
        entries.append((name, (lambda e, o, fn=fn: fn(e, o, random_rng)), native_env))
    rows = []
    for name, fn, env in entries:
        for s, r in zip(test_seeds, rollout_rewards(env, fn, test_seeds, a.steps)):
            rows.append({"policy": name, "test_seed": s, "reward": r})
    final = pd.DataFrame(rows)
    final.to_csv(os.path.join(a.out, "final_eval.csv"), index=False)

    summary = (final.groupby("policy")["reward"].agg(reward_mean="mean", reward_std="std")
               .sort_values("reward_mean", ascending=False).round(3))
    print("\n=== Held-out evaluation (test seeds) ===")
    print(summary.to_string())
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()
