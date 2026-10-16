"""Options added for docs/NEXT_EXPERIMENTS.md: tunable heuristic constants (E1),
PPO hyperparameter flags (E2) and the new training queues (E2, E3). The
defaults must reproduce every published result exactly."""
import os

import numpy as np
import pandas as pd
import pytest

from carla_gym import baselines as B
from carla_gym.env import CarlaGymEnv


# ---------------------------------------------------------------- E1 heuristics
def _old_bandwidth(env, obs):                      # 0.4.0 code, copied verbatim
    e = env.num_edges
    pending = obs[e:2 * e]
    if pending.max() <= 1e-9:
        return np.full(e, 0.5)
    return np.clip(pending / pending.max(), 0.05, 1.0)


def _old_routing(env, obs):                        # 0.4.0 code, copied verbatim
    e = env.num_edges
    pending, occupancy = obs[e:2 * e], obs[2 * e:]
    index = {n: i for i, n in enumerate(env.network.graph.nodes())}
    endpoint = np.array([(occupancy[index[u]] + occupancy[index[v]]) / 2.0
                         for u, v in env.network.edge_list()])
    return 0.1 + 0.9 * np.maximum(pending, endpoint)


@pytest.mark.parametrize("topology", ["grid", "ring", "random", "star", "mesh"])
def test_default_heuristics_are_bit_identical_to_0_4_0(topology):
    env = CarlaGymEnv(topology=topology, num_nodes=9, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = rng.uniform(0, 1, env.observation_space.shape).astype(np.float32)
        old = {"queue_proportional": np.concatenate([_old_bandwidth(env, obs), np.ones(env.num_edges)]),
               "load_aware_routing": np.concatenate([np.ones(env.num_edges), _old_routing(env, obs)]),
               "congestion_aware": np.concatenate([_old_bandwidth(env, obs), _old_routing(env, obs)])}
        for name, want in old.items():
            assert np.array_equal(B.POLICIES[name](env, obs), want)
            assert np.array_equal(B.make_policy(name)(env, obs), want)


def test_committed_benchmark_episodes_reproduce_with_defaults():
    from scripts.run_benchmark import run_episode, scenarios
    # round_trip: the default fast parser can be one unit in the last place off
    committed = pd.read_csv(os.path.join("results", "benchmark", "episodes.csv"), float_precision="round_trip")
    sc = [s for s in scenarios(False) if s["name"] == "grid/load x1/budget 0.5"][0]
    for name in ("queue_proportional", "load_aware_routing", "congestion_aware"):
        for seed in (2000, 2001):
            r = run_episode(sc, B.POLICIES[name], seed, 200, 20)
            row = committed[(committed.scenario == sc["name"]) & (committed.policy == name) & (committed.seed == seed)]
            # The published numbers were produced on Python 3.14 / Windows. Other
            # platforms and NumPy versions may differ in the last bits of a sum
            # (seen in CI: 2e-17 on Python 3.10/3.11, Linux), so allow rounding
            # noise. Bit-identical code paths are checked in the test above.
            assert r["reward"] == pytest.approx(row.reward.iloc[0], rel=1e-12, abs=1e-15)


def test_heuristic_parameters_take_effect():
    env = CarlaGymEnv(seed=0)
    obs = np.random.default_rng(2).uniform(0, 1, env.observation_space.shape).astype(np.float32)
    e = env.num_edges
    obs[e] = 1e-3                                                     # one link with almost no demand
    assert B.make_policy("queue_proportional", floor=0.2)(env, obs)[:e].min() == pytest.approx(0.2)
    assert B.make_policy("queue_proportional", floor=0.0)(env, obs)[:e].min() < 0.01
    r = B.make_policy("load_aware_routing", base=0.0, slope=1.0, signal="pending")(env, obs)[e:]
    assert np.allclose(r, obs[e:2 * e])                               # cost = pending demand only
    occ_changed = obs.copy(); occ_changed[2 * e:] = 0.0
    assert np.array_equal(B.make_policy("load_aware_routing", signal="pending")(env, occ_changed)[e:],
                          B.make_policy("load_aware_routing", signal="pending")(env, obs)[e:])


def test_make_policy_rejects_bad_input():
    with pytest.raises(ValueError):
        B.make_policy("no_such_policy")
    env = CarlaGymEnv(seed=0)
    with pytest.raises(ValueError):
        B.make_policy("congestion_aware", signal="median")(env, np.zeros(env.observation_space.shape))


# ---------------------------------------------------------------- E2 hyperparameters
def test_train_agent_hyperparameter_defaults_are_unchanged():
    pytest.importorskip("stable_baselines3")
    from examples.train_agent import parse_args
    a = parse_args([])
    assert (a.gamma, a.lr, a.ent_coef) == (0.95, 3e-4, 0.0)
    b = parse_args(["--gamma", "0.99", "--lr", "1e-4", "--ent-coef", "0.01"])
    assert (b.gamma, b.lr, b.ent_coef) == (0.99, 1e-4, 0.01)


def test_hyperparameters_reach_the_model():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from examples.train_agent import build_model, make_env, parse_args
    for argv, want in [([], (0.95, 3e-4, 0.0)), (["--gamma", "0.99", "--lr", "1e-4", "--ent-coef", "0.01"],
                                                 (0.99, 1e-4, 0.01))]:
        a = parse_args(argv)
        m = build_model("ppo", DummyVecEnv([make_env(a, 0)]), a)
        assert (m.gamma, m.learning_rate, m.ent_coef) == want


# ---------------------------------------------------------------- queues
def test_queues_have_the_planned_size():
    from scripts.run_training_matrix import jobs
    assert len(jobs()) == 55                                          # the published 0.4.0 matrix, unchanged
    assert len(jobs(range(3), queue="hp-sweep")) == 24                # E2: 8 configurations x 3 seeds
    assert len(jobs(queue="ablations-e3")) == 15                      # E3: 3 ablations x 5 seeds
    one = jobs(range(5), queue="hp-sweep", only=["g0.99_lr1e-4_ent0.01"])
    assert len(one) == 5 and all(d.startswith(os.path.join("results", "rl_hp")) for d, _ in one)
    args = one[0][1]
    assert args[args.index("--gamma") + 1] == "0.99" and "--load-scale" in args
    with pytest.raises(SystemExit):
        jobs(queue="hp-sweep", only=["no_such_config"])


def test_matrix_queue_is_identical_to_the_published_one():
    """The default queue must still describe exactly the 55 runs in results/rl."""
    from scripts.run_training_matrix import jobs
    budget = {"ppo": ("500000", "25000"), "sac": ("100000", "10000")}
    for d, args in jobs(budget=budget):
        assert os.path.exists(os.path.join(d, "final_eval.csv")), d
