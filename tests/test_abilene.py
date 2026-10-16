"""The Abilene topology, the SNDlib-derived trace, and random episode starts."""
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from carla_gym.core.network import ABILENE_LINKS, ABILENE_NODES, NetworkModel
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv

TRACE = os.path.join("data", "abilene_trace_20040301_7d.csv.gz")


def test_abilene_topology_matches_sndlib():
    net = NetworkModel("abilene", 12, link_capacity=30.0)
    g = net.graph
    assert g.number_of_nodes() == 12 and g.number_of_edges() == 15 and nx.is_connected(g)
    idx = {n: i for i, n in enumerate(ABILENE_NODES)}
    want = {net._key(idx[a], idx[b]): c / 9920.0 * 30.0 for a, b, c in ABILENE_LINKS}
    got = {net._key(u, v): float(c) for (u, v), c in zip(net.edge_list(), net.get_edge_capacities())}
    assert got == pytest.approx(want)
    assert sorted(set(got.values())) == pytest.approx([7.5, 30.0])        # Atlanta-Indianapolis at a quarter


def test_abilene_needs_twelve_nodes():
    with pytest.raises(ValueError, match="12"):
        NetworkModel("abilene", 9)


def test_abilene_with_a_bottleneck_combines_both_scales():
    net = NetworkModel("abilene", 12, link_capacity=30.0, bottlenecks={5: 0.5})   # IPLSng
    caps = dict(zip(net.edge_list(), net.get_edge_capacities()))
    assert min(caps.values()) == pytest.approx(30.0 * 0.25 * 0.5)


def test_committed_trace_is_one_week_on_twelve_nodes():
    df = pd.read_csv(TRACE)
    assert list(df.columns) == ["step", "src", "dst", "volume"]
    assert df.step.min() == 0 and df.step.max() == 2015
    assert set(df.src) | set(df.dst) <= set(range(12)) and (df.src != df.dst).all() and (df.volume > 0).all()
    assert df.groupby(["src", "dst"]).ngroups == 132


def test_random_start_is_reproducible_and_covers_the_trace():
    def offsets(seed, n):
        g = TraceTrafficGenerator(list(range(12)), TRACE, target_load=40, seed=0, random_start=True)
        out = []
        g.reset(seed=seed)
        for _ in range(n):
            out.append(g.offset)
            g.reset()                                    # unseeded: continue the stream
        return out
    a, b = offsets(7, 20), offsets(7, 20)
    assert a == b                                        # same seed, same starts
    assert len(set(a)) > 15 and max(a) - min(a) > 1000   # spread over the week
    assert offsets(8, 1) != offsets(7, 1)


def test_default_trace_generator_still_starts_at_zero():
    g = TraceTrafficGenerator(list(range(12)), TRACE, target_load=40, seed=0)
    g.reset(seed=3)
    assert g.offset == 0


def test_env_runs_on_abilene_with_the_trace():
    traffic = TraceTrafficGenerator(list(range(12)), TRACE, target_load=40, seed=0, random_start=True)
    env = CarlaGymEnv(topology="abilene", num_nodes=12, traffic=traffic, bandwidth_budget=0.6, max_steps=50, seed=0)
    obs, _ = env.reset(seed=2000)
    offered = 0.0
    for _ in range(50):
        obs, r, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
        offered += info["offered_units"]
        assert np.isfinite(r)
    assert offered / 50 == pytest.approx(40, rel=0.5)    # mean load near the target (bursty, short window)


def test_random_start_stamps_packets_with_the_simulator_step():
    """Regression guard: the shifted trace step was stamped on packets as their
    birth step, so delays came out negative (about minus the offset) and the
    reward's delay and deadline terms vanished on random-start traces."""
    g = TraceTrafficGenerator(list(range(12)), TRACE, target_load=120, seed=0, random_start=True)
    g.reset(seed=2000)
    assert g.offset > 0
    pkts = g.generate(7)
    assert pkts and all(p.birth_step == 7 for p in pkts)


def test_delays_are_positive_on_a_random_start_trace():
    traffic = TraceTrafficGenerator(list(range(12)), TRACE, target_load=120, seed=0, random_start=True)
    env = CarlaGymEnv(topology="abilene", num_nodes=12, traffic=traffic, bandwidth_budget=0.7, max_steps=60, seed=0)
    env.reset(seed=2000)
    lat = []
    for _ in range(60):
        _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
        if info["delivered_count"]:
            lat.append(info["mean_latency_steps"])
        assert info["mean_queue_age_steps"] >= 0
    assert lat and min(lat) >= 1.0                       # at least one step per hop
