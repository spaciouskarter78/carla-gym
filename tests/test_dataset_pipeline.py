"""
tests/test_dataset_pipeline.py

Covers the trace-data integration: raw dataset -> dataset_adapters ->
TraceTrafficGenerator -> CarlaGymEnv. Adapter tests run against the real raw
CSVs when a local copy exists in data/raw/ (they are not distributed); the
committed derived traces are tested everywhere.

Two regressions are specifically guarded against because they were found
and fixed during integration:
    * a heavy traffic node landing on a low-degree topology position
      (throttles throughput regardless of bandwidth policy -- nothing to
      do with load calibration)
    * packet units leaking when traffic comes from a trace instead of the
      Poisson generator the original conservation test covered
"""

import os

import numpy as np
import pandas as pd
import pytest

from carla_gym.core.dataset_adapters import (
    convert_cs448b, convert_packet_capture, detect_format, _topology_degree_order,
)
from carla_gym.core.network import NetworkModel
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
# The derived traces are committed. The raw datasets they were made from are
# not distributed with the repository; tests that need them run only where a
# local copy exists in data/raw/ and are skipped otherwise.
RAW = os.path.join(DATA, "raw")
CS448B = os.path.join(RAW, "cs448b_ipasn.csv")
PCAP_NUMERIC = os.path.join(RAW, "output1.csv")
PCAP_TEXT = os.path.join(RAW, "Midterm_53_group.csv")
# The distributed trace: one week of SNDlib's Abilene traffic matrices
# (data/ABILENE_TRACE.md). The pcap and cs448b traces were removed in 0.4.x:
# their origin and licence could not be established.
ABILENE_TRACE = os.path.join(DATA, "abilene_trace_20040301_7d.csv.gz")
SNDLIB_ARCHIVE = os.path.join(RAW, "sndlib", "archives", "directed-abilene-zhang-5min-over-6months-ALL-native.tgz")

needs_raw = pytest.mark.skipif(
    not all(os.path.exists(f) for f in (CS448B, PCAP_NUMERIC, PCAP_TEXT)),
    reason="raw datasets are not distributed; place them in data/raw/ to run this test",
)


# ---------------------------------------------------------------- format detection
@needs_raw
def test_detect_format():
    assert detect_format(CS448B) == "cs448b"
    assert detect_format(PCAP_NUMERIC) == "pcap"
    assert detect_format(PCAP_TEXT) == "pcap"


# ---------------------------------------------------------------- cs448b adapter
@needs_raw
def test_cs448b_schema_and_bounds():
    df = convert_cs448b(CS448B, internal_nodes=10, external_buckets=6)
    assert set(df.columns) == {"step", "src", "dst", "volume"}
    assert df.attrs["num_nodes"] == 16
    assert df["src"].between(0, 15).all() and df["dst"].between(0, 15).all()
    assert (df["src"] != df["dst"]).all()


@needs_raw
def test_cs448b_rejects_l_ipn_overflow():
    with pytest.raises(ValueError):
        convert_cs448b(CS448B, internal_nodes=5, external_buckets=6)  # l_ipn goes to 9


# ---------------------------------------------------------------- pcap adapter
@needs_raw
def test_pcap_numeric_and_text_agree():
    """output1.csv and Midterm_53_group.csv are the same capture exported two
    ways; after conversion their per-step load should be close (not
    identical -- the text export drops a few hundred non-IP rows the
    numeric export already lacks)."""
    a = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0)
    b = convert_packet_capture(PCAP_TEXT, num_nodes=16, bin_seconds=5.0)
    load_a = a.groupby("step")["volume"].sum().mean()
    load_b = b.groupby("step")["volume"].sum().mean()
    assert load_a == pytest.approx(load_b, rel=0.1)


@needs_raw
def test_pcap_schema():
    df = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0)
    assert set(df.columns) == {"step", "src", "dst", "volume"}
    assert (df["src"] != df["dst"]).all()
    assert df["src"].between(0, 15).all() and df["dst"].between(0, 15).all()


def test_pcap_rejects_unknown_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        convert_packet_capture(str(bad))


# ---------------------------------------------------------------- topology-aware placement (regression)
@needs_raw
def test_adapter_output_reproducible_across_processes(tmp_path):
    """Regression guard: the adapters used to bucket IPs/ASNs with
    Python's builtin hash(), which is randomized per-process by default
    (PYTHONHASHSEED). That made every re-run of prepare_trace.py produce
    a different node mapping for the SAME raw CSV -- a serious
    reproducibility break, not a cosmetic one. Runs the adapter in two
    genuinely separate subprocesses (distinct hash seeds) and requires
    identical output."""
    import subprocess
    import sys
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    script = (
        "from carla_gym.core.dataset_adapters import convert_packet_capture; "
        "convert_packet_capture({src!r}, {dst!r}, num_nodes=16, bin_seconds=5.0)"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
    env_b = {**os.environ, "PYTHONHASHSEED": "2", "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
    subprocess.run([sys.executable, "-c", script.format(src=PCAP_NUMERIC, dst=str(out_a))],
                   env=env_a, check=True)
    subprocess.run([sys.executable, "-c", script.format(src=PCAP_NUMERIC, dst=str(out_b))],
                   env=env_b, check=True)
    df_a = pd.read_csv(out_a)
    df_b = pd.read_csv(out_b)
    pd.testing.assert_frame_equal(df_a, df_b)


@needs_raw
def test_heaviest_node_lands_on_best_connected_position():
    """Regression guard: before this was fixed, frequency/hash-based node
    assignment could put the single busiest traffic concentration on an
    arbitrary topology position -- including, by chance, a 2-link grid
    corner, which throttles throughput no matter the bandwidth policy."""
    df = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0, topology="grid")
    net = NetworkModel(topology="grid", num_nodes=16)
    degrees = dict(net.graph.degree())

    load = df.groupby("dst")["volume"].sum().add(
        df.groupby("src")["volume"].sum(), fill_value=0.0
    ).sort_values(ascending=False)
    busiest_node = int(load.index[0])
    assert degrees[busiest_node] == max(degrees.values())


def test_degree_order_is_well_formed():
    order = _topology_degree_order("grid", 16)
    assert sorted(order) == list(range(16))
    net = NetworkModel(topology="grid", num_nodes=16)
    degs = dict(net.graph.degree())
    assert all(degs[order[i]] >= degs[order[i + 1]] for i in range(len(order) - 1))


# ---------------------------------------------------------------- full pipeline
@needs_raw
@pytest.mark.parametrize("raw_path,kind", [
    (CS448B, "cs448b"), (PCAP_NUMERIC, "pcap"), (PCAP_TEXT, "pcap"),
])
def test_full_pipeline_runs_without_crashing(raw_path, kind, tmp_path):
    out_csv = tmp_path / "trace.csv"
    if kind == "cs448b":
        convert_cs448b(raw_path, str(out_csv))
        num_nodes = 16
    else:
        convert_packet_capture(raw_path, str(out_csv), num_nodes=16, bin_seconds=5.0)
        num_nodes = 16

    traffic = TraceTrafficGenerator(nodes=list(range(num_nodes)), csv_path=str(out_csv),
                                    target_load=100.0, seed=0)
    env = CarlaGymEnv(topology="grid", num_nodes=num_nodes, bandwidth_budget=0.5,
                      max_steps=30, traffic=traffic, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)


def test_trace_driven_env_conserves_packet_units(tmp_path):
    """Same invariant as the Poisson-traffic conservation test, but now
    with trace-driven demand -- the simulator's bookkeeping shouldn't
    care where packets came from."""
    traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE,
                                    target_load=100.0, seed=0, random_start=True)
    env = CarlaGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=0.5,
                      max_steps=50, traffic=traffic, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    offered = delivered = dropped = 0.0
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(0, 1, 2 * env.num_edges))
        offered += info["offered_units"]
        delivered += info["delivered_units"]
        dropped += info["dropped_units"]
    backlog = env.network.backlog_units()
    assert offered == pytest.approx(delivered + dropped + backlog, rel=1e-6)


def test_reference_policies_callable_with_and_without_rng():
    """Regression guard: train_ppo.py's rollout() calls policy functions as
    fn(env, obs) (no rng), while random_agent_test.py's own run_policy()
    calls them as fn(env, obs, rng). Both call sites must work."""
    from examples.random_agent_test import policy_max, policy_min, policy_queue_proportional, policy_random

    env = CarlaGymEnv(topology="grid", num_nodes=16, max_steps=5, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for fn in (policy_max, policy_min, policy_queue_proportional, policy_random):
        a1 = fn(env, obs)          # train_ppo.py call style
        a2 = fn(env, obs, rng)     # random_agent_test.py call style
        assert a1.shape == (2 * env.num_edges,) and a2.shape == (2 * env.num_edges,)


def test_rollout_works_with_monitor_wrapped_env():
    """Regression guard: train_ppo.py's held-out eval wraps the env in
    stable_baselines3's Monitor, which does NOT forward custom attributes
    like num_edges/num_nodes -- calling a reference policy directly on the
    wrapped env raises AttributeError. rollout() must hand policies the
    unwrapped env instead."""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from examples.random_agent_test import policy_queue_proportional
    from examples.train_ppo import rollout

    env = Monitor(CarlaGymEnv(topology="grid", num_nodes=16, max_steps=40, seed=0))
    m, sd = rollout(env, policy_queue_proportional, seeds=[0, 1], steps=40)  # > default warmup=20
    assert np.isfinite(m) and np.isfinite(sd)


def test_bandwidth_budget_changes_outcome_on_real_trace():
    """The central claim of the trace-driven setup: under a tight budget,
    congestion-aware allocation should not do worse than uniform -- if it
    did, there would be nothing for an agent to learn from this trace."""
    def run(policy_weight_fn, budget):
        traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE,
                                        target_load=120.0, seed=0, random_start=True)
        env = CarlaGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=budget,
                          max_steps=100, traffic=traffic, seed=0)
        obs, _ = env.reset(seed=0)
        total = 0.0
        for t in range(100):
            obs, reward, *_ = env.step(policy_weight_fn(env, obs))
            if t >= 20:
                total += reward
        return total

    uniform = lambda e, o: np.ones(2 * e.num_edges)
    def queue_prop(e, o):
        pending = o[e.num_edges:2 * e.num_edges]
        bw = np.clip(pending / max(pending.max(), 1e-9), 0.05, 1.0)
        return np.concatenate([bw, np.ones(e.num_edges)])

    r_uniform = run(uniform, budget=0.5)
    r_queue_prop = run(queue_prop, budget=0.5)
    assert r_queue_prop >= r_uniform


# ---------------------------------------------------------------- trace load calibration / committed traces
def test_trace_mean_load_matches_target():
    gen = TraceTrafficGenerator(list(range(12)), ABILENE_TRACE, target_load=110.0, seed=0)
    expected = sum(rows[:, 2].sum() for rows in gen.by_step.values()) / gen.n_steps
    assert expected == pytest.approx(110.0)
    realised = np.mean([sum(p.size for p in gen.generate(t)) for t in range(gen.n_steps)])
    assert realised == pytest.approx(110.0, rel=0.05)


def test_trace_mean_load_counts_empty_bins(tmp_path):
    """Regression guard: the rescaling averaged over non-empty bins only, so a
    trace with gaps under-delivered."""
    csv = tmp_path / "gappy.csv"
    pd.DataFrame({"step": [0, 3, 9], "src": [0, 1, 2], "dst": [1, 2, 0],
                  "volume": [5.0, 1.0, 2.0]}).to_csv(csv, index=False)        # bins 1-2, 4-8 empty
    gen = TraceTrafficGenerator(list(range(9)), str(csv), target_load=50.0, seed=0)
    assert gen.n_steps == 10
    expected = sum(rows[:, 2].sum() for rows in gen.by_step.values()) / gen.n_steps
    assert expected == pytest.approx(50.0)
    realised = np.mean([sum(p.size for p in gen.generate(t)) for t in range(300 * gen.n_steps)])
    assert realised == pytest.approx(50.0, rel=0.05)


def test_trace_with_no_usable_rows_is_rejected(tmp_path):
    csv = tmp_path / "trace.csv"
    pd.DataFrame({"step": [0], "src": [50], "dst": [51], "volume": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no usable rows"):
        TraceTrafficGenerator(list(range(9)), str(csv))


@pytest.mark.skipif(not os.path.exists(SNDLIB_ARCHIVE),
                    reason="the SNDlib archive is not distributed; download it to run this test")
def test_committed_abilene_trace_matches_the_conversion_script(tmp_path):
    """data/abilene_trace_20040301_7d.csv.gz must be exactly what
    scripts/prepare_sndlib_trace.py produces from SNDlib's archive."""
    from scripts.prepare_sndlib_trace import main as prepare
    out = tmp_path / "abilene.csv.gz"
    prepare(["--tgz", SNDLIB_ARCHIVE, "--start", "20040301", "--days", "7", "--out", str(out)])
    pd.testing.assert_frame_equal(pd.read_csv(out), pd.read_csv(ABILENE_TRACE))


@needs_raw
def test_random_topology_placement_uses_the_env_topology_seed():
    """Regression guard: placement used an unseeded random graph, which need
    not be the graph the env builds."""
    order = _topology_degree_order("random", 16, seed=7)
    net = NetworkModel(topology="random", num_nodes=16, topology_seed=7)
    degs = dict(net.graph.degree())
    assert order == sorted(degs, key=lambda n: (-degs[n], n))
    df = convert_packet_capture(PCAP_NUMERIC, topology="random", topology_seed=7)
    load = df.groupby("dst")["volume"].sum().add(
        df.groupby("src")["volume"].sum(), fill_value=0.0)
    assert degs[int(load.idxmax())] == max(degs.values())


# ---------------------------------------------------------------- committed trace (no raw data needed)
def test_committed_trace_is_well_formed_and_anonymous():
    """The distributed trace holds node ids only: no addresses, no self-loops,
    one row per (step, src, dst)."""
    df = pd.read_csv(ABILENE_TRACE)
    assert list(df.columns) == ["step", "src", "dst", "volume"]
    assert df["src"].between(0, 11).all() and df["dst"].between(0, 11).all()
    assert (df["src"] != df["dst"]).all() and (df["volume"] > 0).all()
    assert not df.duplicated(["step", "src", "dst"]).any()
    assert df["step"].min() == 0
    for col in ("step", "src", "dst"):
        assert pd.api.types.is_integer_dtype(df[col])


def test_committed_trace_drives_the_env():
    traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE, target_load=100.0,
                                    seed=0, random_start=True)
    env = CarlaGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=0.5, max_steps=30,
                      traffic=traffic, seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, reward, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(reward)
