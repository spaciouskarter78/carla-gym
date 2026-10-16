"""
carla_gym/core/trace_traffic.py

Dataset-driven traffic. Drop-in replacement for TrafficGenerator: same
`generate(step)` / `reset(seed)` interface, but demand comes from a real
traffic-matrix trace instead of Poisson sampling.

Expected CSV (one row per demand record; long format):
    step, src, dst, volume
`step` = time bin, `src`/`dst` = node ids (0..N-1), `volume` = bytes/Mbps/etc.
Volume is rescaled so the trace's mean offered load per step (averaged over
every time bin, empty ones included) equals `target_load` units/step -- this
is how you match it to link_capacity.
Each record becomes a Poisson number of packets (volume-conserving) with a QoS
class by `qos_mix` (traces rarely carry class labels).
A CSV may be gzip-compressed (`.csv.gz`).

random_start=True starts every episode at a random time bin of the trace,
drawn from the generator's own stream when it is reset, so a long trace (a
week of 5-minute bins) is covered across episodes instead of every episode
replaying its first bins. A seeded reset gives the same start every time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from carla_gym.core.traffic import Packet, QoSClass, DEFAULT_QOS_PROFILES


class TraceTrafficGenerator:
    def __init__(self, nodes, csv_path, target_load=104.0, qos_mix=None, loop=True, seed=None,
                 random_start=False):
        self.nodes = list(nodes)
        df = pd.read_csv(csv_path)
        need = {"step", "src", "dst", "volume"}
        if not need <= set(df.columns):
            raise ValueError(f"CSV needs columns {need}, got {set(df.columns)}")
        df = df[df.src.isin(self.nodes) & df.dst.isin(self.nodes) & (df.src != df.dst)]
        if df.empty:
            raise ValueError("Trace has no usable rows for these node ids "
                             "(need src != dst, both in `nodes`).")
        df = df.copy()
        df["step"] = df["step"] - df["step"].min()
        self.n_steps = int(df["step"].max()) + 1
        # Mean over ALL time bins, including bins with no traffic, so the
        # realised load per simulation step equals target_load.
        df["volume"] = df["volume"] * target_load * self.n_steps / df["volume"].sum()
        self.by_step = {int(s): g[["src", "dst", "volume"]].to_numpy()
                        for s, g in df.groupby("step")}
        self.loop = loop
        self.qos = list(QoSClass)
        mix = qos_mix or {QoSClass.VOICE: .2, QoSClass.GAMING: .1,
                          QoSClass.VIDEO: .3, QoSClass.BEST_EFFORT: .4}
        self.mix = np.array([mix[q] for q in self.qos], dtype=float)
        self.mix /= self.mix.sum()
        # expected packet size under the QoS mix -> volume-conserving packet counts
        self.mean_size = float(sum(m * DEFAULT_QOS_PROFILES[q]["packet_size"]
                                   for m, q in zip(self.mix, self.qos)))
        self.rng = np.random.default_rng(seed)
        self.random_start = random_start
        self.offset = int(self.rng.integers(self.n_steps)) if random_start else 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.random_start:
            self.offset = int(self.rng.integers(self.n_steps))

    def generate(self, step):
        # The offset only chooses which time bin of the trace to read. Packets
        # are stamped with the simulator's own step: their delay is measured
        # against it, so stamping the shifted step would make delays negative.
        shifted = step + self.offset
        t = shifted % self.n_steps if self.loop else shifted
        packets = []
        for src, dst, vol in self.by_step.get(t, []):
            n = self.rng.poisson(vol / self.mean_size)   # conserves volume on average
            for _ in range(n):
                q = self.qos[self.rng.choice(len(self.qos), p=self.mix)]
                prof = DEFAULT_QOS_PROFILES[q]
                size = float(max(self.rng.normal(prof["packet_size"],
                                                 prof["packet_size"] * .2), .1))
                packets.append(Packet(int(src), int(dst), q, size,
                                      prof["delay_sensitivity"], prof["deadline"], step))
        return packets
