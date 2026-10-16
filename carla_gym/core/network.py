"""
carla_gym/core/network.py

Store-and-forward network simulator on a NetworkX graph.

Model
-----
* Each NODE has a finite buffer (units). Packets physically sit in node
  buffers -- this is what makes node occupancy a real observable rather
  than a decorative zero.
* Each LINK has a service rate (units/step). The RL action scales that
  rate, so bandwidth allocation has a direct, measurable effect.
* Packets advance AT MOST ONE HOP PER STEP, subject to link budget and
  downstream buffer space. Latency is therefore *measured in steps*,
  not proxied by utilization.
* Service is FRACTIONAL: a packet larger than the budget a link has left
  is partly transmitted and finishes the hop on a later step, so any
  positive allocation eventually moves traffic.
* Drops occur on admission overflow, downstream buffer overflow, or when
  no route exists.

* ROUTING is shortest path on the agent's link weights. Among equal-cost
  next hops a packet picks one at random (per-packet ECMP), so symmetric
  positions get symmetric service. `tie_break="first"` takes the lowest node
  id instead.
* QUEUEING is one FIFO per outgoing link by default, all sharing the node's
  buffer: a packet waiting for an exhausted link blocks only traffic for
  that link. `queueing="shared_fifo"` keeps one FIFO per node, where the head
  packet blocks every link (head-of-line blocking across links).
* Nodes are served in a fresh random order every step (`service_order="id"`
  serves them in id order), so no node gets first use of a shared link by
  virtue of its id.

Randomness: the graph is drawn from `topology_seed` (only the "random"
topology uses it), so it does not change with the episode seed; tie-breaks
and service order are drawn from `seed`, reseeded by `reset(seed=...)`.
That stream is derived from the seed with a fixed tag (`_STREAM_TAG`), so
it is independent of a traffic generator given the same seed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from carla_gym.core.traffic import Packet


# The Abilene research backbone as distributed by SNDlib (network "abilene":
# 12 routers, 15 links, capacities in Mbit/s); https://sndlib.put.poznan.pl.
# Node ids follow this (alphabetical) order. Link capacities are used
# relative to the largest (9920 Mbit/s), so one link runs at a quarter.
ABILENE_NODES = ("ATLAM5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
                 "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng")
ABILENE_LINKS = (("ATLAng", "ATLAM5", 9920.0), ("HSTNng", "ATLAng", 9920.0),
                 ("IPLSng", "ATLAng", 2480.0), ("WASHng", "ATLAng", 9920.0),
                 ("IPLSng", "CHINng", 9920.0), ("NYCMng", "CHINng", 9920.0),
                 ("KSCYng", "DNVRng", 9920.0), ("SNVAng", "DNVRng", 9920.0),
                 ("STTLng", "DNVRng", 9920.0), ("KSCYng", "HSTNng", 9920.0),
                 ("LOSAng", "HSTNng", 9920.0), ("KSCYng", "IPLSng", 9920.0),
                 ("SNVAng", "LOSAng", 9920.0), ("WASHng", "NYCMng", 9920.0),
                 ("STTLng", "SNVAng", 9920.0))


class NetworkModel:
    SUPPORTED_TOPOLOGIES = ("grid", "random", "star", "mesh", "ring", "abilene")
    QUEUEING = ("per_link", "shared_fifo")
    TIE_BREAK = ("random", "first")
    SERVICE_ORDER = ("random", "id")

    def __init__(self, topology: str = "grid", num_nodes: int = 9,
                 link_capacity: float = 30.0, node_buffer: float = 120.0,
                 edge_prob: float = 0.35, seed: int | None = None,
                 bottlenecks: dict[int, float] | None = None,
                 topology_seed: int | None = 0, queueing: str = "per_link",
                 tie_break: str = "random", service_order: str = "random"):
        """
        bottlenecks : optional {node id: scale in (0, 1]}. A bottleneck node
            keeps only `scale` of its buffer, and every link touching it keeps
            only `scale` of its capacity (the smaller scale wins on a link
            between two bottleneck nodes). Models a degraded or under-
            provisioned node that traffic should be steered around.
        seed : seeds tie-breaks and service order (not the graph).
        topology_seed : seeds the "random" topology. Fixed by default so the
            graph, and with it the action space, is the same for every
            episode seed. None draws a fresh graph.
        queueing, tie_break, service_order : see the module docstring.
        """
        if topology not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unknown topology '{topology}'. "
                             f"Choose from {self.SUPPORTED_TOPOLOGIES}")
        for name, value, allowed in (("queueing", queueing, self.QUEUEING),
                                     ("tie_break", tie_break, self.TIE_BREAK),
                                     ("service_order", service_order, self.SERVICE_ORDER)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
        if link_capacity <= 0 or node_buffer <= 0:
            raise ValueError("link_capacity and node_buffer must be positive")
        if topology == "abilene" and num_nodes != len(ABILENE_NODES):
            raise ValueError(f"the abilene topology has {len(ABILENE_NODES)} nodes; pass "
                             f"num_nodes={len(ABILENE_NODES)}, got {num_nodes}")
        if topology == "grid" and int(round(np.sqrt(num_nodes))) ** 2 != num_nodes:
            raise ValueError(f"grid topology needs a perfect-square num_nodes "
                             f"(4, 9, 16, ...), got {num_nodes}")
        self.topology = topology
        self.num_nodes = num_nodes
        self.link_capacity = link_capacity
        self.node_buffer = node_buffer
        self.edge_prob = edge_prob
        self.topology_seed = topology_seed
        self.queueing = queueing
        self.tie_break = tie_break
        self.service_order = service_order
        self.rng = self._stream(seed)

        self.graph = self._build_topology()
        self.num_nodes = self.graph.number_of_nodes()
        if sorted(self.graph.nodes()) != list(range(self.num_nodes)):  # pragma: no cover
            raise RuntimeError("topology builders must label nodes 0..N-1")
        self._nodes = list(self.graph.nodes())
        self._edges = list(self.graph.edges())
        self._edge_index = {self._key(u, v): i for i, (u, v) in enumerate(self._edges)}
        self._edge_u = np.array([u for u, _ in self._edges], dtype=np.intp)
        self._edge_v = np.array([v for _, v in self._edges], dtype=np.intp)
        self._neighbors = {n: np.array(sorted(self.graph.neighbors(n)), dtype=np.intp)
                           for n in self._nodes}

        self.bottlenecks = dict(bottlenecks or {})
        for n, scale in self.bottlenecks.items():
            if n not in self.graph:
                raise ValueError(f"bottleneck node {n} is not in the topology")
            if not 0.0 < scale <= 1.0:
                raise ValueError(f"bottleneck scale must be in (0, 1], got {scale} for node {n}")
        scale = {n: self.bottlenecks.get(n, 1.0) for n in self.graph.nodes()}
        # Per-link capacity scale of the topology itself (1.0 except on Abilene).
        self._edge_scale = np.array([self.graph[u][v].get("capacity_scale", 1.0) for u, v in self._edges])
        self._edge_capacity = np.array(
            [self.link_capacity * min(scale[u], scale[v]) for u, v in self._edges]) * self._edge_scale
        self._node_buffer = {n: self.node_buffer * scale[n] for n in self.graph.nodes()}
        self.reset()

    # ------------------------------------------------------------------ #
    # Topology
    # ------------------------------------------------------------------ #
    def _build_topology(self) -> nx.Graph:
        t = self.topology
        if t == "grid":
            side = int(round(np.sqrt(self.num_nodes)))
            g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        elif t == "random":
            topo_rng = np.random.default_rng(self.topology_seed)
            g = nx.erdos_renyi_graph(self.num_nodes, self.edge_prob,
                                     seed=int(topo_rng.integers(1e9)))
            comps = list(nx.connected_components(g))
            for a, b in zip(comps, comps[1:]):
                g.add_edge(next(iter(a)), next(iter(b)))
        elif t == "star":
            g = nx.star_graph(self.num_nodes - 1)
        elif t == "mesh":
            g = nx.complete_graph(self.num_nodes)
        elif t == "ring":
            g = nx.cycle_graph(self.num_nodes)
        elif t == "abilene":
            idx = {name: i for i, name in enumerate(ABILENE_NODES)}
            top = max(c for _, _, c in ABILENE_LINKS)
            g = nx.Graph()
            g.add_nodes_from(range(len(ABILENE_NODES)))
            for a, b, c in ABILENE_LINKS:
                g.add_edge(idx[a], idx[b], capacity_scale=c / top)
        else:  # pragma: no cover
            raise ValueError(t)
        return g

    # Envs seed the traffic generator with default_rng(seed). Seeding this
    # model the same way would give it the very same raw numbers, coupling
    # tie-breaks and service order to packet arrivals. Mixing in a tag gives
    # an independent stream that is still a pure function of the seed.
    _STREAM_TAG = 0x6E6574

    @classmethod
    def _stream(cls, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(None if seed is None else [int(seed), cls._STREAM_TAG])

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        """Undirected links share one budget; normalize endpoint order."""
        return (u, v) if u <= v else (v, u)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> None:
        """Empty the network. An explicit `seed` reseeds tie-breaks and service
        order; without one the existing stream continues (as for traffic)."""
        if seed is not None:
            self.rng = self._stream(seed)
        self.buffers: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        self._served = np.zeros(len(self._edges), dtype=np.float64)
        for i, (u, v) in enumerate(self._edges):
            self.graph[u][v]["capacity"] = float(self._edge_capacity[i])
            self.graph[u][v]["utilization"] = 0.0
        for n in self.graph.nodes():
            self.graph.nodes[n]["buffer_size"] = self._node_buffer[n]
            self.graph.nodes[n]["queue"] = 0.0

        # Initialize default uniform routing weights
        self.update_routing(np.ones(len(self._edges)))

    # Link weights get this added so a zero weight cannot make a free cycle.
    # Every hop therefore lowers the remaining distance by at least this much,
    # which is what keeps tie tolerance (_TIE_RTOL) from ever admitting a loop.
    _WEIGHT_FLOOR = 1e-4
    _TIE_RTOL = 1e-9

    def update_routing(self, routing_weights: np.ndarray) -> None:
        """Set link weights and recompute all-pairs shortest distances."""
        routing_weights = np.asarray(routing_weights, dtype=np.float64)
        if routing_weights.shape != (len(self._edges),) or not np.all(np.isfinite(routing_weights)) \
                or np.any(routing_weights < 0):
            raise ValueError("routing_weights must be one finite, non-negative weight per edge")
        w = routing_weights + self._WEIGHT_FLOOR
        for i, (u, v) in enumerate(self._edges):
            self.graph[u][v]["routing_weight"] = float(w[i])

        # Floyd-Warshall on a dense matrix: O(N^3) but vectorised, and much
        # faster than per-source Dijkstra in Python at the sizes used here.
        n = self.num_nodes
        weight = np.full((n, n), np.inf)
        weight[self._edge_u, self._edge_v] = w
        weight[self._edge_v, self._edge_u] = w
        dist = weight.copy()
        np.fill_diagonal(dist, 0.0)
        for k in range(n):
            np.minimum(dist, dist[:, k, None] + dist[None, k, :], out=dist)
        self._weight = weight
        self._dist = dist

    def _next_hops(self, node: int, dst: int) -> np.ndarray:
        """Neighbours of `node` that lie on a shortest path to `dst`."""
        nbrs = self._neighbors[node]
        via = self._weight[node, nbrs] + self._dist[nbrs, dst]
        target = self._dist[node, dst]
        return nbrs[via <= target + self._TIE_RTOL * max(target, 1.0)]

    def path(self, src: int, dst: int) -> list[int]:
        """A shortest path on the current link weights; [] if none exists.
        Equal-cost next hops are chosen at random, or the lowest id first
        under tie_break="first"."""
        if src == dst or not np.isfinite(self._dist[src, dst]):
            return []
        route = [src]
        node = src
        while node != dst:
            hops = self._next_hops(node, dst)
            if len(hops) == 0:  # pragma: no cover - excluded by _TIE_RTOL << _WEIGHT_FLOOR
                raise RuntimeError(f"no shortest-path next hop from {node} to {dst}")
            if len(hops) == 1 or self.tie_break == "first":
                node = int(hops[0])
            else:
                node = int(hops[self.rng.integers(len(hops))])
            route.append(node)
        return route

    def _occupancy(self, node: int) -> float:
        return sum(p.size for p in self.buffers[node])

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def admit(self, packets: list[Packet]) -> tuple[float, float]:
        """Try to place new packets in their source node's buffer.
        Returns (offered_units, dropped_units)."""
        offered = dropped = 0.0
        occ = {n: self._occupancy(n) for n in self.graph.nodes()}
        for pkt in packets:
            offered += pkt.size
            route = self.path(pkt.src, pkt.dst)
            if len(route) < 2:
                dropped += pkt.size
                continue
            if occ[pkt.src] + pkt.size > self._node_buffer[pkt.src]:
                dropped += pkt.size          # admission overflow
                continue
            pkt.path = route
            pkt.hop_index = 0
            self.buffers[pkt.src].append(pkt)
            occ[pkt.src] += pkt.size
        return offered, dropped

    def forward(self, allocation: np.ndarray, step: int) -> dict:
        """Advance every packet at most one hop, honouring per-link budget
        and downstream buffer space. `allocation` is one weight per edge in
        the canonical edge order (see edge_list())."""
        alloc = np.asarray(allocation, dtype=np.float64)
        # NaN survives clip and compares false against everything, which would
        # make a link look unlimited. Refuse it here, whoever the caller is.
        if alloc.shape != (len(self._edges),) or not np.all(np.isfinite(alloc)):
            raise ValueError("allocation must be one finite weight per edge")
        alloc = np.clip(alloc, 0.0, 1.0)
        budget = {self._key(u, v): self._edge_capacity[i] * alloc[i]
                  for i, (u, v) in enumerate(self._edges)}

        retained: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        arrivals: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        pending_units = np.zeros(len(self._edges))
        self._served[:] = 0.0

        delivered_units = 0.0
        dropped_units = 0.0
        latencies: list[tuple[float, float, float]] = []  # (delay, sensitivity, size)
        deadline_misses = 0
        delivered_count = 0

        # Downstream space is judged against occupancy at the START of the
        # step plus what has arrived since. Space freed by departures becomes
        # usable on the next step. This is independent of node iteration order
        # and guarantees a buffer never ends a step above `node_buffer`.
        start_occ = {n: self._occupancy(n) for n in self.graph.nodes()}
        arrived = {n: 0.0 for n in self.graph.nodes()}

        def occ(n: int) -> float:
            return start_occ[n] + arrived[n]

        order = (self.rng.permutation(self._nodes).tolist()
                 if self.service_order == "random" else self._nodes)
        shared_fifo = self.queueing == "shared_fifo"
        for node in order:
            # Outgoing links whose queue is stalled this step. Under a shared
            # FIFO the first stall blocks the whole node (marked by None).
            blocked: set = set()
            for pkt in self.buffers[node]:
                nxt = pkt.next_hop
                if nxt is None:                       # defensive: already home
                    continue
                ei = self._edge_index[self._key(node, nxt)]
                pending_units[ei] += pkt.size

                if None in blocked or nxt in blocked:  # head-of-line blocking
                    retained[node].append(pkt)
                    continue

                key = self._key(node, nxt)
                need = pkt.size - pkt.sent
                if budget[key] + 1e-9 < need:
                    # Fractional service: spend what the link has left on this
                    # packet and finish the hop on a later step.
                    pkt.sent += budget[key]
                    self._served[ei] += budget[key]
                    budget[key] = 0.0
                    blocked.add(None if shared_fifo else nxt)
                    retained[node].append(pkt)
                    continue

                # Link can finish the hop -- check the downstream buffer next.
                budget[key] = max(budget[key] - need, 0.0)
                self._served[ei] += need
                pkt.sent = 0.0
                if pkt.hop_index + 1 == len(pkt.path) - 1:
                    pkt.hop_index += 1
                    delivered_units += pkt.size
                    delivered_count += 1
                    delay = float(step - pkt.birth_step + 1)
                    latencies.append((delay, pkt.delay_sensitivity, pkt.size))
                    if delay > pkt.deadline and not pkt.missed:
                        pkt.missed = True
                        deadline_misses += 1
                elif occ(nxt) + pkt.size > self._node_buffer[nxt]:
                    dropped_units += pkt.size          # downstream overflow
                else:
                    pkt.hop_index += 1
                    arrivals[nxt].append(pkt)
                    arrived[nxt] += pkt.size

        self.buffers = {n: retained[n] + arrivals[n] for n in self.graph.nodes()}

        # Packets still queued: record their age, and count a deadline miss
        # the first step a packet outlives its deadline while waiting. Without
        # this, traffic that never leaves a buffer is invisible to the delay
        # and deadline terms of the reward.
        queue_expired = 0
        queued_count = 0
        queued_units = 0.0
        queued_age_sum = 0.0
        queued_weighted_age_sum = 0.0
        for pkts in self.buffers.values():
            for pkt in pkts:
                age = float(step - pkt.birth_step + 1)
                queued_count += 1
                queued_units += pkt.size
                queued_age_sum += age
                queued_weighted_age_sum += age * pkt.delay_sensitivity * pkt.size
                if age > pkt.deadline and not pkt.missed:
                    pkt.missed = True
                    queue_expired += 1

        util = np.clip(self._served / np.maximum(self._edge_capacity, 1e-9), 0.0, 1.0)
        for i, (u, v) in enumerate(self._edges):
            self.graph[u][v]["utilization"] = float(util[i])
        for n in self.graph.nodes():
            self.graph.nodes[n]["queue"] = self._occupancy(n)

        return {
            "delivered_units": delivered_units,
            "dropped_units": dropped_units,
            "latencies": latencies,
            "delivered_count": delivered_count,
            "deadline_misses": deadline_misses,
            "queue_expired": queue_expired,
            "queued_count": queued_count,
            "queued_units": queued_units,
            "queued_age_sum": queued_age_sum,
            "queued_weighted_age_sum": queued_weighted_age_sum,
            "pending_units": pending_units,
            "utilization": util,
        }

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #
    def edge_list(self) -> list[tuple[int, int]]:
        return list(self._edges)

    def num_edges(self) -> int:
        return len(self._edges)

    def get_edge_utilization(self) -> np.ndarray:
        return np.array([self.graph[u][v]["utilization"] for u, v in self._edges],
                        dtype=np.float32)

    def get_node_occupancy(self) -> np.ndarray:
        return np.array([self._occupancy(n) for n in self.graph.nodes()],
                        dtype=np.float32)

    def get_edge_capacities(self) -> np.ndarray:
        return self._edge_capacity.astype(np.float32)

    def get_node_buffers(self) -> np.ndarray:
        return np.array([self._node_buffer[n] for n in self.graph.nodes()],
                        dtype=np.float32)

    def backlog_units(self) -> float:
        return float(sum(self._occupancy(n) for n in self.graph.nodes()))
