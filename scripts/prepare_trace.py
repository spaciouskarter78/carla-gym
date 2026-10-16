"""
scripts/prepare_trace.py

Convert a raw dataset into the (step, src, dst, volume) CSV that
TraceTrafficGenerator / train_ppo.py --trace expect, and report enough
stats to tell whether it will calibrate into a usable load regime before
you spend a training run finding out.

Usage (PowerShell: $env:PYTHONPATH = "."   bash: export PYTHONPATH=.):
    python scripts/prepare_trace.py --in data/raw/my_flows.csv --out data/raw/my_trace.csv
    python scripts/prepare_trace.py --in data/raw/my_capture.csv --out data/raw/my_trace.csv --bin-seconds 5

Converts your own flow logs or packet captures (formats: see
carla_gym/core/dataset_adapters.py). The repository ships no trace made with
this script: the distributed trace comes from SNDlib
(scripts/prepare_sndlib_trace.py, data/ABILENE_TRACE.md). Keep traces built
from private captures out of version control.

The raw inputs are not distributed with the repository; data/raw/ is ignored
by git. Use your own flow-count or packet-capture CSV in one of the two
supported formats (see carla_gym/core/dataset_adapters.py).

Format is auto-detected by column names; pass --format to override.
"""
from __future__ import annotations

import argparse

import numpy as np

from carla_gym.core.dataset_adapters import ADAPTERS, detect_format


def main():
    p = argparse.ArgumentParser(description="Convert a raw dataset into CarlaGym trace format")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--format", choices=["auto", "cs448b", "pcap"], default="auto")
    p.add_argument("--num-nodes", type=int, default=16,
                   help="pcap adapter only (cs448b fixes its own node count)")
    p.add_argument("--bin-seconds", type=float, default=5.0, help="pcap adapter only")
    p.add_argument("--internal-nodes", type=int, default=10, help="cs448b adapter only")
    p.add_argument("--external-buckets", type=int, default=6, help="cs448b adapter only")
    p.add_argument("--topology", default="grid", help="topology to place heavy nodes against")
    p.add_argument("--topology-seed", type=int, default=0,
                   help="for --topology random: the topology_seed the env will be built with (env default 0)")
    args = p.parse_args()

    fmt = detect_format(args.in_path) if args.format == "auto" else args.format
    print(f"Detected format: {fmt}")

    if fmt == "cs448b":
        df = ADAPTERS[fmt](args.in_path, args.out_path,
                           internal_nodes=args.internal_nodes,
                           external_buckets=args.external_buckets,
                           topology=args.topology, topology_seed=args.topology_seed)
    else:
        df = ADAPTERS[fmt](args.in_path, args.out_path,
                           num_nodes=args.num_nodes, bin_seconds=args.bin_seconds,
                           topology=args.topology, topology_seed=args.topology_seed)

    n_nodes = df.attrs["num_nodes"]
    per_step = df.groupby("step")["volume"].sum()

    # TraceTrafficGenerator always rescales to whatever target_load you pass
    # it, so the trace's own raw per-step mean is not a usable suggestion --
    # it would just reproduce whatever load (over- or under-saturated) the
    # raw units happened to imply. Instead, base the suggestion on the
    # default env's link budget (link_capacity=30 across num_edges links),
    # using the fraction our own calibration runs found non-degenerate.
    from carla_gym.core.network import NetworkModel
    net = NetworkModel(topology=args.topology, num_nodes=n_nodes, topology_seed=args.topology_seed)
    default_link_capacity = 30.0
    total_capacity_per_step = net.num_edges() * default_link_capacity
    # About 5% of total link capacity: a trace concentrates its load on few
    # node pairs and few time bins, so it saturates far below the level that
    # evenly spread synthetic traffic tolerates (an earlier packet-capture trace delivered
    # roughly 77% at load 40 on the 4x4 grid and 43% at load 110).
    suggested_load = round(0.05 * total_capacity_per_step, -1) or 10.0

    print(f"Wrote {args.out_path}")
    print(f"  rows            : {len(df)}")
    print(f"  steps           : {df['step'].nunique()} "
          f"(0..{df['step'].max()})")
    print(f"  node ids used   : 0..{n_nodes - 1}  (pass num_nodes={n_nodes} to the env)")
    print(f"  src/dst coverage: {df['src'].nunique()} src nodes, "
          f"{df['dst'].nunique()} dst nodes")
    print(f"  volume/step (raw, pre-rescale): mean={per_step.mean():.1f} "
          f"median={per_step.median():.1f} std={per_step.std():.1f} "
          f"max={per_step.max():.1f}")
    if "dropped_non_ip_rows" in df.attrs:
        print(f"  dropped (no routable IP): {df.attrs['dropped_non_ip_rows']} rows")
    print(f"\n{net.num_edges()} links x link_capacity=30 = {total_capacity_per_step:.0f} "
          f"units/step of total budget at default env settings.")
    print(f"Starting point (NOT a substitute for sweeping --load yourself --\n"
          f"this trace's volume is heavy-tailed, so the right value still\n"
          f"depends on how much burst headroom you want):")
    print(f"  --trace {args.out_path} --nodes {n_nodes} --load {suggested_load:.0f} --budget 0.5")
    print("Check the delivered / dropped fraction, not only the reward: on bursty\n"
          "traces the mean per-step reward can look fine while most traffic is lost.")
    print(f"\nNote: --budget 0.5, not the 0.8 used for synthetic Poisson traffic. Real\n"
          f"traces concentrate load on fewer src/dst pairs, so a congestion-aware\n"
          f"policy only beats uniform allocation once the budget is tighter -- see\n"
          f"README.md's 'real traces vs synthetic traffic' section.")


if __name__ == "__main__":
    main()
