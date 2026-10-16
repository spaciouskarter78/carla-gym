"""
scripts/prepare_sndlib_trace.py

Convert SNDlib's Abilene dynamic traffic matrices (5-minute demand matrices,
Mar-Sep 2004, measured by Yin Zhang) into a CarlaGym trace for the "abilene"
topology: one row per (time bin, source, destination) with the demand in
Mbit/s.

    # 1. download (about 49 MB) from https://sndlib.put.poznan.pl/dynamicmatrices.overview.action
    #    directed-abilene-zhang-5min-over-6months-ALL-native.tgz  ->  data/raw/sndlib/archives/
    # 2. convert one week
    python scripts/prepare_sndlib_trace.py \
        --tgz data/raw/sndlib/archives/directed-abilene-zhang-5min-over-6months-ALL-native.tgz \
        --start 20040301 --days 7 --out data/abilene_trace_20040301_7d.csv.gz

The archive is read directly; nothing is extracted to disk. Node ids follow
carla_gym.core.network.ABILENE_NODES. Rows with zero demand are dropped
(TraceTrafficGenerator treats missing rows as zero).

When using this data, cite SNDlib and its dynamic-traffic references; see
data/ABILENE_TRACE.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import tarfile

import pandas as pd

from carla_gym.core.network import ABILENE_NODES

NAME = re.compile(r"demandMatrix-abilene-zhang-5min-(\d{8})-(\d{4})\.txt$")
DEMAND = re.compile(r"^\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([0-9.eE+-]+)")


def parse_demands(text: str):
    """(source, target, value) for every line of the DEMANDS section."""
    inside, out = False, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("DEMANDS ("):
            inside = True
            continue
        if inside:
            if s.startswith(")"):
                break
            m = DEMAND.match(line)
            if m:
                out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--tgz", required=True)
    p.add_argument("--start", required=True, help="first day, YYYYMMDD")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    first = dt.datetime.strptime(a.start, "%Y%m%d")
    days = {(first + dt.timedelta(d)).strftime("%Y%m%d") for d in range(a.days)}
    idx = {name: i for i, name in enumerate(ABILENE_NODES)}

    picked = []
    with tarfile.open(a.tgz, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            hit = NAME.search(m.name)
            if hit and hit.group(1) in days:
                picked.append((hit.group(1) + hit.group(2), m))
        picked.sort(key=lambda x: x[0])
        expected = 288 * a.days
        if len(picked) != expected:
            raise SystemExit(f"expected {expected} 5-minute matrices for {a.days} days from {a.start}, found {len(picked)}")
        rows = []
        for step, (_, m) in enumerate(picked):
            text = tf.extractfile(m).read().decode("utf-8", errors="replace")
            for s, t, v in parse_demands(text):
                if v > 0 and s in idx and t in idx and s != t:
                    rows.append((step, idx[s], idx[t], v))
    df = pd.DataFrame(rows, columns=["step", "src", "dst", "volume"])
    df.to_csv(a.out, index=False, float_format="%.4f")
    print(f"wrote {a.out}: {len(picked)} time bins ({picked[0][0]} to {picked[-1][0]}), "
          f"{len(df)} rows, {df.groupby(['src', 'dst']).ngroups} source-destination pairs, "
          f"mean total demand {df.groupby('step').volume.sum().mean():.1f} Mbit/s")


if __name__ == "__main__":
    main()
