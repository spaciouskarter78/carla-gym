# `abilene_trace_20040301_7d.csv.gz`

One week (Monday 1 March to Sunday 7 March 2004) of measured traffic on the
Abilene research backbone, as 2,016 five-minute traffic matrices between its
12 routers, in Mbit/s. Format: `step,src,dst,volume`; node ids follow
`carla_gym.core.network.ABILENE_NODES` (the `abilene` topology).

## Source

[SNDlib](https://sndlib.put.poznan.pl), "Networks with multiple demand
matrices": `directed-abilene-zhang-5min-over-6months-ALL-native.tgz`
(48,096 five-minute matrices, March to September 2004), downloaded
16 Oct 2026. SNDlib states that the matrices "are based on detailed
measurements of traffic in real IP networks with a certain level of
anonymization" and credits Yin Zhang for the Abilene data. The topology
(12 nodes, 15 links, capacities) is SNDlib's `abilene` network.

The week was chosen by a fixed rule before any experiment: the first
complete Monday-to-Sunday week of the data (all 167 days are complete).

## Converting it yourself

```bash
python scripts/prepare_sndlib_trace.py \
    --tgz data/raw/sndlib/archives/directed-abilene-zhang-5min-over-6months-ALL-native.tgz \
    --start 20040301 --days 7 --out data/abilene_trace_20040301_7d.csv.gz
```

The raw archive is not distributed here; download it from SNDlib.

## Terms and citation

SNDlib states no licence. It asks: "When using the data for publications,
please cite SNDlib", and "Please cite [3] and [4], when using the dynamic
traffic data in your publications":

* S. Orlowski, R. Wessäly, M. Pióro, A. Tomaszewski. SNDlib 1.0 -
  Survivable Network Design Library. *Networks* 55(3):276-286, 2010.
* [3] F. Idzikowski, S. Orlowski, C. Raack, H. Woesner, A. Wolisz. Dynamic
  routing at different layers in IP-over-WDM networks - Maximizing energy
  savings. *Optical Switching and Networking*, Special Issue on Green
  Communications, 2011.
* [4] A. M. C. A. Koster, M. Kutschka, C. Raack. Towards Robust Network
  Design using Integer Linear Programming Techniques. *Proceedings of NGI
  2010*, Paris, 2010.

Whether redistributing a derived trace like this one is acceptable is not
covered by SNDlib's pages; confirming with the maintainer
(sndlib@put.poznan.pl) before publication is advisable.
