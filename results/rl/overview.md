Delivered % of offered traffic, test seeds 2000-2009, steps 20..199, mean ± 95% CI (agents: over 5 training seeds; rule-based: over test episodes). The best rule-based policy is picked per row on the test seeds. Ablation rows: PPO controls one half of the action, the rule-based policies both. Flow bound: carla_gym/bounds.py, an upper limit that is loose (see docs/RL_RESULTS_0.4.0.md, section 2.6).

| Training folder | Flags | PPO | SAC | Best rule-based policy | Flow bound |
|---|---|---:|---:|---|---:|
| `grid_l125_b06` | `--load-scale 1.25 --budget 0.6` | 73.5 ± 0.6 | 68.6 ± 1.3 | congestion_aware 74.1 ± 1.4 | 90.3 |
| `grid_bottleneck_b08` | `--bottleneck 4:0.5` | 94.6 ± 0.3 | 90.3 ± 1.4 | load_aware_routing 85.0 ± 0.8 | 100.0 |
| `abilene_l120_b08` | `--topology abilene --nodes 12 --trace data/abilene_trace_20040301_7d.csv.gz --load 120 --budget 0.8 --trace-random-start` | 75.4 ± 3.3 | 78.5 ± 2.3 | congestion_aware 80.7 ± 8.5 | 92.5 |
| `random9_l15_b06` | `--topology random --topology-seed 0 --load-scale 1.5 --budget 0.6` | 81.3 ± 0.4 | 76.1 ± 1.8 | queue_proportional 79.7 ± 0.7 | 99.4 |
| `grid_l125_b06_bandwidth_only` | `--load-scale 1.25 --budget 0.6 --control bandwidth` | 72.9 ± 0.3 | n/a | congestion_aware 74.1 ± 1.4 | 90.3 |
| `grid_l125_b06_routing_only` | `--load-scale 1.25 --budget 0.6 --control routing` | 72.2 ± 0.2 | n/a | congestion_aware 74.1 ± 1.4 | 90.3 |
| `random9_l15_b06_routing_only` | `--topology random --topology-seed 0 --load-scale 1.5 --budget 0.6 --control routing` | 66.3 ± 0.2 | n/a | queue_proportional 79.7 ± 0.7 | 99.4 |
| `grid_bottleneck_b08_bandwidth_only` | `--bottleneck 4:0.5 --control bandwidth` | 72.8 ± 1.1 | n/a | load_aware_routing 85.0 ± 0.8 | 100.0 |
| `grid_bottleneck_b08_routing_only` | `--bottleneck 4:0.5 --control routing` | 93.8 ± 0.2 | n/a | load_aware_routing 85.0 ± 0.8 | 100.0 |
| `random9_l15_b06_bandwidth_only` | `--topology random --topology-seed 0 --load-scale 1.5 --budget 0.6 --control bandwidth` | 80.8 ± 0.2 | n/a | queue_proportional 79.7 ± 0.7 | 99.4 |
