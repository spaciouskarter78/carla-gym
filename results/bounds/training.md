Fluid flow upper bound on the delivered fraction (%), from carla_gym/bounds.py: the expected traffic matrix, link capacities and the bandwidth budget, with the allocation chosen optimally. No policy can exceed it in the long run; a finite window can exceed it slightly through backlog carried in. Policy figures: delivered % on test seeds 2000-2009, steps 20..199. The bound is loose (docs/RL_RESULTS_0.4.0.md, section 2.6): the room below it is an upper limit on what any policy could gain, not a measure of it. The trace row's bound uses the trace's mean matrix over the week and ignores its daily cycle, so it is looser still.

| Scenario | Bound, free routing | Bound, shortest paths | Best heuristic | PPO | SAC |
|---|---:|---:|---|---:|---:|
| grid_l125_b06 | 90.3 | 90.3 | congestion_aware 74.1 | 73.5 | 68.6 |
| grid_bottleneck_b08 | 100.0 | 100.0 | load_aware_routing 85.0 | 94.6 | 90.3 |
| abilene_l120_b08 | 92.5 | 92.5 | congestion_aware 80.7 | 75.4 | 78.5 |
| random9_l15_b06 | 99.4 | 99.4 | queue_proportional 79.7 | 81.3 | 76.1 |
