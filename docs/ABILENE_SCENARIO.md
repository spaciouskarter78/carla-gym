# Abilene trace scenario: choice and a recorded deviation

Researcher role, 16 Oct 2026. Replaces the pcap trace scenario
(`pcap40_grid16_b05`), whose data has no established provenance, with
citable public data (`data/ABILENE_TRACE.md`).

## Decision (corrected, 16 Oct 2026, 23:30)

**Abilene topology (SNDlib, 12 routers, 15 links), one-week SNDlib trace
(1-7 March 2004) with random episode starts, mean load 120, bandwidth budget
0.8.** Training flags:

```
--topology abilene --nodes 12 --trace data/abilene_trace_20040301_7d.csv.gz
--load 120 --budget 0.8 --trace-random-start
```

Queue name: `abilene_l120_b08`.

> **Correction.** The first version of this document chose budget 0.7
> (`abilene_l120_b07`, commit `61d028f`). That choice was made with a bug in
> the random-start option: packets were stamped with the shifted trace step,
> so measured delays were negative and the reward's delay and deadline terms
> were zero on this trace. Delivered and dropped traffic were not affected.
> The bug was found from the impossible delays in the first training run
> (which was discarded, never committed or reported), fixed, and guarded by
> two regression tests. The selection was then rerun unchanged, under the
> same rules (C1-C4 and the paired C5'), and picks budget 0.8. The buggy
> run's logs are kept as `qa/research/abilene_select*_buggy_reward.*`.

## How it was chosen

`qa/research/abilene_select.py` (`.log`) ran 20 cells (load 40 to 120 x
budget 0.5 to 0.8) x 4 heuristics on **validation seeds 1000-1029**, built
exactly as training builds the env. Test seeds were not used. The rules were
fixed in the script before it ran: C1 to C5 as in
`qa/research/headline_settle30.py`, with a fallback of dropping C3 if no cell
qualified.

**Result: no cell met the rules.** Every cell congested enough for C1 failed
C5 (95% CI of the best heuristic's reward at most 0.03): the intervals were
0.058 to 0.111 with the corrected code (0.046 to 0.085 in the buggy first
run). The fallback did not help, because C3 (routing helps) was
never the problem: routing moves delivered traffic by 8 to 19 points on
Abilene, more than on any synthetic topology tested.

## Deviation from the pre-set rules (approved by the user)

Why C5 failed: with random episode starts over a real week, each test
episode samples a different time of day, and the offered load varies by a
factor of about two over a day. That moves every policy's reward up and down
together. C5's threshold was calibrated on stationary synthetic traffic,
where this does not happen.

The comparisons between policies are what the scenario is for, and they are
much more stable than the raw rewards: the paired 95% CI of (best minus
second-best heuristic) reward is 0.027 to 0.045 across the congested cells
with the corrected code (0.019 to 0.031 in the buggy first run), against
unpaired intervals of 0.058 to 0.111, with gaps of +0.08 to +0.30
(`qa/research/abilene_select_paired.py`, `.log`).

**Revised rule, for trace scenarios:** C5' = paired CI of (best minus
second-best heuristic) reward at most 0.03, in place of C5. Everything else
as pre-set, including the tie-break (best-heuristic delivered share closest to
76.5%, then lower load).

This rule was written **after** looking at the validation results; it is a
deviation and must be reported as one. It was approved by the user before
any training or any test-seed evaluation of this scenario.

With the corrected code (`qa/research/abilene_select.log`,
`abilene_select_paired.log`), the cells meeting C1-C4 and their paired
stability:

| Load | Budget | Best heuristic delivers | Paired CI (best minus 2nd) | C1-C4 | C5' |
|---|---|---|---|---|---|
| 80 | 0.5 | 87.0% | 0.029 | yes | yes |
| 100 | 0.5 | 73.8% | 0.042 | yes | no |
| 100 | 0.6 | 81.4% | 0.038 | yes | no |
| 100 | 0.7 | 85.8% | 0.045 | yes | no |
| 100 | 0.8 | 86.6% | 0.035 | yes | no |
| 120 | 0.6 | 71.8% | 0.036 | yes | no |
| 120 | 0.7 | 77.0% | 0.033 | yes | no |
| **120** | **0.8** | **78.8%** | **0.027** | yes | yes |

Selected: load 120, budget 0.8 (of the two qualifying cells, the closer to
76.5%). With the delay and deadline terms restored, the paired intervals are
wider than in the buggy run, and 120 / 0.7 now just misses C5' (0.033). On
validation seeds at 120 / 0.8, congestion_aware leads, bandwidth control is
worth 6.4 points and routing 12.9 points of delivered traffic.

## Consequences
* The training queue replaces `pcap40_grid16_b05` with `abilene_l120_b08`
  (PPO and SAC, 5 seeds each).
* The paper should also report the bug and the corrected choice: the first
  selection used a reward without its delay and deadline terms.
* For the paper: report that the trace scenario's stability criterion was
  revised after a first look at validation data, why, and what the
  original rule would have done (no cell qualified).
