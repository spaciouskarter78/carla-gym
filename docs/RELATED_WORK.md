# Related work: CarlaGym against existing RL environments for networking

Researcher role, 16 Oct 2026. Compiled from the papers themselves, not from
search summaries (section 4 says how deeply each was read). Every claim
about another project below should be re-checked against its paper before
it is cited; in particular, nothing here was run or installed.

## 1. Comparison

| Project | Simulator, fidelity | What the agent controls | Scale used in the paper | Install effort | Licence |
|---|---|---|---|---|---|
| **CarlaGym** (this work) | Own slotted store-and-forward packet simulator, pure Python; one hop per step, no TCP | **Per-link bandwidth allocation under a budget and per-link routing weights**, jointly or one half at a time | 9 and 16 nodes | `pip install`; NumPy, NetworkX, Gymnasium | BSD 2-clause |
| PackeRL (Boltres et al., TMLR 2024) | ns-3, packet level, UDP and TCP | Routing only: link weights for shortest paths (M-Slim) or next-hop ratings per destination (FieldLines); link rates fixed | Trained on up to 10 nodes and NSFNet/GEANT2; evaluated up to 101-250 nodes | ns-3 build | Paper CC BY 4.0; code licence not checked |
| PRISMA (Alliche et al., IFIP Networking 2022) | ns-3, packet level, ZeroMQ to Python | Routing only: per-packet next hop, one agent per node (multi-agent) | Abilene (11 nodes), GEANT (23 nodes) | ns-3 build, ZeroMQ | Code at github.com/rapariciopardo/PRISMA; licence not checked |
| ns3-gym (Gawlowicz and Zubow, 2018/19) | ns-3, full stack | A bridge, not a task: examples control 802.11 contention windows and channel selection; no routing or bandwidth environment shipped | n/a | ns-3 build, ZeroMQ, Protocol Buffers | GPL |
| NetworkGym (Haider et al., NeurIPS 2024 D&B) | ns-3, full stack, environment hosted in the cloud and reached through a client | Per-user Wi-Fi/LTE traffic-splitting ratio (33 levels), 0.1 s steps | 4 users in the experiments | Client package; environment runs remotely | Code at github.com/IntelLabs/networkgym; licence not checked |
| mobile-env (Schneider et al., NOMS 2022) | Own lightweight simulator of users and base stations | Which cells each user connects to (multi-cell selection with coordinated multipoint); designed for cell selection, power control or scheduling | Not checked | `pip install mobile-env` (25 kB), Gym interface, single- and multi-agent | MIT |
| Park (Mao et al., NeurIPS 2019) | 12 systems environments, 7 on real systems and 5 simulated | Networking ones: video bitrate, congestion control, active queue management, load balancing, switch scheduling, CDN caching. None controls routing or per-link bandwidth | n/a | Per environment | Not checked |
| NetSimGym (Cha et al., EuCNC/6G Summit 2025) | NetSim (a commercial simulator), Protocol Buffers bridge | A bridge; its two examples are cellular: downlink power control and base-station load balancing | n/a | NetSim 14.1 and a NetSim licence file; Python 3.12+ | Wrapper: licence file in the repository, not read; NetSim is not open source |
| MAGNNETO (Bernardez et al., IEEE TCCN 2023) | An RL method, not an environment; its training environment computes link utilisation analytically from one traffic matrix per episode and Dijkstra over the link weights (no packet simulation) | Raises OSPF link weights, one agent per link; objective: minimise maximum link utilisation; capacities fixed | 82 topologies, largest 153 nodes and 354 links | n/a | Code and data released (github.com/BNN-UPC/Papers/wiki/MAGNNETO-TE); licence not checked |

## 2. Where CarlaGym sits

**What no project in this list does.** None of the environments lets an
agent control per-link **bandwidth allocation**: PackeRL and PRISMA fix the
link rates and control routing; NetworkGym controls traffic splits at the
access network; ns3-gym and NetSimGym provide bridges, not tasks. A broad
search on 16 Oct for an open-source Gymnasium environment that jointly
controls routing and bandwidth allocation found none. A search is not proof
of absence, and the paper should say "to our knowledge".

**What CarlaGym gives up.** Fidelity. It is a slotted abstraction (one hop per
step, no TCP, no propagation delay), against ns-3's packet- and
protocol-level models in PackeRL, PRISMA, ns3-gym and NetworkGym. PackeRL
argues directly that packet-level dynamics are required for sub-second
routing and reports that policies trained in a fluid model generalise
poorly to packet level. CarlaGym is not a fluid model (it moves discrete
packets through finite buffers), but it is much simpler than ns-3, so the
same concern applies to transfer: **no CarlaGym result says anything about
behaviour in ns-3 or in a real network** until it is validated (readiness
plan item 8). CarlaGym is also small: 9 and 16 nodes, against PackeRL's
evaluation on up to 250 nodes and MAGNNETO's 153.

**What it offers in exchange.**
1. **Joint control of both levers** (bandwidth and routing) under a budget,
   with ablations that switch either half off; the results show which half
   matters depends on the network.
2. **Low barrier:** pure Python, `pip install`, about 600 steps per second
   on a laptop at 9 nodes; no ns-3 build, no remote service, no commercial
   simulator.
3. **A complete experimental protocol** shipped with the environment:
   tuned rule-based baselines, a fluid upper bound, train / validation /
   test seed separation, and scripts that regenerate every reported number.

**Build vs. contribute** (JOSS asks for this explicitly). Extending PackeRL
or PRISMA would add bandwidth control to a high-fidelity routing
environment, which is a legitimate alternative design. The case for a
separate environment is the use case: fast iteration on joint
bandwidth-and-routing control without an ns-3 toolchain, at the price of
fidelity. The paper should make that trade explicit, and position CarlaGym
as a lightweight testbed for developing and comparing policies, not as a
substitute for packet-level validation.

## 3. Suggested positioning sentence

"Existing RL environments for networking either control routing at packet
level in ns-3 (PRISMA, PackeRL), control access-network traffic splitting
(NetworkGym), or bridge a general simulator without defining a task
(ns3-gym, NetSimGym). To our knowledge none lets an agent allocate per-link
bandwidth under a budget together with routing. CarlaGym provides this in a
lightweight pure-Python simulator, at lower fidelity than ns-3, with tested
baselines, an upper bound and a reproducible evaluation protocol."

## 4. How deeply each source was read

| Source | Read |
|---|---|
| PackeRL | Full text (arXiv HTML), first 100,000 of 148,000 characters, which cover method, setup and results |
| PRISMA | Full PDF, keyword passages (method, topologies, related work) |
| ns3-gym | Full PDF, keyword passages (design, examples, licence) |
| NetworkGym | Abstract and full text (arXiv HTML), architecture and action sections |
| Park | Full PDF, environment table (Table 2) and related passages |
| mobile-env | Full PDF (3 pages), keyword passages; PyPI page |
| NetSimGym | Abstract, publication listing and the repository README (github.com/etri/NetSimGym); the paper itself is behind IEEE Xplore and was not read |
| MAGNNETO | Full text (ar5iv HTML): environment, actions, traffic, capacities, release |

Before citing: read the NetSimGym paper itself if accessible, check every
licence marked "not checked", and search again for work published after
16 Oct 2026 (routing-and-bandwidth RL environments appear regularly).

## Changes after QA (16 Oct 2026)
QA findings QW-1 and QW-2 (`docs/QA_RELATED_WORK_REPORT.md`): Park's list
now includes active queue management; mobile-env's licence (MIT) and task
(multi-cell selection) are filled in from its PyPI page.

### Second pass (16 Oct 2026)
mobile-env, NetSimGym and MAGNNETO were read from their own texts (MAGNNETO
in full, mobile-env's paper, NetSimGym's repository). Changes: MAGNNETO's
training environment is an analytical utilisation model with one traffic
matrix per episode and fixed capacities, confirmed from the paper rather than
from PackeRL's description; NetSimGym's shipped examples are cellular power
control and load balancing, and it needs a NetSim licence; mobile-env is the
closest project in design philosophy (lightweight, pip-installable, Gym
interface; CarlaGym's architecture follows it), in a different domain
(wireless access). The conclusion is unchanged: none controls per-link
bandwidth or routing together with it.

## Sources
* PackeRL: https://arxiv.org/abs/2410.10377 and https://arxiv.org/html/2410.10377
* PRISMA: https://dl.ifip.org/db/conf/networking/networking2022/1570798804.pdf
* ns3-gym: https://arxiv.org/abs/1810.03943
* NetworkGym: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c112271dab20d38b43a1c0669f86b4f6-Abstract-Datasets_and_Benchmarks_Track.html, https://arxiv.org/html/2411.04138v1, https://github.com/IntelLabs/networkgym
* mobile-env: https://ris.uni-paderborn.de/record/30236, https://pypi.org/p/mobile-env
* Park: https://proceedings.neurips.cc/paper/2019/hash/f69e505b08403ad2298b9f262659929a-Abstract.html
* NetSimGym: https://pure.korea.ac.kr/en/publications/netsimgym-the-gymnasium-for-reinforcement-learning-in-networking-/
* MAGNNETO: https://arxiv.org/pdf/2303.18157, https://ar5iv.labs.arxiv.org/html/2303.18157
* mobile-env paper: https://ris.uni-paderborn.de/download/30236/30237
* NetSimGym repository: https://github.com/etri/NetSimGym
