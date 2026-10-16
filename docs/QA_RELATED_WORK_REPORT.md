# QA report: related work and second readiness assessment

Tester/QA role, 16 Oct 2026. Reviewed: `docs/RELATED_WORK.md` and
`docs/PUBLICATION_READINESS_2.md` (uncommitted, worktree). Same-session
caveat: not an independent review.

## Verdict

**Approved, with four small corrections (QW-1 to QW-4).** The comparison's
central claim (no listed environment controls per-link bandwidth; PackeRL
and PRISMA control routing with fixed link rates) holds on every source
re-checked.

## What was re-checked, and how

| Claim | Re-checked against | Result |
|---|---|---|
| PackeRL: ns-3; link rates fixed; largest preset 101-250 nodes; training 3-14 h on 4 cores | Fresh fetch of the arXiv full text, asked for verbatim sentences ("a constant pre-specified datarate") | Confirmed |
| PRISMA: per-packet next hop; Abilene 11 nodes, GEANT 23 | Local text of the PDF, exact strings | Confirmed |
| ns3-gym: GPL; examples are contention-window control and channel selection | Local text of the PDF | Confirmed |
| Park: no routing or per-link bandwidth environment | Local text of Table 2: 11 of 12 rows identified, none controls routing | Confirmed; see QW-1 |
| MAGNNETO: OSPF link weights, more than 75 topologies, up to 153 nodes and 354 links, IEEE TCCN 2023 | arXiv abstract | Confirmed. That it was trained in a fluid model rests on PackeRL's description, as the doc says; the abstract does not say |
| mobile-env: connection (cell) selection between users and base stations | PyPI page | Confirmed; see QW-2 |
| JOSS: commits "concentrated in the last few weeks" are "Not acceptable"; six months of public history | JOSS review criteria, verbatim | Confirmed |
| JOSS: AI disclosure must confirm humans "reviewed, edited, validated all AI-assisted outputs" and "made the core design decisions" | JOSS AI policy (`joss.theoj.org/about`) | Confirmed; see QW-3 |
| Repository: concentrated October history, no tags | `git log`, `git tag` | Confirmed; see QW-4 |
| CarlaGym licence BSD 2-clause | README and `LICENSE` | Confirmed |

## Findings

### QW-1 (low): Park's networking environments are listed incompletely
`RELATED_WORK.md` lists Park's networking environments as video bitrate,
congestion control, load balancing, switch scheduling and CDN caching.
Park's Table 2 also has **network active queue management**. It does not
control routing or link bandwidth either, so the conclusion is unchanged;
add it to the list.

### QW-2 (trivial): mobile-env's licence is known
Marked "not checked"; PyPI states **MIT**.

### QW-3 (trivial): cite JOSS's AI policy page for the design-decisions rule
`PUBLICATION_READINESS_2.md` 4.3 quotes the "core design decisions"
requirement while citing the JOSS documentation pages. The requirement is on
JOSS's AI policy (`https://joss.theoj.org/about`, AI policy section), not on
the review-criteria page. Cite that page, and add its further sentence: in
cases of intentional misrepresentation JOSS may notify the authors'
institutions, funders or professional societies.

### QW-4 (trivial): commit counts have moved
The doc says 71 of 84 commits on `main` are from October. After PR #7 was
merged (13:10 today) it is 76 of 89. The point (history concentrated in one
month) is unchanged; update the figures or date them.

## Observation, no action taken
One fetch of the PackeRL arXiv page reported "a block of instructions at the
end" of the page asking for a particular answer format, which the fetching
tool said it ignored. Nothing was acted on; the confirmed facts above come
from the paper's own sentences. Likely harmless, but noted because content
fetched from the web is data, not instructions.

## Not covered
NetworkGym's figures (33 split levels, 0.1 s steps, 4 users) were not
re-fetched; licences marked "not checked" other than mobile-env's; work
published after 16 Oct.
