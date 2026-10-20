# SIGNALROOT × Struktura DFA: paired eight-seed evidence (2026-10-17)

CI source: https://github.com/koscak-labs/struktura/actions/runs/39052542947
Analyzed commit: `c286eb2cedb91864720b54a50099d804e13a7ebb`.
Python and Rust CI jobs both **passed**.

Protocol: 8 paired spatial seeds, 25 observation locations per policy (including 5 shared initialization points). Offline precomputed Rust `struktura::dfa_short` on 96-point synthetic temporal traces at every location. Selection sees only scores at visited locations. Each hit is a **valid** reading within 2 cells of a hidden synthetic source. Note: computed measure is raw DFA alpha deviation above 0.5, NOT a calibrated Struktura Guard alert.

| Field family | Coverage hit | SCA+DFA hit | Greedy+DFA hit | Mean travel coverage / SCA / greedy |
|---|---:|---:|---:|---:|
| Correlation shift | 5/8 | **7/8** | 4/8 | 309.16 / 148.27 / 136.72 |
| Amplitude-only decoy | 5/8 | **7/8** | 4/8 | 309.16 / 148.27 / 136.72 |
| Missing window/dropout | **5/8** | 2/8 | 3/8 | 309.16 / 158.65 / 139.20 |

Important: correlation and amplitude-decoy runs deliberately share normalized temporal traces with scaled amplitude, which explains identical DFA scores and policy routes; they are NOT independent replications. All policies are evaluated on an idealized unobstructed grid; locomotion and compute budgets are not equalized, only observation count. Even eight independent spatial seeds yield broad uncertainty; no significance or novelty claims.

**Negative finding worth pursuing:** Signal-guided search loses its apparent advantage under missing windows: 2/8 versus coverage 5/8. It also underperforms a greedy policy in travel cost despite more hits on complete data. Don't claim a uniformly superior detector/planner.

Next falsifiable experiment:
- Isolate health-state fusion versus naive raw DFA score using equal-window and equal-travel comparisons.
- Require a trusted baseline per channel, use actual Struktura Guard events with severity, validity, cooldown and quarantine, and test withheld sensor faults.
- Implement explicit exploratory fallback after repeated abstentions; compare to a fixed coverage mixture and abstention-blind SCA.
- Add shuffled-series controls, pure-amplitude baselines, independent signal generation families, 30+ unseen seeds, confidence intervals, planner CPU time and energy costs.
- Reuse the anomaly-to-case evidence mechanism with no hardware actuation.

This document records the original data without retrospective threshold tuning.
