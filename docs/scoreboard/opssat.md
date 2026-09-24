# OPS-SAT-AD: structure of ESA satellite telemetry

**Dataset:** OPS-SAT-AD, telemetry from ESA's OPS-SAT satellite, labelled by KP Labs
([Zenodo 12588359](https://doi.org/10.5281/zenodo.12588359)). 2,123 segments over 9 channels,
median length 70 samples. The dataset's own train/test split: 529 test segments, 113 of them anomalous.

**Question:** does the DFA exponent that struktura computes separate anomalous segments from nominal
ones, and is that separation about the order of the values rather than their spread or length?

## Protocol

- One number per segment: `struktura::dfa_short(values).alpha` (works from about 24 samples) or
  `struktura::dfa(values).alpha` (needs 64).
- Per channel, the median and standard deviation of alpha over the **nominal training segments**.
  The test score is `|alpha - median| / std`. This uses the training labels to pick the nominal
  reference, so it is not fully unsupervised; the test labels are used only for scoring.
- No threshold is chosen: AUC-ROC and AUC-PR over all 529 test segments.
- Controls through the same scoring: alpha of the same values shuffled within each segment (keeps
  length and variance, destroys order), segment length alone, log-variance alone, and a random score.

## Results (struktura master b1a2985, 2026-09-25)

All 529 test segments (113 anomalous):

| score | AUC-ROC | AUC-PR |
|---|---|---|
| **struktura `dfa_short` alpha** | **0.943** | **0.882** |
| control: `dfa_short` alpha of shuffled values | 0.554 | 0.225 |
| control: segment length | 0.734 | 0.434 |
| control: log variance | 0.590 | 0.335 |
| struktura `dfa` alpha (unmeasurable below 64 samples, scored 0) | 0.772 | 0.670 |
| random score | 0.500 | 0.214 |

Split by segment length:

| subset | test segments (anomalous) | `dfa_short` AUC-ROC | shuffled control |
|---|---|---|---|
| 64 samples or more | 272 (84) | 0.925 | 0.461 |
| under 64 samples | 257 (29) | 0.908 | 0.743 |

Median alpha on test segments of 64+ samples: nominal 1.987, anomalous 1.667.

## What this does and does not show

- It is segment classification: one score per labelled segment. It is not the streaming `guard`
  monitor, and it says nothing about detection delay or false alarms per hour.
- On segments of 64+ samples the shuffled control falls to chance (0.461), so the separation there
  comes from the order of the values. On short segments the shuffled control still reaches 0.743:
  part of the short-segment separation is not order structure.
- The reference statistics need nominal training segments per channel.
- The old `dfa` scores 0 on the 1,030 segments under 64 samples; `dfa_short` exists for that case.

## Reproduce

```
# download the dataset from Zenodo 12588359 (dataset.csv, segments.csv)
OPSSAT_DIR=path/to/OPS-SAT-AD/data cargo run --release --example opssat_eval
```

Source: [examples/opssat_eval.rs](../../examples/opssat_eval.rs). The run above prints every table
on this page.
