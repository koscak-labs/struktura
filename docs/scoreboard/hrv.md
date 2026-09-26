# Heart-rate variability: heart failure vs healthy (PhysioNet)

**Datasets:**
- [chf2db](https://physionet.org/content/chf2db/): 29 records, congestive heart failure, NYHA class I-III.
- [nsr2db](https://physionet.org/content/nsr2db/): 54 records, normal sinus rhythm.

Each record is about 24 h of Holter beat annotations.

**Question:** is struktura's short-term DFA exponent lower in heart failure, as Peng et al. 1995
report? And does it add anything to conventional HRV measures?

## Protocol

Written down before the first run.

- **Inputs:** NN intervals, taken from consecutive annotation pairs where both beats are normal
  (N), keeping 0.3-2.0 s. The example reads the WFDB `.ecg` annotation files directly.
- **Primary measure:** alpha1, the median `dfa_short` alpha over non-overlapping 64-beat segments
  (box sizes 4..16).
  - Pre-registered direction: lower alpha1 = heart failure.
  - Scored by AUC-ROC.
- **Also reported:**
  - alpha2, the median `dfa` alpha over 2048-beat segments;
  - SDNN, mean NN and RMSSD;
  - a leave-one-out logistic regression: does alpha1 improve on [mean NN, SDNN]?
- **Control:** alpha1 after shuffling the beats inside each 64-beat segment. This keeps each
  segment's intervals and destroys their order. The protocol specified one shuffle; this example
  runs 20 and reports the median AUC and the range.

## Results (struktura master 9d297d1, 2026-09-26)

| measure | healthy median | heart failure median | AUC, lower = heart failure |
|---|---|---|---|
| **alpha1** (`dfa_short`, 64-beat segments) | **1.231** | **0.933** | **0.829**, bootstrap 95% CI [0.725, 0.919] |
| control: alpha1 of shuffled beats (20 shuffles) | 0.577 | 0.577 | 0.488, range 0.400-0.600 |
| alpha2 (`dfa`, 2048-beat segments) | 1.039 | 1.037 | 0.508 |
| SDNN | 0.137 s | 0.067 s | 0.897 |
| mean NN | 0.780 s | 0.690 s | 0.779 |
| RMSSD | 0.025 s | 0.017 s | 0.746 |

**No incremental value over SDNN.**
- SDNN alone separates the groups better than alpha1: AUC 0.897 vs 0.829.
- The difference is not significant (paired bootstrap 95% CI [-0.022, 0.165]).
- Adding alpha1 to [mean NN, SDNN] does not help. The leave-one-out AUC goes from 0.879 to 0.870.

## Confounds (exploratory, added after the first results)

- **Age works against the finding.**
  - The heart-failure group is younger: median 58 vs 65 years.
  - In the healthy group alpha1 falls with age (r = -0.34).
  - Age alone: leave-one-out AUC 0.646. Age plus alpha1: 0.849.
  - Restricted to the overlapping age range 34-76 (52 healthy, 28 heart failure), alpha1 AUC is 0.816.
- **Heart rate is a partial confound.**
  - Box sizes are counted in beats, and alpha1 correlates with mean NN (r = 0.44 over all records).
  - Mean NN alone: leave-one-out AUC 0.762. Mean NN plus alpha1: 0.836.

## What this does and does not show

- alpha1 is lower in heart failure than in healthy subjects in this cohort, in the pre-registered
  direction. The shuffled control stays near chance, so the separation comes from the order of the
  beats.
- The long-term exponent alpha2 does not differ. Heart failure does not push alpha "toward 0.5" here:
  the heart-failure median alpha1 is 0.933.
- alpha1 does not beat SDNN and adds nothing on top of mean NN and SDNN. struktura does not detect or
  diagnose heart failure. This is one cohort of 83 records, with no external validation.

## Reproduce

```
# download nsr2db and chf2db from PhysioNet into one folder (RECORDS, *.hea, *.ecg)
PHYSIONET_DIR=path/to/physionet cargo run --release --example hrv_eval
```

Source: [examples/hrv_eval.rs](../../examples/hrv_eval.rs). The run above prints every number on
this page.
- Every number that does not depend on a random generator matches the first run, which used the
  published Python wheel 1.8.7.
- That run used a single shuffle (control AUC 0.471) and its own resampling (alpha1 CI
  [0.722, 0.919]).
