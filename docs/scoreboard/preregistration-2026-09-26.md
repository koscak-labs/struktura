# Pre-registration and first-run logs, 2026-09-26

These files back the [UCR](ucr.md) and [HRV](hrv.md) scoreboard pages. They are copied verbatim
from the run directory. The paths inside them are the author's machine.

**Order of events**, by the files' last-modified times (local time, CEST):
- `PROTOCOL.md`: 2026-09-26 03:23
- `hrv_chf.log`: 2026-09-26 03:37
- `ucr_archive.log`: 2026-09-26 03:39
- `hrv_age_check.log`: 2026-09-26 03:49
- `verify/gate_check.log`: 2026-09-26 03:47

These are local file times, not an independent timestamp. The age check and the recompute came
after the first results, and the pages mark what they add as exploratory.

**Where the Rust examples differ from this protocol:**
- **HRV control:** 20 shuffles instead of one. Both the examples and the Python run draw their own
  random numbers, so the control AUC and the bootstrap intervals differ. Every other number is the
  same.
- **UCR guard:** the Python run used the binding's `Guard` (cooldown 50). The example drives
  `AutoPilot` directly. The cooldown only suppresses repeated alarms, so the first alarm is the
  same, and the output is identical.

## The protocol

`PROTOCOL.md`

```text
# Empirical use-case studies for struktura, pre-registered 2026-09-26 (before any result below existed)

All numbers come from struktura 1.8.7 as published (Python wheel from py-v1.8.7), used unmodified.
Scripts in this directory; data under C:/Users/philp/space/upstream/empirical/.

## Study A. Heart-rate variability: congestive heart failure vs healthy (PhysioNet)

Data: PhysioNet nsr2db (54 healthy subjects, about 24 h ECG beat annotations) and chf2db (29 subjects with
congestive heart failure, NYHA I-III), beat annotations `.ecg`, 128 Hz.
RR series per record: intervals between consecutive beats both annotated 'N'; intervals outside
0.3-2.0 s dropped; remaining NN intervals concatenated in order (seconds).
Features per record (fixed now):
- alpha1 = median over consecutive non-overlapping 64-interval segments of `struktura.dfa_short(seg).alpha`
  (box sizes 4..16, the classical short-term exponent). Segments where it returns None are skipped.
- alpha2 = median over consecutive non-overlapping 2048-interval segments of `struktura.dfa(seg).alpha`
  (box sizes 16..512, long-term).
- Conventional HRV references: mean NN, SDNN (std of NN), RMSSD.
- Control: alpha1 recomputed after shuffling the NN intervals within each 64-interval segment (keeps the values,
  destroys their order; seed 0).
Endpoints:
- Primary: AUC-ROC of alpha1 for CHF vs healthy, with direction "lower alpha1 = CHF" fixed in advance
  (Peng et al. 1995 report reduced short-term scaling in heart failure), bootstrap 95% CI (2000 resamples,
  seed 0). The shuffled control must fall near 0.5.
- Secondary: AUC of alpha2 (direction not fixed: report max(AUC, 1-AUC) and say which), of mean NN, SDNN, RMSSD;
  leave-one-out logistic regression AUC of [mean NN, SDNN] vs [mean NN, SDNN, alpha1] (standardised features,
  sklearn LogisticRegression default C).
- Check of the claim in struktura's examples/cardiac_hrv.rs ("heart failure drops alpha toward 0.5"):
  report medians of alpha1 and alpha2 per group.

## Study B. UCR Time Series Anomaly Archive 2021 (250 series, one anomaly each)

Data: UCR_Anomaly_FullData, file name gives train end index and anomaly begin/end.
Scoring (KDD Cup 2021 convention used with this archive): one predicted location per series; correct if it lies
within [begin - 100, end + 100]. Accuracy = fraction of the 250 series predicted correctly. Indices as in the
file name, applied 1-based to the file's rows (row r of the file = index r).
Methods, fixed now, no tuning on the test part:
- M1 struktura Guard (cooldown 50): calibrate on the last min(train_end, 20000) training rows (must be >= 768,
  else first 768 rows... if train_end < 768 the series counts as no prediction), stream the test part (rows after
  train_end); prediction = row of the first alarm or rolled_back event; no event = no prediction (a miss).
- M2 struktura offline DFA: rolling `dfa_short` alpha, window 128, stride 16, over the whole series; baseline =
  median and MAD*1.4826 of the alphas of windows fully inside the training part; prediction = centre row of the
  test window with the largest |alpha - median| / scale. Windows with no alpha are skipped.
Baselines and controls:
- B0 random location: expected accuracy computed exactly per series as
  min(1, (end - begin + 1 + 200) / test_length), summed over series.
- B1 largest |z| of the raw value vs training mean and std (argmax over the test part).
- B2 largest |first difference| in the test part.
Report accuracy of each, and the per-series agreement table for M1, M2, B1, B2.

## What will be reported regardless of outcome
All numbers, including where struktura loses. No claim leaves this directory before a skeptic check of the
numbers against the raw logs.
```

## HRV first run (hrv_chf.py)

`hrv_chf.log`

```text
struktura 1.8.7; records: healthy 54, CHF 29
NN intervals per record: median 104478, min 53265
alpha1           median healthy 1.231  CHF 0.933
alpha2           median healthy 1.039  CHF 1.037
alpha1_shuffled  median healthy 0.576  CHF 0.577
mean_nn          median healthy 0.780  CHF 0.690
sdnn             median healthy 0.137  CHF 0.067
rmssd            median healthy 0.025  CHF 0.017
PRIMARY alpha1 AUC (lower = CHF) 0.829  95% CI [0.722, 0.919]
control alpha1 of shuffled NN: AUC 0.471
alpha2   AUC 0.508 (lower = CHF)
mean_nn  AUC 0.779 (lower = CHF)
sdnn     AUC 0.897 (lower = CHF)
rmssd    AUC 0.746 (lower = CHF)
LOO logistic AUC: [mean_nn, sdnn] 0.879   [mean_nn, sdnn, alpha1] 0.870
```

## UCR first run (ucr_archive.py)

`ucr_archive.log`

```text
struktura 1.8.7; series 250
M1_guard  accuracy 0.268 (67/250), no prediction 151
M2_dfa    accuracy 0.244 (61/250), no prediction 0
B1_z      accuracy 0.124 (31/250), no prediction 0
B2_diff   accuracy 0.312 (78/250), no prediction 0
B0_random expected accuracy 0.024 (6.0/250)
M1 and M2 both correct 31; correct by M1 or M2 but by neither baseline 30
elapsed 44 s
Guard alarmed on 99 series; first alarm correct 67 (67.7%), early 20, late 12; series with train_end<768: 0
on those 99 series, B2_diff correct 44
on the 151 silent series: B2_diff correct 34, M2_dfa correct 24
```

## HRV age check (exploratory, added after the results)

`hrv_age_check.log`

```text
age known: 83/83
healthy: age median 65 range 28-76; sex {'M': 30, 'F': 24}
CHF: age median 58 range 34-79; sex {'?': 19, 'M': 8, 'F': 2}
NYHA in CHF: {'III': 17, 'II': 8, 'I': 4}
AUC of age alone (higher = CHF): 0.286
within healthy: corr(alpha1, age) = -0.34 (n=54)
within CHF: corr(alpha1, age) = -0.10 (n=29)
LOO logistic AUC: [age] 0.646  [age, alpha1] 0.849
age-overlap band 34-76: healthy 52, CHF 28; alpha1 AUC (lower = CHF) 0.816
```

## Independent recompute from the raw per-record and per-series outputs (after the results; includes the heart-rate checks)

`verify/gate_check.log`

```text
UCR n 250 alarmed 99 coverage 0.396
M1_guard overall 67 on alarmed 67 on silent 0
M2_dfa overall 61 on alarmed 37 on silent 24
B1_z overall 31 on alarmed 23 on silent 8
B2_diff overall 78 on alarmed 44 on silent 34
on alarmed: M1 only 27 B2 only 4 both 40 neither 28 McNemar p 0.0
overall M1-only 27 B2-only 38 McNemar p 0.2145
overall M2-only 19 B2-only 36 McNemar p 0.03
B2 acc on alarmed 0.444 on silent 0.225
B0 expected on alarmed 2.2 silent 3.8
post-hoc M1-else-B2 (NOT pre-registered) 101
HRV y counts [54 29]
corr alpha1~mean_nn 0.44 alpha1~sdnn 0.682
group 0 corr alpha1~mean_nn -0.107
group 1 corr alpha1~mean_nn 0.553
LOO ['mean_nn'] 0.762
LOO ['mean_nn', 'alpha1'] 0.836
LOO ['alpha1'] 0.807
LOO ['sdnn'] 0.89
LOO ['sdnn', 'alpha1'] 0.881
LOO ['mean_nn', 'sdnn'] 0.879
LOO ['mean_nn', 'sdnn', 'alpha1'] 0.87
AUC(sdnn)-AUC(alpha1) paired boot 95% CI [-0.022  0.165]
mean_nn overlap band 0.736 0.753 n per group [5 4] alpha1 AUC in band 0.6 mean_nn AUC in band 0.1
healthy alpha1 range 0.802 1.498 CHF 0.368 1.321
CHF alpha1<0.75: 8 CHF alpha2 range 0.78 1.274
```
