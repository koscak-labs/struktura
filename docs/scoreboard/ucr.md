# UCR Anomaly Archive: one anomaly per series

**Dataset:** UCR Time Series Anomaly Archive 2021 (Wu and Keogh), 250 univariate series, each with a
training prefix and exactly one labelled anomaly after it
([archive page](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip)).

**Question:** can struktura point at the anomaly, and how does it compare with simple baselines?

## Protocol

Written down before the first run. One predicted location per series. It counts as correct if it
lies within [begin - 100, end + 100], the archive's KDD Cup 2021 convention. That is a tolerance
window, not exact localisation.

- **guard:** the streaming monitor, as `struktura guard` runs it (`AutoPilot` with defaults).
  - Calibrated on the last min(train_end, 20000) training rows.
  - The prediction is the row of the first alarm (or rolled-back adaptation) in the test part.
  - A series with no event has no prediction and counts as a miss.
- **offline dfa:** rolling `dfa_short` alpha, window 128, stride 16.
  - The reference is the median and MAD of the windows inside the training part.
  - The prediction is the centre of the test window whose alpha is furthest from that reference.
- **Baselines:**
  - the largest first difference in the test part;
  - the largest |z| of the raw value against the training mean and std;
  - a random location (expected accuracy, computed exactly).

## Results (struktura master 9d297d1, 2026-09-26)

| method | accuracy | correct / 250 | no prediction |
|---|---|---|---|
| baseline: largest first difference | 0.312 | 78 | 0 |
| **struktura guard, first alarm** | **0.268** | **67** | 151 |
| struktura offline `dfa_short` locator | 0.244 | 61 | 0 |
| baseline: largest raw z-score | 0.124 | 31 | 0 |
| random location (expected) | 0.024 | 6.0 | 0 |

Guard vs the first-difference baseline, per series:

- 27 series are right only by guard and 38 only by the baseline.
- Exact McNemar p = 0.215. The difference is not significant, and guard does not beat the baseline.

The offline `dfa_short` locator is significantly below the first-difference baseline: 19 vs 36
series, p = 0.030.

## What this does and does not show

- **Guard's overall accuracy is 67/250 = 0.268.**
  - It stayed silent on 151 series.
  - On the 99 series where it did alarm, its first alarm was within tolerance on 67.
  - Those 99 series are easier for every method. The first-difference baseline is right on 44 of
    them (0.444) and on only 34 of the other 151 (0.225).
  - So 67/99 is not a head-to-head score. It is only quoted together with the 99/250 coverage.
- Both struktura methods are well above random (0.024) and the raw z-score baseline (0.124). Both
  are below the first-difference baseline.
- On 30 series, guard or the offline locator is right and neither baseline is.
- The protocol differs from published UCR leaderboards. These numbers are not comparable with them.

## Reproduce

```
# download and unzip the archive; point UCR_DIR at the UCR_Anomaly_FullData folder
UCR_DIR=path/to/UCR_Anomaly_FullData cargo run --release --example ucr_eval
```

Source: [examples/ucr_eval.rs](../../examples/ucr_eval.rs). The run above prints every number on
this page. The same numbers came out of the published Python wheel 1.8.7, which ran the protocol first.
