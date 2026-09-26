# UCR Anomaly Archive: one anomaly per series

**Dataset:** UCR Time Series Anomaly Archive 2021 (Wu and Keogh), 250 univariate series, each with a
training prefix and exactly one labelled anomaly after it
([archive download](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip)).

**Question:** can struktura point at the anomaly, and how does it compare with simple baselines?

## Protocol

Written down before the first run: see [the protocol and first-run logs](preregistration-2026-09-26.md).
One predicted location per series. It counts as correct if it lies within [begin - 100, end + 100],
the archive's KDD Cup 2021 convention. That is a tolerance window, not exact localisation.

- **guard:** `AutoPilot` with its default configuration. This is the monitor behind `struktura guard`
  and the Python `Guard`.
  - Calibrated on the last min(train_end, 20000) training rows. The CLI chooses its calibration
    rows differently.
  - The prediction is the row of the first alarm (or rolled-back adaptation) in the test part.
  - A series with no event, or whose training rows cannot calibrate a monitor, counts as a miss.
    In this run no series failed to calibrate.
- **offline dfa:** rolling `dfa_short` alpha, window 128, stride 16.
  - The reference is the median and MAD of the windows inside the training part.
  - The prediction is the centre of the test window whose alpha is furthest from that reference.
- **Baselines:**
  - the largest first difference in the test part;
  - the largest |z| of the raw value against the training mean and std;
  - a random location (expected accuracy, computed exactly).

## Results (struktura ccd86c5, 2026-09-26)

On a49534a guard had 67 (0.268), silent on 151, McNemar p = 0.215. The DFA fix in ccd86c5
(box sizes inside a constant run no longer enter the fit as rounding noise) changes one series:
169_UCR_Anomaly_gait3, whose anomaly is a flat stretch (91 of the 96 samples before the alarm
repeat one value). Its old DFA-leg alarm, 17.9 against a threshold of 5.28, came from that
rounding noise; now guard is silent there.

| method | accuracy | correct / 250 | no prediction |
|---|---|---|---|
| baseline: largest first difference | 0.312 | 78 | 0 |
| **struktura guard, first alarm** | **0.264** | **66** | 152 |
| struktura offline `dfa_short` locator | 0.244 | 61 | 0 |
| baseline: largest raw z-score | 0.124 | 31 | 0 |
| random location (expected) | 0.024 | 6.0 | 0 |

Guard vs the first-difference baseline, per series:

- 27 series are right only by guard and 39 only by the baseline.
- Exact McNemar p = 0.175. The difference is not significant, and guard does not beat the baseline.

The offline `dfa_short` locator is significantly below the first-difference baseline: 19 vs 36
series, p = 0.030.

## What this does and does not show

- **Guard's overall accuracy is 66/250 = 0.264.**
  - It stayed silent on 152 series.
  - On the 98 series where it did alarm, its first alarm was within tolerance on 66.
  - Those 98 series are easier for every method. On them vs on the other 152:
    - first difference: 43/98 (0.439) vs 35/152 (0.230);
    - offline locator: 37/98 vs 24/152;
    - raw z-score: 22/98 vs 9/152.
  - So 66/98 is not a head-to-head score. It is only quoted together with the 98/250 coverage.
- Both struktura methods are well above random (0.024) and the raw z-score baseline (0.124). Both
  are below the first-difference baseline.
- On 30 series, guard or the offline locator is right and neither baseline is.
- The protocol differs from published UCR leaderboards. These numbers are not comparable with them.

## Reproduce

```
curl -fLO https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
echo "ac4b991c701e620ae9cc5ebd57ae45593a36cc9c0b6ed5e3c4b7e466cf4783d4  UCR_TimeSeriesAnomalyDatasets2021.zip" | sha256sum -c -
unzip -q UCR_TimeSeriesAnomalyDatasets2021.zip 'AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/*'
UCR_DIR=AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData \
  cargo run --release --example ucr_eval
```

The archive the numbers came from is 184,066,400 bytes. It holds 250 series in `UCR_Anomaly_FullData`.

Source: [examples/ucr_eval.rs](../../examples/ucr_eval.rs). The run above prints every number on
this page. Master 9d297d1 gave the same output.

The protocol was first run in Python on the published wheel 1.8.7. Its log is in
[preregistration-2026-09-26.md](preregistration-2026-09-26.md), and its accuracies, the 99 alarmed
series and the 44/34 split are the same as above.
