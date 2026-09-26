# ESA-ADB: guard on real satellite telemetry

**Dataset:** the ESA Anomaly Detection Benchmark (ESA-ADB), Mission 1
([kplabs-pl/ESA-ADB](https://github.com/kplabs-pl/ESA-ADB), data on
[Zenodo 12528696](https://doi.org/10.5281/zenodo.12528696)).
- Channels 41-46, the benchmark's lightweight subset.
- The benchmark's 84_months split, resampled to 30 s:
  - training: 2000-2007, 7,364,161 rows;
  - test: 2007-2014, 7,364,161 rows.
- The operators' test labels on these channels: 29 events of category Anomaly and 65 of category
  Rare Event or Anomaly.

**Question:** under the benchmark's own metrics, do guard's alarms land on the labelled anomalies more
often than the same alarms would at other times?

## Protocol

Written before each run: see [the protocol record](esa-adb-protocol.md).

- **Data:** the official preprocessing (`notebooks/data-prep`), restricted to channels 41-46.
  Running the official script on all 76 channels needs far more memory than six channels do.
  - It was checked against the official script on the benchmark's small 3_months split: timestamps,
    float32 values and labels all identical for these channels.
  - The 84_months split was not checked against the official script.
- **guard:** `struktura.Guard` (AutoPilot, default configuration, cooldown 50).
  - Calibrated on the first 20,160 training rows (7 days) in which all six channels are labelled
    nominal.
  - Every test row is pushed in order. A channel's prediction is 1 wherever guard raises an `alarm`
    or `rolled_back` event on it.
  - The test labels are never read.
- **Scoring:** the benchmark's metric code, imported unchanged. This is the pipeline's own path:
  scores min-max scaled per channel, cast to `uint8`, then:
  - `ESAScores` (event-wise and affiliation F0.5) and `ADTQC` on the maximum over channels;
  - `ChannelAwareFScore` (F0.5) per channel.
- **Control:** the whole prediction matrix, shifted in time by one offset (uniform in 10-90% of the
  length, the same for every channel), 20 seeds.
  - The shift keeps every alarm, its length and its timing across channels. Only where the alarms
    fall relative to the labels changes.
  - p = (1 + number of shifts scoring at least as high) / 21. The smallest possible value is 0.048.

## Results (struktura master 38cb83e, 2026-09-26)

The scripts below were first checked against the original evaluation on master d439fb8. Starting
from the raw files, they gave:
- byte-identical preprocessed arrays;
- identical guard predictions and 20 shifts;
- 125 of 126 metric values equal and none different.

The numbers here are from the same scripts on 38cb83e.

Guard over the 7.36 million test rows:
- 136 alarms and no rolled-back adaptations;
- 138 quarantines and 134 releases;
- alarm ticks per channel 41..46: 42, 5, 23, 12, 41, 13.

| labels | metric | guard | 20 shifts: mean | 20 shifts: max | p |
|---|---|---|---|---|---|
| Anomaly (29) | event-wise recall | **0.103** (3 events) | 0.010 | 0.034 | 0.048 |
| | event-wise F0.5 | 0.035 | 0.003 | 0.009 | 0.048 |
| | affiliation F0.5 | 0.429 | 0.333 | 0.421 | 0.048 |
| | channel F0.5 | 0.085 | 0.006 | 0.031 | 0.048 |
| | alarming precision | 0.167 | 0.212 | 1.000 | 0.33 |
| Rare Event or Anomaly (65) | event-wise recall | **0.169** (11 events) | 0.015 | 0.031 | 0.048 |
| | event-wise F0.5 | 0.111 | 0.008 | 0.017 | 0.048 |
| | affiliation F0.5 | 0.479 | 0.298 | 0.354 | 0.048 |
| | channel F0.5 | 0.124 | 0.010 | 0.019 | 0.048 |
| | alarming precision | 0.289 | 0.529 | 1.000 | 0.71 |

ADTQC (detection timing) is 0.992 and 0.826. It is undefined for shifts that detect nothing, so it
is not compared.

## What this does and does not show

- **Guard's alarms hit more anomaly events than chance would place them.** 3 of 29 Anomaly events,
  and 11 of 65 Rare Event or Anomaly events. That is more than any of 20 time-shifted copies of the
  same alarms (p = 0.048, the smallest this control can give).
- **Recall is low.** Guard misses 26 of the 29 Anomaly events.
- **Precision is not above chance.** Its alarming precision is below the shifts' mean.
- **Multiple comparisons.** Several metrics are tested, and there is no correction for that.
- **It is not a blind test.** Guard has been run on this test split at four versions:

  | version | change | alarms | result |
  |---|---|---|---|
  | 1.8.5 | | 7 | all six channels quarantined for good; matched 1 of 65 events, no better than random |
  | f1f00f2 | recovery added | 2,466 | channels flapping in and out of quarantine |
  | a49534a | recovery back-off | 391 | 3 of 29 Anomaly events, above every shift |
  | 38cb83e | recalibration fixes | 136 | the run reported here |

  - The recovery and back-off changes were written after the previous run on this data showed how
    guard's quarantine behaved.
  - 38cb83e's fixes came from a code audit. The a49534a scores on this data existed before it was
    written.
- **Limited scope.** One mission, six channels, one split. The full 58-channel set was not run.

## Reproduce

Needs Python 3.8 with `numpy==1.21.6 pandas==1.5.3 scikit-learn==1.2.2 portion==2.4.1
python-dateutil`, and struktura's Python module built from this repository.

Timings on a desktop PC: preprocessing 4 minutes, guard 16 seconds, scoring about 25 seconds per
prediction. That is about 14 minutes in all.

```
git clone https://github.com/kplabs-pl/ESA-ADB && export ESA_ADB=$PWD/ESA-ADB
# download Mission 1 from Zenodo 12528696 into $ESA_ADB/data/ESA-Mission1
(cd $ESA_ADB/data/ESA-Mission1 && sha256sum -c -) <<'EOF'
1564d630f1ec1387d699ec81531ce57118051a220c8b7cba050a446a652b9082  labels.csv
e0cadee4ee9697c5ca8d4f9e81067454d7d7a00d063ee0de33cbc77660eec8bc  anomaly_types.csv
92fe4582f1907e42129a764e52afddaef02de8e555e4c47302acd4225222abbf  channels.csv
32519ed13c68067cfafeef0df9ac5f04aeaf534cf03e526ce9e76c02c2a5c09d  telecommands.csv
32cabd39524c5c5fe972255f158e0ba933cdb0609efe88782e6217f3782d20d0  channels/channel_41.zip
ef01d2afdfd6926f69fc7b427eaa63565a7b218d81317027420d22f2aa34854a  channels/channel_42.zip
e664ea5b64ccea49796f32c9fd97cb6584c16d837442aebb2a6926c6f7413289  channels/channel_43.zip
b5fa421b1e5828441a0e61fe489fb3fca3036db2de8c6fbb599be39acdd5de18  channels/channel_44.zip
d98bce78727df6125ad82de764af68c20027b85ef3fe64c8b3dc402d3c58d57e  channels/channel_45.zip
9cf536c62cf3db9e54c0da0c1b9b2988fb92b21e3652da472ff3f5725799ea08  channels/channel_46.zip
EOF
pip install maturin && (cd crates/struktura-py && maturin build --release -o dist) && pip install crates/struktura-py/dist/*.whl
W=esa-adb-work
python esa-adb/eval/prep.py $W/data 84_months 2007-01-01
python esa-adb/eval/guard.py $W guard
for s in $(seq 0 19); do python esa-adb/eval/guard.py $W shift $s; done
python esa-adb/eval/score.py $W $W/preds/guard.npz $W/preds/shift_s*.npz
python esa-adb/eval/summarize.py $W
```

`prep.py` also opens the other 70 channel files and 11 telecommand files, because the official
preprocessing takes its time range from all of them. The checksums cover every file whose values or
labels reach channels 41-46.

The metric and preprocessing code is unchanged between ESA-ADB commit aeebcd9, which these numbers
used, and `main` as of 2026-09-26.

Source: [esa-adb/eval/](../../esa-adb/eval/). `summarize.py` prints the table above.
