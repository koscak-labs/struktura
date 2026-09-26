# ESA-ADB evaluation record, 2026-09-25/26

Backs the [ESA-ADB scoreboard page](esa-adb.md). The protocol and the analysis report below are copied
verbatim from the evaluation directory. The paths inside them are the author's machine.

**What the record shows:**
- The protocol was written before the first run.
- It was amended thirteen times, and each amendment says what had been seen when it was written.
- It covers guard at struktura 1.8.5 (Amendment 3), f1f00f2 (Amendment 9), a49534a (Amendment 10),
  38cb83e (Amendment 11) and ccd86c5, the DFA flat-box fix (Amendment 12).
- Amendment 13 is branch fix/parity-isolation (45b92ec). It did worse on this data and was not
  merged.
- The report's follow-up to Amendment 13 checks guard's recovery back-off on channel 45.
- It also covers the benchmark's `struktura_dfa` algorithm (Amendments 4-8), which is not on the
  scoreboard page.

**File times (local, CEST).** The protocol file is appended to, so its time is that of the last
amendment. These are local file times, not an independent timestamp.
- `PROTOCOL.md`: 2026-09-26 14:43:15 (Amendment 12 appended)
- 38cb83e prediction `run-38cb83e/preds/guard.npz`: 2026-09-26 08:39:27
- ccd86c5 prediction `run-ccd86c5/preds/guard.npz`: 2026-09-26 12:18:46

The scripts in `esa-adb/eval/` were checked against this pipeline on master d439fb8, starting from
the raw files:
- identical preprocessed arrays;
- identical guard predictions and 20 shifts;
- 125 of 126 metric values equal and none different.

The scoreboard numbers come from those scripts on 38cb83e (Amendment 11 in the report). On ccd86c5
(Amendment 12) the guard prediction matrix is byte-identical, so the same numbers hold.

## Protocol and amendments

`PROTOCOL.md`

```text
# Pre-registered protocol (written 2026-09-25 before any test-split prediction or score existed)

Data: ESA-ADB Mission1, 84_months split, channels 41-46 (the benchmark's lightweight subset),
reproduced with prep_subset.py (restricted replication of the official prep script; verified
identical to the official 3_months.train.csv for these channels: timestamps, float32 values, labels).

## struktura (1.8.5, PyPI-equivalent wheel from koscak-labs/struktura py-v1.8.5, default MonitorConfig)
- Calibration rows: the FIRST contiguous block of N rows of 84_months.train in which all six
  is_anomaly_channel_4x columns are 0 (nominal: no Anomaly, no Rare Event, no gap label).
- Primary run: N = 768 (as instructed). Sensitivity run, also fixed now: N = 20160 (7 days at 30 s).
  Both are reported whatever they score. No other N is tried.
- Values: float32 from the prep, passed as Python floats. Channel order 41..46.
- Stream every row of 84_months.test through Monitor.push in order. When push returns non-None,
  prediction[tick, last_alarm()['channel']] = 1. All other cells 0. No post-processing, no
  threshold change, no alarm extension.
- Info only (no selection made on it): alarm count when the rest of the train split is streamed.

## Controls (scored with the same code path)
- (a) random: per channel, the same number of positive ticks as struktura (primary run), placed
  uniformly at random over the test ticks; 10 seeds (0..9), report mean, sd, min, max.
- (a2) random-events: struktura's own alarm runs (same count, same lengths, per channel) moved to
  uniformly random start ticks; 10 seeds.
- (b) std, tol = 3: means/stds per channel from 84_months.train rows with is_anomaly == 0 (float32
  np.mean / np.std as in TimeEval-algorithms/std/algorithm.py), stds == 0 -> 1, prediction =
  value > mean + 3 std or value < mean - 3 std.
- (c) all-zero.

## Scoring (replicates timeeval/core/experiments.py Experiment.evaluate for this setup)
- labels.csv (tz dropped), filtered to StartTime >= first test timestamp and EndTime <= last test
  timestamp, anomaly_types.csv last 4 columns merged in, restricted to Channel in channels 41-46.
- y_scores -> MinMax per channel (scale_scores) -> astype(uint8).
- ESAScores(betas=0.5) with Category in {Rare Event, Anomaly} and {Anomaly}: global y_pred =
  max over channels. ChannelAwareFScore(beta=0.5, both selections, with subsystem mapping from
  channels.csv). ADTQC (both selections, global as in the pipeline).
- The (timestamp, score) series is passed compressed to change points (first row, every row whose
  value differs from the previous one, last row). This yields the identical event list in
  convert_time_series_to_events and the same full_range; equivalence is checked numerically on a
  test slice before use.
- Winner rule for the summary: struktura "beats" a control on a metric if its value is higher
  (for random: higher than the max over seeds for "clearly beats", higher than the mean for
  "beats on average").

## Amendment 1 (2026-09-25, after seeing ONLY alarm counts, before any struktura metric was computed)
Observation: with the protocol above, struktura raised exactly 1 alarm on the 7.36M-row train
remainder and exactly 1 alarm on the 7.36M-row test split (N=768 and N=20160 alike). Cause, from
struktura source (src/monitor.rs:846): "After an alarm, the monitor latches (returns None) until
reset". The literal protocol therefore measures a one-shot detector.
Amendment (API usage, not tuned on labels): add a second mode that calls Monitor.reset()
immediately after every alarm (reset clears the latch and detector streaks; ring contents persist,
src/monitor.rs:1098/1187). Both the literal (latched) runs and the reset runs are reported, for
N=768 and N=20160. The random controls (a)/(a2) are matched to the primary reset run (N=768, reset),
because matching to a single alarm would be a degenerate control; they are additionally matched to
the latched N=768 run for completeness.
Disclosure: I (the analyst) saw the test alarm count (1) before this amendment; I had not seen any
label overlap or metric for struktura.

## Amendment 2 (2026-09-25, feasibility; disclosed)
The official metric code materialises every alarm event as Python objects. Scoring the N=20160
reset run (1.12M global alarm events) took 632 s for one ESAScores call and brought machine free
RAM down to 1.31 GB. The N=768 reset run has 3.30M events and the tick-matched random controls
for it would have a comparable number, which does not fit in this machine's RAM. Changes:
- Random controls (a) and (a2) are matched to the N=20160 reset run instead of the N=768 reset run.
- Seeds reduced from 10 to 3 (0,1,2) per random type because of scoring time (~30 min per file).
- The N=768 reset run is still attempted last, one metric per process with a memory guard; any
  metric the guard kills is reported as NOT COMPUTED.
- The latched runs keep the 10-seed random control (1 alarm tick, cheap).
Disclosure: when this was decided I had seen, for the N=20160 reset run only, ADTQC(R+A) and
ESAScores(R+A); plus all std/zeros/latched scores.

## Amendment 3 (2026-09-25, written BEFORE any Guard prediction existed)
Decided after all Monitor results (latched and reset-after-alarm, with their controls) had been seen.
No label information was used to choose it; it is set by what the shipped CLI does.
Why: `struktura guard` (the CLI) does not use the bare Monitor. It runs AutoPilot (src/autopilot.rs),
which suppresses a repeat of the same leg within 50 samples. Neither "latched" nor "reset after every
alarm" is the shipped behaviour. The Python binding exposes it as struktura.Guard(clean_channels,
cooldown=50); push(sample) returns a list of event dicts, kind in {alarm, quarantined,
adaptation_started, recalibrated, rolled_back}.
Protocol:
- Same data, channels 41-46, same calibration blocks (N=768 rows 0..767, N=20160 rows 0..20159),
  cooldown=50 (CLI default; no other value is tried). No NaNs occur in the prep output.
- prediction[tick, event["channel"]] = 1 for every event with kind == "alarm" or kind == "rolled_back"
  (a confirmed fault). quarantined / adaptation_started / recalibrated are not alarms. Counts of every
  kind are reported.
- Same official-metric code path, one metric per process, memory guard, killed = NOT COMPUTED.
- Controls: random ticks and random events matched to the Guard N=20160 run, 3 seeds each (5 if memory
  allows); std tol=3 and all-zero reuse the existing scores.
- Guard wheel: struktura-1.8.5-cp38-abi3-win_amd64.whl from C:/Users/philp/AppData/Local/Temp/guardwheel
  (local build that adds Guard; version string still 1.8.5), installed --force-reinstall --no-deps.

## Amendment 4 (2026-09-25, written BEFORE any prediction of the fixed algorithm existed)
Scores the ESA-ADB struktura_dfa algorithm itself (the one merged in kplabs-pl/ESA-ADB #40), after a fix
on branch fix/struktura-dfa-binary: headerless output, one 0/1 column per target channel (sorted),
window flagged when robust z > threshold. The merged original cannot be scored at all: its header
line makes the official loader read n+1 rows (ValueError on assignment). Default parameters only
(window 1024, stride 256, baseline_fraction 0.2, threshold 3.5, the standard modified-z cutoff);
no other value is tried. It sees only the 84_months test file, as the docker adapter would pass it.
Controls: random ticks and random events matched to it, seeds 0-4, same scoring code; std/zeros reuse.

## Amendment 5 (2026-09-25, decided after seeing the Amendment 4 scores of the fixed algorithm, before any causal prediction existed)
Why: the fixed algorithm (window mode) flags every sample of a flagged 1024-sample window, including up to
~512 samples before the data that triggered it. That is look-ahead, visible in ADTQC (37 of 39 detections
start before the anomaly). A causal mode is added on the same branch: when a window's last value arrives
and its robust z > threshold, it flags the next `stride` samples only (never anything before the decision).
Run: same data, defaults + {"causal": true}, no other value. Controls: random events (same run lengths)
seeds 0-4 and random ticks seeds 0-2, same scoring code. Both window and causal results are reported.

## Amendment 6 (2026-09-26, after the Amendment 4/5 chains died mid-run; decided before any control of the fixed algorithm beyond random_s0 was scored)
Random-TICK controls are dropped for the fixed struktura_dfa runs: matched to 344,832 alarm ticks they
scatter single ticks into ~300k separate events (the algorithm raises 177 events), which is not a
same-budget control and does not fit this machine's RAM (the memory guard killed seed 1 twice).
random_s0 (ticks) stays reported as scored. The control is random EVENTS: per channel, the algorithm's
own alarm runs (same count, same lengths) moved to uniformly random start rows, seeds 0-4, same
scoring code. Order: controls for the window mode first, then the causal mode and its controls.

## Amendment 7 (2026-09-26, written BEFORE any prediction on the training split existed)
Question: can the CAUSAL mode of struktura_dfa (branch fix/struktura-dfa-binary) detect better than chance
when its configuration is chosen on labelled TRAINING data only?
- Selection data: 84_months.train (2000-01-01 .. 2007-01-01, channels 41-46), scored with the same official
  metric code against the labels inside that period (score.py with SPLIT=train; labels are filtered by the
  file's time range, as for test).
- Grid (12 configs, causal = true, baseline_fraction = 0.2): window in {128, 256, 512, 1024}, stride = window/4,
  threshold in {2.5, 3.5, 5.0}. No other values.
- Selection rule, fixed now: highest ChannelAwareFScore (Category Anomaly) channel_F0.50 on train; ties broken by
  ESAScores (Anomaly) EW_F_0.50, then by the smaller window. Configs with zero alarms are excluded.
- Then exactly ONE run of the selected config on 84_months.test, with random-events controls seeds 0-4, same
  scoring. Reported whatever it scores, next to the default causal run (Amendment 5) and std.

## Amendment 8 (2026-09-26, after an adversarial review of the fix branch; written BEFORE any prediction below existed)
Two review findings change the evaluation, both upheld by independent skeptics:
1. The causal mode still looked ahead inside the baseline: decisions for windows in the first
   baseline_fraction were judged against a baseline that included later windows. Fixed on the branch:
   in causal mode no window is judged until all baseline windows have ended. (Also fixed, not affecting
   these runs: read errors now exit 1, explicit flush, BOM/quoted headers, minimum window 72, at least
   10 baseline windows per channel.) The Amendment 5 causal predictions are superseded; the window mode
   is re-run to confirm its output is unchanged by the fix.
2. The random-events control (Amendment 6) is matched per channel, but ESAScores and ADTQC are scored on
   the max over channels. The algorithm's channels alarm together, the per-channel shuffle scatters them,
   so the control's global alarm budget was about 3.5 times larger. New primary control for all metrics:
   CIRCULAR SHIFT of the whole prediction matrix by one offset shared by all channels, offset uniform in
   [10%, 90%] of the series length, 20 seeds (0-19). This keeps every alarm, its length, and the
   cross-channel coincidence, and moves only where alarms fall relative to the labels.
Reported: window mode and fixed causal mode, each with its circular-shift controls (max, mean, and the
empirical p-value = (1 + #controls >= observed) / 21 per metric); the random-events controls stay
reported
for the channel-aware metric. Amendment 7 is re-run afterwards with the fixed binary (same grid, same
rule).
(Amendment 7 re-run note: same grid and rule, prefix trgrid2, fixed binary; the trgrid runs of the first attempt used the look-ahead causal mode and are superseded.)

## Amendment 9 (2026-09-26, written BEFORE the run below; requested by the struktura maintainers)
Re-run of the Amendment 3 Guard protocol on struktura master f1f00f2, which adds quarantine recovery
(a quarantined channel is released after 192 consecutive readings consistent with its calibration;
Event::Unquarantined). Wheel built from f1f00f2 (version string still 1.8.7). Same data, same calibration block
(N=20160), cooldown 50, prediction = alarm and rolled_back events per channel. Output name guard_n20160_f1f00f2.
Controls: circular shift of the whole prediction matrix, 20 seeds (Amendment 8 rule). Reported with the count of
every event kind (including quarantined / unquarantined). Maintainers' stated expectation before the run: the
forward-filled 30 s steps may keep resetting the recovery count, so channels may stay quarantined.

## Amendment 10 (2026-09-26, written BEFORE the run below; requested by the struktura maintainers)
Re-run of the Amendment 9 Guard protocol on struktura master a49534a, which adds recovery back-off:
a channel that fails again less than its current recovery span after a release has its span doubled,
recovery_span(flaps) = 192 << min(flaps, 10); staying healthy for the span resets it. Every event kind is
still emitted. Wheel built from a49534a (version string still 1.8.7) and installed into the same venv.
Pre-check (backoff_check.py, a sensor stuck 300 of every 550 samples over 20,000): f1f00f2 wheel
37 quarantined / 36 unquarantined, a49534a wheel 2 / 1, so the installed module carries the back-off.
Same data, same calibration block (N=20160, rows 0..20159 of train), cooldown 50, prediction = alarm and
rolled_back events per channel. Output name guard_n20160_a49534a. Controls: circular shift of the whole
prediction matrix, 20 seeds (Amendment 8 rule). NOT_COMPUTED metrics are retried once (RETRY_NC).
Reported: the count of every event kind and alarm leg against f1f00f2, and the six official metrics
against f1f00f2 and against the controls. Other commits between f1f00f2 and a49534a (3f11658 docs,
9d297d1 blank CSV cell, 55ffb3d time-named column dropped) touch the CLI's CSV path, not the Guard
object this protocol drives, so any change in counts is attributed to the back-off only if the event
stream differs in quarantine/unquarantine first.

## Amendment 11 (2026-09-26, written BEFORE the run below; requested by the struktura maintainers)
Re-run of the Amendment 10 Guard protocol on struktura master 38cb83e, which changes Guard during
recalibration (a channel quarantined during baseline collection keeps its previous calibration; a sensor
failing during collection ends it and is quarantined; the current monitor is fed during the trial).
Same data (the esa-adb/eval/prep.py output, byte-identical to the original prep), same calibration block
(N=20160, rows 0..20159 of train), cooldown 50, prediction = alarm and rolled_back events per channel.
Run with the ported scripts (struktura repo esa-adb/eval/, validated in PR #35 against the original pipeline:
125/126 metric values equal, 0 different). Controls: 20 circular shifts of the new predictions (same seeds and
offset rule). Reported whatever it scores, with every event kind against a49534a, and the scoreboard page
updated to these numbers.

## Amendment 12 (2026-09-26, written BEFORE the run below; requested by the struktura maintainers)
Re-run of the Amendment 11 Guard protocol on struktura branch fix/dfa-flat-boxes, commit ccd86c5 (not yet on
master). It changes every Rust DFA path (dfa, dfa_into, dfa_scratch, dfa_fast_into, dfa_short): a box whose
one-pass residual is within 1e-9 of its own terms is recomputed from centred explicit residuals, and a box size
with F^2 <= 1e-24 * mean(profile^2) no longer enters the fit. Guard's DFA leg uses dfa_fast_into, so predictions
may change. Same data (prep output), same calibration block, cooldown 50, same scripts (esa-adb/eval). If the
prediction matrix is byte-identical to 38cb83e/a342c16, that is reported and nothing else is re-scored; otherwise
the 20 circular shifts are regenerated and all 21 predictions scored with the official code, reported whatever
they score.

## Amendment 13 (2026-09-26, written BEFORE the run below; requested by the struktura maintainers)
Guard on branch fix/parity-isolation, head 45b92ec (NOT on master; on top of master fe32e6a, whose src equals
ccd86c5). Parity leg only: a channel is quarantined only if it stays inconsistent whichever single other channel
is left out, otherwise the alarm is class "cross_channel_ambiguous" and nothing is quarantined (with a hold);
channels whose calibration values or reconstruction residual are >= 50% explained by a line in time are not
parity targets. Wheel built from a FRESH clone at 45b92ec (own target dir), installed .pyd hash-checked. Same
data, calibration block, cooldown 50, esa-adb/eval scripts. Baseline = the ccd86c5 run (Amendment 12, identical
to 38cb83e). Reported whatever it gives: event kinds, alarm legs, alarm classes (separate counting pass with the
same push loop), quarantine/unquarantine counts, and, if the prediction matrix differs, all six official metrics
against 20 regenerated circular shifts. Decision rule stated by the maintainers before the run: the branch stays
off master if quarantines or false alarms get worse or recall drops.
```

## Analysis report

`REPORT.md`

```text
# ESA-ADB Mission 1, channels 41-46: struktura under the official metrics

Written 2026-09-25 from the files in this directory. Numbers: `tables.md` (built by `make_tables.py`
from `scores/*.json`). Protocol and its three disclosed amendments: `PROTOCOL.md`.

## Verdict

On ESA-ADB Mission 1, channels 41-46 (the benchmark's own lightweight subset), 84_months split,
7,364,161 test rows, 65 labelled events (29 of category Anomaly), **no struktura mode shows detection
skill beyond a random control with the same number of alarms.** The benchmark's `std` (tol=3) baseline
has the best affiliation F0.5. Nothing here supports any ESA-ADB detection claim.

## What was run

- Data: `prep_subset.py`, a restriction to channels 41-46 of the official preprocessing; verified
  identical to the official `3_months.train.csv` for these channels (timestamps, float32 values,
  labels): `logs/check_prep_3_months.log`. The full 84_months official prep was not run (memory).
- Metrics: the official `ESAScores(betas=0.5)`, `ChannelAwareFScore(beta=0.5)` and `ADTQC`, both label
  selections, with the official scaling (MinMax per channel, then `astype(uint8)`). The score series
  is passed compressed to change points; equivalence checked on a test slice (`logs/check_compression.log`).
- Detectors (all struktura 1.8.5 core, Python binding): bare Monitor latched (literal protocol),
  Monitor with reset after every alarm (Amendment 1), Guard = AutoPilot + 50-sample same-leg dedupe,
  i.e. what `struktura guard` runs (Amendment 3). Calibration N = 768 and N = 20160 rows, fixed in advance.
- Controls: random ticks and random events matched in count to a struktura run (3-10 seeds), `std`
  tol=3 replicated from `TimeEval-algorithms/std/algorithm.py`, all-zero.

## Results (Category Rare Event + Anomaly; the Anomaly-only table in `tables.md` shows the same pattern)

| method | alarm events | event recall | EW F0.5 | AFF F0.5 | channel F0.5 | ADTQC |
|---|---|---|---|---|---|---|
| Guard N=20160, cooldown 50 (N=768 identical) | 7 | 0.015 (1/65) | 0.056 | 0.0002 | 0.008 | 1.0 (1 event) |
| random, same 7 ticks, 5 seeds | 7 | 0.003 [0..0.015] | 0.011 [0..0.054] | 0.166 [0.106..0.194] | 0.002 | 1 seed detected |
| Monitor latched (N=768, N=20160) | 1 | 0 | 0 | 0.00003 | 0 | nan |
| Monitor reset-after-alarm N=20160 | 1,121,809 | 0.708 | 4.4e-5 | 0.446 | 0.541 | 0.832 |
| random ticks matched to it, 3 seeds | ~952,000 | 0.667 [0.63..0.71] | 4.9e-5 | 0.445 [0.441..0.447] | 0.512 [0.487..0.538] | 0.875 (2 of 3 seeds) |
| std tol=3 | 32,499 | 0.431 | 0.0011 | **0.509** | 0.351 | 0.770 |
| all-zero | 0 | official code raises | | | 0 | nan |

## Why it failed here

1. **Latching.** The bare Monitor stops reporting after its first alarm (src/monitor.rs:846): 1 alarm
   in 7.36M rows. Python/JS users only had this class until `Guard` was added (koscak-labs/struktura #32,
   released in 1.8.6).
2. **Flat stretches read as dead sensors.** Guard raised 7 alarms; its one hit is anomaly id_116,
   14 s after onset. By 2007-04-14 the `repeated_value` leg had quarantined all six channels and Guard
   stayed silent for the remaining 6.7 years. The official preprocessing forward-fills each 30 s step
   with the last value, so constant runs are normal in this data. (Not checked: which runs are fill and
   which are genuinely constant readings.)
3. **Flooding when forced to re-arm.** Resetting after every alarm makes it alarm on 16% (N=20160) to
   46% (N=768) of all test rows, which is indistinguishable from random at the same alarm budget.
4. **A core bug found on the way** (reported to the struktura maintainers with a repro, proofs in
   `../monitor-latch-skip-2026-09-25/`): when one channel alarms, `push_with_validity` skips that tick's
   sample for every later channel, so their clocks drift. In the reset-mode runs the lag reached 3.3M
   ticks, so those runs did not see all the data.

## Not done / not verified

- Not computed (memory guard killed them): ESAScores (Rare Event + Anomaly) for the N=768 reset run,
  ADTQC for one random seed, both ESAScores for one random-events seed. No random control matched to
  the N=768 reset run (over 3M events; does not fit in memory).
- `std` replicated from its Python source, not run through the official Docker image.
- The Guard run used a local build of the #32 branch whose version string says 1.8.5; the released
  1.8.6 wheel contains the same Guard code but was not the one used here.
- Only channels 41-46; the full 58-channel target set was not run.

## Update (2026-09-25): the core bug in point 4 is fixed

Fixed on struktura master 097b8e6 (independently confirmed with the repro: channel 1 now reports tick
3000 after 3001 samples, was 2689). Effect on the runs above: the Guard runs had 7 alarm ticks, so at
most a handful of samples were skipped per channel and the Guard conclusion (quarantine of all six
channels via repeated_value on forward-filled data) does not depend on it. The reset-after-alarm runs
were affected heavily (lag up to 3.3M ticks) and were not re-run.

## Amendments 4-6 (2026-09-26): the ESA-ADB struktura_dfa algorithm itself

The algorithm merged in kplabs-pl/ESA-ADB #40 cannot be scored (header line -> n+1 rows; `logs/contract_merged.log`).
Fixed on branch fix/struktura-dfa-binary (philphauler/ESA-ADB, commits 1eeceb1, 74e58e2, 7793fd1; `logs/contract_fixed.log`).
Official metrics, channels 41-46, Anomaly labels (29 events), defaults; control = own alarm runs moved to random
positions (random events), 5 seeds, max shown:

| | EW recall | EW F0.5 | AFF F0.5 | channel F0.5 | ADTQC |
|---|---|---|---|---|---|
| window (default) | 0.862 | 0.179 | 0.816 | 0.861 | 0.099 |
| random runs | 0.241 | 0.015 | 0.495 | 0.140 | 0.400 |
| causal | 0.172 | 0.033 | 0.820 | 0.172 | 0.715 |
| random runs | 0.172 | 0.008 | 0.507 | 0.117 | 0.663 |
| std tol=3 | 0.310 | 0.0003 | 0.495 | 0.293 | 0.826 |

Verdict: the window mode localises anomalies in hindsight far better than chance; most of its recall is
look-ahead (ADTQC). Causally, recall is at chance; affiliation and event-wise F0.5 stay above every control.
Random-tick controls were dropped for these runs (Amendment 6); random_s0 (ticks) was scored before that.
NOTE: the "causal" row above is the first causal version, which could decide inside its own baseline (fixed
in ce79d3f). Superseded by Amendments 7-8 below.

## Amendments 7-8 (2026-09-26): fixed causal mode, circular-shift controls, tuning check

Control from Amendment 8 on: the whole prediction matrix is shifted by one offset (10-90% of the length), the same
for every channel, 20 seeds. p = (1 + #shifts >= observed) / 21, floor 0.048. `logs/amend8.log`,
`logs/summary_amend8_9.log` (from `summarize_controls.py`). Window-mode predictions are unchanged by the fix
(WINDOW_IDENTICAL True, 1,316,864 flags). Anomaly labels, max over 20 shifts:

| | EW recall | EW F0.5 | AFF F0.5 | channel F0.5 | ADTQC |
|---|---|---|---|---|---|
| window (default) | 0.862 | 0.179 | 0.816 | 0.861 | 0.099 |
| shifts, max | 0.138 | 0.026 | 0.506 | 0.125 | |
| causal v2 | 0.103 | 0.028 | 0.749 | 0.103 | 0.859 |
| shifts, max | 0.103 | 0.026 | 0.485 | 0.100 | |

- Causal v2 ties the best shift on EW recall. Under the older random-runs control it sits below 4 of 5 placements
  (0.138), so there is no recall claim for causal mode. Affiliation F0.5 beats every shift and every random-runs
  control.
- Amendment 7 rerun: grid of 12 (window 128-1024, threshold 2.5/3.5/5.0, causal), selection on train by channel
  F0.5 picked w128 t2.5. On test it beats its shifts on no metric (smallest p 0.095). Tuning does not help;
  defaults kept.

## Amendments 9-10 (2026-09-26): Guard with quarantine recovery (f1f00f2) and recovery back-off (a49534a)

Same Guard protocol as Amendment 3 (N=20160, cooldown 50). Wheel for a49534a pre-checked with `backoff_check.py`
(flapping sensor: f1f00f2 37 quarantined / 36 released, a49534a 2 / 1). `logs/amend9.log`, `logs/amend10.log`,
`logs/summary_amend10.log`.

| event counts | f1f00f2 | a49534a |
|---|---|---|
| alarm | 2466 | 391 |
| quarantined / unquarantined | 2568 / 2564 | 370 / 366 |
| repeated_value / parity alarms | 2356 / 33 | 341 / 0 |

Anomaly labels: EW recall 0.103 both, EW F0.5 0.0015 -> 0.0099, alarming precision 0.130 -> 0.167, AFF F0.5
0.401 -> 0.410, channel F0.5 0.087 -> 0.082. a49534a vs 20 shifts: EW recall, EW F0.5, channel F0.5 beat all
(p 0.048); AFF F0.5 p 0.095; alarming precision p 0.52. RareEvent+Anomaly: EW recall, EW F0.5, AFF F0.5 beat
all 20 (p 0.048); channel F0.5 p 0.050 (19 shifts, seed 10 not computed: memory guard). f1f00f2 has only 9 shifts
scored (the scorer was reaped under memory pressure; not retried). Parity drops to 0 because the monitor skips
parity for all channels while any channel is quarantined (owner's reading of monitor.rs, not verified here).

## Amendment 11 (2026-09-26): guard on 38cb83e (recalibration fixes), ported scripts

Run with the struktura repo's esa-adb/eval scripts (validated against this pipeline on d439fb8: identical prep,
predictions and shifts, 125/126 metric values equal, 0 different). Wheel built from 38cb83e, installed binary
hash-checked against the build. `run-38cb83e/run.log`, `run-38cb83e/summary.md`.
Events a49534a -> 38cb83e: alarm 391 -> 136, rolled_back 8 -> 0, quarantined 370 -> 138, unquarantined
366 -> 134, adaptation_started 29 -> 10, recalibrated 21 -> 7; legs repeated_value 341 -> 104, level_shift
40 -> 9, residual_cusum 14 -> 18, dfa 3 -> 0, parity 0 -> 4.
vs 20 circular shifts: Anomaly EW recall 0.103 (3/29, max shift 0.034), EW F0.5 0.035, AFF F0.5 0.429
(max 0.421), channel F0.5 0.085: all p 0.048; alarming precision 0.167 p 0.333. RareEvent+Anomaly EW recall
0.169 (11/65, max 0.031), EW F0.5 0.111, AFF F0.5 0.479, channel F0.5 0.124: all p 0.048; alarming precision
0.289 p 0.714. ADTQC 0.992 / 0.826 (undefined for most shifts, not compared).

## Amendment 12 (2026-09-26): guard on ccd86c5 (DFA flat-box fix, branch fix/dfa-flat-boxes)

Wheel built from ccd86c5, installed binary hash-checked against the build; the installed struktura.dfa on the
known flat-edge window (test row 2535177, channel 42, W=256) gives alpha 1.612071 / r2 0.835197 where 797d785
gave -1.606352 / 0.023052, so the fix is in the module. Guard prediction matrix byte-identical to 38cb83e
(136 alarm ticks, 0 cells differ), event kinds and alarm legs identical: the DFA leg raised no alarm on this data
before or after. Per the amendment, nothing re-scored; the page's 38cb83e scores hold for ccd86c5.
`run-ccd86c5/guard.log`, `run-ccd86c5/compare.log`.

## Amendment 13 (2026-09-26): guard on branch fix/parity-isolation 45b92ec (not on master)

Fresh clone, installed .pyd hash-checked; class counts from `guard_classes.py`, each run with its own hash-checked
wheel. Prediction matrix differs from master (ccd86c5): 228 vs 136 alarm ticks, 196 cells.
Master -> 45b92ec: alarm 136 -> 228; quarantined 138 -> 232 (channel 45: 48 -> 146, others within 3);
unquarantined 134 -> 229; adaptation_started 10 -> 9; repeated_value/stuck 104 -> 201; parity 4
cross_channel_inconsistency -> 2 cross_channel_ambiguous (reported channels 41 and 43); level_shift 9 -> 8;
residual_cusum 18 -> 15; residual 1 -> 2. Alarm ticks per channel 41..46 [42,5,23,12,41,13] -> [42,5,20,11,140,10].
Official metrics vs 20 regenerated shifts (master in brackets): Anomaly EW recall 0.103 (0.103) p 0.048, EW F0.5
0.018 (0.035) p 0.048, channel F0.5 0.082 (0.085) p 0.048, AFF F0.5 0.362 (0.429) p 0.238 (was 0.048), alarming
precision 0.167 (0.167) p 0.476. RareEvent+Anomaly EW recall 0.154 = 10/65 (0.169 = 11/65) p 0.048, EW F0.5 0.056
(0.111), AFF F0.5 0.403 (0.479) p 0.048, channel F0.5 0.111 (0.124), alarming precision 0.303 (0.289) p 0.381.
ADTQC 0.992 / 0.895. Under the rule stated before the run (stays off master if quarantines or false alarms get
worse or recall drops): quarantines +94, alarms +92, R+A recall -1 event, so it stays off master on this data.
`run-45b92ec/{guard.log,compare.log,classes.log,summary.md,run.log}`.

Follow-up (maintainers asked whether the recovery back-off counter misbehaves on channel 45): every ch45
quarantine, release and alarm logged on master (ccd86c5, same predictions as 38cb83e) and on 45b92ec
(`ch45_timeline.py`), then the back-off rule of autopilot.rs (next_flaps, recovery_span) replayed over those
ticks (`ch45_replay.py`, `ch45_replay.log`). Master: 48 quarantines, 47 releases, 0 releases shorter than the
span the replayed flap count implies, 3 quick re-fails (healthy gaps 23, 61, 68 samples), median healthy gap
43,779 rows, median quarantine 192 samples (max 9,585). 45b92ec: 146 / 145, 0 short releases, 3 quick re-fails,
median healthy gap 11,729. The back-off works as designed: ch45 sticks once every ~10k-50k rows of this
forward-filled data, each healthy stretch is longer than the span, so the flap count resets and each stuck run
costs one quarantine of about 192 samples.
```
