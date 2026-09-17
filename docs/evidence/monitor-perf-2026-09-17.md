# monitor-perf evidence, 2026-09-17

- command: `struktura monitor-perf` (no flags; seed is fixed in src/bin/struktura.rs cmd_monitor_perf, no --seed option exists yet)
- binary: target/release/struktura.exe built 2026-09-17 14:46:34 from HEAD 2a821c2 (working tree had uncommitted src changes from the detector session; see git status at run time below)
- host: MINGW64_NT-10.0-26200 3.6.6-1cdd4371.x86_64 x86_64, rustc rustc 1.95.0 (59807616e 2026-04-14)

```
$ git status --short src/

$ struktura monitor-perf

  STREAMING MONITOR PERFORMANCE (flight-relevant timing)
  6 channels, window 96, DFA stride 2 — release build, this host

  Calibration (2048 samples x 6 ch): 6.5946ms
  Stream: 200000 samples x 6 channels
  Mean per-sample cost:           5210 ns
  P50 / P99 / P99.9:              5500 / 14100 / 44300 ns
  Raw max (includes OS jitter): 832700 ns
  Throughput:                      0.2 Msamples/s
  Alarms on clean stream:            0  (200000 samples ≈ 97.7x calib length)
  By leg (res/rep/dfa/level/cusum): 0/0/0/0/0

  Memory: fixed after calibration — per channel 96 + 96 f64 rings;
  no heap allocation in the push path.

  FAULT DETECTION (streaming monitor, 20 seeds, 2048-sample streams)
  | Fault Type         | Detect | Mean Latency |
  |--------------------|--------|--------------|
  | packet_loss        |  100%  |      6       |
  | spike              |  100%  |      1       |
  | stuck              |  100%  |      4       |
  | drift              |  100%  |    108       |
  | regime_shift       |  100%  |     45       |
  | mixed              |  100%  |      6       |
  | correlation_change |  100%  |      4       |

  FAULT-CLASS IDENTIFICATION (from alarm provenance, rule-based)
  | Fault Type         | Correct ID |
  |--------------------|------------|
  | packet_loss        |  100%      |
  | spike              |   90%      |
  | stuck              |  100%      |
  | drift              |  100%      |
  | regime_shift       |  100%      |
  | mixed              |  100%      |
  | correlation_change |   90%      |
  Misclassifications:
    spike -> regime_shift (2x)
    correlation_change -> spike (2x)

exit=0
```
