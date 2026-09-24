<p align="center">
  <img src="assets/social-preview-readme.jpg" alt="struktura — detrended fluctuation analysis anomaly detection for Rust" width="100%">
</p>

# struktura

**Detrended fluctuation analysis (DFA) for Rust: find the moment a time series' structure changed — no thresholds, no training. `no_std` + `alloc` from the next release (1.7.2 on crates.io does not build for `default-features = false` dependents; see [features](#-features)).**

<p align="center">
  <a href="https://crates.io/crates/struktura"><img src="https://img.shields.io/crates/v/struktura.svg" alt="Crates.io"></a>
  <a href="https://github.com/koscak-labs/struktura/actions/workflows/ci.yml"><img src="https://github.com/koscak-labs/struktura/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/koscak-labs/struktura"><img src="https://img.shields.io/crates/l/struktura.svg" alt="License"></a>
  <a href="https://docs.rs/struktura"><img src="https://docs.rs/struktura/badge.svg" alt="docs.rs"></a>
</p>

---

is your data broken? find out in one command. zero config. no model training: baselines are estimated from the signal itself (self-calibrating). one exception: the SMAP/MSL benchmark evaluation fits a closed-form per-channel AR predictor on the train split.

```
$ cargo install struktura
<!-- example:guard-sylv-spike -->
$ struktura guard data/sylv_spike.csv
struktura guard: 1000 samples x 1 channels, calibrated on 333 rows
  row    500  ⚠ ch0 (4.6x threshold): gradual drift — the signal is trending away from its baseline
  row    551  ⚠ ch0 (1.7x threshold): the signal's pattern is changing slowly (structural drift)
  2 faults detected across 667 samples (0 adaptations, 0 quarantines)
$ echo $?
1
<!-- /example -->
```

exit code 0 = healthy, 1 = fault detected, 2 = error. `data/sylv_spike.csv` ships in the repo, so this run is reproducible as-is.

## use in CI

```yaml
- uses: koscak-labs/struktura@v1
  with:
    file: data/telemetry.csv
    fail-on-fault: true
```

fails the step when a fault is detected, and writes the report to the job summary. see [action.yml](action.yml) for all inputs.

**who this is for:** you have a time series and no labeled faults to train on. concretely:
- **spacecraft telemetry anomaly detection** — reaction wheels, magnetometers, batteries (tested on NASA/ESA data)
- **bearing fault detection in Rust** — vibration analysis for rotating machinery (CWRU + NASA IMS datasets)
- **time series health monitoring** for DevOps — latency, error rate, throughput drift, piped from stdin
- **cardiac HRV monitoring** — heart-rate-variability structure per ESC/AHA-referenced α metric
- **embedded / `no_std` anomaly detection** — zero-alloc flight monitors that compile to C99, ~5.9KB RAM (no DFA leg — AR(1) prediction-residual, stuck-value, and rolling-mean shift detectors only)

**what makes it different:** it calibrates itself, quarantines dead sensors autonomously, adapts to environment changes, and tells you in plain english what happened. no thresholds to set. no model to train. the DFA computation itself measures 85-112x faster in Rust than a published Python (`nolds`) reference timing (not run head-to-head here — see [benchmarks](#speed-benchmarked-not-guessed)). one required dependency (`libm`). `no_std` + `alloc` on this branch: `crate-type = ["lib"]`, an allocation-free `dfa_scratch`, and a CI job in which an external consumer of the packaged crate builds on `thumbv7em-none-eabihf` with its own allocator and panic handler and calls the DFA path (logs in [docs/evidence](docs/evidence/)). 1.7.2 on crates.io does not build for `default-features = false` dependents (observed 2026-09-17: "no global memory allocator found but one is required", "`#[panic_handler]` function required, but not found", "unwinding panics are not supported without std").

built while contributing to [NASA F´ flight software](https://github.com/nasa/fprime). tested against real NASA (SMAP/MSL, IMS bearing, Voyager) and ESA (ESA-ADB) spacecraft telemetry datasets — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact command behind every number below.

<p align="center">
  <img src="assets/terminal-demo.svg" alt="struktura demo" width="100%">
</p>

## 📋 use cases

| you have | you run | you get |
|---|---|---|
| server metrics CSV | `struktura guard metrics.csv` | live anomaly alerts with the observed/threshold ratio, e.g. "(4.6x threshold)" |
| "when did performance change?" | `struktura when latency.csv` | the sample index and z-score of each structural shift, or "no structural changes detected" (needs ≥ 6144 samples) |
| two datasets to compare | `struktura compare before.csv after.csv` | "shift: -0.187 … spread: baseline ±0.142 current ±0.199 … (z=1.5: within the signals' own variability, treat as inconclusive)"; z ≥ 3 means the shift is outside both signals' own variability |
| factory sensor log | `struktura guard --watch machine.csv` | real-time monitoring, auto-adapts |
| healthy baseline to certify | `struktura stamp baseline.csv` | self-certifying CSV (carries its own fingerprint) |
| rover/IoT/embedded | `struktura generate --rover` | zero-alloc C99 flight monitor, ~5.9KB RAM, no DFA (AR(1) prediction-residual + stuck-value + rolling-mean-shift detectors) |

## 🤖 autonomous operation

`struktura guard` doesn't just detect — it makes decisions:

```
<!-- example:guard-rover -->
$ struktura guard examples/rover.csv --baseline 1000
struktura guard: 3000 samples x 5 channels (motor_current_A, wheel_rpm, imu_accel_g, battery_soc, temp_motor_C), calibrated on 1000 rows
  row   1644  ⚠ wheel_rpm (1.3x threshold): the signal's behavior changed — predictions are failing
  row   1715  ⚠ imu_accel_g (1.1x threshold): gradual drift — the signal is trending away from its baseline
  row   1725  ⚠ imu_accel_g (1.2x threshold): the signal shifted to a new operating level
  row   1725  ↻ environment may have changed — learning new baseline...
  row   2200  ✗ not a real environment change — fault confirmed
  row   2201  ⚠ motor_current_A (15.1x threshold): gradual drift — the signal is trending away from its baseline
  row   2211  ⚠ motor_current_A (1.7x threshold): this channel disagrees with what the other channels' physics says it should be
  row   2211  ✗ motor_current_A declared dead — using reconstructed values
  row   2215  ⚠ wheel_rpm (2.0x threshold): the signal's behavior changed — predictions are failing
  6 faults detected across 2000 samples (0 adaptations, 1 quarantines)
<!-- /example -->
```

(full output — nothing omitted; `examples/rover.csv` ships in the repo.)

dead sensor? quarantined, its reading reconstructed from the
other channels' physics (R² > 0.9). environment permanently changed? it
re-learns "normal" through a guarded candidate — and rolls back if the new
regime is actually a fault trying to sneak in. drift disguised as a regime
change? refused. every decision above was made by the monitor alone.

## 🧬 it evolved its own detectors

`struktura evolve` — an adversarial RED/BLUE loop: RED invents faults the
monitor misses, BLUE synthesizes new detector legs from a grammar, accepted
only with ZERO false alarms on clean data:

| generation | fault coverage | detector legs |
|---|---|---|
| 1 | 71% | 2 |
| 4 | 89% | 5 |
| 9 | **97%** | 6 |

the machine independently invented variance monitors, residual-trend
monitors, and derivative-volatility monitors — detector classes nobody
hand-coded. parameter tuning alone plateaued at 75%; structural synthesis
broke through.

## 🛡️ the receipts

- **7/7 telemetry fault taxonomy detected by the hybrid monitor** (packet
  loss, spike, stuck, drift, regime shift, mixed, correlation change) —
  DFA catches structural faults; residual-based legs catch value faults.
  neither alone covers all 7; the combination does. self-calibrated
  thresholds, **0 false alarms on a single synthetic 200,000-sample
  6-channel spacecraft stream** (`struktura monitor-perf`; EVT-calibrated
  on a separate 2,048-sample calibration stream from the same generator;
  re-run on the current binary 2026-09-17, full output in
  [docs/evidence/monitor-perf-2026-09-17.md](docs/evidence/monitor-perf-2026-09-17.md).
  one fixed seed, synthetic data, separate calibration: zero observed alarms
  on that stream is the result; it is not a zero false-alarm rate, and no
  confidence bound is claimed because streaming alarm decisions over
  overlapping windows are not independent trials)
- **`struktura when` controls (2026-09-17, rebuilt detector):** 0 changes
  reported on four stationary controls (shuffled, AR 0.7, AR 0.95, 1/f
  noise, 123K samples each); the white|walk|white positive control lands
  at exactly 8192 and 16384
- **NASA IMS bearing run-to-failure: alarm at recording 970 of 984**
  (about 2 h before the test ended; α spikes from 0.17 to 0.53). A plain
  RMS amplitude threshold trips earlier on the same bearing, so this is
  not an early-warning result
- **generated C99 that compiles clean under `-Wall -Werror`** with a
  self-test: `struktura generate-hybrid` bakes your calibration into a
  dependency-free monitor that detects a stuck sensor in its own
  self-test — not mission-qualified
- every claim above → one command: see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## ⚡ speed (benchmarked, not guessed)

**timings vary by machine and load — not covered by the fixture-checked examples in `docs/examples/`; treat the numbers below as indicative, not reproducible byte-for-byte.**

| signal size | struktura (rust, measured) | python nolds (typical, not run here) | speedup |
|---|---|---|---|
| 4,096 pts | **0.24 ms** | ~15-25 ms | **~85x** |
| 16,384 pts | **0.93 ms** | ~60-100 ms | **~86x** |
| 65,536 pts | **2.89 ms** | ~250-400 ms | **~112x** |

the Rust column is measured by `speed_bench`; the Python column is a published
reference figure, not run head-to-head in this benchmark — see
[examples/speed_bench.rs](examples/speed_bench.rs).

at 1Hz spacecraft telemetry: 0.24ms per analysis = **4,000 channels on one core**. yeah.

reproduce it yourself: `cargo run --release --example speed_bench`

## 🏆 head-to-head: struktura vs alternatives

tested on real datasets against [ankane/AnomalyDetection.rs](https://github.com/ankane/AnomalyDetection.rs) (STL decomposition) and classic 3σ threshold:

| dataset | struktura | ankane (STL) | threshold (3σ) |
|---------|-----------|-------------|----------------|
| **IMS bearing failure** (NASA) | YES, 46μs | YES, 696μs | YES, 0μs |
| **Voyager heliopause** (NASA) | flagged, 570μs (α 1.137 → 1.056, z = 0.6: inconclusive on these slices) | no | YES, 0μs |
| **synthetic correlation shift** | no | no | no |

on the IMS bearing failure both detect it, struktura ~15x faster. on the Voyager heliopause slices bundled here (3,988 and 4,404 rows) struktura flags an α shift of -0.081 that its own subsampling z (0.6) calls inconclusive, so that row is a labelled comparison, not a demonstrated detection; a longer pre/post window is the open test. on the synthetic correlation change, nobody detects via simple API (that's the honest frontier we're working on). threshold is fastest but only catches amplitude anomalies, not structural changes. CRITICAL/WARNING labels from `check`/`compare` are fixed α thresholds, not significance — quote the z on the same line.

different tools for different fault types. DFA detects when the STRUCTURE of a signal changes — not when values go out of bounds. ankane/threshold catch amplitude outliers. use both.

reproduce: `cargo run --release --example comparison`

## 🔧 as a library

```rust
use struktura::{analyze, health_check, HealthVerdict};

let law = analyze(&sensor_data);
let verdict = health_check(&law, baseline_alpha);
// Healthy | Watch | Warning | Critical
```

streaming monitor for production:

```rust
use struktura::BaselineTracker;

let mut monitor = BaselineTracker::new(256, 1000);
for sample in telemetry_stream {
    if let Some(verdict) = monitor.push(sample) {
        match verdict {
            HealthVerdict::Critical => trigger_alert(),
            _ => {}
        }
    }
}
```

spacecraft-specific stuff:

```rust
use struktura::space::{SpacecraftMonitor, Subsystem};

let mut rwa = SpacecraftMonitor::new(Subsystem::ReactionWheel, "RWA_current");
// push samples, get verdicts
```

## 🎯 what it detects (all verified, all real data)

| domain | signal | healthy α | fault α | shift | verdict |
|--------|--------|-----------|---------|-------|---------|
| 🔧 **bearings** | CWRU 12kHz vibration | 0.689 | 0.183 | -0.506 | CRITICAL |
| 🚀 **spacecraft** | Voyager 1 magnetometer, 2021 vs 2022 slices (not the AACS anomaly window; z=1.5, inconclusive) | 0.989 | 0.801 | -0.187 | CRITICAL |
| 🛰️ **ESA satellites** | ESA-ADB Mission 1 | — | — | — | adapter built ([esa-adb/](esa-adb/struktura-dfa/)); PA%K scores pending benchmark run |
| 📖 **text** | Austen vs shuffled | 0.749 | 0.572 | -0.177 | detected |
| 🧬 **genome** | Human chr1 GC% | 0.909 | | | R²=0.991 |
| ❤️ **cardiac** | HRV RR intervals | 0.695 | | | R²=0.985 |

every number from an actual run. reproduce with `struktura demo` / `struktura voyager`.

see [USE_CASES.md](USE_CASES.md) for the full list with citations.

## 🪐 time series health monitoring for Mars rover anomaly detection (NASA SMAP/MSL)

tested on the real NASA SMAP/MSL telemetry benchmark (82 labeled anomaly channels from Mars rovers + soil moisture satellite). no model training: a per-channel AR predictor is fit by closed-form ridge least squares on the train split (no gradient descent, no GPU); zero tuning.

```
<!-- example:smap -->
$ struktura smap --ar 0 --dfa

  NASA SMAP + MSL/CURIOSITY ANOMALY BENCHMARK
  Real spacecraft telemetry, JPL-labeled anomalies (telemanom dataset).
  Protocol: calibrate on the nominal train split, stream the test
  split; a labeled sequence is DETECTED if any alarm lands inside it;
  alarms outside every labeled sequence count as false positives.
  ================================================================

  | Spacecraft | Channels | Sequences | Detected | FP alarms | Precision | Recall |
  |------------|----------|-----------|----------|-----------|-----------|--------|
  | SMAP       |   54 (1sk) |        69 |       42 |         5 |     89.4% |  60.9% |
  | MSL        |   26 (1sk) |        36 |       16 |         9 |     64.0% |  44.4% |
  |------------|----------|-----------|----------|-----------|-----------|--------|
  | TOTAL      |          |       105 |       58 |        14 |     80.6% |  55.2% |

  Overall F1: 0.655   (JPL telemanom LSTM, same data: P=87.5% R=80.0%)
  Recall by anomaly class:
    [contextual]   9/17  =  52.9%
    [point]       38/45  =  84.4%
    contextual]"   8/30  =  26.7%
    point]"        3/11  =  27.3%
  Self-calibrated, no training, no GPU, ~4us/sample. Honest note: the
  values are pre-scaled to (-1,1) by JPL and many channels saturate,
  which auto-disables the repeated-value leg on those channels.

<!-- /example -->
```

F1 0.655 with `--ar 0 --dfa` (precision 0.806, recall 0.552) is the best configuration measured; the bare `struktura smap` scores 0.161 because its default residual leg raises 419 false-positive alarms on JPL's pre-scaled channels. Both are well below the JPL telemanom LSTM on the same data (P=87.5%, R=80.0%). This crate is not competitive with supervised models on SMAP/MSL; the value proposition is no model training, no GPU, and point-adjust honesty (a hit anywhere inside a labeled window counts), not raw detection accuracy on this benchmark. Earlier versions of this README quoted F1=0.788: that number could not be reproduced from any commit or flag combination tried on 2026-09-17 (see [docs/evidence/smap-f1-2026-09-17.md](docs/evidence/smap-f1-2026-09-17.md)); the block above is generated from `docs/examples/smap.cmd` and checked in CI.

## 📖 text analysis

DFA measures the fractal rhythm of writing. sentence lengths in human prose have long-range correlations that disappear when you shuffle them.

```
<!-- example:text -->
$ struktura text data/austen_shuffled.txt data/mechanical_text.txt

  TEXT STRUCTURE ANALYSIS
  DFA on sentence-length sequences — measures writing rhythm
  ====================================================================

  Reference: human prose α≈0.7-0.8 | shuffled/mechanical α≈0.5

  #################............. data/austen_shuffled.txt
                                 sentences=18049  mean_len=125  α=0.572  R²=0.9851
                                 MODERATE RHYTHM

  ################.............. data/mechanical_text.txt
                                 sentences=1000  mean_len=58  α=0.525  R²=0.9757
                                 UNIFORM/MECHANICAL

  ====================================================================
  α > 0.6 = long-range correlations in sentence rhythm (human writing)
  α ≈ 0.5 = random/shuffled/uniform sentence lengths
<!-- /example -->
```

the repo ships a shuffled Austen sentence-length corpus (`data/austen_shuffled.txt`, α=0.572, moderate rhythm) and a mechanical/uniform one (`data/mechanical_text.txt`, α=0.525); it does not ship an unshuffled original, so the three-way "original vs shuffled vs mechanical" comparison from an earlier draft of this README could not be reproduced from a real run and is not shown here. DFA catches sequential structure, not just statistics — see `docs/examples/text.cmd` to reproduce.

## 🛰️ spacecraft telemetry anomaly detection & health monitoring

built-in support for reaction wheels, magnetometers, batteries, thermal sensors, solar arrays, gyroscopes.

```
<!-- example:spacecraft -->
$ struktura spacecraft

  MULTI-CHANNEL SPACECRAFT HEALTH MONITOR
  DFA structural analysis across 4 telemetry channels
  (RWA/BAT/THM synthetic; MAG = real Voyager 1 data)
  ====================================================================

  ############################## [RWA:RWA_current]
                                 alpha=1.308 baseline=1.193 shift=+0.116 R²=0.9570  WARNING

  ############################## [BAT:BAT_voltage]
                                 alpha=1.980 baseline=1.883 shift=+0.097 R²=0.9993  WARNING

  ############################## [THM:THM_panel_A]
                                 alpha=1.888 baseline=1.815 shift=+0.073 R²=0.9944  WATCH

  ############################## [MAG:MAG_B_total]
                                 alpha=1.030 baseline=0.878 shift=+0.152 R²=0.9903  CRITICAL

  ====================================================================
  Each channel: first half = baseline, second half = current period.
  DFA reads structure, not level: the mean can look normal while
  alpha shifts. Whether that shift is a fault needs a control run.
<!-- /example -->
```

these are live alpha/verdict values from the synthetic RWA/BAT/THM channels plus real Voyager 1 magnetometer data; they shift run-to-run only if the underlying data files change (fixture: `docs/examples/spacecraft.cmd`). an earlier draft of this README showed different alpha values and verdicts (e.g. BAT as HEALTHY) that did not match a real run of the current binary.

`dfa_into()` avoids reallocating on the hot path by writing into a caller-supplied `Vec` buffer. `no_std` + `alloc` with `default-features = false`: on this branch `crate-type = ["lib"]`, `dfa_scratch(&[f64], &mut [f64])` is allocation-free, and CI builds an external consumer of the packaged crate on `thumbv7em-none-eabihf` (own `#[global_allocator]` and `#[panic_handler]`, calls `dfa_scratch` and `dfa_into`; logs in [docs/evidence](docs/evidence/)). 1.7.2 on crates.io does not build for such dependents (observed 2026-09-17: "no global memory allocator found but one is required", "`#[panic_handler]` function required, but not found", "unwinding panics are not supported without std"). no embedded flight computer has run this library.

## 🏗️ flight software codegen

generate complete monitoring apps for NASA flight frameworks:

```
struktura generate --cfs    --db channels.json -o dfa_cfs_app/
struktura generate --fprime --db channels.json -o dfa_fprime_component/
struktura generate --ros    --db channels.json -o dfa_ros_node/
```

compatible with [nasa/ogma](https://github.com/nasa/ogma) variable database format.

## 🧠 how detrended fluctuation analysis works — the Hurst exponent in Rust

DFA (Peng et al., Physical Review E, 1994, 3000+ citations) measures long-range correlation:

1. compute the cumulative profile (running sum minus mean)
2. divide into boxes, detrend each with a linear fit
3. measure residual fluctuation vs box size
4. slope in log-log space = α (the scaling exponent)

| α | what it means |
|---|---------|
| ~0.5 | random noise (no structure) |
| 0.5-1.0 | persistent correlations (healthy structure) |
| α shifts | something changed. go look. |

the crate reports R² alongside every α. if R² < 0.3, quality = `Abstain`. it never bluffs.

## 🧬 autonomous evolution (RED/BLUE)

the detection policy evolves itself. RED probes for faults the current config misses, BLUE mutates the policy and only keeps improvements that raise zero false alarms on clean data.

```
<!-- example:redblue -->
$ struktura redblue

  RED/BLUE ADVERSARIAL SELF-IMPROVEMENT
  RED probes the continuous fault space for misses; BLUE evolves the
  detection policy against the accumulated miss corpus under a
  ZERO-clean-alarm law. 6 rounds x 120 probes x 10 mutations.
  ================================================================

  | Round | RED coverage | New misses | Corpus cov. after BLUE | Evolved? |
  |-------|--------------|------------|------------------------|----------|
  |     1 |    60.0%     |         48 |              16.7%     | YES |
  |     2 |    70.8%     |         35 |              21.7%     | YES |
  |     3 |    68.3%     |         38 |              15.7%     | YES |
  |     4 |    74.2%     |         31 |              10.7%     | YES |
  |     5 |    64.2%     |         43 |               3.6%     |       no |
  |     6 |    75.0%     |         30 |               0.7%     |       no |

  RED coverage: 60.0% (round 1) -> 75.0% (round 6)
  Evolved config: res_span=20 dfa_persist=2 roll_persist=7 cusum_k=1.00 horizon=147553
  Zero-clean-alarm law verified on every accepted mutation.
<!-- /example -->
```

it finds its own blind spots and fixes them. no human tuning needed. this command is deterministic (seeded RNG — two independent runs produced byte-identical output) but slow, ~100s wall-clock on this machine; an earlier draft of this README claimed round-6 coverage reached 92% and `dfa_persist 5→2, roll_persist 10→7, horizon 1M→148K` — a real run tops out at 75% coverage with `dfa_persist=2 roll_persist=7 horizon=147553` (no observed starting value of `dfa_persist=5`).

## 🎮 full command list

38 commands. here are the highlights:

| command | what it does |
|---------|-------------|
| `demo` | bearing fault detection on real CWRU data |
| `voyager` | Voyager 1 magnetometer α, 2021 vs 2022 comparison |
| `smap` | NASA SMAP/MSL Mars rover benchmark |
| `spacecraft` | multi-channel spacecraft health monitor |
| `scan <file>` | auto-classify + show trend in one shot |
| `watch <file>` | live monitoring with auto-refresh |
| `text <file>` | writing rhythm analysis (human vs AI) |
| `market <file>` | financial regime detection |
| `genome <file>` | DNA sequence structure |
| `rhythm <file>` | event timing (commits, heartbeats) |
| `fingerprint <file>` | structural DNA of a signal |
| `redblue` | autonomous fault-coverage evolution |
| `evolve` | generational policy optimization |
| `mission` | full autonomous monitoring mission |
| `guard <file>` | prognosis: when will this cross the threshold? |
| `when <file>` | time-to-failure estimation |
| `batch *.csv` | CI/CD: analyze many files, JSON output |
| `alert <cmd>` | exit-code monitoring for cron/systemd |
| `self-test` | verify all claims against real data |
| `nasa` | run all NASA benchmarks |

| `pipe` | stream DFA from stdin (Prometheus, MQTT, tail) |

run `struktura --help` for the full list.

## 🐳 docker / python / devops

```bash
# docker — zero install
docker build -t struktura . && docker run -v ./data:/data struktura guard /data/sensor.csv

# python — ~85x faster than a published nolds reference timing (not run head-to-head)
pip install maturin && maturin develop --features python
python -c "import struktura; print(struktura.py_dfa([1.0]*256))"

# pipe anything through DFA
tail -f /var/log/metrics.csv | struktura pipe --json
curl prometheus:9090/query | struktura pipe --window 128

# cron alert with slack webhook
*/5 * * * * struktura guard /data/sensor.csv --webhook $SLACK_URL
```

see `examples/devops_integration.sh` for more.

## ⚠️ gotchas

stuff to know before you rely on this:

- **DFA catches structural shifts, not point anomalies.** a single spike won't move alpha much. use a residual detector alongside DFA for spike/outlier detection.
- **preprocessing changes alpha.** if you add a filter (notch, bandpass, artifact rejection) upstream, your baseline is invalid — recalibrate after any preprocessing change. ([#8](https://github.com/koscak-labs/struktura/issues/8))
- **alpha alone isn't a decision.** you still need to decide what "shifted enough" means for your domain. the `HealthVerdict` thresholds (0.03/0.08/0.15) are reasonable defaults, not universal truth.
- **F1 on SMAP/MSL is 0.655, not 0.95.** supervised models beat this. the value prop is no model training + speed + embedded, not raw detection accuracy.

## 🧰 features

- **adaptive box sizes** geometrically spaced to signal length
- **`no_std` + `alloc`** `default-features = false`; verified on this branch by a packaged-crate consumer build on `thumbv7em-none-eabihf` (see the library API note above). 1.7.2 on crates.io fails that build; use a git dependency or wait for the next release.
- **C FFI** `struktura.h` header, call from C/C++/whatever
- **serde** optional `features = ["serde"]`
- **self-test** `struktura self-test` verifies everything

## 🏆 alternatives

| crate | DFA | speed vs python (typical, not run head-to-head) | license | deps | `no_std` |
|-------|-----|----------------|---------|------|---------|
| **struktura** | native | **~85-112x** | MIT/Apache | **1** (libm) | **yes (+ alloc)** |
| anomaly_detection | no | | GPL-3.0 | many | no |
| extended-isolation-forest | no | | MIT | many | no |

## 📚 references

1. C.-K. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. C.-K. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
4. NASA SPDF Voyager Data: https://spdf.gsfc.nasa.gov/pub/data/voyager/

## license

MIT OR Apache-2.0

---

<p align="center">
  if this is useful to you, <a href="https://github.com/koscak-labs/struktura">⭐ star it</a>. if it's not, open an issue and tell me why 🫡
</p>
