# real-world use cases for DFA anomaly detection

struktura works on anything with a time axis. here's where people actually use DFA — with citations.

## 🚀 spacecraft telemetry

NASA JPL's MSL Curiosity team monitors temperatures, currents, voltages, and RF power with ML anomaly detection — reduced workload by 90%. DFA catches structural shifts their threshold system misses.

**what we proved:** F1=0.655 on NASA SMAP/MSL benchmark (no model training: a per-channel AR predictor is fit by closed-form ridge least squares on the train split; a detection counts as a hit anywhere inside the labeled anomaly window). On public Voyager 1 magnetometer data, α differs between 2021 and 2022 (0.989 vs 0.801 on the bundled demo slices, z=1.5, inconclusive by struktura's own subsampling test); the 2022 AACS anomaly window itself is not a DFA shift (during-vs-pre p=0.52), so this is a year-over-year structural comparison, not anomaly detection.

```
struktura voyager     # Voyager 1 magnetometer α, 2021 vs 2022
struktura smap        # Mars rover benchmark
struktura spacecraft  # multi-channel monitor
```

**citation:** [MSL Telecom Automated Anomaly Detection (NASA TRS, 2022)](https://ntrs.nasa.gov/citations/20220000760)

## 🔧 bearing & rotating machinery

DFA scaling exponent increases consistently toward bearing failure — proven on the CWRU bearing dataset (3 fault types, all detected, R² > 0.99). this is predictive maintenance: know the bearing is degrading before it fails.

```
struktura demo                    # built-in CWRU data
struktura check vibration.csv     # your own sensor
```

**citation:** [Early Warning Signals for Bearing Failure Using DFA (MDPI, 2020)](https://mdpi.com/2076-3417/10/23/8489/htm)

## ❤️ cardiac health (HRV)

the DFA exponent of beat-to-beat (RR) intervals is a heart-rate research measure. in heart failure the short-term exponent (alpha1, about 4-16 beats) is lower than in healthy hearts, while the long-term exponent changes much less (Peng et al. 1995, Chaos 5:82). it does not drop to 0.5. the example below is a DFA demo, not a medical device.

```
cargo run --example cardiac_hrv                    # synthetic demo
cargo run --example cardiac_hrv -- apple_hrv.csv   # your export
```

**citation:** Peng et al., "Long-range anticorrelations and non-Gaussian behavior of the heartbeat," Physical Review Letters 70(9), 1993.

## 🛰️ GNSS satellite signals

DFA detects ionospheric scintillation — when the atmosphere disrupts GPS/Galileo signal structure. catches positioning degradation before the receiver reports errors.

```
cargo run --example gnss_signal                  # synthetic demo
cargo run --example gnss_signal -- snr_log.csv   # your receiver data
```

**citation:** [Combined iCEEMDAN and VMD for GNSS Scintillation (Springer, 2021)](https://link.springer.com/article/10.1007/s11600-021-00629-y)

## 📖 text & writing rhythm

human prose has long-range correlations in sentence lengths (α ≈ 0.7-0.9). AI-generated text has different structure. shuffling destroys it — proof that DFA measures sequential organization, not statistics.

```
struktura text novel.txt
```

## 📈 financial regime detection

α distinguishes trending (α > 0.6), mean-reverting (α < 0.4), and random walk (α ≈ 0.5) regimes. regime shifts = α shifts.

```
struktura market prices.csv
```

## 🧬 genome sequences

chromosomal DNA has long-range correlation structure. α > 0.5 across all tested chromosomes (8 chromosomes, R² > 0.99). structural patterns correlate with functional boundaries.

```
struktura genome sequence.fa
```

## 🏭 DevOps / production monitoring

pipe any metric through DFA. catches structural degradation in latency, error rates, throughput — before thresholds fire.

```
tail -f metrics.csv | struktura pipe --json
struktura guard sensor.csv --watch --webhook $SLACK_URL
```

---

**the pattern:** DFA works because structural shifts are universal. a bearing, a heart, a spacecraft, a market — they all have correlation structure that changes when something goes wrong. one algorithm, many domains.
