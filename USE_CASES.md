# real-world use cases for DFA anomaly detection

struktura works on anything with a time axis. here's where people actually use DFA — with citations.

## 🚀 spacecraft telemetry

NASA JPL's MSL Curiosity team monitors temperatures, currents, voltages, and RF power with ML anomaly detection — reduced workload by 90%. DFA catches structural shifts their threshold system misses.

**what we proved:** detected Voyager 1's 2022 AACS anomaly from public data. F1=0.755 on NASA SMAP/MSL benchmark (zero training).

```
struktura voyager     # Voyager 1 anomaly
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

heart rate variability α is a clinical metric. healthy hearts: α ≈ 1.0 (complex fractal dynamics). congestive heart failure: α drops toward 0.5 (loss of adaptability). approved by ESC/AHA guidelines.

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
