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

a bearing fault can change the structure of its vibration. on the bundled CWRU excerpts (normal vs inner-race fault) α drops from 0.689 to 0.183 while RMS amplitude rises 11%. these are two separate recordings, not a run to failure; on the IMS run-to-failure bearing a plain RMS threshold trips earlier than the α alarm, so this is not an early-warning result (REPRODUCIBILITY.md).

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

ionospheric scintillation disrupts GPS/Galileo signal structure, and DFA of receiver SNR is one way to look at it (see the citation). the example below is a synthetic demo; struktura has not been evaluated on real receiver data.

```
cargo run --example gnss_signal                  # synthetic demo
cargo run --example gnss_signal -- snr_log.csv   # your receiver data
```

**citation:** [Combined iCEEMDAN and VMD for GNSS Scintillation (Springer, 2021)](https://link.springer.com/article/10.1007/s11600-021-00629-y)

## 📖 text & writing rhythm

prose can have long-range correlations in sentence lengths, and shuffling the sentences destroys that order while keeping the length distribution, which is how to check that α measures sequence and not statistics. the bundled sample is already shuffled Austen (α=0.573); the unshuffled original is not shipped, and no human-vs-AI comparison has been measured.

```
struktura text novel.txt
```

## 📈 financial regime detection

α distinguishes trending (α > 0.6), mean-reverting (α < 0.4), and random walk (α ≈ 0.5) regimes. regime shifts = α shifts.

```
struktura market prices.csv
```

## 🧬 genome sequences

chromosomal DNA has long-range correlation structure in base composition. bundled human chr1, GC% in 1 kb windows: α=0.967, R²=0.980 (docs/claims.tsv: genome-chr1-alpha). one chromosome is bundled; other chromosomes are not measured here.

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
