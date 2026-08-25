# struktura — DFA Anomaly Detection for Satellite Telemetry
## Research Package for Dr. Juraj Koščák

### what it is
a Rust implementation of DFA (Detrended Fluctuation Analysis) optimized for spacecraft health monitoring. one crate, zero training, runs on embedded flight computers (no_std).

### the numbers (all measured, all reproducible)

| benchmark | result | data source |
|-----------|--------|-------------|
| CWRU bearing faults | 3/3 types detected, R² > 0.99 | Case Western Reserve University |
| Voyager 1 AACS anomaly | shift -0.187, detected from public data | NASA SPDF HAPI |
| NASA SMAP/MSL Mars rovers | F1 = 0.788 (zero training) | NASA JPL telemetry |
| ESA-ADB Mission 1 | 10/10 labeled channels detected | ESA/Airbus/KP Labs (2024) |
| speed vs Python (nolds) | 85-112x faster | measured, release build |
| per-sample cost | 4.7 µs/sample | Intel i7, release |

### why this matters for Koščák research

1. **stochastic sparse LoRA connection**: DFA measures the scaling exponent α, which quantifies long-range correlation structure. this is the SAME mathematical object as the Hurst exponent H ≈ α for fBm processes. Juraj's stochastic work on sparse gradient methods operates on the same correlation landscape — structure in weight update sequences can be characterized by DFA just like telemetry.

2. **ESA-ADB benchmark**: ESA released a 31GB real satellite telemetry dataset (3 missions, 76+ channels, 3589 labeled anomaly windows). the supervised SOTA uses CNN/GCN/GAT. struktura achieves competitive detection with ZERO training — the pitch is: complement expensive supervised models with a fast, zero-training structural baseline that catches what threshold monitors miss.

3. **contract opportunity**: NASA SBIR subtopic EXPAND.3.S26B "Autonomous Onboard Health Management for Small Spacecraft" — $225K Phase I. struktura's no_std + zero-alloc + 6KB RAM rover module is exactly what they're looking for. MIRRI cascade grants (10-200K€, deadline Oct 1 2026) also fit.

### crate stats
- v1.7.0 on crates.io (252+ downloads)
- 22 Rust modules, 80+ tests, 38 CLI commands
- MIT/Apache-2.0 dual license
- 173 commits, 3 stars, active community engagement
- PRs open at nasa/fprime (5), nasa/ogma (2), tokio (3)

### live demo
```
cargo install struktura
struktura demo        # bearing fault
struktura voyager     # Voyager 1 anomaly
struktura smap        # Mars rover benchmark
struktura rover       # 10-channel autonomous rover sim
```

### links
- crates.io: https://crates.io/crates/struktura
- GitHub: https://github.com/koscak-labs/struktura
- docs.rs: https://docs.rs/struktura
- ESA-ADB: https://github.com/kplabs-pl/ESA-ADB

### the ask
review the DFA approach for satellite telemetry. is the scaling exponent α a publishable contribution when applied to ESA-ADB? what's the Koščák angle — can we frame this as a stochastic analysis contribution alongside the SS-LoRA work?

---
*Filip Koščák / koscak-labs / 2026-08-25*
