# struktura vs. the alternatives

An honest comparison for anyone evaluating tools for structural/DFA-based
anomaly detection or Hurst-exponent estimation. "not verified" means we did
not independently confirm the cell — check the linked source yourself.

| tool | language | `no_std` | zero-config | training needed | speed claim (source) | license |
|---|---|---|---|---|---|---|
| **struktura** | Rust (`no_std` + `alloc`) | yes | yes | no† | ~85–112x vs. a published `nolds` reference timing — not run head-to-head; `speed_bench` only times the Rust side (this repo, `cargo run --release --example speed_bench`; see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md)) | MIT OR Apache-2.0 |
| [nolitsa](https://github.com/manu-mannattil/nolitsa) | Python (NumPy/SciPy/Numba) | no | yes (function calls, no CLI) | no | not verified — no published benchmark found | BSD-3-Clause |
| [nolds](https://github.com/CSchoel/nolds) | Python (NumPy) | no | yes | no | not verified — this is the baseline struktura benchmarks against, not an independent third-party number | MIT |
| [hurst](https://github.com/Mottl/hurst) (PyPI `hurst`) | Python | no | yes | no | not verified | MIT |
| [hurst](https://github.com/evrom/hurst) (Rust crate) | Rust | not verified | yes | no | not verified | GPL-3.0 (derived from R `pracma`) |
| [MFDFA](https://github.com/LRydin/MFDFA) | Python | no | yes | no | not verified | MIT |
| scikit-learn `IsolationForest` | Python (Cython/C core) | no | no — needs `contamination`/tree params tuned | yes, unsupervised but fits a model per dataset | not verified | BSD-3-Clause |
| [ADTK](https://github.com/arundo/adtk) | Python | no | partial — rule-based detectors need thresholds set | no (rule-based) | not verified | MPL-2.0 |
| [Merlion](https://github.com/salesforce/Merlion) | Python | no | no — model selection + config required | yes, most detectors are trained/fitted | not verified | BSD-3-Clause |
| [NASA telemanom](https://github.com/khundman/telemanom) | Python (TensorFlow/Keras) | no | no — LSTM must be trained per channel | yes, supervised LSTM training required | not verified | Apache-2.0 |

## Notes

- **`no_std`**: struktura's core DFA/monitor logic builds with `default-features = false` (see `Cargo.toml`) for embedded targets, but still requires `extern crate alloc` and a global allocator (`no_std` ≠ allocator-free) — `dfa_into()` avoids reallocating on the hot path by writing into a caller-supplied `Vec` buffer, it does not avoid the allocator dependency. None of the Python tools apply here (Python has no `no_std` concept); the Rust `hurst` crate's `no_std` status was not verified — check its source before relying on it. The separate flight-computer variant (`src/rover_flight.rs`) is genuinely allocation-free (fixed arrays only, ~5.9KB) but drops the DFA leg, keeping only AR(1) prediction-residual, stuck-value, and rolling-mean-shift detectors.
- **Zero-config**: struktura self-calibrates thresholds from the signal itself (see [GUARANTEES.md](../GUARANTEES.md) — "does NOT train on data, no hyperparameters to tune"). `IsolationForest`, `Merlion`, and `telemanom` all require choosing/fitting model parameters per dataset; ADTK is rule-based and needs domain thresholds set per detector.
- **Speed**: the only speed number we independently reproduce is struktura's own timing; the `nolds` side is a published reference figure, not measured head-to-head in this repo's benchmark (`examples/speed_bench.rs` only calls the Rust `dfa()` in a loop and prints hard-coded "typical" Python numbers as a comment). We did not benchmark against the other tools in this table — treat any speed comparison beyond that as unverified until someone runs it.
- †**training needed**: the core DFA/monitor API fits no model. The `smap --ar` NASA benchmark mode is an exception — it fits a per-channel AR predictor by closed-form ridge least squares on the train split (no gradient descent, no GPU) before scoring the test split; see `src/smap_eval.rs`.
- **DFA vs. general anomaly detection**: `IsolationForest`, ADTK, Merlion, and telemanom solve a broader/different problem (point or forecast-residual anomalies) than DFA specifically. They are included because someone evaluating "anomaly detection" tooling will likely compare against them, not because they use the same algorithm.
- This table will drift as upstream projects change. If you find a stale or wrong cell, open an issue.
