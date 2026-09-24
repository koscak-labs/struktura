# struktura (Python)

Python access to [struktura](https://github.com/koscak-labs/struktura), a Rust crate for
time-series anomaly detection without training data. Every call runs the Rust
code; the Python package only converts arguments.

```python
import struktura

r = struktura.dfa_short(values)        # DFA alpha, works from about 24 samples; None if unmeasurable
print(r.alpha, r.r_squared)

m = struktura.Monitor([ch0_clean, ch1_clean])   # calibrate on clean data, one list per channel
for sample in stream:                           # one value per channel
    if (leg := m.push(sample)) is not None:
        print(leg, m.last_alarm())
```

Build from source (needs a Rust toolchain):

```
pip install maturin
maturin build --release        # run inside crates/struktura-py
```
