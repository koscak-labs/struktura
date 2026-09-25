# struktura (Python)

Python access to [struktura](https://github.com/koscak-labs/struktura), a Rust crate for
time-series anomaly detection without training data. Every call runs the Rust
code; the Python package only converts arguments.

```python
import struktura

r = struktura.dfa_short(values)        # DFA alpha, works from about 24 samples; None if unmeasurable
print(r.alpha, r.r_squared)

g = struktura.Guard([ch0_clean, ch1_clean])     # calibrate on clean data, one list per channel
for sample in stream:                           # one value per channel; NaN = missing reading
    for event in g.push(sample):                # [] in the steady state
        print(event["tick"], event["kind"], event.get("explanation"))
```

`Guard` is what the `struktura guard` command runs (the monitor inside struktura's
AutoPilot), and reports the same events at the same rows: it keeps watching after an
alarm, quarantines a dead channel, and recalibrates after a level shift that settles
into a new normal. Repeats of the same detector within `cooldown` samples (default 50,
as in the CLI) are not reported again; `cooldown=0` reports every alarm.

`Monitor` is the bare detector underneath: `push()` returns the leg that fired, and
after one alarm it stays silent until you call `reset()`.

Build from source (needs a Rust toolchain):

```
pip install maturin
maturin build --release        # run inside crates/struktura-py
```
