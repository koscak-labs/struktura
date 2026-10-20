# GitHub Discussion — "Show and tell" or "General"

Post at: https://github.com/koscak-labs/struktura/discussions/new

> **Status 2026-10-20: DRAFT, not posted.** Rewritten from the 2026-09-17 version, which
> carried two claims the claims audit had already killed: "caught the 2022 AACS anomaly"
> and "F1 ~0.75". Both are corrected below.

## Title: What struktura actually is (and isn't)

hey, opening this to be straight about what this crate is.

**what it is:** a Rust implementation of DFA (Detrended Fluctuation Analysis) — a
well-established statistical method from 1994 (Peng et al., "Mosaic organization of
DNA nucleotide sequences", Phys. Rev. E 49(2) — 4,316 citations per Crossref) for
measuring long-range correlation in time-series data. it gives you one number (the
scaling exponent) that shifts when a signal's structure changes.

**what it's not:** machine learning, a full anomaly detection suite, or a silver bullet.
it's one algorithm that's good at one thing — detecting structural shifts — packaged fast
and simple in Rust.

**why I built it:** most DFA implementations are in Python/R, which don't fit production
monitoring or embedded systems. wanted a `no_std`, zero-training, single-dependency
version that's actually fast enough to run in real time on real hardware.

**what's tested:**
- CWRU bearing fault data — α 0.689 (normal) → 0.183 (inner-race fault), R² 0.953 / 0.918
- NASA SMAP/MSL Mars rover telemetry — **F1 0.655** with `--ar 0 --dfa`, against JPL's
  telemanom LSTM at P=87.5% / R=80.0% on the same data. So: a zero-training baseline,
  not a replacement for supervised methods.
- ESA OPS-SAT-AD — AUC-ROC 0.943, with a shuffle control that drops it to 0.554
- Numenta's anomaly benchmark — `guard` catches 45 of 116 labelled windows with 45 false
  alarms; the fewest false alarms of any detector there, and the fewest windows caught
- text rhythm, genome sequences, cardiac HRV (exploratory, less rigorously validated)

**on Voyager 1:** the magnetometer comparison is year-over-year (α 0.989 in 2021 vs 0.801
in 2022 May–Jul) and is currently **inconclusive** — a pre-anomaly vs during-anomaly
comparison within 2022 shows no significant shift (p=0.52, z=1.5). I don't claim it
detected the AACS anomaly, and `struktura voyager` says the same thing in its own output.

**honest limitations:**
- DFA catches structural/correlation shifts, not point anomalies (a single spike might not
  move alpha much)
- the Mars rover F1 score is honest but not competitive with trained models — this is a
  zero-training baseline, not a replacement for supervised methods
- the SMAP path fits a closed-form AR predictor, so "zero training" has that one exception
- I'm one person building this, not a team — expect rough edges

if you've got a time-series problem and want to try it, `cargo add struktura`. if you find
it doesn't work for your case, open an issue — I want to know where the limits are.

Every public number above has a command and a control behind it in
[`docs/claims.tsv`](https://github.com/koscak-labs/struktura/blob/master/docs/claims.tsv),
and CI fails if the output drifts.
