# Reddit r/rust — launch assets

> **Status 2026-10-20: POSTED on 2026-08-26, body below is a REWRITE, not the
> live text.** The live thread is at the URL in `docs/promotion/` history and
> `.local-backup-20261020/FEEDBACK.md`; its title leads with a speed ratio that
> the project's own claims audit (docs/CLAIMS-AUDIT-2026-09-17.md) later
> withdrew, and the body carried four other killed claims. Reddit blocks
> unauthenticated fetch from this box, so the live text could not be re-read to
> confirm the exact wording — it is reconstructed from FEEDBACK.md.
>
> **Treat the text below as a NEW post, not an edit of the live one.** Posting
> to a thread you already own is a self-post; re-posting the same crate to
> r/rust again would need a different angle and Phil's word.
>
> The 2026-09-17 version carried four claims the claims audit had already
> killed. The corrections are marked ⚠️ below. Do not restore the old wording.

Every number here is checkable against `docs/claims.tsv` and `scripts/check-claims.sh`.

---

## Title

I made a Rust anomaly detector that finds *structural* shifts, not thresholds — one function, one number, no training

## Body

Built **struktura** — DFA-based anomaly detection. Not ML, not thresholds: one function,
one number (the fractal scaling exponent α). When α shifts from baseline, the signal's
structure changed.

Why Rust:
- **~0.24 ms on 4K points**, release build, same window. ⚠️ The "85× vs Python
  `nolds`" figure that used to be in the title is a *published reference* number,
  not a head-to-head run on this machine — the README labels the Python column
  "published figure, not run here", and `examples/nolds_compare.py` is in the repo
  so you can run it yourself.
- **`no_std`**, single dependency (`libm`), zero heap on the hot path (`dfa_into()`)
- **C FFI**, and it compiles clean under `-Wall -Werror` with a self-test. ⚠️ Not
  "flight-ready" — it is not mission-qualified.
- Single-threaded benchmark method documented in `REPRODUCIBILITY.md`.

Validated on real NASA data: F1 **0.655** on the SMAP/MSL Mars rover benchmark with
`--ar 0 --dfa`, against JPL's telemanom LSTM at P=87.5% / R=80.0% on the same data —
so it does **not** beat supervised models there. ⚠️ (An earlier F1 of 0.755/0.788 was
not reproducible on any commit or flag and has been withdrawn.) Also AUC-ROC 0.943 on
ESA's OPS-SAT-AD telemetry, CWRU bearings, Numenta's anomaly benchmark, HRV, human chr1.

`cargo install struktura && struktura demo`
docs.rs/struktura · github.com/koscak-labs/struktura

Genuinely stuck on direction: embedded/edge targets? API polish? Domains beyond what it
does? Roast the design.

---

## Pinned comment

For the curious: docs.rs/struktura · full command list via `struktura --help`.

**On the Voyager 1 claim:** ⚠️ the year-over-year magnetometer comparison (α 0.989 in
2021 vs 0.801 in 2022 May–Jul) is **inconclusive**, not a detection. A pre-anomaly vs
during-anomaly comparison within 2022 shows no significant shift (p=0.52) and gives
z=1.5. The `CRITICAL` label the CLI prints is a fixed α threshold, not significance.
`struktura voyager` says all of this in its own output.

**On "zero training":** the SMAP path fits a closed-form AR predictor — that's a
closed-form fit, not gradient training, but it is an exception, so I name it.

**On the core size:** the DFA reference implementation is a public module
(`src/dfa_core.rs`) and is verified to agree with the optimized production path to 1e-9.
⚠️ It is not "98 lines" — that number was a miscount of a different file. The module is
~114 lines of code; the `dfa_reference` function itself is ~68, of which the box loop and
the log-log fit are the actual math.

**On the Numenta benchmark:** `guard` catches 45 of 116 labelled windows with 45 false
alarms (0.15 per 1,000 samples) — the fewest false alarms of any detector there, and the
fewest windows caught too. That trade is deliberate: it suits paging a person, where false
alarms are what gets a monitor switched off. `--sensitivity high` catches 55 with 53 false
alarms; BOCPD catches 77 but with 1,461 false alarms.

**On the speed:** it's Rust vs CPython `nolds` — pure Python — not Rust vs a
PyO3-bound equivalent, so the gap overstates what a same-language binding
would give. The Rust column is measured on my machine; the Python column is a
published reference figure, not a run I did. `examples/nolds_compare.py` is
in the repo if you want to do the head-to-head yourself.

The whole detector core is auditable in one file — no matrices, no gradients, no
hyperparameters. Benchmarks: `cargo run --release --example speed_bench`.

---

## Reply bank

**"Is the benchmark doing heavy lifting?"**
The DFA core is the textbook algorithm. `dfa_reference()` is the reference implementation
and there's a test asserting it matches the shipped fast path to 1e-9, so what you read is
what runs. The `guard` command is separate: it self-calibrates a threshold per channel and
runs several detector legs (level shift, residual-CUSUM, repeated-value, DFA).

**"How is this different from a moving-average threshold?"**
Thresholds compare a value to a bound. DFA measures how fluctuation scales with time window
— it can see a channel that stays inside its limits but whose correlation structure changed.
The repo ships controls for exactly this (variance-matched and AR(0.9) comparisons) so the
claim is testable rather than rhetorical.

**"Doesn't supervised ML win on this?"**
Yes, on SMAP/MSL — the README says so plainly. The pitch is no training, no GPU, and
running on a microcontroller with no heap.

**"Is the Voyager claim real?"**
No, and I say so in the output. It was a year-over-year comparison, currently inconclusive.
The honest version of the claim is "DFA measures a structural shift in Voyager's
magnetometer data across a year boundary", not "it detected the AACS anomaly".
