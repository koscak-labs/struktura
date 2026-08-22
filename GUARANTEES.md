# Struktura Guarantees

## What this crate DOES

1. Computes the DFA scaling exponent (alpha) and its R-squared for any f64 slice.
2. Reports alpha, R-squared, Hurst, kurtosis, and a quality classification.
3. Compares current alpha against a baseline and returns a health verdict.
4. Provides shuffle proof that the detected structure is real, not an artifact.
5. Handles NaN, Inf, and constant signals gracefully (never panics on valid input).

## What this crate DOES NOT

1. It does NOT do frequency analysis (FFT, wavelets). DFA measures correlation structure, not frequency content.
2. It does NOT train on data. There are no hyperparameters to tune. The baseline is established from the signal itself.
3. It does NOT guarantee that a structural shift means a specific fault. It detects THAT something changed, not WHAT changed.
4. It does NOT replace domain-specific monitoring. Use it alongside threshold monitors, not instead of them.

## Exact-or-Abstain

If R-squared < 0.3, quality = Abstain. The crate will not diagnose a signal it cannot reliably characterize. This is by design.

## Stability (from v1.0.0)

- All public types and functions follow semantic versioning.
- Enums are `#[non_exhaustive]` — new variants may be added in minor versions.
- The DFA algorithm is deterministic: same input always produces the same output.
- No unsafe code anywhere in the crate.
