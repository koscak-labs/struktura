# DFA structural health monitoring — custom template prototype

Hey @ivanperez-keera 👋

Following up from [cFS#1096](https://github.com/nasa/cFS/issues/1096) — took your advice and went deep into ogma's template system. Followed the cfs-001 example, the ros-copilot variable DB pattern, and extra-vars.json exactly as you showed me. Got it working 🔥

## What I got running

Built a custom cFS template that swaps Copilot temporal monitors for DFA scaling analysis. Uses `--template-dir` with the same Mustache vars (`{{#variables}}`, `{{#msgHandlers}}` etc):

```
ogma cfs \
  --input-file dfa-spec.json \
  --input-format json-format.cfg \
  --template-dir ogma-template/cfs/dfa_monitor \
  --variable-db ogma-template/example/db.json \
  --template-vars ogma-template/example/extra-vars.json \
  --target-dir output
```

Generates a full cFS app — per-channel DFA monitors with circular buffers, baseline learning, shift detection, EVS events. The DFA algorithm is a self-contained `dfa_core.h` (98 lines, zero deps beyond `<math.h>`, fixed stack buffers — flight-computer safe).

DFA parameters (window size, threshold, learning period, R² gate) come through `extra-vars.json` exactly like you described. No Haskell needed on the user's side.

## Quick validation — Voyager 1

Ran DFA on public Voyager 1 magnetometer data (NASA SPDF, 48-second averages). Picks up the 2022 AACS anomaly period:

| Period | DFA α | R² |
|--------|-------|----|
| 2021 healthy | 0.875 | 0.9999 |
| May-Jul 2022 (anomaly) | 0.827 | 0.9996 |

Shift = -0.048, fully reproducible: `cargo install struktura && struktura voyager`

## Also — spotted a couple things

PR [#552](https://github.com/nasa/ogma/pull/552): `mergeSpecs` was doing `s2 ++ s2` instead of `s1 ++ s2` on external variables, and `cannotCopyTemplate` was swallowing the exception message.

## The template + guide

Everything at [koscak-labs/struktura/ogma-template](https://github.com/koscak-labs/struktura/tree/master/ogma-template). I also wrote a [template guide](https://github.com/koscak-labs/struktura/blob/master/ogma-template/TEMPLATE_GUIDE.md) documenting the whole process — might be useful for [#315](https://github.com/nasa/ogma/discussions/315).

## Where I'd love your take

Could DFA work as a Copilot extern? The computation stays in C, the monitoring logic (shift detection, R² gating) lives in the Copilot DSL. Gets both Copilot's guarantees on the monitor and DFA's on the measurement. Sketched it [here](https://github.com/koscak-labs/struktura/blob/master/ogma-template/copilot_dfa_spec.hs) but you'd know way better than me whether that fits.

What direction would you steer this? 🙏

Crate: https://crates.io/crates/struktura
