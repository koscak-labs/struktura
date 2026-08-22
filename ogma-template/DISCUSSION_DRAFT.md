# DFA structural health monitoring — exploring integration with ogma

Hey @ivanperez-keera 👋

Following up from [cFS#1096](https://github.com/nasa/cFS/issues/1096) — you pointed me to ogma and I spent some time reading through the source and trying things out.

## What I found while reading

First, I spotted a couple of things in the code — submitted as PR [#552](https://github.com/nasa/ogma/pull/552):
- `mergeSpecs` had a copy-paste on `externalVariables` (`s2 ++ s2` instead of `s1 ++ s2`)
- `cannotCopyTemplate` was swallowing the actual exception message

Small stuff but figured it's worth fixing.

## What I tried

I wanted to understand how the template system works, so I built a custom cFS template that does DFA monitoring instead of Copilot checks. It plugs in through `--template-dir` and reuses the same Mustache variables (`{{#variables}}`, `{{#msgHandlers}}` etc). Documented the process in case it's useful for [#315](https://github.com/nasa/ogma/discussions/315).

I also tried running DFA on public Voyager 1 magnetometer data from NASA SPDF (48-second averages). The scaling exponent picks up a structural shift during the 2022 AACS anomaly period — alpha goes from 0.875 (2021 healthy) to 0.827 (May-Jul 2022), R² > 0.999 on all windows. Not dramatic but real, and fully reproducible from public data.

Everything is at [koscak-labs/struktura/ogma-template](https://github.com/koscak-labs/struktura/tree/master/ogma-template) if you're curious.

## Question

The thing I keep coming back to is whether DFA could work as a Copilot extern — the computation in C, the monitoring logic (shift detection, R² gating) in the Copilot DSL. That way it gets both Copilot's guarantees on the monitor and DFA's on the measurement. I sketched what that [might look like](https://github.com/koscak-labs/struktura/blob/master/ogma-template/copilot_dfa_spec.hs) but you'd know way better than me whether that fits.

Would love to hear your thoughts on the right direction here 🙏

Crate: https://crates.io/crates/struktura
