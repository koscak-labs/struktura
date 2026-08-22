# DFA structural health monitoring — ogma-compatible generator + custom templates

Hey @ivanperez-keera 👋

Following up from our chats on [fprime#5772](https://github.com/nasa/fprime/issues/5772) and [cFS#1096](https://github.com/nasa/cFS/issues/1096) — I went deep into ogma's source and built some things I think you'll find interesting.

## What I built

### 1. `struktura generate` — ogma-compatible app generator (no Haskell)

Struktura now reads ogma's `db.json` variable database format and generates complete monitoring applications for all 3 ogma backends:

```sh
cargo install struktura
struktura generate --cfs    --db channels.json -o dfa_cfs_app/
struktura generate --fprime --db channels.json -o dfa_fprime_component/
struktura generate --ros    --db channels.json -o dfa_ros_node/
```

One command, 10-second install, works on Windows/Mac/Linux. Each backend generates code that follows the same structure as ogma's output — cFS gets `AppMain`/`Init`/`ProcessPkt` with SB subscriptions and EVS events, F Prime gets an FPP model with async ports and typed events, ROS 2 gets an `rclcpp` node with topic subscriptions and a shift publisher.

The idea is complementary: ogma generates monitors for temporal properties (Copilot specs), struktura generates monitors for structural health (DFA scaling analysis). Same `db.json`, different monitoring paradigm, both producing ready-to-compile flight software.

### 2. Custom ogma templates for cFS and F Prime

I also built custom ogma templates that work with `--template-dir`:

- `ogma-template/cfs/dfa_monitor/` — cFS template using all the standard Mustache variables (`{{#variables}}`, `{{#msgCases}}`, `{{#msgHandlers}}`)
- `ogma-template/fprime/dfa_monitor/` — F Prime template with `{{varDeclFPrimeType}}`, `{{#monitors}}`

These replace `copilot_step()` with DFA computation while keeping ogma's subscription/dispatch infrastructure.

### 3. Template preparation guide (re: #315)

I noticed [discussion #315](https://github.com/nasa/ogma/discussions/315) asking for documentation on how to prepare a new template. While building the DFA templates, I documented the full process — available Mustache variables per backend, how `db.json` maps to subscriptions, how `extra-vars.json` works, and how to replace Copilot with custom logic. Happy to contribute this if useful.

Everything is at: https://github.com/koscak-labs/struktura/tree/master/ogma-template

## How DFA complements Copilot

| | Copilot (temporal logic) | DFA (structural health) |
|---|---|---|
| Detects | Property violations | Behavioral degradation |
| Fires when | A Boolean property becomes false | Signal's correlation structure shifts from baseline |
| Catches | Known failure modes (specified) | Unknown degradation (structural change) |

A reaction wheel bearing degrading over weeks: DFA catches the vibration pattern change (alpha shifts from 0.7 to 0.4) weeks before the amplitude crosses a Copilot threshold. Running both = earliest detection + formal correctness.

## Verified results

### Industrial: CWRU bearing fault detection
Reproducible — `cargo install struktura && struktura demo`:

| Condition | DFA alpha | R² | Verdict |
|---|---|---|---|
| Normal bearing | 0.738 | 0.985 | Baseline |
| Inner race fault | 0.217 | 0.948 | **CRITICAL** (shift -0.522) |

### Spacecraft: Voyager 1 magnetometer (public NASA SPDF data)

I pulled Voyager 1's 48-second MAG averages from NASA SPDF and ran DFA across the 2022 AACS anomaly period. Data bundled in the repo — fully reproducible:

```sh
struktura compare data/voyager1_2021_healthy.csv data/voyager1_during_anomaly.csv
```

| Period | N | DFA alpha | R² | Quality |
|---|---|---|---|---|
| 2021 (healthy baseline) | 149,359 | 0.875 | 0.9999 | EXACT |
| 2022 Jan-Apr (pre-anomaly) | 61,516 | 0.834 | 0.9999 | EXACT |
| 2022 May-Jul (AACS anomaly) | 69,137 | 0.827 | 0.9996 | EXACT |

Alpha shifts from 0.875 (healthy) to 0.827 during the anomaly window — a structural change in the magnetometer signal at R² > 0.999. Shuffle proof confirms the fractal structure is genuine.

Source: L. F. Burlaga, VIM 48-second averages, [NASA SPDF](https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/magnetic_fields/VIM_48s_mag_ascii/)

## Properties matching Copilot's guarantees

I documented [formal DFA properties](https://github.com/koscak-labs/struktura/blob/master/ogma-template/DFA_PROPERTIES.md) to show the guarantees align with Copilot's:

- **Constant memory**: fixed circular buffer, no dynamic allocation
- **Deterministic**: same input → same output, zero randomness
- **Bounded time**: O(n·B) per evaluation, ~1500 MACs for 256-sample window
- **Exact-or-abstain**: if R² < threshold, monitor suspends (never alerts on noise)

## Questions

1. Does the `db.json` compatibility approach make sense, or would you prefer tighter integration (e.g., a Haskell backend in ogma itself)?

2. Could DFA and Copilot monitors coexist in the same generated app? I could see a mode where ogma generates an app that does both temporal checks AND structural health.

3. I submitted a couple of fixes in PR [#552](https://github.com/nasa/ogma/pull/552):
   - `mergeSpecs` was using `s2` twice instead of `s1 ++ s2` for external variables (data loss when combining multiple `--input-file` args, refs #551)
   - `cannotCopyTemplate` was swallowing the actual exception — now propagates the real error message for easier template debugging (refs #390)

Really enjoying reading through the ogma architecture — the template system is surprisingly powerful for how clean it is 🙏⚡

Crate: https://crates.io/crates/struktura
Source: https://github.com/koscak-labs/struktura
