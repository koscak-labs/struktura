# SEO / discoverability audit — struktura

Date: 2026-09-17. Baseline: 4 stars, 0 forks, v1.7.2, 393 crates.io downloads.

## What I changed

1. **README.md** — rewrote the first screen only (title line, one-sentence
   hook, badges, a 6-line copy-paste quickstart with real CLI output, and a
   "who this is for" section naming 5 concrete use cases with search-intent
   phrases worked in naturally). Renamed 4 existing headings further down to
   carry the remaining target phrases without changing their content:
   - `## how DFA works` → `## how detrended fluctuation analysis works — the Hurst exponent in Rust`
   - `## spacecraft health monitoring` → `## spacecraft telemetry anomaly detection & health monitoring`
   - `## features` → `## features — anomaly detection no_std support included`
   - `## mars rover anomaly detection (NASA SMAP/MSL)` → `## time series health monitoring for Mars rover anomaly detection (NASA SMAP/MSL)`
   - Fixed a factual error found while reading: the IMS bearing early-warning
     claim said "~105 hours before failure"; REPRODUCIBILITY.md line 19 (the
     repo's own source-of-truth for every measured claim) says "~2h before
     failure," and the math checks out (recording 970/984 at ~10-min
     intervals ≈ 2.3h). Corrected to "~2 hours."
2. **docs/COMPARISON.md** (new) — honest table vs. nolitsa, nolds, `hurst`
   (PyPI + the GPL-3.0 Rust crate), MFDFA, IsolationForest, ADTK, Merlion,
   NASA telemanom. Unverified cells marked "not verified" — I confirmed each
   project exists and its license via search, but did not independently
   benchmark speed for anything except struktura-vs-nolds (the only
   benchmark actually shipped in this repo).
3. **llms.txt** (new, repo root) — llmstxt.org-format summary for AI
   crawlers, including an explicit "what it is NOT" section so crawlers
   don't hallucinate an onboard-flight-software claim.
4. **Cargo.toml** — updated `keywords` and `categories`:
   - keywords: `["anomaly-detection", "dfa", "time-series", "spacecraft", "hurst"]`
     (dropped `health-monitor`, added `hurst` — there's an existing
     competing Hurst-exponent Rust crate people search for by that exact
     term, and it wasn't covered).
   - categories: `["science", "algorithms", "no-std", "aerospace", "embedded"]`
     (added `aerospace` and `embedded`, both valid crates.io category slugs
     per `https://crates.io/api/v1/categories`, and both a strong fit given
     the NASA/ESA telemetry focus and the `no_std`/C99-codegen story).
5. **.github/FUNDING.yml** — left untouched. It already exists and points
   at `github: [philphauler]` (personal Sponsors), so per the task's "only
   if Sponsors is not already configured" condition, nothing to add. See
   the "needs Phil" item below on whether it should point at the org
   instead.

## Unverified claims removed or softened

- Removed: "the same engine that monitors Mars rovers runs on your laptop"
  — struktura has been tested offline against public NASA/ESA telemetry
  datasets (SMAP/MSL, IMS bearing, Voyager, ESA-ADB); nothing in
  GUARANTEES.md, REPRODUCIBILITY.md, or the git history supports it being
  deployed on any actual mission or rover. Replaced with a factual
  "tested against real NASA and ESA spacecraft telemetry datasets" line
  that links to REPRODUCIBILITY.md.
- Corrected: "~105 hours before failure" → "~2 hours before failure" (see
  above) — this was a real, checkable error, not a hedge.
- Kept, with caveats intact: "85-112x faster than Python" — I narrowed the
  claim's phrasing to "vs. Python (`nolds`)" in the hero since that's what
  the benchmark table actually measures, and it already links to a
  reproduce command. The broader "Python" framing in the mid-README speed
  table was left as-is (out of first-screen scope) but is worth narrowing
  too — see below.

## Findings not acted on (in scope for a future pass, not this one)

- **USE_CASES.md** still says "F1=0.755 on NASA SMAP/MSL" while README.md
  says F1=0.655 (per commit `3724137`, which updated the README but not
  USE_CASES.md). `git status` shows USE_CASES.md and PROMO_PACK.md have
  no pending changes from this session or any other agent — this
  inconsistency is pre-existing and untouched. Recommend a follow-up fix.
- The mid-README "⚡ speed (benchmarked, not guessed)" table still says
  "python nolds" generically as "python" in its narrative sentence below
  the table ("at 1Hz spacecraft telemetry...") — fine as written, no
  change needed there.

## What needs Phil's hand

0. **Publish the `struktura` GitHub Action to the Marketplace.** `action.yml`
   (composite action, root of this repo) and `.github/workflows/action-selftest.yml`
   are added and self-test clean on `data/sylv_spike.csv` (2 known alarms).
   The README's "use in CI" snippet references `koscak-labs/struktura@v1`,
   which does not exist yet — needs repo-admin access I don't have:
   1. Tag the current commit (once action.yml is merged to `main`/`master`):
      `git tag v1 && git push origin v1` — then create a lightweight `v1`
      major-version tag that always points at the latest `v1.x.y` release
      going forward (`git tag -f v1 <new-commit> && git push origin v1 --force`
      on each subsequent release, the standard GitHub Actions convention).
   2. On GitHub: Releases → Draft a new release → choose the `v1` tag →
      check "Publish this Action to the GitHub Marketplace" → pick a
      primary category (e.g. "Testing", "Utilities") → Publish release.
   3. Verify `assets/social-preview.png` (see also item 1 below) renders
      correctly on the Marketplace listing card before announcing.
1. **GitHub repo description and social preview image.** Live repo
   description currently reads "Predict failure before it happens. DFA
   anomaly detection for time-series. Spacecraft, bearings, finance,
   genome. 85x faster than Python. no_std." — this is *not* the same
   string as the Cargo.toml `description` field (which says "Is your data
   broken?..."). Pick one and sync both; I did not touch GitHub settings
   (no repo-admin access from here). Topics already present (15, good
   coverage: anomaly-detection, bearings, dfa, embedded, fault-detection,
   health-monitoring, mars-rover, nasa, no-std, predictive-maintenance,
   rust, signal-processing, spacecraft, structural-health, time-series) —
   consider adding `hurst-exponent` and `dfa-analysis` to match the new
   Cargo.toml keyword. Social preview image: recommend 1280x640px with the
   one-liner "Detrended fluctuation analysis for Rust — find the moment
   your signal's structure changed" plus the α-spike visual already used
   in `assets/terminal-demo.svg`; I can't generate/upload images from here.
   Discussions tab: enable it — right now there's no lightweight place for
   "does this work for my domain" questions, which is exactly the kind of
   engagement that turns into stars.
2. **GitHub Sponsors target.** `.github/FUNDING.yml` points at the personal
   `philphauler` account, not the `koscak-labs` org. Decide whether
   Sponsors should be org-level (needs Sponsors enabled for the org on
   GitHub's side first) or stay personal — I did not change this since it
   was "already configured" per the task's condition, but flagging since
   the task brief assumed org-level.
3. **awesome-list PRs — eligibility gate, not just a list to submit to.**
   Verified via WebFetch/WebSearch on 2026-09-17:
   - **awesome-rust** (`rust-unofficial/awesome-rust`) — CONTRIBUTING.md
     requires **>50 GitHub stars OR >2,000 crates.io downloads** (or
     documented equivalent). struktura has 4 stars / 393 downloads — **not
     yet eligible**. Entry format when eligible: `[koscak-labs/struktura](https://github.com/koscak-labs/struktura) [[struktura](https://crates.io/crates/struktura)] - DESCRIPTION`, alphabetical, under a science/algorithms section.
   - **awesome-embedded-rust** (`rust-embedded/awesome-embedded-rust`) —
     actively maintained (1,034+ commits). No visible star/download gate
     in what I could fetch; place under its "no-std crates" section, format:
     `*   [`struktura`](https://crates.io/crates/struktura) - Brief description - [![crates.io](badge)](https://crates.io/crates/struktura)`.
     Verify the exact gate in its CONTRIBUTING.md before opening the PR.
   - **AeroRust/awesome-space** — exists, but WebFetch showed only ~26
     commits total; maintenance cadence looks low. It does explicitly
     welcome PRs ("Feel free to open PR to add additional links and
     resources ... including your own crates!"). Low risk to try, low
     certainty it gets merged promptly — confirm it's not stale before
     relying on it for traffic.
   - **awesome-anomaly-detection / awesome-time-series** — no single
     canonical, actively-maintained list matched both terms exactly. The
     closest actively-maintained match is `rob-med/awesome-TS-anomaly-detection`
     (3.2k stars, explicitly prioritizes maintained tools, table format:
     Name | Language | Pitch | License | Maintained) — target this one
     under "Anomaly Detection Software," not a generic awesome-anomaly-detection.
   - **awesome-nasa** — I could not find a maintained list by that name;
     do not spend effort on it. `nasa/ogma` Discussion #557 (per
     PROMO_PACK.md) is a more direct, existing channel into the NASA open-
     source world than searching for a generic awesome-nasa list.
   - Bottom line: **the awesome-rust PR is blocked on star count**, which
     is itself downstream of the HN/TWiR promo sequence already planned in
     PROMO_PACK.md. Do the promo push first; awesome-embedded-rust and
     awesome-TS-anomaly-detection PRs can go out now since neither has a
     star gate I could confirm.
