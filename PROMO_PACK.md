# Struktura promo pack — ALL DRAFTS, NOTHING POSTED WITHOUT PHIL'S EXPLICIT WORD

The story got 10x stronger this week. The old pitch was "fast DFA crate."
The new pitch: **a self-calibrating, self-healing, self-DESIGNING telemetry
monitor, flight-software-grade, every claim reproducible in one command.**

## The one-liner (use everywhere)
"A pure-Rust spacecraft health monitor that calibrates itself, quarantines
its own dead sensors, and evolved its own detectors — 71%→92% fault
coverage through adversarial self-play, zero false alarms in 200K samples,
and it compiles to flight-ready C99. Every number: one command."

## Channel drafts (fire individually on Phil's word)

### 1. This Week in Rust nomination (lowest effort, high reach)
> struktura 1.7: a no_std spacecraft telemetry monitor that calibrates its
> own thresholds (extreme-value statistics), reconstructs dead sensors from
> the surviving channels, and uses an adversarial RED/BLUE loop to evolve
> new detector legs — measured 71%→92% fault coverage, 0 false alarms in
> 200K clean samples. Reproducible: `cargo install struktura && struktura mission`.
Submit at: github.com/rust-lang/this-week-in-rust (PR to Interesting Projects)

### 2. r/rust post (title options)
- "My no_std telemetry monitor evolved its own detectors (71%→92% coverage, zero false alarms) — every claim is one command"
- "I built a spacecraft health monitor in Rust that quarantines dead sensors and survives double faults — then made it design its own detectors"
Body: the mission log screenshot (struktura mission output), the evolution
table, REPRODUCIBILITY.md link, honest-limits section (Rust crowd loves it).

### 3. dev.to / blog article
"Detect-fast, diagnose-precise: building a flight-grade health monitor in
Rust" — the 5 scars structure (each bug found by reading raw output):
clamp-riding sensors, EVT vs max thresholds, CUSUM transfer bias,
drift-chasing autopilot, ill-conditioned reconstruction. Engineers share
war stories, not feature lists.

### 4. Hacker News (Show HN) — only after TWiR/reddit validate the framing
"Show HN: A telemetry monitor that evolved its own fault detectors (Rust, no_std)"

## More space stuff — target map (ranked by fit x credibility)

### Tier 1 — DO (direct fit, doors already open)
- **NASA SMAP/MSL benchmark (telemanom)**: REAL NASA spacecraft telemetry,
  82 labeled channels. Run the hybrid monitor against it → precision/recall
  vs JPL's own LSTM detector. IN PROGRESS (data downloading). This is the
  single highest-credibility move available.
- **nasa/ogma (Ivan)**: Discussion #557 is our open thread. Follow-up when
  Ivan engages: the self-calibrating monitor as an ogma template/backend
  (generate-hybrid already emits cFS-style C99).
- **sylvester (xlstm-telemetry-assurance)**: draft ready + now the guarded-
  adaptation implementation (his discipline, in Rust) + evolve results.
- **nasa/cFS Limit Checker (LC app)**: our monitor IS a next-generation LC
  (their LC = static thresholds). A comparative note or contrib fits the
  existing cFS #1096 thread.

### Tier 2 — PREP (fit is real, relationship not yet)
- **ESA OPS-SAT anomaly dataset**: labeled REAL satellite telemetry from
  ESA's flying lab; the European counterpart to SMAP/MSL. Also sylvester-
  adjacent (ESA world).
- **nasa/fprime**: Svc components do telemetry checking; our F Prime
  codegen path exists. Wait for zimri thread resolution first.
- **Space ROS**: Ivan is active there; ROS 2 node generation exists in
  struktura already.
- **Yamcs (mission control)** / **OpenMCT**: ground-segment monitoring —
  our monitor as a Yamcs plugin/algorithm would meet operators where they
  live.
- **satnogs**: community ground-station network, open telemetry firehose —
  live public demo material.

### Rover track (mdj is building a rover)
- **MSL half of the NASA benchmark IS rover telemetry** (Curiosity) — the
  SMAP/MSL results double as rover credibility automatically.
- **F Prime is THE rover/heli flight software** (Ingenuity flew it);
  struktura already generates F Prime components — a rover health-monitor
  component is a natural demo.
- **mdj's physical rover**: struktura monitoring its real telemetry (motor
  currents, IMU, wheel vibration — literally our bearing math) over serial/
  CSV = a hardware demo video. Real hardware converts stars like nothing
  else. Offer mdj the health-monitor stack for it.

### Excluded on purpose
- ArduPilot/PX4: drones-not-space (Phil's rule) + prior pushback scars.

## GitHub growth mechanics (stars + FOLLOWERS, honestly)
Current baseline: 1 star, 229 crates downloads. HN upvote ≈ 1.4 stars (own memory).
- Stars land on koscak-labs/struktura; FOLLOWERS land on philphauler —
  so: personal profile README (DRAFT_PROFILE_README.md ready) + pinned
  repos + public org membership, so repo traffic converts to follows.
- Traffic order that compounds: crates publish → TWiR (steady drip) →
  r/rust (spike) → HN Show HN (biggest; a front-page hour can be 200-800
  stars) → GitHub Trending (stars-in-window triggers it → free compounding).
- README must be at maximum punch BEFORE the spike (done: mission log +
  evolution table + receipts now in the hero).
- Awesome-list PRs after the spike (they get accepted easier with stars):
  awesome-rust (science), awesome-embedded-rust (no_std), awesome-space.
- Repeatable content engine: every future measured result = 1 tweet-length
  line + 1 repo commit + occasional dev.to post. Consistency beats bursts
  for followers.

## Sequencing (Juraj logic)
1. SMAP/MSL results first — they upgrade EVERY other conversation.
2. Then sylvester message (his benchmark + his discipline + NASA data results).
3. Then ogma follow-up when Ivan surfaces (or after PR #552/#556 review).
4. Public promo (TWiR → r/rust → blog → HN) after crates.io publish so
   `cargo install struktura` works for everyone who clicks.

## Prerequisites before ANY promo fires
- [ ] crates.io publish (repro commands must work for strangers)
- [ ] 200K clean battery on the evolved organism (ship-bar)
- [ ] Phil's explicit word per channel
