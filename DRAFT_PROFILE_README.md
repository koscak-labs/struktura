# DRAFT — personal GitHub profile README (repo: philphauler/philphauler)
# NEEDS PHIL'S WORD to create/push. Stars land on repos; FOLLOWERS land here.

## yo, I'm Phil 👋

I teach rust to find broken things before they break. spacecraft mostly.

🛰️ **[struktura](https://github.com/koscak-labs/struktura)** — a health
monitor that calibrates itself, quarantines its own dead sensors, and got
bored of my hand-written detectors so it **evolved its own**. adversarial
self-play, 71%→97% fault coverage, zero false alarms in 200,000 samples.
not thresholds. not ML. not vibes. receipts:

```
cargo install struktura && struktura mission
```

that command runs a 24,000-sample spacecraft mission where a sensor dies,
the environment changes, and a drift tries to sneak in disguised as a new
normal. the monitor handles all three alone. no human. no cloud. 4 microseconds
per sample. yeah.

### stuff this one algorithm has caught 🎯
- 🔩 a bearing failing — **105 hours before it died** (NASA run-to-failure data)
- 🚀 Voyager 1's 2022 attitude anomaly (public magnetometer data)
- 🌌 the **edge of the solar system** (heliopause crossing, α shift in the field)
- 🧬 what makes human DNA structurally different from a chimp's

zero training. works on anything with a time axis.

### currently cooking 🔭
- benchmarking against NASA's SMAP satellite + **Curiosity rover** labeled
  anomalies — mars-rover-grade receipts loading
- contributing runtime monitoring to [nasa/ogma](https://github.com/nasa/ogma)
- it compiles to flight-ready C99, because spacecraft don't run cargo

📫 into spacecraft telemetry, runtime assurance, or weird fractal math?
let's talk. I answer everything.

---
# Profile checklist (do together with the README):
- [ ] Pin: struktura + best fork contributions
- [ ] Profile bio: "teaching rust to find broken things before they break 🛰️"
- [ ] Location + koscak.ai link
- [ ] Ensure koscak-labs org membership is PUBLIC (visitors hop org -> person)
