# struktura (JavaScript / WebAssembly)

JavaScript access to [struktura](https://github.com/koscak-labs/struktura), a Rust crate for
time-series anomaly detection without training data. Every call runs the Rust code compiled
to WebAssembly; the package only converts arguments.

```js
const s = require('struktura');            // Node build

const r = s.dfaShort(Float64Array.from(values));   // works from about 24 samples; undefined if unmeasurable
console.log(r.alpha, r.rSquared);

const g = new s.Guard(cleanFlat, 2);       // clean data, channel-major: all of ch0, then all of ch1
for (const e of g.push(Float64Array.of(x0, x1))) {   // [] in the steady state; NaN = missing reading
  console.log(e.tick, e.kind, e.explanation);
}
```

`Guard` is what the `struktura guard` command runs (the monitor inside struktura's AutoPilot),
and reports the same events at the same rows: it keeps watching after an alarm, quarantines a
dead channel, and recalibrates after a level shift that settles into a new normal. Repeats of
the same detector within `cooldown` samples (third argument, default 50 as in the CLI) are not
reported again; `0` reports every alarm.

`Monitor` is the bare detector underneath: `push()` returns the leg that fired, and after one
alarm it stays silent until `reset()`.

Two builds are attached to each `wasm-v*` GitHub release: `nodejs` (CommonJS, for Node and CI)
and `web` (ES module for browsers; call the default `init()` export first).
