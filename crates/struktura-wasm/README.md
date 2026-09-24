# struktura (JavaScript / WebAssembly)

JavaScript access to [struktura](https://github.com/koscak-labs/struktura), a Rust crate for
time-series anomaly detection without training data. Every call runs the Rust code compiled
to WebAssembly; the package only converts arguments.

```js
const s = require('struktura');            // Node build

const r = s.dfaShort(Float64Array.from(values));   // works from about 24 samples; undefined if unmeasurable
console.log(r.alpha, r.rSquared);

const m = new s.Monitor(cleanFlat, 2);     // clean data, channel-major: all of ch0, then all of ch1
const leg = m.push(Float64Array.of(x0, x1)); // undefined, or the detector that fired
if (leg) console.log(m.lastAlarm().explanation);
```

Two builds are attached to each `wasm-v*` GitHub release: `nodejs` (CommonJS, for Node and CI)
and `web` (ES module for browsers; call the default `init()` export first).
