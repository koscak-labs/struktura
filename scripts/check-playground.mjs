// The playground page shows live numbers from the WebAssembly build. This
// checks that build gives the same 30-seed results as the native example
// (docs/claims.tsv rows structure-vs-amp-*), so the page cannot drift.
//
// Usage: node scripts/check-playground.mjs [playground/pkg]
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { pathToFileURL } from 'url';

const dir = resolve(process.argv[2] || 'playground/pkg');
const m = await import(pathToFileURL(dir + '/struktura_playground.js').href);
m.initSync({ module: readFileSync(dir + '/struktura_playground_bg.wasm') });

const CALIB = 768, CHANGE = 1000;
const firstAlarm = v => {
  const a = JSON.parse(m.guard(v, CALIB)).events.find(e => e.kind === 'alarm');
  return a ? a.t : -1;
};
const median = a => (a.sort((x, y) => x - y), a[Math.floor(a.length / 2)]);

let g = 0, l = 0; const gd = [], ld = [];
for (let s = 0; s < 30; s++) {
  const v = m.demo_stream('matched', 1000 + s);
  const a = firstAlarm(v), b = m.limit_check(v, CALIB);
  if (a >= CHANGE) { g++; gd.push(a - CHANGE); }
  if (b >= CHANGE) { l++; ld.push(b - CHANGE); }
}
let gf = 0, lf = 0;
for (let s = 0; s < 30; s++) {
  const v = m.demo_stream('wander', s);
  if (firstAlarm(v) >= 0) gf++;
  if (m.limit_check(v, CALIB) >= 0) lf++;
}

const got = `matched guard ${g}/30 median ${median(gd)}, limit ${l}/30 median ${median(ld)}; wander false alarms guard ${gf}/30, limit ${lf}/30`;
const want = 'matched guard 30/30 median 100, limit 7/30 median 371; wander false alarms guard 0/30, limit 15/30';
console.log(got);
if (got !== want) {
  console.log('FAIL: expected ' + want);
  process.exit(1);
}
console.log('check-playground: PASS');
