// Smoke test for the Node build (wasm-pack --target nodejs). Run: node tests/smoke.mjs <pkg dir>
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import path from 'node:path';

const require = createRequire(import.meta.url);
const s = require(path.resolve(process.argv[2]));

let seed = 1;
const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
const white = (n) => Float64Array.from({ length: n }, gauss);

const r = s.dfaShort(white(512));
assert.ok(r && r.alpha > 0.3 && r.alpha < 0.7, `dfaShort white alpha ${r && r.alpha}`);
assert.equal(s.dfaShort(new Float64Array(10)), undefined, 'dfaShort on 10 samples is undefined');
let c = 0;
const walk = Float64Array.from(white(2048), (v) => (c += v));
assert.ok(s.dfa(walk).alpha > 1.2, 'random walk alpha > 1.2');

const m = new s.Monitor(white(2000), 1);
assert.equal(m.channels, 1);
for (let i = 0; i < 300; i++) assert.equal(m.push(Float64Array.of(gauss())), undefined, `clean sample ${i} alarmed`);
let leg;
for (let i = 0; i < 200 && leg === undefined; i++) leg = m.push(Float64Array.of(gauss() + 8));
assert.ok(typeof leg === 'string', 'step raised an alarm');
const a = m.lastAlarm();
assert.equal(a.leg, leg);
assert.throws(() => m.push(Float64Array.of(1, 2)), /channels/);
assert.throws(() => new s.Monitor(white(10), 3), /multiple of channels/);
console.log('ok', 'white', r.alpha.toFixed(3), 'step alarm', leg, '-', a.explanation);
