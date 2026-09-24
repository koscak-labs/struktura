// Node test for analysis.js against the real WebAssembly web build.
// Run: node playground/csv/analysis.test.mjs <path to struktura-wasm web pkg dir>
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { parseCsv, analyze, exampleCsv, MIN_CALIB } from './analysis.js';

const pkg = path.resolve(process.argv[2]);
const s = await import(pathToFileURL(path.join(pkg, 'struktura.js')).href);
s.initSync({ module: fs.readFileSync(path.join(pkg, 'struktura_bg.wasm')) });

let seed = 7;
const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());

// CSV parsing: header, delimiter, non-numeric column dropped, bad cells dropped.
const p = parseCsv('time;temp;label\n1;20.5;ok\n2;20.7;ok\n3;x;ok\n');
assert.deepEqual(p.columns.map((c) => c.name), ['time']);
const q = parseCsv('1,2\n3,4\n5,6\n');
assert.equal(q.columns.length, 2);
assert.equal(q.columns[0].name, 'column 1');

// Clean white noise: no alarms.
const clean = Float64Array.from({ length: 3000 }, gauss);
const a = analyze(s, clean);
assert.equal(a.error, null);
assert.equal(a.alarms.length, 0, `clean stream raised ${a.alarms.length} alarms`);
assert.ok(a.whole && a.whole.alpha > 0.3 && a.whole.alpha < 0.7, `white alpha ${a.whole && a.whole.alpha}`);

// Step of 6 sigma at row 2000: at least one alarm, the first one at or after the step.
const step = Float64Array.from({ length: 3000 }, (_, i) => gauss() + (i >= 2000 ? 6 : 0));
const b = analyze(s, step);
assert.ok(b.alarms.length >= 1, 'step raised no alarm');
assert.ok(b.alarms[0].at >= 2000, `first alarm at ${b.alarms[0].at}, before the step`);
assert.ok(b.alarms[0].at < 2100, `first alarm at ${b.alarms[0].at}, too late`);

// Too short: clear error, no crash.
const short = analyze(s, Float64Array.from({ length: MIN_CALIB }, gauss));
assert.ok(short.error && short.error.includes('Need at least'), short.error);

// Short calibration warns.
const w = analyze(s, Float64Array.from({ length: 700 }, gauss));
assert.equal(w.error, null, w.error);
assert.ok(w.warnings.some((t) => t.includes('Learned "normal" from only')), 'expected a short-calibration warning');

// Problem inside the calibration window: the self-check warns.
const early = Float64Array.from({ length: 4000 }, (_, i) => gauss() + (i >= 700 ? 6 : 0));
const e = analyze(s, early);
assert.ok(e.calibrationSuspectRow >= 600, `calibration self-check: ${e.calibrationSuspectRow}`);
assert.equal(a.calibrationSuspectRow, undefined, 'clean stream flagged its calibration');

// The page's own example must be flagged after its change at row 2200.
const ex = parseCsv(exampleCsv());
const er = analyze(s, ex.columns[ex.columns.length - 1].values);
assert.ok(er.alarms.length >= 1 && er.alarms[0].at >= 2200, `example: alarms ${JSON.stringify(er.alarms.slice(0, 2))}`);

console.log(`example flagged at row ${er.alarms[0].at + 1}`);
console.log(`ok parse; clean 0 alarms (alpha ${a.whole.alpha.toFixed(3)}); step alarm at ${b.alarms[0].at} (${b.alarms[0].leg}); short input rejected; short calibration warned`);
