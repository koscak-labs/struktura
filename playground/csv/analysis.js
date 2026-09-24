// Pure analysis for the "drop your CSV" page: no DOM, so it runs in Node tests
// against the same WebAssembly build the browser loads.

export const MIN_CALIB = 192; // HybridMonitor needs 2 * WINDOW (96) clean samples
export const SAFE_CALIB = 768; // below this, guard reported level-shift false alarms on clean data

/** Parse CSV text into numeric columns. Accepts comma, semicolon or tab, with or without a header row. */
export function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
  if (lines.length === 0) return { columns: [], rows: 0 };
  const delim = [',', ';', '\t'].reduce((best, d) =>
    lines[0].split(d).length > lines[0].split(best).length ? d : best, ',');
  const split = (l) => l.split(delim).map((c) => c.trim().replace(/^"|"$/g, ''));
  const first = split(lines[0]);
  const hasHeader = first.some((c) => c !== '' && !Number.isFinite(Number(c)));
  const names = hasHeader ? first : first.map((_, i) => `column ${i + 1}`);
  const body = (hasHeader ? lines.slice(1) : lines).map(split);
  const columns = [];
  names.forEach((name, j) => {
    const raw = body.map((r) => Number(r[j]));
    const ok = raw.filter((v) => Number.isFinite(v)).length;
    if (body.length > 0 && ok / body.length >= 0.9) {
      columns.push({ name, values: Float64Array.from(raw.filter((v) => Number.isFinite(v))), dropped: body.length - ok });
    }
  });
  return { columns, rows: body.length };
}

/**
 * Analyse one series.
 * s: the struktura WebAssembly module (dfaShort, Monitor).
 * Returns the whole-series alpha, a rolling alpha, and monitor alarms after a
 * calibration prefix that is assumed normal.
 */
export function analyze(s, values, { calibRows } = {}) {
  const n = values.length;
  const out = { n, whole: null, rolling: [], calib: 0, alarms: [], warnings: [], error: null };
  const w = s.dfaShort(values);
  out.whole = w ? { alpha: w.alpha, rSquared: w.rSquared } : null;

  const win = Math.min(256, Math.max(64, Math.floor(n / 20)));
  const step = Math.max(1, Math.floor(win / 4));
  for (let i = 0; i + win <= n; i += step) {
    const r = s.dfaShort(values.subarray(i, i + win));
    if (r) out.rolling.push({ at: i + win, alpha: r.alpha });
  }

  const calib = calibRows ?? Math.min(Math.max(Math.floor(n * 0.3), Math.min(SAFE_CALIB, n)), 5000);
  if (calib < MIN_CALIB || n - calib < 20) {
    out.error = `Need at least ${MIN_CALIB + 20} rows (${MIN_CALIB} to learn what normal looks like, then some to check); this column has ${n}.`;
    return out;
  }
  if (calib < SAFE_CALIB) {
    out.warnings.push(`Learned "normal" from only ${calib} rows. Under ${SAFE_CALIB} rows the level-shift detector can raise false alarms.`);
  }
  out.calib = calib;
  let m;
  try {
    m = new s.Monitor(values.subarray(0, calib), 1);
  } catch (e) {
    out.error = `Could not calibrate on the first ${calib} rows: ${e.message || e}`;
    return out;
  }
  // Report the start of each alarm episode, not every alarming tick.
  let lastAlarmAt = -Infinity;
  const quiet = 50;
  for (let i = calib; i < n; i++) {
    const leg = m.push(values.subarray(i, i + 1));
    if (leg !== undefined) {
      if (i - lastAlarmAt > quiet) {
        const a = m.lastAlarm();
        out.alarms.push({ at: i, leg, explanation: a ? a.explanation : '' });
      }
      lastAlarmAt = i;
    }
  }
  return out;
}

/** Example for the page: healthy sensor, then at row 2200 the noise turns correlated with the same amplitude. */
export function exampleCsv() {
  // Healthy sensor, then at row 2200 the noise turns correlated (same amplitude): a limit check sees nothing.
  let seed = 3, out = 'time,sensor\n', prev = 0;
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  for (let i = 0; i < 4000; i++) {
    const e = gauss();
    const v = i < 2200 ? e : (prev = 0.9 * prev + Math.sqrt(1 - 0.81) * e);
    out += `${i},${(20 + v).toFixed(4)}\n`;
  }
  return out;
}
