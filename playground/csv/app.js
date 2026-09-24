import init, * as s from './pkg/struktura.js';
import { parseCsv, analyze, exampleCsv } from './analysis.js';

const $ = (id) => document.getElementById(id);
let parsed = null;
let ready = null;

function show(el, on) { el.hidden = !on; }

async function load(text, label) {
  await ready;
  parsed = parseCsv(text);
  if (parsed.columns.length === 0) {
    show($('colrow'), false);
    render({ error: 'No numeric column found. Each column should hold numbers, one row per time step.' });
    return;
  }
  const sel = $('col');
  sel.innerHTML = '';
  parsed.columns.forEach((c, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${c.name} (${c.values.length} rows)`;
    sel.appendChild(o);
  });
  // Default to the last numeric column: often the measured value after a time/index column.
  sel.value = String(parsed.columns.length - 1);
  $('fileinfo').textContent = label;
  show($('colrow'), true);
  run();
}

function run() {
  const col = parsed.columns[Number($('col').value)];
  const r = analyze(s, col.values);
  render(r, col);
}

function render(r, col) {
  show($('result'), true);
  const v = $('verdict');
  $('warnings').innerHTML = '';
  $('alarms').innerHTML = '';
  if (r.error) {
    v.className = 'err';
    v.textContent = r.error;
    $('detail').textContent = '';
    ['c-rows', 'c-alarms', 'c-alpha'].forEach((id) => { $(id).textContent = '-'; });
    clearChart();
    return;
  }
  const first = r.alarms[0];
  if (first) {
    v.className = 'alarm';
    v.textContent = `Change detected at row ${first.at + 1}`;
    $('detail').textContent = `${first.explanation} (detector: ${first.leg.replace(/_/g, ' ')}). Learned "normal" from rows 1 to ${r.calib}.`;
  } else {
    v.className = 'ok';
    v.textContent = 'No change detected';
    $('detail').textContent = `Rows ${r.calib + 1} to ${r.n} look like rows 1 to ${r.calib}, which were taken as normal.`;
  }
  for (const w of r.warnings) {
    const p = document.createElement('p');
    p.className = 'warn';
    p.textContent = w;
    $('warnings').appendChild(p);
  }
  $('c-rows').textContent = String(r.n);
  $('c-alarms').textContent = String(r.alarms.length);
  $('c-alpha').textContent = r.whole ? r.whole.alpha.toFixed(2) : 'n/a';
  for (const a of r.alarms.slice(0, 20)) {
    const li = document.createElement('li');
    li.innerHTML = `<code>row ${a.at + 1}</code> &nbsp;${escapeHtml(a.explanation)} <span style="color:var(--dim)">(${a.leg.replace(/_/g, ' ')})</span>`;
    $('alarms').appendChild(li);
  }
  if (r.alarms.length > 20) {
    const li = document.createElement('li');
    li.textContent = `…and ${r.alarms.length - 20} more`;
    $('alarms').appendChild(li);
  }
  drawChart(col.values, r);
}

function escapeHtml(t) {
  return String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function clearChart() {
  const c = $('chart');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function drawChart(values, r) {
  const c = $('chart');
  const g = c.getContext('2d');
  const W = c.width, H = c.height, pad = 16;
  g.clearRect(0, 0, W, H);
  const n = values.length;
  // Min/max per pixel column so spikes survive downsampling.
  const cols = W - 2 * pad;
  let lo = Infinity, hi = -Infinity;
  for (const x of values) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (hi === lo) { hi += 1; lo -= 1; }
  const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  const x = (i) => pad + (i / Math.max(1, n - 1)) * cols;
  g.fillStyle = 'rgba(108,182,255,0.18)';
  g.fillRect(x(0), 0, x(r.calib) - x(0), H);
  g.strokeStyle = '#fff';
  g.lineWidth = 1.2;
  g.beginPath();
  for (let p = 0; p < cols; p++) {
    const a = Math.floor((p / cols) * n), b = Math.max(a + 1, Math.floor(((p + 1) / cols) * n));
    let mn = Infinity, mx = -Infinity;
    for (let i = a; i < b && i < n; i++) { if (values[i] < mn) mn = values[i]; if (values[i] > mx) mx = values[i]; }
    if (!Number.isFinite(mn)) continue;
    g.moveTo(pad + p, y(mn));
    g.lineTo(pad + p, y(mx) - 0.5);
  }
  g.stroke();
  g.strokeStyle = '#ff5e57';
  g.lineWidth = 3;
  for (const a of r.alarms) {
    g.beginPath();
    g.moveTo(x(a.at), 0);
    g.lineTo(x(a.at), H);
    g.stroke();
  }
}

ready = init();
$('pick').onclick = () => $('file').click();
$('file').onchange = async (e) => {
  const f = e.target.files[0];
  if (f) load(await f.text(), `${f.name}, ${(f.size / 1024).toFixed(0)} KB`);
};
$('example').onclick = () => load(exampleCsv(), 'example: healthy, then the noise structure changes at row 2200 with the same amplitude');
$('col').onchange = run;
const drop = $('drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) load(await f.text(), `${f.name}, ${(f.size / 1024).toFixed(0)} KB`);
});
