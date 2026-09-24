// Headless-browser test of the real page: load it over HTTP, click "try an example",
// wait for the verdict. Run: node playground/csv/browser.test.mjs <dir containing index.html and pkg/>
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const root = path.resolve(process.argv[2]);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(req.url.split('?')[0]).replace(/\/$/, '/index.html'));
  if (!p.startsWith(root) || !fs.existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
try {
  await page.goto(`http://localhost:${port}/`);
  await page.click('#example');
  await page.waitForSelector('#result:not([hidden])', { timeout: 30000 });
  const verdict = await page.textContent('#verdict');
  const alarms = await page.textContent('#c-alarms');
  await page.screenshot({ path: path.join(root, 'screenshot.png'), fullPage: true });
  console.log(`verdict: ${verdict} | alarms: ${alarms}`);
  if (!/Change detected at row (22\d\d|2[3-9]\d\d)/.test(verdict)) throw new Error(`unexpected verdict: ${verdict}`);
  if (errors.length) throw new Error(`page errors: ${errors.join(' | ')}`);
  console.log('ok browser');
} finally {
  await browser.close();
  server.close();
}
