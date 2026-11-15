// Times smaltml on WebGPU against the CPU, in Chromium on this machine's GPU, from the built package (it builds first).
// Both sides run in the same page, so they share a JavaScript engine: the CPU column is smaltml's CPU path in the
// browser, on one thread unless it says otherwise. Each time is the median of several runs after a warm-up, and a GPU
// call is timed end to end: converting to 32-bit floats, uploading, the shader, and reading the result back.
//
// It refuses to run on a software adapter (SwiftShader): that is the CPU pretending to be a GPU, and timing it says
// nothing. Writes bench/results/gpu.json; `npm run bench:report` adds it to the benchmarks page.
//
//   node bench/gpu.mjs [--browser PATH] [--no-build]
//
// --browser runs another Chromium (the system's, say) instead of Playwright's own.
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, statSync, mkdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import { makeData, SIZES } from './data.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name) => {
  const at = process.argv.indexOf(name);
  return at > 0 ? process.argv[at + 1] : undefined;
};
const executablePath = arg('--browser');
if (!executablePath && !existsSync(chromium.executablePath())) {
  console.error(`Playwright's Chromium isn't installed (${chromium.executablePath()}).`);
  console.error('Run `npx playwright install chromium`, or pass another Chromium: node bench/gpu.mjs --browser /usr/bin/chromium');
  process.exit(1);
}
if (!process.argv.includes('--no-build')) execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root, stdio: 'inherit' });
makeData(join(root, 'bench/data'));

const PAGE = `<!doctype html><meta charset="utf-8"><title>smaltml GPU bench</title><script type="module">
import * as ml from '/dist/smaltml.js';
const say = (...a) => console.log(a.join(' '));

/** The median of the runs after a warm-up: at least 5 runs and a second in all, or 3 when one run takes over a second. */
async function time(f) {
  await f();
  const ts = [];
  let total = 0;
  while (ts.length < 25) {
    const t = performance.now();
    await f();
    const dt = performance.now() - t;
    ts.push(dt);
    total += dt;
    if (ts.length >= 5 && total >= 1000) break;
    if (ts.length >= 3 && ts[0] >= 1000) break;
  }
  return ts.sort((a, b) => a - b)[Math.floor(ts.length / 2)];
}
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}
const matrix = (n, p, r) => new ml.Matrix(n, p, Float64Array.from({ length: n * p }, r));
const load = async (key, s) => new Float64Array(await (await fetch('/bench/data/' + key + '.' + s)).arrayBuffer());
const worst = (a, b) => { let w = 0; for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i])); return w; };

window.bench = async (sizes) => {
  const d = await ml.gpuDevice();
  if (!d) throw new Error('no WebGPU device');
  const info = d.adapterInfo ?? (await navigator.gpu.requestAdapter()).info;
  const adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: !!info.isFallbackAdapter };
  if (adapter.isFallbackAdapter || /swiftshader/i.test(adapter.architecture + ' ' + adapter.description)) {
    return { adapter, refused: 'a software adapter (' + (adapter.architecture || adapter.description || 'fallback') + '), not a GPU' };
  }
  say('adapter:', adapter.vendor, adapter.architecture, adapter.description);
  const r = lcg(11);

  // 1. matMul, n × p times p × k: the product behind every prediction on the GPU, against GPU_MIN_WORK (n·p·k).
  const matMul = [];
  for (const n of [1_000, 4_000, 16_000, 64_000, 256_000]) {
    for (const p of [16, 64]) {
      const A = matrix(n, p, r);
      for (const k of [1, 3, 8]) {
        const B = matrix(p, k, r);
        const cpu = await time(() => ml.matMul(A, B, { device: 'cpu' }));
        const gpu = await time(() => ml.matMul(A, B, { device: 'gpu' }));
        const auto = n * p * k >= ml.GPU_MIN_WORK ? 'gpu' : 'cpu';
        matMul.push({ n, p, k, work: n * p * k, cpu, gpu, auto });
        say('matMul', n + '×' + p + ' · ' + p + '×' + k, 'cpu', cpu.toFixed(2), 'gpu', gpu.toFixed(2), 'auto →', auto);
      }
    }
  }

  // 2. Predictions and fits on the CPU benchmark's data, with its parameters.
  const models = {};
  for (const { name, key, n, p } of sizes) {
    const X = new ml.Matrix(n, p, await load(key, 'X'));
    const [y, y2, y3] = await Promise.all(['y', 'y2', 'y3'].map((s) => load(key, s)));
    const lin = new ml.LinearRegression().fitSync(X, y);
    const log3 = new ml.LogisticRegression().fitSync(X, y3);
    const row = {};
    const pair = async (what, run) => {
      row[what] = { cpu: await time(() => run('cpu')), gpu: await time(() => run('gpu')) };
      say(name, what, 'cpu', row[what].cpu.toFixed(2), 'gpu', row[what].gpu.toFixed(2));
    };
    await pair('LinearRegression predict', (device) => lin.predict(X, { device }));
    await pair('LogisticRegression predictProba, 3 classes', (device) => log3.predictProba(X, { device }));
    for (const [what, target] of [['LogisticRegression fit, 2 classes', y2], ['LogisticRegression fit, 3 classes', y3]]) {
      const cpuFit = new ml.LogisticRegression().fitSync(X, target);
      const gpuFit = await new ml.LogisticRegression().fit(X, target, { device: 'gpu' });
      const cpu = await time(() => new ml.LogisticRegression().fitSync(X, target));
      const threads4 = await time(() => new ml.LogisticRegression().fit(X, target, { threads: 4 }));
      const gpu = await time(() => new ml.LogisticRegression().fit(X, target, { device: 'gpu' }));
      // The answers compared only once both fits have finished: at the default 100 iterations neither converges on
      // this data (unscaled columns), and two unfinished fits stop at different points on the way.
      const cpuEnd = new ml.LogisticRegression({ maxIter: 20000 }).fitSync(X, target);
      const gpuEnd = await new ml.LogisticRegression({ maxIter: 20000 }).fit(X, target, { device: 'gpu' });
      const diff = worst(gpuEnd.predictProbaSync(X).data, cpuEnd.predictProbaSync(X).data);
      row[what] = { cpu, threads4, gpu, iterations: { cpu: cpuFit.nIter, gpu: gpuFit.nIter }, converged: { iterations: { cpu: cpuEnd.nIter, gpu: gpuEnd.nIter }, probabilityDifference: diff } };
      say(name, what, 'cpu', cpu.toFixed(1), '4 threads', threads4.toFixed(1), 'gpu', gpu.toFixed(1), 'iterations', cpuFit.nIter + '/' + gpuFit.nIter, 'run to the end: iterations', cpuEnd.nIter + '/' + gpuEnd.nIter, 'Δp', diff.toExponential(1));
    }
    models[name] = row;
  }

  // 3. LogisticRegression fits by rows, up to 30 iterations with tol 1e-12, so neither side stops on the gradient. The
  // line search can still stop either side early, and the GPU's 32-bit objective takes its own path: both iteration
  // counts are recorded.
  const fits = [];
  for (const n of [10_000, 50_000, 200_000, 1_000_000]) {
    const p = 20;
    const X = matrix(n, p, r);
    const w = Float64Array.from({ length: p }, r);
    const score = Array.from({ length: n }, (_, i) => { let s = 0; for (let j = 0; j < p; j++) s += X.data[j * n + i] * w[j]; return s + 0.5 * r(); });
    for (const [classes, labels] of [[2, score.map((s) => (s > 0 ? 1 : 0))], [3, score.map((s) => (s < -0.6 ? 0 : s < 0.6 ? 1 : 2))]]) {
      const make = () => new ml.LogisticRegression({ maxIter: 30, tol: 1e-12 });
      const iterations = { cpu: make().fitSync(X, labels).nIter, gpu: (await make().fit(X, labels, { device: 'gpu' })).nIter };
      const cpu = await time(() => make().fitSync(X, labels));
      const threads4 = await time(() => make().fit(X, labels, { threads: 4 }));
      const gpu = await time(() => make().fit(X, labels, { device: 'gpu' }));
      fits.push({ n, p, classes, cpu, threads4, gpu, iterations });
      say('fit', n + '×' + p, classes, 'classes', 'cpu', cpu.toFixed(1), '4 threads', threads4.toFixed(1), 'gpu', gpu.toFixed(1), 'iterations', iterations.cpu + '/' + iterations.gpu);
    }
  }
  await ml.stopThreads();
  return { adapter, gpuMinWork: ml.GPU_MIN_WORK, library: 'smaltml ' + ml.VERSION, matMul, models, fits };
};
</script>`;

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.map': 'application/json' };
const server = http.createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  // Cross-origin isolated, so fits can share memory with the thread pool's workers.
  const isolated = { 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' };
  if (p === '/') return res.writeHead(200, { 'content-type': 'text/html', ...isolated }).end(PAGE);
  if (p === '/favicon.ico') return res.writeHead(204).end();
  const file = join(root, p);
  if (!(p.startsWith('/dist/') || p.startsWith('/bench/data/')) || p.includes('..') || !existsSync(file) || statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', ...isolated });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, r));

// The real GPU: Vulkan, past Chromium's GPU blocklist, the high-performance adapter where there are two (a laptop's
// discrete GPU rather than its integrated one).
const browser = await chromium.launch({
  executablePath,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--use-webgpu-power-preference=force-high-performance'],
});
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log('  ' + m.text()));
  page.on('pageerror', (e) => {
    console.error('page error: ' + e.message);
    failed = true;
  });
  await page.goto(`http://localhost:${server.address().port}/`);
  await page.waitForFunction(() => typeof window.bench === 'function');
  const result = await page.evaluate((sizes) => window.bench(sizes), SIZES);
  if (result.refused) {
    console.error(`Refused: WebGPU gave ${result.refused}. Timing it would measure the CPU.`);
    failed = true;
  } else {
    const out = {
      ...result,
      browser: `${executablePath ? 'Chromium' : 'Playwright Chromium'} ${browser.version()}`,
      cpu: `${cpus()[0].model}, ${cpus().length} cores`,
      date: new Date().toISOString().slice(0, 10),
    };
    mkdirSync(join(root, 'bench/results'), { recursive: true });
    writeFileSync(join(root, 'bench/results/gpu.json'), JSON.stringify(out, null, 2) + '\n');
    // GPU_MIN_WORK is the line 'auto' draws: the cases measured where it picks the slower device.
    const wrong = out.matMul.filter((c) => (c.auto === 'gpu') !== c.gpu < c.cpu);
    console.log(`\nmatMul: 'auto' (GPU_MIN_WORK ${out.gpuMinWork.toLocaleString('en-US')}) picks the slower device in ${wrong.length} of ${out.matMul.length} cases:`);
    for (const c of wrong) console.log(`  ${c.n}×${c.p} · ${c.p}×${c.k}: auto → ${c.auto}, cpu ${c.cpu.toFixed(2)} ms, gpu ${c.gpu.toFixed(2)} ms`);
    console.log('Wrote bench/results/gpu.json');
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
