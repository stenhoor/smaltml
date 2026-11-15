// Browser test for the hardware: WebGPU and the worker, on the built package in Chromium (WebGPU on SwiftShader, as in
// smaltcy's CI). The GPU's results must agree with the CPU's within the stated tolerance; the worker's fits must equal
// local fits bit for bit. Fails on any page error.
//
//   node test/browser/hardware.mjs [--no-build]
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, resolve, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
if (!process.argv.includes('--no-build')) execFileSync(process.execPath, ['scripts/build.mjs'], { cwd: root, stdio: 'inherit' });

const PAGE = `<!doctype html><meta charset="utf-8"><title>smaltml hardware</title><script type="module">
import * as ml from '/dist/smaltml.js';
import { startWorker } from '/dist/worker.js';
window.results = (async () => {
  const out = [];
  const check = (ok, msg) => out.push({ ok: !!ok, msg });
  const rand = (seed) => { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1; };
  const r = rand(7);
  const matrix = (n, p, scale = 1) => new ml.Matrix(n, p, Float64Array.from({ length: n * p }, () => r() * scale));
  /** |gpu − cpu| within 1e-5 of the size of the terms summed: 32-bit floats. */
  const agree = (gpu, cpu, A, B, what) => {
    const absA = new ml.Matrix(A.rows, A.cols, A.data.map(Math.abs));
    const absB = new ml.Matrix(B.rows, B.cols, B.data.map(Math.abs));
    const size = ml.matMulSync(absA, absB).data;
    let worst = 0;
    for (let i = 0; i < cpu.data.length; i++) worst = Math.max(worst, Math.abs(gpu.data[i] - cpu.data[i]) / (size[i] + 1e-30));
    check(worst <= 1e-5, what + ': worst relative difference ' + worst.toExponential(2));
  };

  check(await ml.gpuAvailable(), 'WebGPU is available');
  for (const [n, p, k] of [[1, 1, 1], [17, 3, 1], [1000, 33, 5], [5000, 128, 16], [300, 1, 40]]) {
    const A = matrix(n, p, 10), B = matrix(p, k);
    agree(await ml.matMul(A, B, { device: 'gpu' }), ml.matMulSync(A, B), A, B, 'matMul ' + n + '×' + p + ' · ' + p + '×' + k);
  }

  // Models: predictions on the GPU against the CPU's.
  const X = matrix(4000, 20, 3);
  const w = Float64Array.from({ length: 20 }, () => r());
  const y = Array.from({ length: 4000 }, (_, i) => { let s = 0; for (let j = 0; j < 20; j++) s += X.data[j * 4000 + i] * w[j]; return s + 0.1 * r(); });
  const lin = new ml.LinearRegression().fitSync(X, y);
  const gpuLin = await lin.predict(X, { device: 'gpu' });
  const cpuLin = lin.predictSync(X);
  let worst = 0;
  for (let i = 0; i < cpuLin.length; i++) worst = Math.max(worst, Math.abs(gpuLin[i] - cpuLin[i]));
  check(worst < 1e-3, 'LinearRegression on the GPU: worst difference ' + worst.toExponential(2));
  const labels = y.map((v) => (v > 0.5 ? 'high' : v < -0.5 ? 'low' : 'mid'));
  const log = new ml.LogisticRegression({ maxIter: 50 }).fitSync(X, labels);
  const gp = await log.predictProba(X, { device: 'gpu' });
  const cp = log.predictProbaSync(X);
  worst = 0;
  for (let i = 0; i < cp.data.length; i++) worst = Math.max(worst, Math.abs(gp.data[i] - cp.data[i]));
  check(worst < 1e-4, 'LogisticRegression probabilities on the GPU: worst difference ' + worst.toExponential(2));
  const pipe = ml.pipeline(new ml.StandardScaler(), new ml.LogisticRegression()).fitSync(X, labels);
  const gl = await pipe.predict(X, { device: 'gpu' });
  const cl = pipe.predictSync(X);
  const same = gl.filter((v, i) => v === cl[i]).length / cl.length;
  check(same > 0.999, 'pipeline labels on the GPU: ' + (same * 100).toFixed(2) + '% the same as the CPU');

  // 'auto' predicts on the CPU even with WebGPU here and X above GPU_MIN_WORK: exactly the *Sync result (ADR 0034).
  const big = matrix(Math.ceil(ml.GPU_MIN_WORK / 20) + 1000, 20, 3);
  const [autoLin, syncLin] = [await lin.predict(big), lin.predictSync(big)];
  check(autoLin.every((v, i) => v === syncLin[i]), "LinearRegression.predict with 'auto', above GPU_MIN_WORK: the CPU's result, bit for bit");
  const [autoProba, syncProba] = [(await log.predictProba(big, { device: 'auto' })).data, log.predictProbaSync(big).data];
  check(autoProba.every((v, i) => v === syncProba[i]), "LogisticRegression.predictProba with 'auto', above GPU_MIN_WORK: the CPU's result, bit for bit");

  // Fits on the GPU: the objective in 32-bit floats, so close to the CPU's fit, not equal to it.
  for (const [what, labels_] of [['2 classes', labels.map((l) => (l === 'high' ? 'high' : 'other'))], ['3 classes', labels]]) {
    const onGpu = await new ml.LogisticRegression({ maxIter: 200 }).fit(X, labels_, { device: 'gpu' });
    const onCpu = new ml.LogisticRegression({ maxIter: 200 }).fitSync(X, labels_);
    const pg = onGpu.predictProbaSync(X).data;
    const pc = onCpu.predictProbaSync(X).data;
    let worstP = 0;
    for (let i = 0; i < pc.length; i++) worstP = Math.max(worstP, Math.abs(pg[i] - pc[i]));
    check(worstP < 2e-3, 'LogisticRegression fit on the GPU, ' + what + ': probabilities within ' + worstP.toExponential(2) + ' of the CPU fit');
  }

  // Threads: the page is cross-origin isolated, so fits run on a pool of workers sharing X, and equal fitSync.
  const par = await import('/dist/parallel.js');
  check(ml.sharedMemoryAvailable(), 'shared memory on a cross-origin isolated page');
  check((await par.threadPool(4, 1e9)) !== null, 'the thread pool starts its workers');
  const T = matrix(30000, 40, 2);
  const tLabels = Array.from({ length: 30000 }, (_, i) => (T.data[i] + T.data[30000 + i] > 0 ? 'a' : T.data[60000 + i] > 0 ? 'b' : 'c'));
  const tY = Array.from({ length: 30000 }, (_, i) => T.data[i] * 2 - T.data[90000 + i]);
  for (const [what, make, target] of [
    ['LogisticRegression, 3 classes', () => new ml.LogisticRegression({ maxIter: 30 }), tLabels],
    ['Ridge', () => new ml.Ridge(), tY],
    ['Lasso', () => new ml.Lasso({ alpha: 0.01 }), tY],
  ]) {
    const many = await make().fit(T, target, { threads: 4 });
    const one = make().fitSync(T, target);
    check(JSON.stringify(many.toJSON()) === JSON.stringify(one.toJSON()), what + ' on 4 threads equals fitSync, bit for bit');
  }
  await ml.stopThreads();

  // The worker: the same fits, off the page's thread.
  const worker = startWorker();
  try {
    const fitted = await worker.fit(new ml.LogisticRegression({ C: 0.5 }), X, labels);
    const local = new ml.LogisticRegression({ C: 0.5 }).fitSync(X, labels);
    check(fitted.coef.data.every((v, i) => v === local.coef.data[i]), 'a fit in the worker equals the local fit, bit for bit');
    const wp = await worker.predict(fitted, X);
    check(JSON.stringify(wp) === JSON.stringify(local.predictSync(X)), 'predictions in the worker equal local ones');
    const wg = await worker.predictProba(fitted, X, { device: 'gpu' });
    let ww = 0;
    for (let i = 0; i < wg.data.length; i++) ww = Math.max(ww, Math.abs(wg.data[i] - local.predictProbaSync(X).data[i]));
    check(ww < 1e-4, 'WebGPU in the worker: worst difference ' + ww.toExponential(2));
    let code = null;
    try { await worker.fit(new ml.LinearRegression(), X, [1, 2]); } catch (e) { code = e.code; }
    check(code === 'ML_SHAPE', 'errors in the worker come back with their code (' + code + ')');
  } finally {
    worker.terminate();
  }
  return out;
})().catch((e) => [{ ok: false, msg: 'threw: ' + (e && e.stack || e) }]);
</script>`;

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.map': 'application/json' };
const server = http.createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  // Cross-origin isolated, so the page can share memory with the thread pool's workers.
  const isolated = { 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' };
  if (p === '/') return res.writeHead(200, { 'content-type': 'text/html', ...isolated }).end(PAGE);
  const file = join(root, p);
  if (!p.startsWith('/dist/') || !existsSync(file) || statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', ...isolated });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, r));

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-gl=angle', '--use-angle=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(`http://localhost:${server.address().port}/`);
const results = await page.evaluate(() => window.results);
let failures = 0;
for (const { ok, msg } of results) {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failures++;
}
if (errors.length) {
  console.log(`  FAIL page errors: ${errors.join(' | ')}`);
  failures++;
}
await browser.close();
server.close();
process.exit(failures ? 1 : 0);
