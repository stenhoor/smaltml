// Times smaltml on the benchmark data, from the built package (npm run build first): the median of several runs after
// a warm-up. On one thread, synchronously, it writes bench/results/smaltml.json; with --threads N, the fits run with
// fit(X, y, { threads: N }) and it writes bench/results/smaltml-N.json. With --simd, smaltml's SIMD kernels run
// (dist/simd.js's useSimd(), its defaults, as smaltml turns them on by itself), and it writes smaltml-simd.json or
// smaltml-simd-N.json. Without it, smaltml's own TypeScript kernels (useKernels(null)).
// With --simd --logistic, the logistic row steps on SIMD too (useSimd({ logistic: true })), written to
// smaltml-simd-logistic.json or smaltml-simd-logistic-N.json.
//   node bench/smaltml.mjs [--threads N] [--simd [--logistic]]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { cpus } from 'node:os';
import { performance } from 'node:perf_hooks';
import { makeData, SIZES } from './data.mjs';

const ml = await import('../dist/smaltml.js');
const dir = new URL('./data', import.meta.url).pathname;
makeData(dir);
const f64 = (f) => {
  const b = readFileSync(f);
  return new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8).slice();
};
const at = process.argv.indexOf('--threads');
const threads = at > 0 ? Number(process.argv[at + 1]) : 1;
const simdOn = process.argv.includes('--simd');
const logistic = simdOn && process.argv.includes('--logistic');
// smaltml's own kernels, unless --simd: its SIMD kernels, as smaltml turns them on (the same copy of smaltml).
let kernels = ml.useKernels(null);
if (simdOn) {
  const { useSimd } = await import('../dist/simd.js');
  if (!(await useSimd({ logistic }))) throw new Error(`the SIMD kernels weren't used: ${JSON.stringify(ml.kernelsInUse())}`);
  kernels = ml.kernelsInUse();
}
const median = async (f, reps) => {
  await f();
  const ts = [];
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    await f();
    ts.push(performance.now() - t);
  }
  return ts.sort((a, b) => a - b)[Math.floor(ts.length / 2)];
};
// Up to N threads, as threads: 'auto' chooses them: small fits stay on one thread.
if (threads > 1) ml.configureThreads({ max: threads });
/** A fit: fitSync on one thread, else fit() with threads 'auto'. */
const fit = (model, X, y) => (threads === 1 ? model.fitSync(X, y) : model.fit(X, y, { threads: 'auto' }));

const results = {};
for (const { name, key, n, p } of SIZES) {
  const X = new ml.Matrix(n, p, f64(`${dir}/${key}.X`));
  const [y, y2, y3] = ['y', 'y2', 'y3'].map((s) => f64(`${dir}/${key}.${s}`));
  const reps = n < 100_000 ? 7 : 3;
  const lr = new ml.LinearRegression().fitSync(X, y);
  const l3 = new ml.LogisticRegression().fitSync(X, y3);
  results[name] = {
    'StandardScaler fit + transform': await median(() => new ml.StandardScaler().fitSync(X).transformSync(X), reps),
    'LinearRegression fit': await median(() => fit(new ml.LinearRegression(), X, y), reps),
    'LinearRegression predict': await median(() => lr.predictSync(X), reps),
    'Ridge fit': await median(() => fit(new ml.Ridge({ alpha: 1 }), X, y), reps),
    'Lasso fit (alpha 0.1)': await median(() => fit(new ml.Lasso({ alpha: 0.1 }), X, y), reps),
    'LogisticRegression fit, 2 classes': await median(() => fit(new ml.LogisticRegression(), X, y2), reps),
    'LogisticRegression fit, 2 classes, newton-cholesky': await median(() => fit(new ml.LogisticRegression({ solver: 'newton-cholesky' }), X, y2), reps),
    'LogisticRegression fit, 3 classes': await median(() => fit(new ml.LogisticRegression(), X, y3), reps),
    'LogisticRegression predictProba, 3 classes': await median(() => l3.predictProbaSync(X), reps),
    'DecisionTreeRegressor fit, depth 10': await median(() => new ml.DecisionTreeRegressor({ maxDepth: 10, seed: 0 }).fitSync(X, y), reps),
    'DecisionTreeClassifier fit, 3 classes, depth 10': await median(() => new ml.DecisionTreeClassifier({ maxDepth: 10, seed: 0 }).fitSync(X, y3), reps),
  };
  results[name].iterations = {
    lasso: new ml.Lasso({ alpha: 0.1 }).fitSync(X, y).nIter,
    logistic2: new ml.LogisticRegression().fitSync(X, y2).nIter,
    logistic3: l3.nIter,
  };
}
const out = {
  library: `smaltml ${ml.VERSION}`,
  runtime: `Node ${process.versions.node}`,
  threads,
  simd: simdOn ? kernels : null,
  cpu: `${cpus()[0].model}, ${cpus().length} cores`,
  date: new Date().toISOString().slice(0, 10),
  milliseconds: results,
};
mkdirSync(new URL('./results', import.meta.url).pathname, { recursive: true });
writeFileSync(new URL(`./results/smaltml${simdOn ? '-simd' : ''}${logistic ? '-logistic' : ''}${threads === 1 ? '' : `-${threads}`}.json`, import.meta.url).pathname, JSON.stringify(out, null, 2) + '\n');
console.log(JSON.stringify(out, null, 2));
await ml.stopThreads();
