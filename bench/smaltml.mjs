// Times smaltml on the benchmark data, from the built package (npm run build first): the median of several runs after
// a warm-up, on one thread, synchronously. Writes bench/results/smaltml.json.
//   node bench/smaltml.mjs
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
const median = (f, reps) => {
  f();
  const ts = [];
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    f();
    ts.push(performance.now() - t);
  }
  return ts.sort((a, b) => a - b)[Math.floor(ts.length / 2)];
};

const results = {};
for (const { name, key, n, p } of SIZES) {
  const X = new ml.Matrix(n, p, f64(`${dir}/${key}.X`));
  const [y, y2, y3] = ['y', 'y2', 'y3'].map((s) => f64(`${dir}/${key}.${s}`));
  const reps = n < 100_000 ? 7 : 3;
  const lr = new ml.LinearRegression().fitSync(X, y);
  const l3 = new ml.LogisticRegression().fitSync(X, y3);
  results[name] = {
    'StandardScaler fit + transform': median(() => new ml.StandardScaler().fitSync(X).transformSync(X), reps),
    'LinearRegression fit': median(() => new ml.LinearRegression().fitSync(X, y), reps),
    'LinearRegression predict': median(() => lr.predictSync(X), reps),
    'Ridge fit': median(() => new ml.Ridge({ alpha: 1 }).fitSync(X, y), reps),
    'Lasso fit (alpha 0.1)': median(() => new ml.Lasso({ alpha: 0.1 }).fitSync(X, y), reps),
    'LogisticRegression fit, 2 classes': median(() => new ml.LogisticRegression().fitSync(X, y2), reps),
    'LogisticRegression fit, 3 classes': median(() => new ml.LogisticRegression().fitSync(X, y3), reps),
    'LogisticRegression predictProba, 3 classes': median(() => l3.predictProbaSync(X), reps),
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
  threads: 1,
  cpu: `${cpus()[0].model}, ${cpus().length} cores`,
  date: new Date().toISOString().slice(0, 10),
  milliseconds: results,
};
mkdirSync(new URL('./results', import.meta.url).pathname, { recursive: true });
writeFileSync(new URL('./results/smaltml.json', import.meta.url).pathname, JSON.stringify(out, null, 2) + '\n');
console.log(JSON.stringify(out, null, 2));
