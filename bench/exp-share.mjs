// How much of the logistic objective's time goes to exp, log1p and log (smaltml ADR 0035, "Still to decide", item 4):
// smaltml's binary and multinomial objectives on the benchmark data, timed as they are and with Math.exp, Math.log1p
// and Math.log replaced by cheap arithmetic of the same shape. The difference is their share: what a faster exp could
// save at most. The median of many runs after a warm-up; of five rounds, the one with the median time.
//   node bench/exp-share.mjs
import * as esbuild from 'esbuild';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { makeData, SIZES } from './data.mjs';

const dir = mkdtempSync(join(tmpdir(), 'smaltml-exp-'));
const entry = join(dir, 'entry.js');
writeFileSync(entry, `export { binaryLogisticPart, multinomialLogisticPart } from ${JSON.stringify(new URL('../src/core/kernels.ts', import.meta.url).pathname)};
export { Matrix } from ${JSON.stringify(new URL('../src/core/matrix.ts', import.meta.url).pathname)};`);
await esbuild.build({ entryPoints: [entry], bundle: true, format: 'esm', platform: 'node', outfile: join(dir, 'real.mjs'), logLevel: 'warning' });
const real = readFileSync(join(dir, 'real.mjs'), 'utf8');
const cheap = 'const __exp = (x) => 1 + x * x * 0.5; const __log1p = (x) => x * 0.5; const __log = (x) => x * 0.5;\n' +
  real.replace(/Math\.exp\(/g, '__exp(').replace(/Math\.log1p\(/g, '__log1p(').replace(/Math\.log\(/g, '__log(');
writeFileSync(join(dir, 'cheap.mjs'), cheap);
const R = await import(join(dir, 'real.mjs'));
const C = await import(join(dir, 'cheap.mjs'));

const data = new URL('./data', import.meta.url).pathname;
makeData(data);
const f64 = (f) => {
  const b = readFileSync(f);
  return new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8).slice();
};
const median = (f, reps) => {
  for (let i = 0; i < 20; i++) f();
  const t = [];
  for (let i = 0; i < reps; i++) {
    const s = performance.now();
    f();
    t.push(performance.now() - s);
  }
  return t.sort((a, b) => a - b)[reps >> 1];
};

console.log('| Data | Objective | ms per evaluation | Without exp, log1p, log | Their share |');
console.log('| --- | --- | ---: | ---: | ---: |');
for (const { name, key, n, p } of SIZES) {
  const X = new R.Matrix(n, p, f64(`${data}/${key}.X`));
  const y2 = Int32Array.from(f64(`${data}/${key}.y2`));
  const y3 = Int32Array.from(f64(`${data}/${key}.y3`));
  const x2 = Float64Array.from({ length: p + 1 }, (_, j) => Math.sin(j) * 0.01);
  const x3 = Float64Array.from({ length: 3 * (p + 1) }, (_, j) => Math.sin(j) * 0.01);
  const g = new Float64Array(3 * (p + 1));
  const reps = n * p < 1e6 ? 201 : 31;
  for (const [objective, run] of [
    ['binary', (M) => M.binaryLogisticPart(X, y2, true, x2, g, 0, n)],
    ['3 classes', (M) => M.multinomialLogisticPart(X, y3, 3, true, x3, g, 0, n)],
  ]) {
    const rounds = [0, 1, 2, 3, 4].map(() => [median(() => run(R), reps), median(() => run(C), reps)]);
    const [real, cheap] = rounds.sort((a, b) => a[0] - b[0])[2];
    console.log(`| ${name} | ${objective} | ${real.toFixed(2)} | ${cheap.toFixed(2)} | ${((100 * (real - cheap)) / real).toFixed(0)}% |`);
  }
}
