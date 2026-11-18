// smaltml's SIMD kernels with its models (ADR 0040, carrying ADR 0035's items 7 and 8): useSimd() replaces the
// kernels, fits agree with smaltml's own within rounding, fits on several threads equal fitSync bit for bit, and the
// kernel set the workers load (one file, no imports: dist/smaltml-simd-kernels.js, bundled here from the source as the
// build bundles it) computes what the source's does.
import assert from 'node:assert/strict';
import * as esbuild from 'esbuild';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, afterEach, beforeAll, test } from 'vitest';
import { configureThreads, kernelsInUse, LinearRegression, LogisticRegression, Matrix, matMulSync, Ridge, stopThreads, useKernels } from '../src/index.js';
import { threadPool } from '../src/parallel/index.js';
import source from '../src/simd/kernels.js';
import { useSimd, type SimdOptions } from '../src/simd/index.js';

const dir = mkdtempSync(join(tmpdir(), 'smaltml-simd-test-'));
const bundled = join(dir, 'smaltml-simd-kernels.js');
let kernelsModule = '';
beforeAll(async () => {
  const common = { bundle: true, format: 'esm', platform: 'neutral', logLevel: 'silent' } as const;
  await esbuild.build({ ...common, entryPoints: ['src/parallel/worker.ts'], outfile: join(dir, 'smaltml-threads.js') });
  await esbuild.build({ ...common, entryPoints: ['src/simd/kernels.ts'], outfile: bundled });
  configureThreads({ script: pathToFileURL(join(dir, 'smaltml-threads.js')) });
  kernelsModule = pathToFileURL(bundled).href;
});
afterEach(() => {
  useKernels(null);
});
afterAll(() => stopThreads());

/** useSimd(), with the workers' kernel set from the bundle. */
const turnOn = (options: SimdOptions = {}) => useSimd({ ...options, kernelsModule });

function random(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

/** 40,000 × 12: over smaltml's parallel threshold, in three chunks of rows. */
const n = 40_000;
const p = 12;
const r = random(17);
const X = new Matrix(n, p, Float64Array.from({ length: n * p }, (_, k) => r() * (1 + (k % 5))));
const w = Float64Array.from({ length: p }, () => r());
const y = Float64Array.from({ length: n }, (_, i) => { let s = 0.3; for (let j = 0; j < p; j++) s += X.data[j * n + i]! * w[j]!; return s + 0.1 * r(); });
const labels = Float64Array.from(y, (v) => (v + 0.2 * r() > 0 ? 1 : 0));

const near = (a: ArrayLike<number>, b: ArrayLike<number>, rel: number) => {
  const scale = Math.max(...Array.from(b, Math.abs));
  for (let i = 0; i < b.length; i++) assert.ok(Math.abs(a[i]! - b[i]!) <= rel * scale, `${a[i]} vs ${b[i]}`);
};

test('simd() replaces gramPart, and useKernels(null) puts smaltml\'s back', async () => {
  assert.equal(await turnOn(), true);
  assert.deepEqual(kernelsInUse(), { name: 'smaltml-simd', kernels: ['gramPart', 'matMul', 'binaryLogisticRows', 'multinomialLogisticRows', 'logisticProducts'], rejected: [] });
  useKernels(null);
  assert.equal(kernelsInUse().name, 'smaltml');
  assert.equal(await turnOn({ fusedMultiplyAdd: true }), true);
  assert.equal(kernelsInUse().name, 'smaltml-simd, fused multiply-add');
});

test('fits with SIMD agree with smaltml\'s own within rounding, and move in the last bits', async () => {
  const coef = (m: { coef: Float64Array | Matrix }) => (m.coef instanceof Matrix ? m.coef.data : m.coef);
  const fits: (() => { coef: Float64Array | Matrix })[] = [
    () => new LinearRegression().fitSync(X, y),
    () => new Ridge({ alpha: 2 }).fitSync(X, y),
    () => new LogisticRegression({ solver: 'newton-cholesky' }).fitSync(X, labels),
  ];
  for (const fit of fits) {
    const own = fit();
    for (const fusedMultiplyAdd of [false, true]) {
      await turnOn({ fusedMultiplyAdd });
      const simd = fit();
      near(coef(simd), coef(own), 1e-12);
      useKernels(null);
    }
  }
});

test('fits on several threads with SIMD equal fitSync, bit for bit', async () => {
  for (const fusedMultiplyAdd of [false, true]) {
    await turnOn({ fusedMultiplyAdd });
    // The pool starts only if every worker loaded the SIMD kernels; otherwise fits would quietly run on one thread.
    assert.notEqual(await threadPool(4, Infinity), null, 'the workers load the kernels');
    const one = new Ridge({ alpha: 2 }).fitSync(X, y).toJSON();
    for (const threads of [2, 4]) {
      assert.deepEqual((await new Ridge({ alpha: 2 }).fit(X, y, { threads })).toJSON(), one, `${threads} threads`);
    }
    const lone = new LogisticRegression({ solver: 'newton-cholesky' }).fitSync(X, labels).toJSON();
    assert.deepEqual((await new LogisticRegression({ solver: 'newton-cholesky' }).fit(X, labels, { threads: 3 })).toJSON(), lone);
    useKernels(null);
  }
});

test('matMulSync with SIMD gives smaltml\'s result bit for bit, and fused within rounding', async () => {
  const A = new Matrix(1001, 7, Float64Array.from({ length: 1001 * 7 }, () => r() * 5));
  const B = new Matrix(7, 3, Float64Array.from({ length: 21 }, r));
  const own = matMulSync(A, B).data;
  await turnOn();
  assert.deepEqual(matMulSync(A, B).data, own);
  await turnOn({ fusedMultiplyAdd: true });
  near(matMulSync(A, B).data, own, 1e-14);
});

test('lbfgs logistic fits with the SIMD row kernels agree with smaltml\'s own within rounding, and threads equal fitSync', async () => {
  const labels3 = Float64Array.from(y, (v) => (v < -0.5 ? 0 : v < 0.5 ? 1 : 2));
  for (const target of [labels, labels3]) {
    const own = new LogisticRegression({ maxIter: 30 }).fitSync(X, target);
    for (const fusedMultiplyAdd of [false, true]) {
      await turnOn({ fusedMultiplyAdd, logistic: true });
      const simd = new LogisticRegression({ maxIter: 30 }).fitSync(X, target);
      near((simd.coef as Matrix).data, (own.coef as Matrix).data, 1e-9);
      assert.deepEqual((await new LogisticRegression({ maxIter: 30 }).fit(X, target, { threads: 3 })).toJSON(), simd.toJSON());
      useKernels(null);
    }
  }
});

test('the logistic row kernels are on by default, and off with logistic: false', async () => {
  await turnOn();
  assert.deepEqual(kernelsInUse().kernels, ['gramPart', 'matMul', 'binaryLogisticRows', 'multinomialLogisticRows', 'logisticProducts']);
  await turnOn({ logistic: false });
  assert.deepEqual(kernelsInUse().kernels, ['gramPart', 'matMul', 'logisticProducts']);
});

test('the bundle imports nothing', () => {
  assert.doesNotMatch(readFileSync(bundled, 'utf8'), /^\s*import\b/m);
});

test('its kernel set computes what the source\'s does, bit for bit', async () => {
  const bundled = (await import(kernelsModule)) as { default: typeof source };
  const X = { rows: 777, cols: 9, data: Float64Array.from({ length: 777 * 9 }, (_, i) => Math.sin(i) * 7) };
  for (const fusedMultiplyAdd of [false, true]) {
    const [a, b] = [await bundled.default({ fusedMultiplyAdd }), await source({ fusedMultiplyAdd })];
    assert.equal(a.name, b.name);
    assert.deepEqual(a.kernels.gramPart!(X as never, undefined, 0, 777), b.kernels.gramPart!(X as never, undefined, 0, 777));
  }
});

test('smaltml\'s workers load the bundle given as kernelsModule, and threaded fits equal fitSync', async () => {
  const n = 40_000;
  const p = 6;
  const X = new Matrix(n, p, Float64Array.from({ length: n * p }, (_, i) => Math.cos(i * 0.37) * (1 + (i % 3))));
  const y = Float64Array.from({ length: n }, (_, i) => X.data[i]! - 2 * X.data[n + i]! + Math.sin(i));
  assert.equal(await useSimd({ kernelsModule }), true);
  assert.notEqual(await threadPool(3, Infinity), null, 'the workers load the bundle');
  const one = new Ridge().fitSync(X, y).toJSON();
  assert.deepEqual((await new Ridge().fit(X, y, { threads: 3 })).toJSON(), one);
});
