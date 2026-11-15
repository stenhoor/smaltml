import * as esbuild from 'esbuild';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { gram, xtv, xw } from '../src/core/kernels.js';
import { Matrix } from '../src/core/matrix.js';
import { Random } from '../src/core/random.js';
import { ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, StandardScaler, configureThreads, pipeline, stopThreads } from '../src/index.js';
import { gramParallel, threadPool, xtvParallel, xwParallel } from '../src/parallel/index.js';

// The workers run the built script; the tests bundle it from the source once.
beforeAll(async () => {
  const dir = mkdtempSync(join(tmpdir(), 'smaltml-threads-'));
  const file = join(dir, 'smaltml-threads.js');
  await esbuild.build({ entryPoints: ['src/parallel/worker.ts'], bundle: true, format: 'esm', platform: 'neutral', outfile: file, logLevel: 'silent' });
  configureThreads({ script: pathToFileURL(file) });
});
afterAll(() => stopThreads());

/** 40,000 × 30: over the parallel threshold, and three chunks of rows split unevenly between workers. */
function data() {
  const r = new Random(11);
  const n = 40_000;
  const p = 30;
  const X = new Matrix(n, p, Float64Array.from({ length: n * p }, (_, k) => (r.next() - 0.5) * (1 + (k % 7))));
  const w = Float64Array.from({ length: p }, () => r.next() - 0.5);
  const score = xw(X, w, 0);
  const y = Float64Array.from(score, (s) => s + 0.1 * (r.next() - 0.5));
  const y2 = Float64Array.from(score, (s) => (s + 0.3 * (r.next() - 0.5) > 0 ? 1 : 0));
  const y3 = Float64Array.from(score, (s) => (s < -0.5 ? 0 : s < 0.5 ? 1 : 2));
  return { X, y, y2, y3 };
}
const { X, y, y2, y3 } = data();

describe('the thread pool', () => {
  test('starts in Node, and stays out of small or one-thread work', async () => {
    expect(await threadPool(4, X.rows * X.cols)).not.toBeNull();
    expect(await threadPool('auto', 100)).toBeNull();
    expect(await threadPool(1, X.rows * X.cols)).toBeNull();
    await expect(threadPool(0, 10)).rejects.toMatchObject({ code: 'ML_PARAM' });
  });

  test('the parallel kernels equal the single-threaded ones, bit for bit', async () => {
    const pool = (await threadPool(3, X.rows * X.cols))!;
    const mean = Float64Array.from({ length: X.cols }, (_, j) => j / 10);
    expect(Array.from(await gramParallel(pool, X, mean))).toEqual(Array.from(gram(X, mean)));
    expect(Array.from(await xtvParallel(pool, X, y, mean))).toEqual(Array.from(xtv(X, y, mean)));
    const w = Float64Array.from({ length: X.cols }, (_, j) => Math.sin(j));
    expect(Array.from(await xwParallel(pool, X, w, 2, mean))).toEqual(Array.from(xw(X, w, 2, mean)));
  });
});

describe('fits on several threads equal fitSync, bit for bit', () => {
  const cases = [
    ['LinearRegression', () => new LinearRegression(), () => y],
    ['Ridge', () => new Ridge({ alpha: 3 }), () => y],
    ['Lasso', () => new Lasso({ alpha: 0.01 }), () => y],
    ['ElasticNet', () => new ElasticNet({ alpha: 0.01, l1Ratio: 0.3 }), () => y],
    ['LogisticRegression, 2 classes', () => new LogisticRegression({ maxIter: 30 }), () => y2],
    ['LogisticRegression, 3 classes', () => new LogisticRegression({ maxIter: 30 }), () => y3],
    ['LogisticRegression, newton-cholesky', () => new LogisticRegression({ solver: 'newton-cholesky' }), () => y2],
    ['LogisticRegression, newton-cholesky, 3 classes', () => new LogisticRegression({ solver: 'newton-cholesky' }), () => y3],
  ] as const;
  for (const [name, make, target] of cases) {
    test(name, async () => {
      const one = (make() as { fitSync(X: Matrix, y: Float64Array): { toJSON(): unknown } }).fitSync(X, target());
      for (const threads of [2, 4]) {
        const many = await (make() as { fit(X: Matrix, y: Float64Array, o: object): Promise<{ toJSON(): unknown }> }).fit(X, target(), { threads });
        expect(many.toJSON(), `${threads} threads`).toEqual(one.toJSON());
      }
    });
  }

  test('a pipeline passes threads to its last step', async () => {
    const one = pipeline(new StandardScaler(), new LogisticRegression({ maxIter: 20 })).fitSync(X, y3);
    const many = await pipeline(new StandardScaler(), new LogisticRegression({ maxIter: 20 })).fit(X, y3, { threads: 4 });
    expect(many.toJSON()).toEqual(one.toJSON());
  });
});

describe('weighted fits on several threads equal fitSync, bit for bit', () => {
  const r = new Random(5);
  const w = Float64Array.from({ length: X.rows }, () => 2 * r.next());
  for (const [name, make] of [
    ['LinearRegression', () => new LinearRegression()],
    ['Ridge', () => new Ridge({ alpha: 3 })],
    ['Lasso', () => new Lasso({ alpha: 0.01 })],
  ] as const) {
    test(name, async () => {
      const one = make().fitSync(X, y, { sampleWeight: w });
      const many = await make().fit(X, y, { threads: 4, sampleWeight: w });
      expect(many.toJSON()).toEqual(one.toJSON());
    });
  }
  for (const [name, make, target] of [
    ['LogisticRegression, 3 classes, balanced', () => new LogisticRegression({ maxIter: 30, classWeight: 'balanced' }), y3],
    ['LogisticRegression, newton-cholesky', () => new LogisticRegression({ solver: 'newton-cholesky' }), y2],
    ['LogisticRegression, newton-cholesky, 3 classes, balanced', () => new LogisticRegression({ solver: 'newton-cholesky', classWeight: 'balanced' }), y3],
  ] as const) {
    test(name, async () => {
      const one = make().fitSync(X, target, { sampleWeight: w });
      const many = await make().fit(X, target, { threads: 4, sampleWeight: w });
      expect(many.toJSON()).toEqual(one.toJSON());
    });
  }
});
