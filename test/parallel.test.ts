import * as esbuild from 'esbuild';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { gram, xtv, xw } from '../src/core/kernels.js';
import { Matrix } from '../src/core/matrix.js';
import { Random } from '../src/core/random.js';
import { ElasticNet, ExtraTreesClassifier, ExtraTreesRegressor, Lasso, LinearRegression, LogisticRegression, RandomForestClassifier, RandomForestRegressor, Ridge, StandardScaler, configureThreads, pipeline, stopThreads } from '../src/index.js';
import { gramParallel, threadPool, xtvParallel, xwParallel } from '../src/parallel/index.js';
import { growOnThreads } from '../src/ensemble/threads.js';
import { asPrepared, prepareX } from '../src/tree/cart.js';

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

/** For the tests that fit logistic models on one thread and on several: slow next to other test files on few cores. */
const FITS_TIMEOUT = 30_000;

describe('fits on several threads equal fitSync, bit for bit', () => {
  const cases = [
    ['LinearRegression', () => new LinearRegression(), () => y],
    ['Ridge', () => new Ridge({ alpha: 3 }), () => y],
    ['Lasso', () => new Lasso({ alpha: 0.01 }), () => y],
    ['ElasticNet', () => new ElasticNet({ alpha: 0.01, l1Ratio: 0.3 }), () => y],
    ['LinearRegression, positive', () => new LinearRegression({ positive: true }), () => y],
    ['Ridge, positive', () => new Ridge({ alpha: 3, positive: true }), () => y],
    ['Lasso, positive', () => new Lasso({ alpha: 0.01, positive: true }), () => y],
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
      // Three fits; a 3-class newton-cholesky fit takes 4–5 s on one 4-core machine alone, past vitest's default 5 s.
    }, FITS_TIMEOUT);
  }

  test('several targets on several threads equal fitSync', async () => {
    const Y = Array.from({ length: X.rows }, (_, i) => [y[i]!, -2 * y[i]! + i]);
    for (const make of [() => new LinearRegression(), () => new Ridge({ alpha: 3 }), () => new Lasso({ alpha: 0.01 })]) {
      const one = make().fitSync(X, Y);
      const many = await make().fit(X, Y, { threads: 4 });
      expect(many.toJSON()).toEqual(one.toJSON());
    }
  });

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
    }, FITS_TIMEOUT);
  }
});

describe('forests grown on several threads equal fitSync, bit for bit (ADR 0039, item 5)', () => {
  // 600 named rows: threads asked for by number, so the pool runs below the 'auto' threshold.
  const r = new Random(23);
  const cols = { a: [] as number[], b: [] as number[], c: [] as number[], d: [] as number[] };
  for (let i = 0; i < 600; i++) for (const k of ['a', 'b', 'c', 'd'] as const) cols[k].push(r.next() * 4 - 2);
  const yr = cols.a.map((a, i) => a * 2 - cols.b[i]! + Math.sin(3 * cols.c[i]!) + 0.2 * r.next());
  const yc = yr.map((v) => (v > 1 ? 'high' : v > -1 ? 'mid' : 'low'));
  const weights = yr.map((_, i) => 0.5 + (i % 5));

  test.each([
    ['defaults, out of bag', { nEstimators: 13, oobScore: true }, undefined],
    ['sample weights, a share of the rows', { nEstimators: 9, maxSamples: 0.6, oobScore: true }, weights],
    ['no bootstrap', { nEstimators: 5, bootstrap: false, maxFeatures: 2 }, weights],
  ] as const)('regressor: %s', async (_name, params, sampleWeight) => {
    const opts = sampleWeight ? { sampleWeight } : {};
    const one = new RandomForestRegressor({ ...params, seed: 4 }).fitSync(cols, yr, opts);
    const three = await new RandomForestRegressor({ ...params, seed: 4 }).fit(cols, yr, { ...opts, threads: 3 });
    expect(three.toJSON()).toEqual(one.toJSON());
    expect(Array.from(three.predictSync(cols))).toEqual(Array.from(one.predictSync(cols)));
    if ('oobScore' in params) expect(Array.from(three.oobPrediction)).toEqual(Array.from(one.oobPrediction));
  }, FITS_TIMEOUT);

  test.each([
    ['defaults, out of bag', { nEstimators: 11, oobScore: true }, undefined],
    ['balanced, sample weights', { nEstimators: 7, classWeight: 'balanced', criterion: 'entropy', oobScore: true }, weights],
  ] as const)('classifier: %s', async (_name, params, sampleWeight) => {
    const opts = sampleWeight ? { sampleWeight } : {};
    const one = new RandomForestClassifier({ ...params, seed: 8 }).fitSync(cols, yc, opts);
    const three = await new RandomForestClassifier({ ...params, seed: 8 }).fit(cols, yc, { ...opts, threads: 3 });
    expect(three.toJSON()).toEqual(one.toJSON());
    expect(Array.from(three.predictProbaSync(cols).data)).toEqual(Array.from(one.predictProbaSync(cols).data));
    expect(Array.from(three.oobDecisionFunction.data)).toEqual(Array.from(one.oobDecisionFunction.data));
  }, FITS_TIMEOUT);

  test.each([
    ['extra trees regressor, defaults', () => new ExtraTreesRegressor({ nEstimators: 9, seed: 5 }), yr, undefined],
    ['extra trees regressor, bootstrap, out of bag, weights', () => new ExtraTreesRegressor({ nEstimators: 9, bootstrap: true, oobScore: true, seed: 5 }), yr, weights],
    ['extra trees classifier, defaults', () => new ExtraTreesClassifier({ nEstimators: 7, seed: 5 }), yc, undefined],
  ] as const)('%s', async (_name, make, target, sampleWeight) => {
    const opts = sampleWeight ? { sampleWeight } : {};
    const one = (make() as ExtraTreesRegressor).fitSync(cols, target as number[], opts);
    const three = await (make() as ExtraTreesRegressor).fit(cols, target as number[], { ...opts, threads: 3 });
    expect(three.toJSON()).toEqual(one.toJSON());
  }, FITS_TIMEOUT);

  test('the pool grows the trees, not this thread', async () => {
    // X as the forest gives it: prepared for trees.
    const m = asPrepared(prepareX(cols), true);
    const job = { classifier: false, X: m, y: Float64Array.from(yr), weights: null, params: {}, bootstrap: true, draws: m.rows, oob: false };
    const grown = await growOnThreads(3, job, [1, 2, 3, 4]);
    expect(grown).not.toBeNull();
    expect(grown!.map((g) => g.tree.kind)).toEqual(Array(4).fill('DecisionTreeRegressor'));
  });

  test('a missing value fails with its code before any thread starts', async () => {
    await expect(new RandomForestRegressor({ nEstimators: 2 }).fit({ a: [1, NaN, 3] }, [1, 2, 3], { threads: 3 })).rejects.toMatchObject({ code: 'ML_VALUE' });
  });
});
