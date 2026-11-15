import { describe, expect, test } from 'vitest';
import {
  ElasticNet,
  GridSearchCV,
  KFold,
  Lasso,
  LinearRegression,
  LogisticRegression,
  Ridge,
  SmaltmlError,
  StandardScaler,
  StratifiedKFold,
  cloneModel,
  crossValScore,
  loadModel,
  parameterGrid,
  trainTestSplit,
  pipeline,
  saveModel,
} from '../src/index.js';

const code = async (f: () => unknown) => {
  try {
    await f();
  } catch (e) {
    expect(e).toBeInstanceOf(SmaltmlError);
    return (e as SmaltmlError).code;
  }
  return 'no error';
};

describe('StratifiedKFold keeps its documented guarantees', () => {
  const y = Array.from({ length: 53 }, (_, i) => (i % 7 < 4 ? 'a' : i % 7 < 6 ? 'b' : 'c'));

  test('every row is tested once; test folds differ by at most one row; classes in proportion', () => {
    for (const k of [2, 3, 5, 7]) {
      const folds = new StratifiedKFold({ nSplits: k }).split(y.length, y);
      const tested = folds.flatMap((f) => Array.from(f.test)).sort((a, b) => a - b);
      expect(tested).toEqual(y.map((_, i) => i));
      const sizes = folds.map((f) => f.test.length);
      expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThanOrEqual(1);
      for (const label of ['a', 'b', 'c']) {
        const total = y.filter((v) => v === label).length;
        for (const f of folds) {
          const inFold = Array.from(f.test).filter((i) => y[i] === label).length;
          expect(Math.abs(inFold - total / k)).toBeLessThan(1);
        }
      }
      for (const f of folds) expect(f.train.length + f.test.length).toBe(y.length);
    }
  });

  test('the folds don\'t depend on the labels\' names; each class keeps its order', () => {
    const renamed = y.map((v) => ({ a: 'zebra', b: 'apple', c: 'moon' })[v]!);
    const a = new StratifiedKFold({ nSplits: 4 }).split(y.length, y).map((f) => Array.from(f.test));
    const b = new StratifiedKFold({ nSplits: 4 }).split(y.length, renamed).map((f) => Array.from(f.test));
    expect(b).toEqual(a);
    // Without shuffling, a class's rows in one test fold are consecutive among that class's rows.
    for (const test of a) {
      for (const label of ['a', 'b', 'c']) {
        const rows = y.map((v, i) => (v === label ? i : -1)).filter((i) => i >= 0);
        const pos = test.filter((i) => y[i] === label).map((i) => rows.indexOf(i));
        for (let j = 1; j < pos.length; j++) expect(pos[j]).toBe(pos[j - 1]! + 1);
      }
    }
  });

  test('a million rows split without spreading a class onto the call stack', () => {
    const big = Array.from({ length: 1_000_000 }, (_, i) => (i % 3 === 0 ? 'x' : 'y'));
    const folds = new StratifiedKFold({ nSplits: 2 }).split(big.length, big);
    expect(folds.map((f) => f.test.length)).toEqual([500_000, 500_000]);
  });

  test('a seed repeats a shuffle; parameters are checked', async () => {
    const a = new KFold({ nSplits: 3, shuffle: true, seed: 9 }).split(20);
    expect(new KFold({ nSplits: 3, shuffle: true, seed: 9 }).split(20)).toEqual(a);
    expect(await code(() => new KFold({ nSplits: 1 }))).toBe('ML_PARAM');
    expect(await code(() => new KFold({ seed: 1 }))).toBe('ML_PARAM');
    expect(await code(() => new KFold({ nSplits: 5 }).split(3))).toBe('ML_PARAM');
    expect(await code(() => new StratifiedKFold().split(10))).toBe('ML_PARAM');
  });
});

describe('cloning and searching', () => {
  const X = Array.from({ length: 40 }, (_, i) => [i % 7, (i * 3) % 11, i / 4]);
  const y = X.map(([a, b, c]) => 2 * a! - b! + 0.5 * c! + ((X.indexOf([a!, b!, c!]) % 3) - 1));
  const labels = X.map(([a, b]) => (a! + b! > 8 ? 'high' : 'low'));

  test('cloneModel copies parameters, unfitted, and changes nested ones', () => {
    const p = pipeline(new StandardScaler({ withMean: false }), new Ridge({ alpha: 3 })).fitSync(X, y);
    const c = cloneModel(p, { ridge__alpha: 10 });
    expect(c.fitted).toBe(false);
    expect(p.fitted).toBe(true);
    expect((c.step('ridge') as Ridge).alpha).toBe(10);
    expect((c.step('standardscaler') as StandardScaler).withMean).toBe(false);
    expect((cloneModel(new LogisticRegression(), { C: 0.5 }) as LogisticRegression).C).toBe(0.5);
  });

  test('bad parameter names fail', async () => {
    const p = pipeline(new StandardScaler(), new Ridge());
    expect(await code(() => cloneModel(p, { alpha: 1 }))).toBe('ML_PARAM');
    expect(await code(() => cloneModel(p, { lasso__alpha: 1 }))).toBe('ML_PARAM');
    expect(await code(() => crossValScore(new Ridge(), X, y, { scoring: 'nope' as 'r2' }))).toBe('ML_PARAM');
    expect(await code(() => new GridSearchCV(new Ridge(), { alpha: [] }))).toBe('ML_PARAM');
  });

  test('parameterGrid follows scikit-learn\'s order: names sorted, the last changing fastest', () => {
    expect(parameterGrid({ b: [1, 2], a: ['x', 'y'] })).toEqual([
      { a: 'x', b: 1 },
      { a: 'x', b: 2 },
      { a: 'y', b: 1 },
      { a: 'y', b: 2 },
    ]);
    expect(parameterGrid([{ a: [1] }, { b: [2, 3] }])).toEqual([{ a: 1 }, { b: 2 }, { b: 3 }]);
  });

  test('a classifier\'s default is stratified folds and accuracy; a pipeline is a classifier when its last step is', async () => {
    const s = await crossValScore(pipeline(new StandardScaler(), new LogisticRegression()), X, labels, { cv: 4 });
    expect(s.length).toBe(4);
    for (const v of s) expect(v).toBeGreaterThan(0.5);
    const r = await crossValScore(pipeline(new StandardScaler(), new LinearRegression()), X, y, { cv: 4 });
    for (const v of r) expect(v).toBeLessThanOrEqual(1);
  });

  test('a fitted search predicts with its best estimator and saves', async () => {
    const g = await new GridSearchCV(pipeline(new StandardScaler(), new Ridge()), { ridge__alpha: [0.1, 10, 1000] }, { cv: 4 }).fit(X, y);
    expect(g.bestParams).toEqual({ ridge__alpha: 0.1 });
    expect(Array.from((await g.predict(X)) as Float64Array)).toEqual(Array.from((g.bestEstimator as ReturnType<typeof pipeline>).predictSync(X) as Float64Array));
    const back = loadModel(saveModel(g)) as GridSearchCV;
    expect(back.bestParams).toEqual(g.bestParams);
    expect(Array.from(back.cvResults.meanTestScore)).toEqual(Array.from(g.cvResults.meanTestScore));
    expect(Array.from((await back.predict(X)) as Float64Array)).toEqual(Array.from((await g.predict(X)) as Float64Array));
  });
});

describe('GridSearchCV keeps XᵀX per fold', () => {
  // Candidates on one fold share its rows, so the search works XᵀX, Xᵀy and the means out once per fold. Its scores
  // must be those of fitting each candidate on its own, bit for bit.
  let s = 11;
  const r = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
  const X = Array.from({ length: 120 }, () => [r() * 3, r(), r() + 2, r() * 0.5]);
  const y = X.map((row) => 2 * row[0]! - row[1]! + 0.5 * row[3]! + 0.1 * r());
  test.each([
    ['Ridge', () => new Ridge()],
    ['Lasso', () => new Lasso()],
    ['ElasticNet', () => new ElasticNet()],
    ['LinearRegression', () => new LinearRegression()],
  ] as const)('%s: the same scores as separate fits', async (_, make) => {
    const grid: Record<string, unknown[]> = make() instanceof LinearRegression ? { fitIntercept: [true, false] } : { alpha: [0.001, 0.1, 1], fitIntercept: [true, false] };
    const search = await new GridSearchCV(make(), grid, { cv: 4 }).fit(X, y);
    const each = await Promise.all(parameterGrid(grid).map((p) => crossValScore(cloneModel(make(), p), X, y, { cv: 4 })));
    search.cvResults.params.forEach((_, c) => {
      expect(Array.from(search.cvResults.splitTestScores, (row) => row[c])).toEqual(Array.from(each[c]!));
    });
  });
});

describe('trainTestSplit with stratify', () => {
  const y = Array.from({ length: 60 }, (_, i) => (i % 6 === 0 ? 'rare' : i % 2 ? 'odd' : 'even'));
  const X = y.map((_, i) => [i]);

  test('keeps the classes in proportion, with no row twice, the same for a seed', () => {
    const a = trainTestSplit(X, y, { stratify: y, testSize: 0.3, seed: 7 });
    const b = trainTestSplit(X, y, { stratify: y, testSize: 0.3, seed: 7 });
    expect(Array.from(a.XTrain.data)).toEqual(Array.from(b.XTrain.data));
    const rows = [...a.XTrain.data, ...a.XTest.data];
    expect(new Set(rows).size).toBe(rows.length);
    expect(a.XTest.rows).toBe(18);
    // rare is 10 of 60 rows: 3 of the 18 test rows.
    expect(a.yTest.filter((v) => v === 'rare').length).toBe(3);
    a.yTest.forEach((v, i) => expect(y[a.XTest.data[i]!]).toBe(v));
  });

  test('fails where it can\'t keep the classes in proportion', () => {
    const one = [...y.slice(0, 59), 'single'];
    expect(() => trainTestSplit(X, one, { stratify: one })).toThrow(expect.objectContaining({ code: 'ML_LABELS' }));
    expect(() => trainTestSplit(X, y, { stratify: y, shuffle: false })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => trainTestSplit(X, y, { stratify: y, testSize: 2 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => trainTestSplit(X, y, { stratify: y.slice(1) })).toThrow(expect.objectContaining({ code: 'ML_SHAPE' }));
  });
});
