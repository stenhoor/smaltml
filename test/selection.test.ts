import { describe, expect, test } from 'vitest';
import {
  GridSearchCV,
  KFold,
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
