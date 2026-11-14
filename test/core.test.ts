import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import {
  LinearRegression,
  LogisticRegression,
  Matrix,
  Random,
  SmaltmlError,
  StandardScaler,
  VERSION,
  loadModel,
  logLoss,
  pipeline,
  saveModel,
  trainTestSplit,
  type Model,
} from '../src/index.js';

const code = (f: () => unknown) => {
  try {
    f();
  } catch (e) {
    expect(e).toBeInstanceOf(SmaltmlError);
    return (e as SmaltmlError).code;
  }
  return 'no error';
};

describe('Matrix', () => {
  test('from rows and from columns hold the same values, column by column', () => {
    const a = Matrix.from([[1, 2], [3, 4], [5, 6]]);
    const b = Matrix.from({ x: [1, 3, 5], y: new Float64Array([2, 4, 6]) });
    expect([a.rows, a.cols]).toEqual([3, 2]);
    expect(Array.from(a.data)).toEqual([1, 3, 5, 2, 4, 6]);
    expect(Array.from(b.data)).toEqual(Array.from(a.data));
    expect(b.names).toEqual(['x', 'y']);
    expect(a.get(2, 1)).toBe(6);
    expect(a.toRows()).toEqual([[1, 2], [3, 4], [5, 6]]);
    expect(a.take([2, 0]).toRows()).toEqual([[5, 6], [1, 2]]);
  });

  test('null is missing (NaN); ragged rows, flat arrays and strings fail', () => {
    expect(Number.isNaN(Matrix.from({ x: [1, null as unknown as number] }).get(1, 0))).toBe(true);
    expect(code(() => Matrix.from([[1, 2], [3]]))).toBe('ML_SHAPE');
    expect(code(() => Matrix.from([1, 2, 3] as unknown as number[][]))).toBe('ML_SHAPE');
    expect(code(() => Matrix.from(new Float64Array(3) as unknown as number[][]))).toBe('ML_SHAPE');
    expect(code(() => Matrix.from({ a: [1, 2], b: [1] }))).toBe('ML_SHAPE');
    expect(code(() => Matrix.from([['a']] as unknown as number[][]))).toBe('ML_VALUE');
  });
});

describe('Random', () => {
  test('the same seed gives the same numbers; numbers are in [0, 1)', () => {
    const a = new Random(42);
    const b = new Random(42);
    const xs = Array.from({ length: 1000 }, () => a.next());
    expect(xs).toEqual(Array.from({ length: 1000 }, () => b.next()));
    expect(xs.every((x) => x >= 0 && x < 1)).toBe(true);
    expect(Math.abs(xs.reduce((s, x) => s + x, 0) / xs.length - 0.5)).toBeLessThan(0.05);
    expect(new Random(1).next()).not.toBe(new Random(2).next());
  });

  test('a permutation holds every index once', () => {
    const p = Array.from(new Random(7).permutation(50));
    expect([...p].sort((x, y) => x - y)).toEqual(Array.from({ length: 50 }, (_, i) => i));
  });
});

describe('models', () => {
  const X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1]];
  const y = [3, 7, 7, 11, 11, 15]; // 2·x₀ + 2·x₁ + 1

  test('linear regression recovers an exact line', async () => {
    const m = await new LinearRegression().fit(X, y);
    expect(Array.from(m.coef).map((v) => +v.toFixed(10))).toEqual([2, 2]);
    expect(+m.intercept.toFixed(10)).toBe(1);
    expect(Array.from(await m.predict([[10, 1]])).map((v) => +v.toFixed(10))).toEqual([23]);
  });

  test('errors carry codes', () => {
    expect(code(() => new LinearRegression().predictSync(X))).toBe('ML_NOT_FITTED');
    expect(code(() => new LinearRegression().fitSync(X, [1, 2]))).toBe('ML_SHAPE');
    expect(code(() => new LinearRegression().fitSync([[1, NaN], [2, 3]], [1, 2]))).toBe('ML_VALUE');
    expect(code(() => new LinearRegression().fitSync([[1, 2], [2, 4], [3, 6]], [1, 2, 3]))).toBe('ML_SINGULAR');
    expect(code(() => new LinearRegression().fitSync(X, y).predictSync([[1, 2, 3]]))).toBe('ML_SHAPE');
    expect(code(() => new LogisticRegression({ C: 0 }))).toBe('ML_PARAM');
    expect(code(() => new LogisticRegression({ l1Ratio: 0.5 }))).toBe('ML_PARAM');
    expect(code(() => new LogisticRegression().fitSync(X, [1, 1, 1, 1, 1, 1]))).toBe('ML_LABELS');
    expect(code(() => new LogisticRegression().fitSync(X, [1, 'a', 1, 1, 1, 1] as unknown as number[]))).toBe('ML_LABELS');
  });

  test('a missing value names its column', () => {
    expect(() => new LinearRegression().fitSync({ age: [1, 2, NaN], spend: [1, 2, 3] }, [1, 2, 3])).toThrow(/column age, row 2/);
  });

  test('logistic regression probabilities sum to 1 and predict the likeliest class', () => {
    const Xc = [[0], [1], [2], [3], [4], [5], [6], [7], [8]];
    const yc = ['a', 'a', 'b', 'a', 'b', 'b', 'c', 'b', 'c'];
    const m = new LogisticRegression().fitSync(Xc, yc);
    expect(m.classes).toEqual(['a', 'b', 'c']);
    const P = m.predictProbaSync(Xc);
    for (let i = 0; i < P.rows; i++) expect(P.row(i).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    const pred = m.predictSync(Xc);
    for (let i = 0; i < P.rows; i++) {
      const row = Array.from(P.row(i));
      expect(pred[i]).toBe(m.classes[row.indexOf(Math.max(...row))]);
    }
    expect(logLoss(yc, P)).toBeGreaterThan(0);
  });
});

describe('trainTestSplit', () => {
  const X = Array.from({ length: 20 }, (_, i) => [i]);
  const y = Array.from({ length: 20 }, (_, i) => i * 10);

  test('a seed gives the same split; rows and labels stay together', () => {
    const a = trainTestSplit(X, y, { seed: 3 });
    const b = trainTestSplit(X, y, { seed: 3 });
    expect(a.XTest.toRows()).toEqual(b.XTest.toRows());
    expect([a.XTrain.rows, a.XTest.rows]).toEqual([15, 5]);
    a.yTest.forEach((v, i) => expect(v).toBe(a.XTest.get(i, 0) * 10));
    const all = [...a.XTrain.column(0), ...a.XTest.column(0)].sort((p, q) => p - q);
    expect(all).toEqual(X.map((r) => r[0]));
  });

  test('sizes out of range fail', () => {
    expect(code(() => trainTestSplit(X, y, { testSize: 1.5 }))).toBe('ML_PARAM');
    expect(code(() => trainTestSplit(X, y, { testSize: 20 }))).toBe('ML_PARAM');
    expect(code(() => trainTestSplit(X, y, { testSize: 0.6, trainSize: 0.6 }))).toBe('ML_PARAM');
  });
});

describe('saving and loading', () => {
  const X = [[1, 20], [2, 18], [3, 25], [4, 30], [5, 28], [6, 40]];
  const labels = ['no', 'no', 'no', 'yes', 'yes', 'yes'];

  const roundTrip = (m: Model) => loadModel(JSON.parse(JSON.stringify(m)));

  test('every model loads back and predicts the same', () => {
    const lin = new LinearRegression().fitSync(X, [1, 2, 3, 4, 5, 6]);
    expect(Array.from((roundTrip(lin) as LinearRegression).predictSync(X))).toEqual(Array.from(lin.predictSync(X)));
    const log = new LogisticRegression({ C: Infinity, maxIter: 20 }).fitSync(X, labels);
    const back = loadModel(saveModel(log)) as LogisticRegression;
    expect(back.C).toBe(Infinity);
    expect(back.predictProbaSync(X).data).toEqual(log.predictProbaSync(X).data);
    const sc = new StandardScaler({ withMean: false }).fitSync({ a: [1, 2, 3], b: [4, 5, 9] });
    const sc2 = roundTrip(sc) as StandardScaler;
    expect(sc2.withMean).toBe(false);
    expect(sc2.featureNamesIn).toEqual(['a', 'b']);
    expect(sc2.transformSync([[1, 1]]).data).toEqual(sc.transformSync([[1, 1]]).data);
  });

  test('a pipeline saves its steps', () => {
    const p = pipeline(new StandardScaler(), ['model', new LogisticRegression()]).fitSync(X, labels);
    const doc = p.toJSON();
    expect(doc.params).toEqual({ steps: ['standardscaler', 'model'] });
    const back = loadModel(saveModel(p));
    expect((back as typeof p).predictSync(X)).toEqual(p.predictSync(X));
    expect((back as typeof p).classes).toEqual(['no', 'yes']);
  });

  test('a document says its format and version; others are refused', () => {
    const doc = new LinearRegression().toJSON();
    expect(doc).toEqual({ format: 'smaltml', version: 1, model: 'LinearRegression', smaltml: VERSION, params: { fitIntercept: true }, state: null });
    expect(code(() => loadModel('{"format":"other"}'))).toBe('ML_FORMAT');
    expect(code(() => loadModel({ ...doc, version: 99 }))).toBe('ML_FORMAT');
    expect(code(() => loadModel({ ...doc, model: 'Mystery' }))).toBe('ML_FORMAT');
    expect(code(() => loadModel('not json'))).toBe('ML_FORMAT');
  });
});

describe('pipeline', () => {
  test('only the last step may be other than a transformer; names are unique', () => {
    expect(code(() => pipeline(new LinearRegression(), new StandardScaler()))).toBe('ML_PARAM');
    expect(code(() => pipeline(new StandardScaler(), new StandardScaler()))).toBe('ML_PARAM');
    expect(code(() => pipeline())).toBe('ML_PARAM');
  });
});

test('VERSION is package.json\'s version', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { version: string };
  expect(VERSION).toBe(pkg.version);
});
