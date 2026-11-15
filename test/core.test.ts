import { readFileSync } from 'node:fs';
import { describe, expect, test, vi } from 'vitest';
import {
  ColumnTransformer,
  ElasticNet,
  Lasso,
  Ridge,
  LinearRegression,
  MinMaxScaler,
  OneHotEncoder,
  OrdinalEncoder,
  RobustScaler,
  SimpleImputer,
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

  test('linear regression fits a long column (no spread onto the call stack)', () => {
    const n = 300_000;
    const x = Float64Array.from({ length: n }, (_, i) => i / n);
    const m = new LinearRegression().fitSync({ x }, Float64Array.from(x, (v) => 3 * v + 1));
    expect(m.coef[0]).toBeCloseTo(3, 9);
    expect(m.intercept).toBeCloseTo(1, 9);
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

  test('the penalized linear models load back and predict the same', () => {
    const X = [[1, 2, 0], [2, 1, 1], [3, 4, 0], [4, 3, 1], [5, 6, 1], [6, 5, 0], [7, 8, 1]];
    const y = [3, 4, 8, 8, 13, 12, 17];
    for (const m of [new Ridge({ alpha: 0.5 }), new Lasso({ alpha: 0.1 }), new ElasticNet({ alpha: 0.1, l1Ratio: 0.3 })]) {
      m.fitSync(X, y);
      const back = loadModel(saveModel(m)) as typeof m;
      expect(back.kind).toBe(m.kind);
      expect(back.getParams()).toEqual(m.getParams());
      expect(Array.from(back.predictSync(X))).toEqual(Array.from(m.predictSync(X)));
    }
    expect('l1Ratio' in new Lasso().getParams()).toBe(false);
  });

  test('the penalized linear models check their parameters', () => {
    expect(code(() => new Ridge({ alpha: -1 }))).toBe('ML_PARAM');
    expect(code(() => new ElasticNet({ l1Ratio: 1.5 }))).toBe('ML_PARAM');
    expect(code(() => new Lasso({ maxIter: 0 }))).toBe('ML_PARAM');
    expect(code(() => new Ridge({ alpha: 0 }).fitSync([[1, 2], [2, 4], [3, 6]], [1, 2, 3]))).toBe('ML_SINGULAR');
    // A penalty on repeated columns has one answer.
    expect(new Ridge({ alpha: 1 }).fitSync([[1, 2], [2, 4], [3, 6]], [1, 2, 3]).coef.length).toBe(2);
  });

  test('the data preparation models load back and transform the same', () => {
    const table = { city: ['Leeds', 'York', null, 'Bath', 'Leeds', 'York'], size: [3, NaN, 5, 3, 1, 5], age: [31, 45, 27, 52, 38, 29] };
    const nums = [[1, NaN, 3], [4, 5, NaN], [7, 8, 9], [NaN, 2, 1]];
    const models = [
      [new MinMaxScaler({ featureRange: [-1, 1], clip: true }), nums],
      [new RobustScaler({ unitVariance: true }), nums],
      [new SimpleImputer({ strategy: 'median' }), nums],
      [new OrdinalEncoder({ handleUnknown: 'use_encoded_value', unknownValue: NaN }), table],
      [new OneHotEncoder({ drop: 'if_binary', handleUnknown: 'ignore' }), table],
      [new ColumnTransformer([['scale', new StandardScaler(), ['age']], ['enc', new OneHotEncoder(), ['city', 'size']]]), table],
    ] as const;
    for (const [model, data] of models) {
      (model as { fitSync(x: unknown): unknown }).fitSync(data);
      const back = loadModel(saveModel(model)) as typeof model;
      const a = (model as { transformSync(x: unknown): Matrix }).transformSync(data);
      const b = (back as { transformSync(x: unknown): Matrix }).transformSync(data);
      expect(b.names, model.kind).toEqual(a.names);
      expect(Array.from(b.data), model.kind).toEqual(Array.from(a.data));
    }
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
    expect(doc).toEqual({ format: 'smaltml', version: 2, model: 'LinearRegression', smaltml: VERSION, params: { fitIntercept: true }, state: null });
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

describe("LogisticRegression's newton-cholesky solver", () => {
  const X = [[0, 1], [1, 0], [2, 1], [3, 3], [4, 2], [5, 4], [1, 2], [4, 4]];
  test('lands where lbfgs does, in fewer iterations', () => {
    const y = [0, 0, 0, 1, 1, 1, 0, 1];
    const nc = new LogisticRegression({ solver: 'newton-cholesky', tol: 1e-10 }).fitSync(X, y);
    const lb = new LogisticRegression({ tol: 1e-10, maxIter: 10000 }).fitSync(X, y);
    nc.coef.data.forEach((v, i) => expect(v).toBeCloseTo(lb.coef.data[i]!, 6));
    expect(nc.nIter).toBeLessThan(lb.nIter);
    expect(LogisticRegression.fromJSON(nc.toJSON()).getParams().solver).toBe('newton-cholesky');
  });
  test('two classes only so far; not on the GPU; a solver it knows', async () => {
    expect(() => new LogisticRegression({ solver: 'newton-cholesky' }).fitSync(X, [0, 1, 2, 0, 1, 2, 0, 1])).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    await expect(new LogisticRegression({ solver: 'newton-cholesky' }).fit(X, [0, 0, 0, 1, 1, 1, 0, 1], { device: 'gpu' })).rejects.toMatchObject({ code: 'ML_PARAM' });
    expect(() => new LogisticRegression({ solver: 'sag' as 'lbfgs' })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
  });
});

describe('fits that stop before they converge', () => {
  const X = [[1, 20], [2, 18], [3, 25], [4, 30], [5, 28], [6, 40], [2, 30], [5, 19]];
  const labels = ['no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no'];
  const y = [3, 4, 8, 8, 13, 12, 9, 7];
  /** The model the fit returns and the ML_CONVERGENCE warnings it printed. */
  const warnings = <T>(fit: () => T): [T, string[]] => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const m = fit();
      return [m, spy.mock.calls.map(([msg]) => String(msg)).filter((msg) => msg.startsWith('smaltml ML_CONVERGENCE:'))];
    } finally {
      spy.mockRestore();
    }
  };

  test('an unfinished fit warns once and says converged false; a finished one is quiet', () => {
    const [early, w1] = warnings(() => new LogisticRegression({ maxIter: 2 }).fitSync(X, labels));
    expect(early.converged).toBe(false);
    expect(early.nIter).toBe(2);
    expect(w1).toHaveLength(1);
    expect(w1[0]).toContain('LogisticRegression');
    const [done, w2] = warnings(() => new LogisticRegression({ maxIter: 10000 }).fitSync(X, labels));
    expect(done.converged).toBe(true);
    expect(w2).toHaveLength(0);
    for (const make of [() => new Lasso({ alpha: 0.01, maxIter: 1, tol: 1e-12 }), () => new ElasticNet({ alpha: 0.01, maxIter: 1, tol: 1e-12 })]) {
      const [m, w] = warnings(() => make().fitSync(X, y));
      expect(m.converged).toBe(false);
      expect(w).toHaveLength(1);
      expect(w[0]).toContain('duality gap');
    }
    const [ridge, w3] = warnings(() => new Ridge().fitSync(X, y));
    expect(ridge.converged).toBe(true);
    expect(w3).toHaveLength(0);
  });

  test('converged is saved; a version 1 document loads with converged null', () => {
    const [early] = warnings(() => new LogisticRegression({ maxIter: 2 }).fitSync(X, labels));
    expect((loadModel(saveModel(early)) as LogisticRegression).converged).toBe(false);
    const lasso = new Lasso({ alpha: 0.1 }).fitSync(X, y);
    expect((loadModel(saveModel(lasso)) as Lasso).converged).toBe(true);
    const old = (doc: ReturnType<LogisticRegression['toJSON']>) => {
      const { converged: _, ...state } = doc.state!;
      return { ...doc, version: 1, state };
    };
    expect((loadModel(old(early.toJSON())) as LogisticRegression).converged).toBeNull();
    expect((loadModel(old(lasso.toJSON())) as Lasso).converged).toBeNull();
    expect(Array.from((loadModel(old(lasso.toJSON())) as Lasso).predictSync(X))).toEqual(Array.from(lasso.predictSync(X)));
  });

  test("ElasticNet with no L1 term (l1Ratio 0) converges: its duality gap closes", () => {
    const [m, w] = warnings(() => new ElasticNet({ alpha: 2, l1Ratio: 0, tol: 1e-12, maxIter: 100000 }).fitSync(X, y));
    expect(m.converged).toBe(true);
    expect(m.nIter).toBeLessThan(1000);
    expect(w).toHaveLength(0);
  });
});

describe('sampleWeight on the least-squares models', () => {
  const r = new Random(9);
  const rows = Array.from({ length: 60 }, () => [r.next() * 4, r.next() - 0.5, r.next() * 2 + 1]);
  const y = rows.map((x) => 2 * x[0]! - 3 * x[1]! + 0.5 * x[2]! + 1 + 0.2 * (r.next() - 0.5));
  const counts = rows.map((_, i) => 1 + (i % 3));
  // The same rows, each repeated as many times as its weight.
  const repeatedX = rows.flatMap((x, i) => Array.from({ length: counts[i]! }, () => x));
  const repeatedY = y.flatMap((v, i) => Array.from({ length: counts[i]! }, () => v));
  const models = [
    ['LinearRegression', () => new LinearRegression()],
    ['Ridge', () => new Ridge({ alpha: 2 })],
    ['Lasso', () => new Lasso({ alpha: 0.05, tol: 1e-12, maxIter: 100000 })],
    ['ElasticNet', () => new ElasticNet({ alpha: 0.05, l1Ratio: 0.3, tol: 1e-12, maxIter: 100000 })],
  ] as const;

  test.each(models)('%s: whole-number weights equal repeating the rows', (_, make) => {
    const weighted = make().fitSync(rows, y, { sampleWeight: counts });
    const repeated = make().fitSync(repeatedX, repeatedY);
    weighted.coef.forEach((v, j) => expect(v).toBeCloseTo(repeated.coef[j]!, 8));
    expect(weighted.intercept).toBeCloseTo(repeated.intercept, 8);
  });

  test.each(models)('%s: equal weights fit as no weights do', (_, make) => {
    const weighted = make().fitSync(rows, y, { sampleWeight: rows.map(() => 1) });
    const plain = make().fitSync(rows, y);
    weighted.coef.forEach((v, j) => expect(v).toBeCloseTo(plain.coef[j]!, 10));
  });

  test('weights that are wrong fail', () => {
    const fit = (w: unknown) => () => new LinearRegression().fitSync(rows, y, { sampleWeight: w as number[] });
    expect(fit([1, 2])).toThrow(expect.objectContaining({ code: 'ML_SHAPE' }));
    expect(fit(rows.map((_, i) => (i === 3 ? -1 : 1)))).toThrow(expect.objectContaining({ code: 'ML_VALUE' }));
    expect(fit(rows.map((_, i) => (i === 3 ? NaN : 1)))).toThrow(expect.objectContaining({ code: 'ML_VALUE' }));
    expect(fit(rows.map(() => 0))).toThrow(expect.objectContaining({ code: 'ML_VALUE' }));
  });

  test("a pipeline passes them to its last step, and fails where that step doesn't take them", async () => {
    const p = pipeline(new StandardScaler(), new Ridge()).fitSync(rows, y, { sampleWeight: counts });
    const scaled = new StandardScaler().fitSync(rows).transformSync(rows);
    expect(Array.from((p.last as Ridge).coef)).toEqual(Array.from(new Ridge().fitSync(scaled, y, { sampleWeight: counts }).coef));
    const labels = y.map((v) => (v > 3 ? 'high' : 'low'));
    expect(() => pipeline(new StandardScaler(), new LogisticRegression()).fitSync(rows, labels, { sampleWeight: counts })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    await expect(new LogisticRegression().fit(rows, labels, { sampleWeight: counts } as object)).rejects.toMatchObject({ code: 'ML_PARAM' });
  });
});
