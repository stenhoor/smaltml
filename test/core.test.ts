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
  MODEL_FORMAT_VERSION,
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
  MultiOutputRegressor,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  BoostedTreesRegressor,
  BoostedTreesClassifier,
  cloneModel,
  crossValScore,
  crossValidate,
  GridSearchCV,
  GroupKFold,
  KFold,
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
    expect(doc).toEqual({ format: 'smaltml', version: 6, model: 'LinearRegression', smaltml: VERSION, params: { fitIntercept: true }, state: null });
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
  test('more than two classes, where it lands where lbfgs does; not on the GPU; a solver it knows', async () => {
    const y3 = [0, 0, 1, 2, 2, 2, 1, 1];
    const nc = new LogisticRegression({ solver: 'newton-cholesky', tol: 1e-10 }).fitSync(X, y3);
    const lb = new LogisticRegression({ tol: 1e-10, maxIter: 10000 }).fitSync(X, y3);
    nc.coef.data.forEach((v, i) => expect(v).toBeCloseTo(lb.coef.data[i]!, 5));
    nc.intercept.forEach((v, i) => expect(v).toBeCloseTo(lb.intercept[i]!, 5));
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
    expect(() => pipeline(new StandardScaler(), new MinMaxScaler()).fitSync(rows, y, { sampleWeight: counts })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
  });
});

describe('sampleWeight and classWeight on LogisticRegression', () => {
  const r = new Random(4);
  const rows = Array.from({ length: 90 }, () => [r.next() * 4 - 2, r.next() * 2 - 1]);
  const labels = rows.map((x) => (x[0]! + 0.5 * x[1]! + 0.4 * (r.next() - 0.5) > 0.6 ? 'yes' : 'no'));
  const counts = rows.map((_, i) => 1 + (i % 3));
  const repeatedX = rows.flatMap((x, i) => Array.from({ length: counts[i]! }, () => x));
  const repeatedY = labels.flatMap((l, i) => Array.from({ length: counts[i]! }, () => l));

  test.each(['lbfgs', 'newton-cholesky'] as const)('%s: whole-number weights equal repeating the rows', (solver) => {
    const make = () => new LogisticRegression({ solver, tol: 1e-10, maxIter: 10000 });
    const weighted = make().fitSync(rows, labels, { sampleWeight: counts });
    const repeated = make().fitSync(repeatedX, repeatedY);
    weighted.coef.data.forEach((v, j) => expect(v).toBeCloseTo(repeated.coef.data[j]!, 6));
  });

  test("classWeight by class is a sample weight for that class's rows", () => {
    const byClass = new LogisticRegression({ classWeight: { yes: 3 }, tol: 1e-10, maxIter: 10000 }).fitSync(rows, labels);
    const bySample = new LogisticRegression({ tol: 1e-10, maxIter: 10000 }).fitSync(rows, labels, { sampleWeight: labels.map((l) => (l === 'yes' ? 3 : 1)) });
    expect(Array.from(byClass.coef.data)).toEqual(Array.from(bySample.coef.data));
  });

  test('classWeight survives saving, and wrong ones fail', () => {
    const m = new LogisticRegression({ classWeight: 'balanced' }).fitSync(rows, labels);
    expect(LogisticRegression.fromJSON(JSON.parse(JSON.stringify(m.toJSON()))).getParams().classWeight).toBe('balanced');
    expect(() => new LogisticRegression({ classWeight: 'even' as 'balanced' })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ classWeight: { yes: -1 } })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ classWeight: { maybe: 2 } }).fitSync(rows, labels)).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ classWeight: { yes: 0, no: 0 } }).fitSync(rows, labels)).toThrow(expect.objectContaining({ code: 'ML_VALUE' }));
  });

  test('a pipeline passes the weights to it', () => {
    const p = pipeline(new StandardScaler(), new LogisticRegression()).fitSync(rows, labels, { sampleWeight: counts });
    const scaled = new StandardScaler().fitSync(rows).transformSync(rows);
    expect(Array.from((p.last as LogisticRegression).coef.data)).toEqual(Array.from(new LogisticRegression().fitSync(scaled, labels, { sampleWeight: counts }).coef.data));
  });
});

describe("LogisticRegression's saga solver: L1 and elastic-net penalties", () => {
  const r = new Random(12);
  const rows = Array.from({ length: 120 }, () => [r.next() * 2 - 1, r.next() * 2 - 1, r.next() * 2 - 1, r.next() * 2 - 1]);
  const labels = rows.map((x) => (x[0]! - 0.5 * x[1]! + 0.2 * (r.next() - 0.5) > 0 ? 'yes' : 'no'));

  test('an L1 penalty sets the coefficients of unrelated columns to exactly 0', () => {
    const m = new LogisticRegression({ solver: 'saga', l1Ratio: 1, C: 0.1, tol: 1e-8, maxIter: 5000, seed: 1 }).fitSync(rows, labels);
    const coef = Array.from(m.coef.data);
    expect(coef[0]).not.toBe(0);
    expect(coef.filter((v) => v === 0).length).toBeGreaterThan(0);
  });

  test('l1Ratio 0 lands where lbfgs does; a seed repeats the fit; async equals sync', async () => {
    const sg = new LogisticRegression({ solver: 'saga', tol: 1e-10, maxIter: 20000, seed: 3 }).fitSync(rows, labels);
    const lb = new LogisticRegression({ tol: 1e-10, maxIter: 10000 }).fitSync(rows, labels);
    sg.coef.data.forEach((v, i) => expect(v).toBeCloseTo(lb.coef.data[i]!, 5));
    const again = new LogisticRegression({ solver: 'saga', tol: 1e-10, maxIter: 20000, seed: 3 }).fitSync(rows, labels);
    expect(Array.from(again.coef.data)).toEqual(Array.from(sg.coef.data));
    const viaFit = await new LogisticRegression({ solver: 'saga', tol: 1e-10, maxIter: 20000, seed: 3 }).fit(rows, labels, { threads: 2 });
    expect(Array.from(viaFit.coef.data)).toEqual(Array.from(sg.coef.data));
  });

  test('saves its solver, l1Ratio and seed', () => {
    const m = new LogisticRegression({ solver: 'saga', l1Ratio: 0.5, seed: 9 }).fitSync(rows, labels);
    const back = LogisticRegression.fromJSON(JSON.parse(JSON.stringify(m.toJSON())));
    expect(back.getParams()).toMatchObject({ solver: 'saga', l1Ratio: 0.5, seed: 9 });
    expect(Array.from(back.predictProbaSync(rows).data)).toEqual(Array.from(m.predictProbaSync(rows).data));
  });

  test('l1Ratio above 0 needs saga; and its other errors', async () => {
    expect(() => new LogisticRegression({ l1Ratio: 1 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ solver: 'newton-cholesky', l1Ratio: 0.5 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ solver: 'saga', l1Ratio: 1.5 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => new LogisticRegression({ solver: 'saga', seed: 0.5 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    await expect(new LogisticRegression({ solver: 'saga' }).fit(rows, labels, { device: 'gpu' })).rejects.toMatchObject({ code: 'ML_PARAM' });
  });
});

describe("StandardScaler's partialFit", () => {
  const rows = [[1, 10], [2, 30], [4, NaN], [7, 20], [11, 60], [3, 5]];

  test('batches give the statistics of one fit on every row', async () => {
    const whole = new StandardScaler().fitSync(rows);
    const parts = new StandardScaler();
    for (const b of [rows.slice(0, 2), rows.slice(2, 5), rows.slice(5)]) await parts.partialFit(b);
    for (const key of ['mean', 'var', 'scale'] as const) {
      Array.from(parts[key]!).forEach((v, j) => expect(v).toBeCloseTo(whole[key]![j]!, 12));
    }
    expect(Array.from(parts.nSamplesSeen as Float64Array)).toEqual([6, 5]);
  });

  test('before any fit it fits, and fit() starts again', () => {
    const first = new StandardScaler().partialFitSync(rows.slice(0, 3));
    expect(Array.from(first.mean!)).toEqual(Array.from(new StandardScaler().fitSync(rows.slice(0, 3)).mean!));
    first.partialFitSync(rows.slice(3));
    first.fitSync([[0, 0], [2, 2]]);
    expect(Array.from(first.mean!)).toEqual([1, 1]);
    expect(first.nSamplesSeen).toBe(2);
  });

  test('the asynchronous call equals its twin, and the result saves', async () => {
    const a = await new StandardScaler().partialFit({ x: [1, 2, 3], y: [4, 4, 9] });
    await a.partialFit({ x: [8, 9], y: [1, 0] });
    const b = new StandardScaler().partialFitSync({ x: [1, 2, 3], y: [4, 4, 9] }).partialFitSync({ x: [8, 9], y: [1, 0] });
    expect(Array.from(a.var!)).toEqual(Array.from(b.var!));
    const back = loadModel(JSON.parse(JSON.stringify(a))) as StandardScaler;
    expect(back.nSamplesSeen).toBe(5);
    expect(Array.from(back.partialFitSync({ x: [5], y: [5] }).mean!)).toEqual(Array.from(a.partialFitSync({ x: [5], y: [5] }).mean!));
  });

  test('a batch must have the fit\'s columns', () => {
    const s = new StandardScaler().partialFitSync({ x: [1, 2], y: [3, 4] });
    expect(code(() => s.partialFitSync([[1, 2, 3]]))).toBe('ML_SHAPE');
    expect(code(() => s.partialFitSync({ y: [1], x: [2] }))).toBe('ML_SHAPE');
    expect(code(() => s.partialFitSync([]))).toBe('ML_SHAPE');
  });
});

describe('positive: coefficients held at 0 or more', () => {
  const r = new Random(4);
  const rows = Array.from({ length: 50 }, () => [r.next() * 3, r.next() * 2 - 1, r.next() + 1]);
  const y = rows.map((x) => 2 * x[0]! - 4 * x[1]! + 0.5 * x[2]! + 1 + 0.1 * (r.next() - 0.5));
  const models = [
    ['LinearRegression', (positive: boolean) => new LinearRegression({ positive })],
    ['Ridge', (positive: boolean) => new Ridge({ alpha: 0.5, positive })],
    ['Lasso', (positive: boolean) => new Lasso({ alpha: 0.01, positive, tol: 1e-12, maxIter: 100000 })],
    ['ElasticNet', (positive: boolean) => new ElasticNet({ alpha: 0.01, l1Ratio: 0.5, positive, tol: 1e-12, maxIter: 100000 })],
  ] as const;

  test.each(models)('%s: no coefficient below 0, and the negative one at exactly 0', (_, make) => {
    const free = make(false).fitSync(rows, y);
    expect(free.coef[1]).toBeLessThan(0);
    const held = make(true).fitSync(rows, y);
    expect(Array.from(held.coef).every((v) => v >= 0)).toBe(true);
    expect(held.coef[1]).toBe(0);
    expect(held.coef[0]).toBeGreaterThan(0);
  });

  test.each(models)('%s: with every coefficient positive anyway, the same fit as without', (_, make) => {
    const up = rows.map((x) => 2 * x[0]! + 3 * x[1]! + 0.5 * x[2]! + 1);
    const free = make(false).fitSync(rows, up);
    const held = make(true).fitSync(rows, up);
    Array.from(held.coef).forEach((v, j) => expect(v).toBeCloseTo(free.coef[j]!, 8));
  });

  test.each(models)('%s: positive saves and loads back', (_, make) => {
    const m = make(true).fitSync(rows, y);
    const back = loadModel(JSON.parse(JSON.stringify(m))) as typeof m;
    expect(back.positive).toBe(true);
    expect((back.getParams() as { positive?: boolean }).positive).toBe(true);
    expect(Array.from(back.predictSync(rows))).toEqual(Array.from(m.predictSync(rows)));
  });

  test('positive is true or false', () => {
    expect(code(() => new LinearRegression({ positive: 1 as unknown as boolean }))).toBe('ML_PARAM');
    expect(code(() => new Ridge({ positive: 'yes' as unknown as boolean }))).toBe('ML_PARAM');
    expect(code(() => new Lasso({ positive: 0 as unknown as boolean }))).toBe('ML_PARAM');
  });
});

describe('MultiOutputRegressor', () => {
  const r = new Random(6);
  const rows = Array.from({ length: 40 }, () => [r.next() * 3, r.next() - 0.5]);
  const Y = { price: rows.map((x) => 2 * x[0]! - x[1]! + 1), cost: rows.map((x) => -x[0]! + 3 * x[1]!) };

  test('each target fits as its own model would', () => {
    const m = new MultiOutputRegressor(new Ridge({ alpha: 0.5 })).fitSync(rows, Y);
    const alone = new Ridge({ alpha: 0.5 }).fitSync(rows, Y.cost);
    expect(Array.from((m.estimators[1] as Ridge).coef)).toEqual(Array.from(alone.coef));
    const out = m.predictSync(rows.slice(0, 3));
    expect(out.names).toEqual(['price', 'cost']);
    expect(Array.from(out.column(1))).toEqual(Array.from(alone.predictSync(rows.slice(0, 3))));
    expect(m.targetNames).toEqual(['price', 'cost']);
    expect(m.nFeaturesIn).toBe(2);
    // The estimator given stays unfitted: each target gets a copy.
    expect(m.estimator.fitted).toBe(false);
  });

  test('fit equals fitSync, and Y can be rows', async () => {
    const asRows = rows.map((_, i) => [Y.price[i]!, Y.cost[i]!]);
    const a = await new MultiOutputRegressor(new LinearRegression()).fit(rows, asRows);
    const b = new MultiOutputRegressor(new LinearRegression()).fitSync(rows, Y);
    expect(Array.from(a.predictSync(rows).data)).toEqual(Array.from(b.predictSync(rows).data));
    expect(a.targetNames).toBeUndefined();
  });

  test('saves and loads back, and cloneModel reaches the estimator', () => {
    const m = new MultiOutputRegressor(new Lasso({ alpha: 0.01 })).fitSync(rows, Y);
    const back = loadModel(saveModel(m)) as MultiOutputRegressor;
    expect(Array.from(back.predictSync(rows).data)).toEqual(Array.from(m.predictSync(rows).data));
    expect(back.targetNames).toEqual(['price', 'cost']);
    const other = cloneModel(m, { estimator__alpha: 0.5 });
    expect(other.fitted).toBe(false);
    expect((other.estimator as Lasso).alpha).toBe(0.5);
    expect(code(() => cloneModel(m, { alpha: 0.5 }))).toBe('ML_PARAM');
  });

  test('sampleWeight goes to each target', () => {
    const w = rows.map((_, i) => 1 + (i % 2));
    const m = new MultiOutputRegressor(new LinearRegression()).fitSync(rows, Y, { sampleWeight: w });
    const alone = new LinearRegression().fitSync(rows, Y.price, { sampleWeight: w });
    expect(Array.from((m.estimators[0] as LinearRegression).coef)).toEqual(Array.from(alone.coef));
  });

  test('the errors', () => {
    expect(code(() => new MultiOutputRegressor(new LogisticRegression()))).toBe('ML_PARAM');
    expect(code(() => new MultiOutputRegressor(new StandardScaler()))).toBe('ML_PARAM');
    const m = new MultiOutputRegressor(new LinearRegression());
    expect(code(() => m.predictSync(rows))).toBe('ML_NOT_FITTED');
    expect(code(() => m.fitSync(rows, Y.price as unknown as number[][]))).toBe('ML_SHAPE');
    expect(code(() => m.fitSync(rows, rows.slice(1)))).toBe('ML_SHAPE');
    // Not yet: scoring several targets, and a pipeline that ends in one.
    expect(code(() => pipeline(new StandardScaler(), new MultiOutputRegressor(new LinearRegression())))).toBe('ML_PARAM');
  });

  test('crossValScore and GridSearchCV refuse several targets for now', async () => {
    const asRows = rows.map((_, i) => [Y.price[i]!, Y.cost[i]!]);
    await expect(crossValScore(new MultiOutputRegressor(new LinearRegression()), rows, asRows as unknown as number[])).rejects.toMatchObject({ code: 'ML_PARAM' });
    const search = new GridSearchCV(new MultiOutputRegressor(new LinearRegression()), { estimator__fitIntercept: [true, false] });
    await expect(search.fit(rows, asRows as unknown as number[])).rejects.toMatchObject({ code: 'ML_PARAM' });
  });
});

describe('several targets on the linear models (a y with columns)', () => {
  const r = new Random(8);
  const rows = Array.from({ length: 50 }, () => [r.next() * 3, r.next() - 0.5, r.next() * 2]);
  const Y = { a: rows.map((x) => 2 * x[0]! - x[1]! + 1 + 0.05 * r.next()), b: rows.map((x) => -x[0]! + 3 * x[2]! + 0.05 * r.next()) };
  const models = [
    ['LinearRegression', () => new LinearRegression()],
    ['LinearRegression, positive', () => new LinearRegression({ positive: true })],
    ['Ridge', () => new Ridge({ alpha: 0.7 })],
    ['Lasso', () => new Lasso({ alpha: 0.01 })],
    ['ElasticNet', () => new ElasticNet({ alpha: 0.01, l1Ratio: 0.4 })],
  ] as const;

  test.each(models)('%s: each target as it fits alone, to the last bit', (_, make) => {
    const both = make().fitSync({ x: rows.map((v) => v[0]!), y: rows.map((v) => v[1]!), z: rows.map((v) => v[2]!) }, Y);
    const alone = make().fitSync({ x: rows.map((v) => v[0]!), y: rows.map((v) => v[1]!), z: rows.map((v) => v[2]!) }, Y.b);
    expect(both.coefs.rows).toBe(2);
    expect(both.coefs.names).toEqual(['x', 'y', 'z']);
    expect(Array.from(both.coefs.toRows()[1]!)).toEqual(Array.from(alone.coef));
    expect(both.intercepts[1]).toBe(alone.intercept);
    const out = both.predictTargetsSync(rows.slice(0, 4));
    expect(out.names).toEqual(['a', 'b']);
    expect(Array.from(out.column(1))).toEqual(Array.from(alone.predictSync(rows.slice(0, 4))));
    expect(both.targetNames).toEqual(['a', 'b']);
  });

  test.each(models)('%s: the one-target accessors refuse, and say what to read', (_, make) => {
    const both = make().fitSync(rows, Y);
    for (const f of [() => both.coef, () => both.intercept, () => both.predictSync(rows)]) expect(code(f)).toBe('ML_SHAPE');
    expect(() => both.coef).toThrow(/coefs, intercepts and predictTargets/);
  });

  test.each(models)('%s: a one-target fit has coefs, intercepts and predictTargets too', (_, make) => {
    const one = make().fitSync(rows, Y.a);
    expect(Array.from(one.coefs.toRows()[0]!)).toEqual(Array.from(one.coef));
    expect(Array.from(one.intercepts)).toEqual([one.intercept]);
    expect(Array.from(one.predictTargetsSync(rows).column(0))).toEqual(Array.from(one.predictSync(rows)));
    expect(one.targetNames).toBeUndefined();
  });

  test.each(models)('%s: several targets save and load back', async (_, make) => {
    const both = make().fitSync(rows, rows.map((_, i) => [Y.a[i]!, Y.b[i]!]));
    const doc = JSON.parse(saveModel(both));
    expect(doc.version).toBe(MODEL_FORMAT_VERSION);
    expect(doc.state.coefs).toHaveLength(2);
    const back = loadModel(doc) as typeof both;
    expect(Array.from(back.predictTargetsSync(rows).data)).toEqual(Array.from(both.predictTargetsSync(rows).data));
    expect(code(() => back.coef)).toBe('ML_SHAPE');
    expect(Array.from((await both.predictTargets(rows)).data)).toEqual(Array.from(both.predictTargetsSync(rows).data));
  });

  test('a one-target document of format version 2 still loads', () => {
    const doc = JSON.parse(saveModel(new Lasso({ alpha: 0.01 }).fitSync(rows, Y.a)));
    const back = loadModel({ ...doc, version: 2 }) as Lasso;
    expect(back.coef).toHaveLength(3);
    expect(back.converged).toBe(true);
  });

  test('the errors, and scoring refuses several targets for now', async () => {
    expect(code(() => new Ridge().fitSync(rows, rows.slice(1).map((v) => [v[0]!])))).toBe('ML_SHAPE');
    expect(code(() => new Ridge().fitSync(rows, { a: [1, 2] }))).toBe('ML_SHAPE');
    const asRows = rows.map((_, i) => [Y.a[i]!, Y.b[i]!]);
    await expect(crossValScore(new LinearRegression(), rows, asRows as unknown as number[])).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(new GridSearchCV(new Ridge(), { alpha: [1, 2] }).fit(rows, asRows as unknown as number[])).rejects.toMatchObject({ code: 'ML_PARAM' });
  });
});

describe('infrequent categories in the encoders', () => {
  const pets = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'eel', 'fox', 'fox', null];
  const table = { pet: pets, legs: [4, 4, 4, 4, 4, 4, 0, 4, 4, 2] };

  test('OneHotEncoder: rare categories share a column, last', () => {
    const e = new OneHotEncoder({ minFrequency: 2 }).fitSync(table);
    expect(e.infrequentCategories).toEqual([['eel', null], [0, 2]]);
    expect(e.getFeatureNamesOut()).toEqual(['pet_cat', 'pet_dog', 'pet_fox', 'pet_infrequent_sklearn', 'legs_4', 'legs_infrequent_sklearn']);
    const row = e.transformSync({ pet: ['eel'], legs: [2] });
    expect(Array.from(row.data)).toEqual([0, 0, 0, 1, 0, 1]);
    // Every category is still in categories.
    expect(e.categories[0]).toEqual(['cat', 'dog', 'eel', 'fox', null]);
  });

  test('unknown categories: an error, all zeros, or the infrequent column', () => {
    const fail = new OneHotEncoder({ minFrequency: 2 }).fitSync(table);
    expect(code(() => fail.transformSync({ pet: ['yak'], legs: [4] }))).toBe('ML_LABELS');
    const ignore = new OneHotEncoder({ minFrequency: 2, handleUnknown: 'ignore' }).fitSync(table);
    expect(Array.from(ignore.transformSync({ pet: ['yak'], legs: [4] }).data)).toEqual([0, 0, 0, 0, 1, 0]);
    const rare = new OneHotEncoder({ minFrequency: 2, handleUnknown: 'infrequent_if_exist' }).fitSync(table);
    expect(Array.from(rare.transformSync({ pet: ['yak'], legs: [4] }).data)).toEqual([0, 0, 0, 1, 1, 0]);
    // A column without infrequent categories gives all zeros.
    const none = new OneHotEncoder({ handleUnknown: 'infrequent_if_exist' }).fitSync(table);
    expect(none.transformSync({ pet: ['yak'], legs: [4] }).data.slice(0, 5).every((v) => v === 0)).toBe(true);
  });

  test('OrdinalEncoder: rare categories share the code after the others; missing stays NaN', () => {
    const e = new OrdinalEncoder({ maxCategories: 3 }).fitSync(table);
    // legs: 4 is kept, and of 0 and 2 (once each) the one sorted later, as scikit-learn breaks ties.
    expect(e.infrequentCategories).toEqual([['eel', 'fox'], [0]]);
    expect(Array.from(e.transformSync({ pet: ['cat', 'dog', 'eel', 'fox', null], legs: [4, 4, 4, 0, 2] }).column(0))).toEqual([0, 1, 2, 2, NaN]);
  });

  test('save, load, async equals sync, and cloneModel reaches the new parameters', async () => {
    const e = new OneHotEncoder({ maxCategories: 2, drop: 'first' }).fitSync(table);
    const back = loadModel(saveModel(e)) as OneHotEncoder;
    expect(back.maxCategories).toBe(2);
    expect(back.infrequentCategories).toEqual(e.infrequentCategories);
    expect(Array.from(back.transformSync(table).data)).toEqual(Array.from(e.transformSync(table).data));
    expect(Array.from((await new OneHotEncoder({ maxCategories: 2, drop: 'first' }).fitTransform(table)).data)).toEqual(Array.from(e.transformSync(table).data));
    expect(cloneModel(e, { minFrequency: 3 }).minFrequency).toBe(3);
    // An encoder saved before infrequent categories loads with none.
    const doc = JSON.parse(saveModel(new OrdinalEncoder().fitSync(table)));
    delete doc.state.infrequent;
    expect((loadModel({ ...doc, version: 2 }) as OrdinalEncoder).infrequentCategories).toEqual([null, null]);
  });

  test('the parameters are checked', () => {
    for (const bad of [0, -1, 1.5]) expect(code(() => new OneHotEncoder({ minFrequency: bad }))).toBe('ML_PARAM');
    expect(code(() => new OrdinalEncoder({ maxCategories: 0 }))).toBe('ML_PARAM');
    expect(code(() => new OrdinalEncoder({ maxCategories: 2.5 }))).toBe('ML_PARAM');
    expect(code(() => new OneHotEncoder({ handleUnknown: 'drop' as never }))).toBe('ML_PARAM');
    expect(new OneHotEncoder({ minFrequency: 0.25 }).minFrequency).toBe(0.25);
  });
});

describe('GroupKFold and groups', () => {
  const groups = ['ann', 'ann', 'bob', 'bob', 'bob', 'cy', 'dee', 'dee', 'eve', 'eve', 'eve', 'eve'];
  const X = groups.map((_, i) => [i, (i * 7) % 5]);
  const y = X.map((x) => 2 * x[0]! - x[1]! + 1);

  test("a group's rows are never in both training and test", () => {
    for (const fold of new GroupKFold({ nSplits: 3 }).split(groups.length, undefined, groups)) {
      const inTest = new Set(Array.from(fold.test, (i) => groups[i]));
      for (const i of fold.train) expect(inTest.has(groups[i]!)).toBe(false);
    }
  });

  test('crossValScore and GridSearchCV take groups', async () => {
    const cv = new GroupKFold({ nSplits: 3 });
    const scores = await crossValScore(new LinearRegression(), X, y, { cv, groups });
    expect(scores).toHaveLength(3);
    const g = await new GridSearchCV(new Ridge(), { alpha: [0.1, 10] }, { cv }).fit(X, y, { groups });
    expect(g.cvResults.meanTestScore).toHaveLength(2);
  });

  test('the errors', async () => {
    // Groups with a splitter that ignores them fail: scikit-learn warns and splits without them.
    await expect(crossValScore(new LinearRegression(), X, y, { cv: new KFold({ nSplits: 3 }), groups })).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(crossValScore(new LinearRegression(), X, y, { groups })).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(crossValScore(new LinearRegression(), X, y, { cv: new GroupKFold({ nSplits: 3 }) })).rejects.toMatchObject({ code: 'ML_PARAM' });
    expect(code(() => new GroupKFold({ nSplits: 6 }).split(groups.length, undefined, groups))).toBe('ML_PARAM');
    expect(code(() => new GroupKFold({ nSplits: 2 }).split(3, undefined, groups))).toBe('ML_SHAPE');
    expect(code(() => new GroupKFold({ nSplits: 2 }).split(3, undefined, ['a', 1, 'b']))).toBe('ML_LABELS');
    expect(code(() => new GroupKFold({ nSplits: 2, shuffle: true } as never))).toBe('ML_PARAM');
    expect(code(() => new GroupKFold({ nSplits: 1 }))).toBe('ML_PARAM');
  });
});

describe('several scorers, train scores and refit by name', () => {
  const r = new Random(12);
  const X = Array.from({ length: 40 }, () => [r.next() * 4, r.next() - 0.5]);
  const y = X.map((x) => 2 * x[0]! - 3 * x[1]! + 0.3 * r.next());
  const cv = new KFold({ nSplits: 4 });

  test('crossValidate: a score per scorer and fold, train scores, times', async () => {
    const res = await crossValidate(new Ridge(), X, y, { cv, scoring: ['r2', 'neg_mean_squared_error'], returnTrainScore: true });
    expect(Object.keys(res.test)).toEqual(['r2', 'neg_mean_squared_error']);
    expect(Array.from(res.test.r2!)).toEqual(Array.from(await crossValScore(new Ridge(), X, y, { cv, scoring: 'r2' })));
    expect(res.train!.r2).toHaveLength(4);
    expect(res.fitTime.every((t) => t >= 0)).toBe(true);
    const one = await crossValidate(new Ridge(), X, y, { cv });
    expect(Object.keys(one.test)).toEqual(['score']);
    expect(one.train).toBeUndefined();
  });

  test('GridSearchCV: each scorer by name, the best by the one refit names', async () => {
    const g = await new GridSearchCV(new Ridge(), { alpha: [0.01, 10, 1000] }, { cv, scoring: { fit: 'r2', error: 'neg_mean_absolute_error' }, refit: 'error' }).fit(X, y);
    const byScorer = g.cvResultsByScorer;
    expect(Object.keys(byScorer.scores)).toEqual(['fit', 'error']);
    expect(g.bestScore).toBe(Math.max(...byScorer.scores.error!.meanTestScore));
    expect(code(() => g.cvResults)).toBe('ML_PARAM');
    expect((g.bestEstimator as Ridge).alpha).toBe(g.bestParams.alpha);
    // It saves and loads, with the scorers' names.
    const back = loadModel(saveModel(g)) as GridSearchCV;
    expect(back.bestIndex).toBe(g.bestIndex);
    expect(Array.from(back.cvResultsByScorer.scores.fit!.meanTestScore)).toEqual(Array.from(byScorer.scores.fit!.meanTestScore));
  });

  test('one scorer: cvResults as before, train scores with returnTrainScore, and a named refit means true', async () => {
    const g = await new GridSearchCV(new Ridge(), { alpha: [0.01, 10] }, { cv, returnTrainScore: true, refit: 'r2' }).fit(X, y);
    expect(g.cvResults.meanTrainScore).toHaveLength(2);
    expect(g.cvResultsByScorer.scores.score!.meanTestScore).toEqual(g.cvResults.meanTestScore);
    expect(g.bestEstimator.fitted).toBe(true);
  });

  test('the errors', async () => {
    expect(code(() => new GridSearchCV(new Ridge(), { alpha: [1] }, { scoring: ['r2', 'neg_mean_squared_error'] }))).toBe('ML_PARAM');
    expect(code(() => new GridSearchCV(new Ridge(), { alpha: [1] }, { scoring: ['r2'], refit: 'mae' }))).toBe('ML_PARAM');
    const noBest = await new GridSearchCV(new Ridge(), { alpha: [1, 2] }, { cv, scoring: ['r2'], refit: false }).fit(X, y);
    expect(code(() => noBest.bestIndex)).toBe('ML_PARAM');
    expect(code(() => noBest.bestEstimator)).toBe('ML_NOT_FITTED');
    await expect(crossValidate(new Ridge(), X, y, { cv, scoring: ['r2', 'r2'] })).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(crossValidate(new Ridge(), X, y, { cv, scoring: [] })).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(crossValScore(new Ridge(), X, y, { cv, scoring: ['r2'] as never })).rejects.toMatchObject({ code: 'ML_PARAM' });
  });
});

describe('decision trees', () => {
  const r = new Random(21);
  const X = Array.from({ length: 60 }, () => [r.next() * 4, r.next() * 2 - 1, r.next()]);
  const y = X.map((x) => (x[0]! > 2 ? 5 : 1) + (x[1]! > 0 ? 2 : 0));
  const labels = X.map((x) => (x[0]! > 2 ? 'big' : x[1]! > 0 ? 'up' : 'down'));

  test('a regression tree fits a step function exactly, and the root splits on the first column', () => {
    const t = new DecisionTreeRegressor({ seed: 1 }).fitSync(X, y);
    expect(Array.from(t.predictSync(X))).toEqual(y);
    expect(t.tree.feature[0]).toBe(0);
    expect(t.featureImportances[2]).toBe(0);
    expect(t.featureImportances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(t.getNLeaves()).toBe(4);
    expect(t.getDepth()).toBe(2);
  });

  test('a classifier gives class shares that sum to 1, and its predictions', async () => {
    const t = new DecisionTreeClassifier({ maxDepth: 1, seed: 1 }).fitSync(X, labels);
    expect(t.classes).toEqual(['big', 'down', 'up']);
    const P = t.predictProbaSync(X);
    for (let i = 0; i < P.rows; i++) expect(P.column(0)[i]! + P.column(1)[i]! + P.column(2)[i]!).toBeCloseTo(1, 12);
    expect(await t.predict(X)).toEqual(t.predictSync(X));
  });

  test('a seed repeats the tree; fit equals fitSync; it saves and loads', async () => {
    const a = await new DecisionTreeClassifier({ maxFeatures: 1, seed: 7 }).fit(X, labels);
    const b = new DecisionTreeClassifier({ maxFeatures: 1, seed: 7 }).fitSync(X, labels);
    expect(Array.from(a.tree.threshold)).toEqual(Array.from(b.tree.threshold));
    const back = loadModel(saveModel(a)) as DecisionTreeClassifier;
    expect(back.predictSync(X)).toEqual(a.predictSync(X));
    expect(Array.from(back.predictProbaSync(X).data)).toEqual(Array.from(a.predictProbaSync(X).data));
    const reg = new DecisionTreeRegressor({ maxDepth: 2, seed: 3 }).fitSync(X, y, { sampleWeight: X.map((_, i) => 1 + (i % 3)) });
    expect(Array.from((loadModel(saveModel(reg)) as DecisionTreeRegressor).predictSync(X))).toEqual(Array.from(reg.predictSync(X)));
  });

  test('in a pipeline and a grid search', async () => {
    const g = await new GridSearchCV(new DecisionTreeRegressor({ seed: 0 }), { maxDepth: [1, 2, 3] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, y);
    expect(g.bestParams.maxDepth).toBeGreaterThan(1);
    const p = pipeline(new StandardScaler(), new DecisionTreeClassifier({ seed: 0 })).fitSync(X, labels, { sampleWeight: X.map(() => 1) });
    expect(p.predictSync(X)).toEqual(labels);
  });

  test('the errors, and what isn\'t supported yet', () => {
    expect(code(() => new DecisionTreeRegressor({ criterion: 'absolute_error' }))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeRegressor({ ccpAlpha: 0.1 }))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeRegressor({ maxDepth: 0 }))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeRegressor({ minSamplesSplit: 1.5 }))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeClassifier({ maxFeatures: 'all' as never }))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeClassifier({ classWeight: { nope: 2 } }).fitSync(X, labels))).toBe('ML_PARAM');
    expect(code(() => new DecisionTreeRegressor().fitSync([[1], [NaN]], [1, 2]))).toBe('ML_VALUE');
    expect(code(() => new DecisionTreeRegressor().predictSync(X))).toBe('ML_NOT_FITTED');
    const t = new DecisionTreeRegressor({ seed: 0 }).fitSync(X, y);
    expect(code(() => t.predictSync([[1, 2]]))).toBe('ML_SHAPE');
  });
});

describe('random forests', () => {
  const r = new Random(5);
  const X = Array.from({ length: 80 }, () => [r.next() * 4, r.next() * 2 - 1, r.next()]);
  const y = X.map((x) => (x[0]! > 2 ? 5 : 1) + (x[1]! > 0 ? 2 : 0) + x[2]! * 0.1);
  const labels = X.map((x) => (x[0]! > 2 ? 'big' : x[1]! > 0 ? 'up' : 'down'));

  test('without bootstrap and with every column, every tree is the single tree', () => {
    const f = new RandomForestRegressor({ nEstimators: 3, bootstrap: false, seed: 1 }).fitSync(X, y);
    const t = new DecisionTreeRegressor({ seed: 0 }).fitSync(X, y);
    // The mean of three equal numbers, in floating point: equal to the last bit or two.
    const p = f.predictSync(X);
    t.predictSync(X).forEach((v, i) => expect(p[i]).toBeCloseTo(v, 12));
    expect(f.estimators).toHaveLength(3);
  });

  test('a seed repeats the forest; fit equals fitSync; it saves and loads', async () => {
    const a = await new RandomForestClassifier({ nEstimators: 10, seed: 4, oobScore: true }).fit(X, labels);
    const b = new RandomForestClassifier({ nEstimators: 10, seed: 4, oobScore: true }).fitSync(X, labels);
    expect(Array.from(a.predictProbaSync(X).data)).toEqual(Array.from(b.predictProbaSync(X).data));
    expect(a.oobScore).toBe(b.oobScore);
    expect(a.classes).toEqual(['big', 'down', 'up']);
    const back = loadModel(saveModel(a)) as RandomForestClassifier;
    expect(back.predictSync(X)).toEqual(a.predictSync(X));
    expect(Array.from(back.predictProbaSync(X).data)).toEqual(Array.from(a.predictProbaSync(X).data));
    expect(back.oobScore).toBe(a.oobScore);
    expect(back.seed).toBe(4);
    const reg = new RandomForestRegressor({ nEstimators: 5, maxSamples: 0.5, seed: 2 }).fitSync(X, y, { sampleWeight: X.map((_, i) => 1 + (i % 3)) });
    const regBack = loadModel(saveModel(reg)) as RandomForestRegressor;
    expect(Array.from(regBack.predictSync(X))).toEqual(Array.from(reg.predictSync(X)));
    expect(Array.from(regBack.featureImportances)).toEqual(Array.from(reg.featureImportances));
    expect(await reg.predict(X)).toEqual(reg.predictSync(X));
  });

  test('probabilities sum to 1, importances to 1, and the out-of-bag values cover every row', () => {
    const f = new RandomForestClassifier({ nEstimators: 30, seed: 3, oobScore: true }).fitSync(X, labels);
    const P = f.predictProbaSync(X);
    for (let i = 0; i < P.rows; i++) expect(P.column(0)[i]! + P.column(1)[i]! + P.column(2)[i]!).toBeCloseTo(1, 12);
    expect(f.featureImportances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    // The labels come from the first two columns: the third, noise to them, ranks last.
    expect(f.featureImportances[2]).toBeLessThan(Math.min(f.featureImportances[0]!, f.featureImportances[1]!));
    expect(f.oobDecisionFunction.rows).toBe(80);
    expect(f.oobScore).toBeGreaterThan(0.8);
    const g = new RandomForestRegressor({ nEstimators: 30, seed: 3, oobScore: true }).fitSync(X, y);
    expect(g.oobPrediction).toHaveLength(80);
    expect(g.oobScore).toBeGreaterThan(0.8);
  });

  test('rows no tree left out: a warning, and 0s, as scikit-learn', () => {
    const warned = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const f = new RandomForestRegressor({ nEstimators: 1, seed: 1, oobScore: true }).fitSync(X, y);
      expect(warned.mock.calls.filter(([m]) => String(m).startsWith('smaltml ML_OOB:'))).toHaveLength(1);
      expect(Array.from(f.oobPrediction).some((v) => v === 0)).toBe(true);
    } finally {
      warned.mockRestore();
    }
  });

  test('rows of weight 0 are never drawn, so always out of bag', () => {
    const w = X.map((_, i) => (i < 10 ? 0 : 1));
    const f = new RandomForestRegressor({ nEstimators: 5, seed: 6 }).fitSync(X, y, { sampleWeight: w });
    for (const t of f.estimators) expect(t.tree.nSamples[0]).toBeLessThanOrEqual(70);
  });

  test('in a grid search', async () => {
    const g = await new GridSearchCV(new RandomForestRegressor({ nEstimators: 5, seed: 0 }), { maxDepth: [1, 3] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, y);
    expect(g.bestParams.maxDepth).toBe(3);
  });

  test('the errors, and what isn\'t supported yet', () => {
    expect(code(() => new RandomForestRegressor({ nEstimators: 0 }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor({ maxSamples: 0.5, bootstrap: false }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor({ oobScore: true, bootstrap: false }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor({ maxSamples: 1.5 }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor({ maxDepth: 0 }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor({ criterion: 'poisson' }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestClassifier({ classWeight: 'balanced_subsample' as never }))).toBe('ML_PARAM');
    expect(code(() => new RandomForestClassifier({ classWeight: { nope: 2 } }).fitSync(X, labels))).toBe('ML_PARAM');
    expect(code(() => new RandomForestRegressor().fitSync(X, y, { sampleWeight: X.map(() => 0) }))).toBe('ML_VALUE');
    expect(code(() => new RandomForestRegressor().predictSync(X))).toBe('ML_NOT_FITTED');
    const f = new RandomForestRegressor({ nEstimators: 2, seed: 0 }).fitSync(X, y);
    expect(code(() => f.oobScore)).toBe('ML_PARAM');
    expect(code(() => f.oobPrediction)).toBe('ML_PARAM');
    expect(code(() => f.predictSync([[1, 2]]))).toBe('ML_SHAPE');
  });
});

describe('boosted trees (ADR 0044)', () => {
  const r = new Random(13);
  const X = Array.from({ length: 150 }, () => [Math.floor(r.next() * 8), Math.round(r.next() * 20) / 10, Math.floor(r.next() * 3)]);
  const y = X.map((x) => 3 * x[0]! - 2 * x[1]! + (x[2] === 1 ? 4 : 0) + r.next());

  test('fit equals fitSync; it saves and loads, to the last bit', async () => {
    const a = await new BoostedTreesRegressor({ nEstimators: 15, maxDepth: 3 }).fit(X, y);
    const b = new BoostedTreesRegressor({ nEstimators: 15, maxDepth: 3 }).fitSync(X, y);
    expect(a.toJSON()).toEqual(b.toJSON());
    const back = loadModel(JSON.parse(JSON.stringify(saveModel(a)))) as BoostedTreesRegressor;
    expect(back.kind).toBe('BoostedTreesRegressor');
    expect(Array.from(back.predictSync(X))).toEqual(Array.from(a.predictSync(X)));
    expect(back.baseScore).toBe(a.baseScore);
    expect(Array.from(back.featureImportances)).toEqual(Array.from(a.featureImportances));
    expect(back.getParams()).toEqual(a.getParams());
    expect(await a.predict(X)).toEqual(a.predictSync(X));
  });

  test('the base score is the weighted mean of y; each round brings the training error down', () => {
    const w = X.map((_, i) => 1 + (i % 4));
    const m = new BoostedTreesRegressor({ nEstimators: 1, learningRate: 0 }).fitSync(X, y, { sampleWeight: w });
    const mean = y.reduce((s, v, i) => s + v * w[i]!, 0) / w.reduce((s, v) => s + v, 0);
    expect(m.baseScore).toBeCloseTo(mean, 12);
    // A learning rate of 0: every leaf is 0, so every prediction is the base score.
    expect(Array.from(m.predictSync(X)).every((v) => v === m.baseScore)).toBe(true);
    const mse = (k: number) => {
      const p = new BoostedTreesRegressor({ nEstimators: k, maxDepth: 2 }).fitSync(X, y).predictSync(X);
      return y.reduce((s, v, i) => s + (v - p[i]!) ** 2, 0) / y.length;
    };
    expect(mse(10)).toBeLessThan(mse(3));
    expect(mse(3)).toBeLessThan(mse(1));
  });

  test('importances: the mean gain per split, summing to 1; the noise column ranks last', () => {
    const noisy = X.map((x) => [...x, r.next() > 0.5 ? 1 : 0]);
    const m = new BoostedTreesRegressor({ nEstimators: 20, maxDepth: 2, gamma: 1 }).fitSync(noisy, y);
    const imp = m.featureImportances;
    expect(imp.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(imp[3]).toBeLessThan(Math.min(imp[0]!, imp[1]!, imp[2]!));
  });

  test('named columns; in a pipeline with sample weights; in a grid search', async () => {
    const cols = { a: X.map((x) => x[0]!), b: X.map((x) => x[1]!), c: X.map((x) => x[2]!) };
    const m = new BoostedTreesRegressor({ nEstimators: 5 }).fitSync(cols, y);
    expect(m.featureNamesIn).toEqual(['a', 'b', 'c']);
    expect(code(() => m.predictSync([[1, 2]]))).toBe('ML_SHAPE');
    const p = pipeline(new StandardScaler(), new BoostedTreesRegressor({ nEstimators: 5 })).fitSync(X, y, { sampleWeight: X.map(() => 2) });
    expect(p.predictSync(X)).toHaveLength(150);
    const g = await new GridSearchCV(new BoostedTreesRegressor({ nEstimators: 10 }), { maxDepth: [1, 3] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, y);
    expect(g.bestParams.maxDepth).toBe(3);
  });

  test('the errors, and what isn\'t supported yet', () => {
    expect(code(() => new BoostedTreesRegressor({ nEstimators: 0 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ maxDepth: 0 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ regLambda: -1 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ maxBin: 1 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ objective: 'reg:logistic' as never }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ subsample: 0.5 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor({ colsampleBytree: 0.5 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesRegressor().fitSync([[1], [Infinity]], [1, 2]))).toBe('ML_VALUE');
    // More distinct values than maxBin: XGBoost's quantile sketch isn't here yet.
    expect(code(() => new BoostedTreesRegressor({ maxBin: 4 }).fitSync(X, y))).toBe('ML_VALUE');
    expect(code(() => new BoostedTreesRegressor().fitSync(X, y, { sampleWeight: X.map(() => 0) }))).toBe('ML_VALUE');
    expect(code(() => new BoostedTreesRegressor().predictSync(X))).toBe('ML_NOT_FITTED');
  });
});

describe('boosted trees classifier (ADR 0044, step 2)', () => {
  const r = new Random(17);
  const X = Array.from({ length: 160 }, () => [Math.floor(r.next() * 8), Math.round(r.next() * 20) / 10, Math.floor(r.next() * 3)]);
  const score = X.map((x) => x[0]! - 2 * x[1]! + (x[2] === 1 ? 2 : 0) + r.next());
  const two = score.map((v) => (v > 2 ? 'yes' : 'no'));
  const three = score.map((v) => (v < 0 ? 0 : v < 4 ? 1 : 2));

  test.each([
    ['two classes', two],
    ['three classes', three],
  ] as const)('%s: fit equals fitSync; it saves and loads, to the last bit', async (_name, y) => {
    const a = await new BoostedTreesClassifier({ nEstimators: 10, maxDepth: 3 }).fit(X, y);
    const b = new BoostedTreesClassifier({ nEstimators: 10, maxDepth: 3 }).fitSync(X, y);
    expect(a.toJSON()).toEqual(b.toJSON());
    const back = loadModel(JSON.parse(JSON.stringify(saveModel(a)))) as BoostedTreesClassifier;
    expect(back.kind).toBe('BoostedTreesClassifier');
    expect(Array.from(back.predictProbaSync(X).data)).toEqual(Array.from(a.predictProbaSync(X).data));
    expect(back.predictSync(X)).toEqual(a.predictSync(X));
    expect(back.classes).toEqual(a.classes);
    expect(back.objective).toBe(a.objective);
    expect(Array.from(back.baseScore)).toEqual(Array.from(a.baseScore));
    expect(back.getParams()).toEqual(a.getParams());
    expect(await a.predict(X)).toEqual(a.predictSync(X));
    expect((await a.predictProba(X)).data).toEqual(a.predictProbaSync(X).data);
  });

  test('two classes: binary:logistic, one tree a round, the base score the second class\'s weighted share', () => {
    const w = X.map((_, i) => 1 + (i % 3));
    const m = new BoostedTreesClassifier({ nEstimators: 4, learningRate: 0 }).fitSync(X, two, { sampleWeight: w });
    expect(m.classes).toEqual(['no', 'yes']);
    expect(m.objective).toBe('binary:logistic');
    expect(m.trees).toHaveLength(4);
    const share = two.reduce((s, v, i) => s + (v === 'yes' ? w[i]! : 0), 0) / w.reduce((s, v) => s + v, 0);
    expect(m.baseScore[0]).toBeCloseTo(share, 12);
    // A learning rate of 0: every row's probability is the base score.
    const P = m.predictProbaSync(X);
    for (let i = 0; i < P.rows; i++) expect(P.data[P.rows + i]).toBeCloseTo(share, 12);
  });

  test('more classes: multi:softprob, a tree per class a round; base scores sum to 0; probabilities sum to 1', () => {
    const m = new BoostedTreesClassifier({ nEstimators: 5, maxDepth: 2 }).fitSync(X, three);
    expect(m.objective).toBe('multi:softprob');
    expect(m.trees).toHaveLength(15);
    expect(Array.from(m.baseScore).reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12);
    const P = m.predictProbaSync(X);
    expect(P.cols).toBe(3);
    for (let i = 0; i < P.rows; i++) expect(P.data[i]! + P.data[P.rows + i]! + P.data[2 * P.rows + i]!).toBeCloseTo(1, 12);
    const acc = (k: number) => {
      const p = new BoostedTreesClassifier({ nEstimators: k, maxDepth: 2 }).fitSync(X, three).predictSync(X);
      return p.filter((v, i) => v === three[i]).length;
    };
    expect(acc(20)).toBeGreaterThan(acc(1));
  });

  test('in a pipeline with sample weights; in a grid search', async () => {
    const p = pipeline(new StandardScaler(), new BoostedTreesClassifier({ nEstimators: 5 })).fitSync(X, two, { sampleWeight: X.map(() => 2) });
    expect(p.predictSync(X)).toHaveLength(160);
    const g = await new GridSearchCV(new BoostedTreesClassifier({ nEstimators: 10 }), { maxDepth: [1, 3] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, three);
    expect([1, 3]).toContain(g.bestParams.maxDepth);
  });

  test('the errors, and what isn\'t supported yet', () => {
    expect(code(() => new BoostedTreesClassifier({ baseScore: 1 }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesClassifier({ objective: 'reg:squarederror' as never }))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesClassifier({ subsample: 0.5 }))).toBe('ML_PARAM');
    // multi:softprob with two classes, as XGBoost refuses it; a base score with three, not supported yet.
    expect(code(() => new BoostedTreesClassifier({ objective: 'multi:softprob' }).fitSync(X, two))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesClassifier({ baseScore: 0.4 }).fitSync(X, three))).toBe('ML_PARAM');
    expect(code(() => new BoostedTreesClassifier().fitSync(X, X.map(() => 1)))).toBe('ML_LABELS');
    // A class whose rows all weigh 0.
    expect(code(() => new BoostedTreesClassifier().fitSync(X, three, { sampleWeight: three.map((v) => (v === 2 ? 0 : 1)) }))).toBe('ML_VALUE');
    expect(code(() => new BoostedTreesClassifier().fitSync([[1], [-Infinity]], [0, 1]))).toBe('ML_VALUE');
    expect(code(() => new BoostedTreesClassifier().predictProbaSync(X))).toBe('ML_NOT_FITTED');
  });
});

describe('boosted trees with missing values (ADR 0044, step 3)', () => {
  const r = new Random(23);
  const X = Array.from({ length: 200 }, () => [Math.floor(r.next() * 6), Math.round(r.next() * 10) / 10]);
  // Column 0 missing in a fifth of the rows, where y is high: the split learns to send missing values with the high side.
  const y = X.map((x) => (x[0]! >= 3 ? 10 : 0) + x[1]!);
  for (let i = 0; i < X.length; i++) if (i % 5 === 0) {
    X[i]![0] = Number.NaN;
    y[i] = 10 + X[i]![1]!;
  }

  test('a split learns the side for missing values; prediction sends them there', () => {
    const m = new BoostedTreesRegressor({ nEstimators: 1, maxDepth: 1, learningRate: 1, regLambda: 0 }).fitSync(X, y);
    const t = m.trees[0]!;
    expect(t.feature[0]).toBe(0);
    expect(t.threshold[0]).toBe(3);
    // The high side is the right: missing values go right, with x ≥ 3.
    expect(t.defaultLeft[0]).toBe(0);
    const [missing, high] = m.predictSync([[Number.NaN, 0.5], [4, 0.5]]);
    expect(missing).toBe(high);
    // Flipped, the high side is the left, and missing values go left.
    const flipped = new BoostedTreesRegressor({ nEstimators: 1, maxDepth: 1, learningRate: 1, regLambda: 0 }).fitSync(X.map((x) => [Number.isNaN(x[0]!) ? Number.NaN : -x[0]!, x[1]!]), y);
    expect(flipped.trees[0]!.defaultLeft[0]).toBe(1);
  });

  test('missing against present: a column whose present values are all the same', () => {
    const Xc = X.map((x) => [Number.isNaN(x[0]!) ? Number.NaN : 1]);
    const m = new BoostedTreesRegressor({ nEstimators: 1, maxDepth: 1 }).fitSync(Xc, y);
    // The cut past the largest value, as XGBoost's: 1 + 1 + 1e-5, in 32-bit floats.
    expect(m.trees[0]!.threshold[0]).toBe(Math.fround(2.00001));
    expect(m.trees[0]!.defaultLeft[0]).toBe(0);
  });

  test('saved with the sides, loaded back; a version 5 document loads with missing values right', () => {
    const m = new BoostedTreesClassifier({ nEstimators: 5, maxDepth: 2 }).fitSync(X, y.map((v) => (v > 5 ? 'high' : 'low')));
    const back = loadModel(JSON.parse(JSON.stringify(saveModel(m)))) as BoostedTreesClassifier;
    expect(Array.from(back.predictProbaSync(X).data)).toEqual(Array.from(m.predictProbaSync(X).data));
    const old = JSON.parse(saveModel(new BoostedTreesRegressor({ nEstimators: 3 }).fitSync([[1], [2], [3], [4]], [1, 2, 3, 4])));
    old.version = 5;
    for (const t of old.state.trees) delete t.defaultLeft;
    const loaded = loadModel(old) as BoostedTreesRegressor;
    expect(Array.from(loaded.trees[0]!.defaultLeft).every((v) => v === 0)).toBe(true);
    expect(loaded.predictSync([[Number.NaN]])[0]).toBe(loaded.predictSync([[100]])[0]);
  });

  test('a column missing everywhere is never split on; Infinity fails', () => {
    const m = new BoostedTreesRegressor({ nEstimators: 3 }).fitSync(X.map((x) => [Number.NaN, x[1]!]), y);
    for (const t of m.trees) for (let k = 0; k < t.left.length; k++) if (t.left[k] !== -1) expect(t.feature[k]).toBe(1);
    expect(code(() => m.predictSync([[Infinity, 1]]))).toBe('ML_VALUE');
  });
});

describe('maxFeatures and columns constant in a node, as scikit-learn counts them', () => {
  // Column 0 is constant everywhere, column 1 is noise, and y follows column 2 alone.
  const r = new Random(31);
  const X = Array.from({ length: 200 }, () => [4, r.next(), r.next()]);
  const y = X.map((x) => (x[2]! > 0.5 ? 10 : 0) + 0.01 * r.next());

  test.each([
    ['best', 'best'],
    ['random, with the presort', 'random'],
  ] as const)('%s: a constant column drawn counts towards maxFeatures, so the root sometimes looks at noise alone', (_name, splitter) => {
    const roots = new Set<number>();
    for (let seed = 0; seed < 40; seed++) {
      // maxFeatures 2 of 3 columns: half or more, so 'random' takes the presort's way.
      const t = new DecisionTreeRegressor({ maxFeatures: 2, maxDepth: 1, splitter, seed }).fitSync(X, y);
      roots.add(t.tree.feature[0]!);
    }
    // Never the constant column; the informative one where it was drawn; the noise where the constant took a draw.
    expect([...roots].sort()).toEqual([1, 2]);
  });

  test('random, sort-free: constant columns drawn take draws from the varying ones', () => {
    // Nine columns, six constant, and the label follows column 8: maxFeatures 2 is below half, so 'random' never
    // sorts. Not counting the constants, a split would look at 2 of the 3 varying columns, and the root picked column
    // 8 in 38 of these 60 seeds; counting them, as scikit-learn does, it often looks at just one, and picks column 8 in
    // 18 (both measured 2026-11-23). Fewer than 28, the midpoint, holds the count.
    const r9 = new Random(33);
    const X9 = Array.from({ length: 200 }, () => [4, -1, 7, 0, 2, 9, r9.next(), r9.next(), r9.next()]);
    const y9 = X9.map((x) => (x[8]! > 0.5 ? 'a' : 'b'));
    let informative = 0;
    for (let seed = 0; seed < 60; seed++) {
      const f = new DecisionTreeClassifier({ maxFeatures: 2, maxDepth: 1, splitter: 'random', seed }).fitSync(X9, y9).tree.feature[0]!;
      expect(f).toBeGreaterThan(5);
      if (f === 8) informative++;
    }
    expect(informative).toBeLessThan(28);
    expect(informative).toBeGreaterThan(0);
  });

  test('every column constant: a single leaf; a constant column never splits', () => {
    const flat = new DecisionTreeRegressor({ maxFeatures: 1, seed: 0 }).fitSync([[1, 2], [1, 2], [1, 2]], [1, 2, 3]);
    expect(flat.getNLeaves()).toBe(1);
    const t = new DecisionTreeRegressor({ maxFeatures: 1, seed: 3 }).fitSync(X, y);
    expect(Array.from(t.tree.feature).filter((f) => f >= 0).every((f) => f !== 0)).toBe(true);
  });
});

describe('extra trees, and trees with splitter: random', () => {
  const r = new Random(9);
  const X = Array.from({ length: 120 }, () => [r.next() * 4, r.next() * 2 - 1, r.next()]);
  const y = X.map((x) => (x[0]! > 2 ? 5 : 1) + (x[1]! > 0 ? 2 : 0) + x[2]! * 0.1);
  const labels = X.map((x) => (x[0]! > 2 ? 'big' : x[1]! > 0 ? 'up' : 'down'));

  test('a random split: each threshold lies within its node\'s values, and its rows go left at or below it', () => {
    const t = new DecisionTreeRegressor({ splitter: 'random', seed: 2, maxDepth: 6 }).fitSync(X, y);
    const leaf = t.apply(X);
    const { left, right, feature, threshold, nSamples } = t.tree;
    // Walk each row down by the stored thresholds: it reaches the leaf apply() gives, and every node it passes has
    // rows on both sides.
    for (let i = 0; i < X.length; i++) {
      let node = 0;
      while (left[node] !== -1) {
        expect(nSamples[left[node]!]).toBeGreaterThan(0);
        expect(nSamples[right[node]!]).toBeGreaterThan(0);
        node = Math.fround(X[i]![feature[node]!]!) <= threshold[node]! ? left[node]! : right[node]!;
      }
      expect(node).toBe(leaf[i]);
    }
    expect(t.getParams().splitter).toBe('random');
    expect(new DecisionTreeRegressor().getParams()).not.toHaveProperty('splitter');
    // A grown tree with every leaf pure: on the rows it was grown on, its predictions are the targets.
    const full = new DecisionTreeRegressor({ splitter: 'random', seed: 5 }).fitSync(X, y);
    expect(Array.from(full.predictSync(X))).toEqual(y);
  });

  test('a seed repeats a random tree, another seed grows another', () => {
    const a = new DecisionTreeClassifier({ splitter: 'random', seed: 3 }).fitSync(X, labels);
    const b = new DecisionTreeClassifier({ splitter: 'random', seed: 3 }).fitSync(X, labels);
    const c = new DecisionTreeClassifier({ splitter: 'random', seed: 4 }).fitSync(X, labels);
    expect(a.toJSON()).toEqual(b.toJSON());
    expect(Array.from(c.tree.threshold)).not.toEqual(Array.from(a.tree.threshold));
    const back = loadModel(saveModel(a)) as DecisionTreeClassifier;
    expect(back.getParams().splitter).toBe('random');
    expect(back.predictSync(X)).toEqual(a.predictSync(X));
  });

  test('no bootstrap by default; with it, out-of-bag scores; a seed repeats; fit equals fitSync; saved and loaded', async () => {
    const f = new ExtraTreesRegressor({ nEstimators: 10, seed: 1 }).fitSync(X, y);
    expect(f.getParams().bootstrap).toBe(false);
    expect(new ExtraTreesClassifier().getParams()).toMatchObject({ bootstrap: false, maxFeatures: 'sqrt', criterion: 'gini' });
    // Without bootstrap, every tree is grown on every row.
    for (const t of f.estimators) expect(t.tree.nSamples[0]).toBe(120);
    for (const t of f.estimators) expect(t.getParams().splitter).toBe('random');
    expect(code(() => f.oobScore)).toBe('ML_PARAM');
    const a = await new ExtraTreesClassifier({ nEstimators: 10, seed: 4, bootstrap: true, oobScore: true }).fit(X, labels);
    const b = new ExtraTreesClassifier({ nEstimators: 10, seed: 4, bootstrap: true, oobScore: true }).fitSync(X, labels);
    expect(a.toJSON()).toEqual(b.toJSON());
    expect(a.oobScore).toBeGreaterThan(0.7);
    expect(a.oobDecisionFunction.rows).toBe(120);
    const back = loadModel(saveModel(a)) as ExtraTreesClassifier;
    expect(back.kind).toBe('ExtraTreesClassifier');
    expect(Array.from(back.predictProbaSync(X).data)).toEqual(Array.from(a.predictProbaSync(X).data));
    expect(back.oobScore).toBe(a.oobScore);
    const regBack = loadModel(saveModel(f)) as ExtraTreesRegressor;
    expect(regBack.kind).toBe('ExtraTreesRegressor');
    expect(Array.from(regBack.predictSync(X))).toEqual(Array.from(f.predictSync(X)));
    expect(regBack.getParams()).toEqual(f.getParams());
  });

  test('importances rank the noise column last; it fits in a grid search and with sample weights', async () => {
    const f = new ExtraTreesClassifier({ nEstimators: 40, seed: 3 }).fitSync(X, labels);
    expect(f.featureImportances.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(f.featureImportances[2]).toBeLessThan(Math.min(f.featureImportances[0]!, f.featureImportances[1]!));
    const g = await new GridSearchCV(new ExtraTreesRegressor({ nEstimators: 5, seed: 0 }), { maxDepth: [1, 4] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, y);
    expect(g.bestParams.maxDepth).toBe(4);
    const w = X.map((_, i) => (i < 20 ? 0 : 1 + (i % 3)));
    const weighted = new ExtraTreesRegressor({ nEstimators: 5, seed: 6 }).fitSync(X, y, { sampleWeight: w });
    // Rows of weight 0 take no part.
    for (const t of weighted.estimators) expect(t.tree.nSamples[0]).toBe(100);
  });

  test('the errors', () => {
    expect(code(() => new DecisionTreeRegressor({ splitter: 'worst' as never }))).toBe('ML_PARAM');
    expect(code(() => new ExtraTreesRegressor({ oobScore: true }))).toBe('ML_PARAM');
    expect(code(() => new ExtraTreesRegressor({ maxSamples: 0.5 }))).toBe('ML_PARAM');
    expect(code(() => new ExtraTreesClassifier({ classWeight: 'balanced_subsample' as never }))).toBe('ML_PARAM');
    expect(code(() => new ExtraTreesRegressor().predictSync(X))).toBe('ML_NOT_FITTED');
  });
});
