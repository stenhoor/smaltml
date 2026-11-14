import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
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
  StandardScaler,
  accuracyScore,
  logLoss,
  meanSquaredError,
  pipeline,
  r2Score,
  trainTestSplit,
  type Label,
} from '../src/index.js';

/**
 * smaltml against scikit-learn (ADR 0024, section 7). fixtures/compare.json holds each case's data and scikit-learn's
 * results, written by compare/run.py (`npm run compare`); this checks smaltml's results against them within each
 * case's tolerance, with no Python.
 */
interface Case {
  name: string;
  kind: string;
  input: Record<string, any>;
  expected: Record<string, any>;
  tol: { rtol: number; atol: number };
}

const fixture = JSON.parse(readFileSync(new URL('../fixtures/compare.json', import.meta.url), 'utf8')) as { reference: Record<string, string>; cases: Case[] };

/**
 * Where smaltml deliberately differs, the case says what smaltml does instead; each is in docs/reference/divergences.md.
 */
const DIVERGES: Record<string, string> = {
  // scikit-learn returns the minimum-norm solution of many; smaltml refuses rather than pick one.
  'linear: as many rows as columns': 'ML_SINGULAR',
};

/** JSON null back to NaN, nested. */
function nan(v: any): any {
  return v === null ? NaN : Array.isArray(v) ? v.map(nan) : v;
}

/** Python's snake_case parameters in smaltml's camelCase (ADR 0024, section 2). */
function camel(params: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(params).map(([k, v]) => [k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()), v === 'Infinity' ? Infinity : v]),
  );
}

function close(actual: ArrayLike<number> | number, expected: any, tol: Case['tol'], what: string): void {
  const a = typeof actual === 'number' ? [actual] : Array.from(actual);
  const e = (typeof expected === 'number' || expected === null ? [expected] : (expected as any[]).flat(Infinity)).map((v) => (v === null ? NaN : v));
  expect(a.length, `${what}: length`).toBe(e.length);
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = e[i]!;
    if (Number.isNaN(y)) {
      expect(Number.isNaN(x), `${what}[${i}] should be NaN, is ${x}`).toBe(true);
      continue;
    }
    const err = Math.abs(x - y);
    const bound = tol.atol + tol.rtol * Math.abs(y);
    if (!(err <= bound)) expect.fail(`${what}[${i}]: smaltml ${x}, scikit-learn ${y} (difference ${err}, allowed ${bound})`);
  }
}

/** Matrix rows as one flat row-major list, to compare with NumPy's nested lists. */
function rowsOf(m: Matrix): number[] {
  return m.toRows().flat();
}

describe(`compared with scikit-learn ${fixture.reference['scikit-learn']}`, () => {
  test.each(fixture.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const { input: inp, expected: exp, tol } = c;
    const fails = DIVERGES[c.name];
    const run = () => check(c, inp, exp, tol);
    if (fails) {
      expect(run).toThrow(expect.objectContaining({ code: fails }));
      return;
    }
    run();
  });
});

function check(c: Case, inp: Record<string, any>, exp: Record<string, any>, tol: Case['tol']): void {
  switch (c.kind) {
    case 'StandardScaler': {
      const X = nan(inp.X) as number[][];
      const s = new StandardScaler(camel(inp.params)).fitSync(X);
      if (exp.mean) close(s.mean!, exp.mean, tol, 'mean');
      if (exp.var) close(s.var!, exp.var, tol, 'var');
      if (exp.scale) close(s.scale!, exp.scale, tol, 'scale');
      close(rowsOf(s.transformSync(X)), exp.transform, tol, 'transform');
      return;
    }
    case 'LinearRegression': {
      const m = new LinearRegression(camel(inp.params)).fitSync(inp.X, inp.y);
      close(m.coef, exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
      return;
    }
    case 'LogisticRegression': {
      const m = new LogisticRegression(camel(inp.params)).fitSync(inp.X, inp.y);
      expect(m.classes).toEqual(exp.classes);
      close(rowsOf(m.coef), exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(rowsOf(m.predictProbaSync(inp.Xnew)), exp.predictProba, tol, 'predictProba');
      expect(m.predictSync(inp.Xnew)).toEqual(exp.predict as Label[]);
      return;
    }
    case 'Pipeline': {
      const p = pipeline(new StandardScaler(), new LogisticRegression(camel(inp.params))).fitSync(inp.X, inp.y);
      close(rowsOf(p.predictProbaSync(inp.Xnew)), exp.predictProba, tol, 'predictProba');
      expect(p.predictSync(inp.Xnew)).toEqual(exp.predict as Label[]);
      return;
    }
    case 'regression-metrics':
      close(meanSquaredError(inp.yTrue, inp.yPred), exp.meanSquaredError, tol, 'meanSquaredError');
      close(r2Score(inp.yTrue, inp.yPred), exp.r2Score, tol, 'r2Score');
      return;
    case 'classification-metrics':
      close(accuracyScore(inp.yTrue, inp.yPred), exp.accuracyScore, tol, 'accuracyScore');
      close(logLoss(inp.yTrue, inp.yProba ?? inp.yProba1), exp.logLoss, tol, 'logLoss');
      return;
    case 'trainTestSplit': {
      const n = inp.n as number;
      const X = Array.from({ length: n }, (_, i) => [2 * i, 2 * i + 1]);
      const split = trainTestSplit(X, undefined, { ...camel(inp.params), shuffle: false });
      expect(split.XTrain.rows).toBe(exp.nTrain);
      expect(split.XTest.rows).toBe(exp.nTest);
      expect(split.XTrain.toRows()).toEqual(exp.XTrain);
      expect(split.XTest.toRows()).toEqual(exp.XTest);
      return;
    }
    case 'Ridge':
    case 'Lasso':
    case 'ElasticNet': {
      const make = { Ridge: (p: object) => new Ridge(p), Lasso: (p: object) => new Lasso(p), ElasticNet: (p: object) => new ElasticNet(p) }[c.kind]!;
      const m = make(camel(inp.params)).fitSync(inp.X, inp.y);
      close(m.coef, exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
      // The L1 penalty sets coefficients to exactly 0: the same ones.
      if (exp.zero) expect(Array.from(m.coef, (v) => v === 0), 'which coefficients are 0').toEqual(exp.zero);
      return;
    }
    case 'MinMaxScaler': {
      const s = new MinMaxScaler(camel(inp.params) as never).fitSync(nan(inp.X));
      close(s.dataMin, exp.dataMin, tol, 'dataMin');
      close(s.dataMax, exp.dataMax, tol, 'dataMax');
      close(s.scale, exp.scale, tol, 'scale');
      close(s.min, exp.min, tol, 'min');
      close(rowsOf(s.transformSync(nan(inp.Xnew))), exp.transform, tol, 'transform');
      return;
    }
    case 'RobustScaler': {
      const s = new RobustScaler(camel(inp.params) as never).fitSync(nan(inp.X));
      if (exp.center) close(s.center!, exp.center, tol, 'center');
      else expect(s.center).toBeNull();
      if (exp.scale) close(s.scale!, exp.scale, tol, 'scale');
      else expect(s.scale).toBeNull();
      close(rowsOf(s.transformSync(nan(inp.Xnew))), exp.transform, tol, 'transform');
      return;
    }
    case 'SimpleImputer': {
      const s = new SimpleImputer(camel(inp.params) as never).fitSync(nan(inp.X));
      close(s.statistics, exp.statistics, tol, 'statistics');
      close(rowsOf(s.transformSync(nan(inp.X))), exp.transform, tol, 'transform');
      return;
    }
    case 'OrdinalEncoder': {
      const e = new OrdinalEncoder(camel(inp.params) as never).fitSync(inp.X);
      close(rowsOf(e.transformSync(inp.Xnew)), exp.transform, tol, 'transform');
      return;
    }
    case 'OneHotEncoder': {
      const e = new OneHotEncoder(camel(inp.params) as never).fitSync(inp.X);
      close(rowsOf(e.transformSync(inp.Xnew)), exp.transform, tol, 'transform');
      // Names of text categories; numbers are written by JavaScript (3, not 3.0).
      const text = new OneHotEncoder(camel(inp.params) as never).fitSync((inp.X as unknown[][]).map((r) => [r[0]]) as never);
      expect(text.getFeatureNamesOut()).toEqual(exp.textNames);
      return;
    }
    case 'ColumnTransformer': {
      const ct = new ColumnTransformer(
        [['scale', new StandardScaler(), [0, 2]], ['onehot', new OneHotEncoder(), [1]]],
        camel(inp.params) as never,
      ).fitSync(inp.X);
      close(rowsOf(ct.transformSync(inp.X)), exp.transform, tol, 'transform');
      expect(ct.getFeatureNamesOut()).toEqual(exp.names);
      return;
    }
    case 'ColumnPipeline': {
      const p = pipeline(
        new ColumnTransformer([['scale', new StandardScaler(), [0, 2]], ['onehot', new OneHotEncoder({ handleUnknown: 'ignore' }), [1]]]),
        new LogisticRegression({ tol: 1e-10, maxIter: 10000 }),
      ).fitSync(inp.X, inp.y);
      close(rowsOf(p.predictProbaSync(inp.X)), exp.predictProba, tol, 'predictProba');
      return;
    }
    default:
      throw new Error(`no check for ${c.kind}`);
  }
}
