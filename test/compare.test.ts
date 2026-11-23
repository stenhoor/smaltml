import { readFileSync } from 'node:fs';
import { describe, expect, test, vi } from 'vitest';
import {
  ColumnTransformer,
  GridSearchCV,
  KFold,
  StratifiedKFold,
  confusionMatrix,
  crossValScore,
  f1Score,
  meanAbsoluteError,
  precisionScore,
  recallScore,
  rocAucScore,
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
  GroupKFold,
  crossValidate,
  MultiOutputRegressor,
  DecisionTreeRegressor,
  BoostedTreesRegressor,
  BoostedTreesClassifier,
  DecisionTreeClassifier,
  RandomForestRegressor,
  RandomForestClassifier,
  ExtraTreesRegressor,
  ExtraTreesClassifier,
  accuracyScore,
  logLoss,
  meanSquaredError,
  pipeline,
  r2Score,
  trainTestSplit,
  type BoostTree,
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
    // Each part of a step__param name on its own: ridge__alpha stays ridge__alpha, fit_intercept becomes fitIntercept.
    Object.entries(params).map(([k, v]) => [k.split('__').map((part) => part.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())).join('__'), v === 'Infinity' ? Infinity : v]),
  );
}

/** XGBoost's trees, as compare/run.py nests them: [column, cut, left, right] at a split, the value at a leaf. */
type XgbNode = number | [number, number, XgbNode, XgbNode];

/** The same trees as XGBoost's: each split's column and cut exactly (as 32-bit floats), the leaves within tol. */
function sameTrees(trees: readonly BoostTree[], expected: XgbNode[], tol: Case['tol']): void {
  expect(trees.length).toBe(expected.length);
  const leaves: number[] = [];
  const want: number[] = [];
  trees.forEach((t, k) => {
    const walk = (x: XgbNode, node: number, path: string): void => {
      if (typeof x === 'number') {
        expect(t.left[node], `tree ${k} ${path}: XGBoost has a leaf`).toBe(-1);
        leaves.push(t.value[node]!);
        want.push(x);
        return;
      }
      const [f, cut, yes, no] = x;
      expect([t.feature[node], Math.fround(t.threshold[node]!)], `tree ${k} ${path}: XGBoost splits f${f} < ${cut}`).toEqual([f, Math.fround(cut)]);
      walk(yes, t.left[node]!, path + 'L');
      walk(no, t.right[node]!, path + 'R');
    };
    walk(expected[k]!, 0, 'root');
  });
  close(leaves, want, tol, 'leaf values');
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
  test.each(fixture.cases.map((c) => [c.name, c] as const))('%s', async (_name, c) => {
    const { input: inp, expected: exp, tol } = c;
    const fails = DIVERGES[c.name];
    const run = () => check(c, inp, exp, tol);
    if (fails) {
      expect(run).toThrow(expect.objectContaining({ code: fails }));
      return;
    }
    await run();
    // The forests' statistical cases fit hundreds of trees.
  }, 60_000);
});

function check(c: Case, inp: Record<string, any>, exp: Record<string, any>, tol: Case['tol']): void | Promise<void> {
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
    case 'StandardScaler-partial': {
      const s = new StandardScaler(camel(inp.params));
      for (const b of inp.batches) s.partialFitSync(nan(b) as number[][]);
      if (exp.mean) close(s.mean!, exp.mean, tol, 'mean');
      if (exp.var) close(s.var!, exp.var, tol, 'var');
      if (exp.scale) close(s.scale!, exp.scale, tol, 'scale');
      const seen = s.nSamplesSeen;
      if (typeof exp.nSamplesSeen === 'number') expect(seen).toBe(exp.nSamplesSeen);
      else close(seen as Float64Array, exp.nSamplesSeen, tol, 'nSamplesSeen');
      close(rowsOf(s.transformSync(nan(inp.batches.flat()) as number[][])), exp.transform, tol, 'transform');
      return;
    }
    case 'DecisionTreeRegressor':
    case 'DecisionTreeClassifier': {
      const params = camel(inp.params) as Record<string, unknown>;
      const m = c.kind === 'DecisionTreeRegressor' ? new DecisionTreeRegressor({ ...params, seed: 0 }) : new DecisionTreeClassifier({ ...params, seed: 0 });
      (m as DecisionTreeRegressor).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      const t = m.tree;
      expect(Array.from(t.left), 'children left').toEqual(exp.left);
      expect(Array.from(t.right), 'children right').toEqual(exp.right);
      expect(Array.from(t.feature), 'feature').toEqual(exp.feature);
      close(t.threshold, exp.threshold, tol, 'threshold');
      close(t.impurity, exp.impurity, tol, 'impurity');
      expect(Array.from(t.nSamples), 'samples').toEqual(exp.nSamples);
      close(t.weightedNSamples, exp.weightedNSamples, tol, 'weighted samples');
      close(t.value, exp.value, tol, 'value');
      close(m.featureImportances, exp.importances, tol, 'importances');
      expect(m.getDepth()).toBe(exp.depth);
      expect(m.getNLeaves()).toBe(exp.leaves);
      expect(Array.from(m.apply(inp.Xnew)), 'apply').toEqual(exp.apply);
      if (c.kind === 'DecisionTreeClassifier') {
        const k = m as DecisionTreeClassifier;
        expect(k.classes).toEqual(exp.classes);
        expect(k.predictSync(inp.Xnew)).toEqual(exp.predict);
        close(rowsOf(k.predictProbaSync(inp.Xnew)), exp.proba, tol, 'proba');
      } else close((m as DecisionTreeRegressor).predictSync(inp.Xnew), exp.predict, tol, 'predict');
      return;
    }
    case 'forest-exact': {
      const params = camel(inp.params);
      const opts = inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {};
      if (inp.classifier) {
        const m = new RandomForestClassifier({ ...params, seed: 0 }).fitSync(inp.X, inp.y, opts);
        expect(m.classes).toEqual(exp.classes);
        expect(m.predictSync(inp.Xnew)).toEqual(exp.predict);
        close(rowsOf(m.predictProbaSync(inp.Xnew)), exp.proba, tol, 'proba');
        close(m.featureImportances, exp.importances, tol, 'importances');
      } else {
        const m = new RandomForestRegressor({ ...params, seed: 0 }).fitSync(inp.X, inp.y, opts);
        close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
        close(m.featureImportances, exp.importances, tol, 'importances');
      }
      return;
    }
    case 'BoostedTreesRegressor': {
      // XGBoost's XGBRegressor, its parameters in camelCase: the same trees (each split's column and cut), and leaves,
      // predictions, importances and base score within its 32-bit arithmetic.
      const m = new BoostedTreesRegressor(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      close([m.baseScore], [exp.baseScore], tol, 'base score');
      sameTrees(m.trees, exp.trees, tol);
      close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
      close(m.featureImportances, exp.importances, tol, 'importances');
      return;
    }
    case 'BoostedTreesClassifier': {
      // XGBoost's XGBClassifier: the same trees in the same order, and probabilities, importances and base scores within
      // its 32-bit arithmetic; the same predicted labels.
      const m = new BoostedTreesClassifier(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      close(m.baseScore, exp.baseScore, tol, 'base score');
      sameTrees(m.trees, exp.trees, tol);
      const P = m.predictProbaSync(inp.Xnew);
      const rows: number[][] = Array.from({ length: P.rows }, (_, i) => Array.from({ length: P.cols }, (_, k) => P.data[k * P.rows + i]!));
      close(rows.flat(), exp.proba, tol, 'predictProba');
      expect(m.predictSync(inp.Xnew)).toEqual(exp.predict);
      close(m.featureImportances, exp.importances, tol, 'importances');
      return;
    }
    case 'forest-score': {
      // smaltml's own seeded draws: the mean score over the seeds, against scikit-learn's mean, within the margin.
      // `model`: a random forest (the default), extra trees, or a single tree (splitter 'random').
      const params = camel(inp.params);
      const fit = inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {};
      const Classifier = { forest: RandomForestClassifier, extra: ExtraTreesClassifier, tree: DecisionTreeClassifier }[(inp.model ?? 'forest') as 'forest'];
      const Regressor = { forest: RandomForestRegressor, extra: ExtraTreesRegressor, tree: DecisionTreeRegressor }[(inp.model ?? 'forest') as 'forest'];
      let test = 0;
      let oob = 0;
      for (const seed of inp.seeds as number[]) {
        if (inp.classifier) {
          const m = new Classifier({ ...params, seed }).fitSync(inp.X, inp.y, fit);
          const p = m.predictSync(inp.Xnew);
          test += p.filter((l, i) => l === inp.ynew[i]).length / p.length;
          if (exp.oob !== null) oob += (m as RandomForestClassifier).oobScore;
        } else {
          const m = new Regressor({ ...params, seed }).fitSync(inp.X, inp.y, fit);
          test += r2Score(inp.ynew, m.predictSync(inp.Xnew));
          if (exp.oob !== null) oob += (m as RandomForestRegressor).oobScore;
        }
      }
      close(test / inp.seeds.length, exp.test, tol, 'mean test score');
      if (exp.oob !== null) close(oob / inp.seeds.length, exp.oob, tol, 'mean out-of-bag score');
      return;
    }
    case 'MultiOutputRegressor': {
      const models = { LinearRegression: (p: object) => new LinearRegression(p), Ridge: (p: object) => new Ridge(p), Lasso: (p: object) => new Lasso(p), ElasticNet: (p: object) => new ElasticNet(p) };
      const m = new MultiOutputRegressor(models[inp.model as keyof typeof models](camel(inp.params))).fitSync(inp.X, inp.Y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      m.estimators.forEach((e, k) => {
        close((e as LinearRegression).coef, exp.coef[k], tol, `coef ${k}`);
        close([(e as LinearRegression).intercept], [exp.intercept[k]], tol, `intercept ${k}`);
      });
      close(rowsOf(m.predictSync(inp.Xnew)), exp.predict, tol, 'predict');
      return;
    }
    case '2-D y': {
      const models = { LinearRegression: (p: object) => new LinearRegression(p), Ridge: (p: object) => new Ridge(p), Lasso: (p: object) => new Lasso(p), ElasticNet: (p: object) => new ElasticNet(p) };
      const m = models[inp.model as keyof typeof models](camel(inp.params)).fitSync(inp.X, inp.Y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      close(rowsOf(m.coefs), exp.coefs, tol, 'coefs');
      close(m.intercepts, exp.intercepts, tol, 'intercepts');
      close(rowsOf(m.predictTargetsSync(inp.Xnew)), exp.predict, tol, 'predict');
      return;
    }
    case 'LinearRegression': {
      const m = new LinearRegression(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      close(m.coef, exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
      return;
    }
    case 'LogisticRegression': {
      const m = new LogisticRegression(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      expect(m.classes).toEqual(exp.classes);
      close(rowsOf(m.coef), exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(rowsOf(m.predictProbaSync(inp.Xnew)), exp.predictProba, tol, 'predictProba');
      // An L1 penalty sets coefficients to exactly 0: the same ones as scikit-learn.
      if (exp.zero) expect(m.coef.toRows().map((row) => row.map((v) => v === 0)), 'which coefficients are 0').toEqual(exp.zero);
      expect(m.predictSync(inp.Xnew)).toEqual(exp.predict as Label[]);
      if ('converged' in exp) expect(m.converged, 'converged, as scikit-learn (no ConvergenceWarning)').toBe(exp.converged);
      return;
    }
    case 'Pipeline': {
      const p = pipeline(new StandardScaler(), new LogisticRegression(camel(inp.params))).fitSync(inp.X, inp.y);
      close(rowsOf(p.predictProbaSync(inp.Xnew)), exp.predictProba, tol, 'predictProba');
      expect(p.predictSync(inp.Xnew)).toEqual(exp.predict as Label[]);
      return;
    }
    case 'split-stratify': {
      // The rows chosen differ (smaltml's own shuffle); each class's count in each set is the same rule's.
      const y = inp.y as Label[];
      const X = y.map((_, i) => [i]);
      const { yTrain, yTest } = trainTestSplit(X, y as unknown as string[], { ...inp.params, stratify: y as unknown as string[], seed: 3 });
      const count = (ys: readonly Label[]) => (exp.classes as Label[]).map((c) => ys.filter((v) => v === c).length);
      expect(count(yTrain as Label[]), 'training counts').toEqual(exp.train);
      expect(count(yTest as Label[]), 'test counts').toEqual(exp.test);
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
      const m = make(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
      close(m.coef, exp.coef, tol, 'coef');
      close(m.intercept, exp.intercept, tol, 'intercept');
      close(m.predictSync(inp.Xnew), exp.predict, tol, 'predict');
      // The L1 penalty sets coefficients to exactly 0: the same ones.
      if (exp.zero) expect(Array.from(m.coef, (v) => v === 0), 'which coefficients are 0').toEqual(exp.zero);
      if ('converged' in exp) expect(m.converged, 'converged, as scikit-learn (no ConvergenceWarning)').toBe(exp.converged);
      return;
    }
    case 'mae':
      close(meanAbsoluteError(inp.yTrue, inp.yPred), exp.value, tol, 'meanAbsoluteError');
      return;
    case 'prf': {
      const o = { average: inp.params.average as never, zeroDivision: 0 };
      close(precisionScore(inp.yTrue, inp.yPred, o) as never, exp.precision, tol, 'precision');
      close(recallScore(inp.yTrue, inp.yPred, o) as never, exp.recall, tol, 'recall');
      close(f1Score(inp.yTrue, inp.yPred, o) as never, exp.f1, tol, 'f1');
      return;
    }
    case 'confusion': {
      const m = confusionMatrix(inp.yTrue, inp.yPred);
      expect(m.names).toEqual(exp.labels.map(String));
      close(rowsOf(m), exp.matrix, tol, 'confusion matrix');
      return;
    }
    case 'auc':
      close(rocAucScore(inp.yTrue, inp.yScore, camel(inp.params) as never), exp.value, tol, 'rocAucScore');
      return;
    case 'kfold':
      expect(new KFold(camel(inp.params) as never).split(inp.n).map((f) => Array.from(f.test))).toEqual(exp.tests);
      return;
    case 'groupkfold':
      expect(new GroupKFold({ nSplits: inp.params.n_splits }).split(inp.groups.length, undefined, inp.groups).map((f) => Array.from(f.test))).toEqual(exp.tests);
      return;
    case 'cross-validate':
    case 'grid-multi':
    case 'grid-train':
      return checkScorers(c, inp, exp, tol);
    case 'cvscore-groups':
    case 'grid-groups':
      return checkGroups(c, inp, exp, tol);
    case 'stratified': {
      const folds = new StratifiedKFold(camel(inp.params) as never).split(inp.y.length, inp.y);
      expect(folds.map((f) => f.test.length).sort((a, b) => a - b)).toEqual(exp.testSizes);
      return;
    }
    case 'cvscore':
    case 'grid':
      return checkSearch(c, inp, exp, tol);
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
    case 'encoder-infrequent': {
      const e = (inp.encoder === 'OneHotEncoder' ? new OneHotEncoder(camel(inp.params) as never) : new OrdinalEncoder(camel(inp.params) as never)).fitSync(inp.X);
      close(rowsOf(e.transformSync(inp.Xnew)), exp.transform, tol, 'transform');
      expect(e.infrequentCategories, 'infrequent categories').toEqual(exp.infrequent);
      if (exp.names) expect((e as OneHotEncoder).getFeatureNamesOut()).toEqual(exp.names);
      if (exp.dropIdx) expect((e as OneHotEncoder).dropIdx, 'dropIdx').toEqual(exp.dropIdx);
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
    case 'convergence': {
      // A fit stopped early: smaltml says so (converged false, a warning) where scikit-learn raised ConvergenceWarning.
      const make = { LogisticRegression: (p: object) => new LogisticRegression(p), Lasso: (p: object) => new Lasso(p), ElasticNet: (p: object) => new ElasticNet(p) }[inp.model as string]!;
      const warned = vi.spyOn(console, 'warn').mockImplementation(() => {});
      try {
        const m = make(camel(inp.params)).fitSync(inp.X, inp.y, inp.sampleWeight ? { sampleWeight: inp.sampleWeight } : {});
        expect(m.converged, 'converged').toBe(exp.converged);
        expect(m.nIter, 'nIter').toBe(exp.nIter);
        const ours = warned.mock.calls.filter(([msg]) => String(msg).startsWith('smaltml ML_CONVERGENCE:'));
        expect(ours.length, 'ML_CONVERGENCE warnings').toBe(exp.converged ? 0 : 1);
      } finally {
        warned.mockRestore();
      }
      return;
    }
    default:
      throw new Error(`no check for ${c.kind}`);
  }
}

/** The models the cross-validation cases use, by name. */
function searchModel(name: string) {
  if (name === 'scaled-ridge') return pipeline(new StandardScaler(), new Ridge({ alpha: 2 }));
  return new LogisticRegression({ tol: 1e-10, maxIter: 10000 });
}

async function checkScorers(c: Case, inp: Record<string, any>, exp: Record<string, any>, tol: Case['tol']): Promise<void> {
  const cv = new KFold({ nSplits: 5 });
  if (c.kind === 'cross-validate') {
    const r = await crossValidate(searchModel(inp.model), inp.X, inp.y, { cv, scoring: inp.scoring, returnTrainScore: true });
    for (const k of Object.keys(exp.test)) {
      close(r.test[k]!, exp.test[k], tol, `test ${k}`);
      close(r.train![k]!, exp.train[k], tol, `train ${k}`);
    }
    expect(r.fitTime).toHaveLength(5);
    return;
  }
  const grid = camel(inp.grid) as Record<string, unknown[]>;
  if (c.kind === 'grid-train') {
    const g = await new GridSearchCV(searchModel(inp.model), grid, { cv, returnTrainScore: true }).fit(inp.X, inp.y);
    close(g.cvResults.meanTrainScore!, exp.meanTrain, tol, 'mean train score');
    close(g.cvResults.stdTrainScore!, exp.stdTrain, { rtol: tol.rtol * 100, atol: tol.atol * 100 }, 'std train score');
    expect(g.bestParams).toEqual(camel(exp.best));
    return;
  }
  const g = await new GridSearchCV(searchModel(inp.model), grid, { cv, scoring: inp.scoring, refit: inp.refit, returnTrainScore: true }).fit(inp.X, inp.y);
  for (const [k, e] of Object.entries(exp.scores as Record<string, any>)) {
    const r = g.cvResultsByScorer.scores[k]!;
    close(r.meanTestScore, e.meanTest, tol, `${k} mean test`);
    expect(Array.from(r.rankTestScore), `${k} rank`).toEqual(e.rankTest);
    close(r.meanTrainScore!, e.meanTrain, tol, `${k} mean train`);
    close(r.stdTrainScore!, e.stdTrain, { rtol: tol.rtol * 100, atol: tol.atol * 100 }, `${k} std train`);
  }
  expect(g.bestIndex).toBe(exp.bestIndex);
  expect(g.bestParams).toEqual(camel(exp.best));
  close([g.bestScore], [exp.bestScore], tol, 'best score');
}

async function checkGroups(c: Case, inp: Record<string, any>, exp: Record<string, any>, tol: Case['tol']): Promise<void> {
  const cv = new GroupKFold({ nSplits: 4 });
  if (c.kind === 'cvscore-groups') {
    close(await crossValScore(searchModel(inp.model), inp.X, inp.y, { cv, groups: inp.groups }), exp.scores, tol, 'scores');
    return;
  }
  const g = await new GridSearchCV(searchModel(inp.model), camel(inp.grid) as Record<string, unknown[]>, { cv }).fit(inp.X, inp.y, { groups: inp.groups });
  close(g.cvResults.meanTestScore, exp.mean, tol, 'mean test score');
  expect(Array.from(g.cvResults.rankTestScore)).toEqual(exp.rank);
  expect(g.bestParams).toEqual(camel(exp.best));
}

async function checkSearch(c: Case, inp: Record<string, any>, exp: Record<string, any>, tol: Case['tol']): Promise<void> {
  const cv = new KFold({ nSplits: 5 });
  const scoring = inp.params.scoring ?? undefined;
  if (c.kind === 'cvscore') {
    close(await crossValScore(searchModel(inp.model), inp.X, inp.y, { cv, scoring }), exp.scores, tol, 'scores');
    return;
  }
  const grid = camel(inp.grid) as Record<string, unknown[]>;
  const g = await new GridSearchCV(searchModel(inp.model), grid, { cv, scoring }).fit(inp.X, inp.y);
  const camelParams = (exp.params as Record<string, unknown>[]).map((p) => camel(p));
  expect(g.cvResults.params).toEqual(camelParams);
  close(g.cvResults.meanTestScore, exp.mean, tol, 'mean test score');
  close(g.cvResults.stdTestScore, exp.std, { rtol: tol.rtol * 100, atol: tol.atol * 100 }, 'std test score');
  expect(Array.from(g.cvResults.rankTestScore)).toEqual(exp.rank);
  expect(g.bestParams).toEqual(camel(exp.best));
}
