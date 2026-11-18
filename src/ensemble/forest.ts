/**
 * Random forests (ADR 0039): many decision trees, each grown on a bootstrap sample of the rows and, at each split, a
 * random subset of the columns, their predictions averaged. scikit-learn's `RandomForestRegressor` and
 * `RandomForestClassifier`, with smaltml's own seeded draws: a seeded forest repeats itself to the last bit, but isn't
 * scikit-learn's forest for the same random_state.
 */
import { fail, warn } from '../core/errors.js';
import { Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, notFitted, toLabels, uniqueLabels, type Label, type Labels, type ModelDocument } from '../core/model.js';
import { Random } from '../core/random.js';
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  type DecisionTreeClassifierParams,
  type DecisionTreeParams,
  type DecisionTreeRegressorParams,
} from '../tree/index.js';

/** What both forests take, beside their trees' parameters. */
export interface ForestParams extends Omit<DecisionTreeParams, 'seed'> {
  /** How many trees. Default 100. */
  nEstimators?: number;
  /** Whether each tree sees a bootstrap sample of the rows (drawn with replacement), or every row. Default true. */
  bootstrap?: boolean;
  /**
   * How many rows each bootstrap sample draws: a whole number (more than the rows is allowed), a share (above 0, below
   * 1) of the rows or, with weights, of their sum, rounded down and at least 1; or as many as the rows (`null`, the
   * default). Only with `bootstrap`.
   */
  maxSamples?: number | null;
  /** Whether to score the forest on each row with the trees that didn't draw it: `oobScore` after fit. Only with `bootstrap`. */
  oobScore?: boolean;
  /** The seed for the bootstrap samples and each tree's column draws. Default: a new one. */
  seed?: number;
}

export interface RandomForestRegressorParams extends ForestParams {
  /** How each split is chosen, as the trees'. Default `'squared_error'`. */
  criterion?: DecisionTreeRegressorParams['criterion'];
}

export interface RandomForestClassifierParams extends ForestParams {
  /** How each split is chosen, as the trees'. Default `'gini'`. */
  criterion?: DecisionTreeClassifierParams['criterion'];
  /**
   * Class weights: `'balanced'` (the total weight over classes × the class's weight, so the classes weigh the same), or
   * weights by label. They multiply the rows' weights, so they change the bootstrap's chances. Default none.
   */
  classWeight?: DecisionTreeClassifierParams['classWeight'];
}

export interface ForestFitOptions {
  /**
   * Each row's weight, 0 or more. With bootstrap, a row's chance of being drawn is in proportion to its weight; without,
   * each tree weighs the rows by it. Default 1 for every row.
   */
  sampleWeight?: VectorInput;
}

const isWhole = (v: unknown, min: number) => typeof v === 'number' && Number.isInteger(v) && v >= min;

function checkForest(p: ForestParams) {
  const nEstimators = p.nEstimators ?? 100;
  if (!isWhole(nEstimators, 1)) fail('ML_PARAM', `nEstimators is a whole number of at least 1, not ${String(nEstimators)}`);
  const bootstrap = p.bootstrap ?? true;
  if (typeof bootstrap !== 'boolean') fail('ML_PARAM', `bootstrap is true or false, not ${String(bootstrap)}`);
  const maxSamples = p.maxSamples ?? null;
  if (maxSamples !== null && !(isWhole(maxSamples, 1) || (typeof maxSamples === 'number' && maxSamples > 0 && maxSamples < 1))) {
    fail('ML_PARAM', `maxSamples is a whole number of at least 1, a share between 0 and 1, or null; not ${String(maxSamples)}`);
  }
  if (maxSamples !== null && !bootstrap) fail('ML_PARAM', 'maxSamples needs bootstrap: set bootstrap to true, or maxSamples to null');
  const oobScore = p.oobScore ?? false;
  if (typeof oobScore !== 'boolean') fail('ML_PARAM', `oobScore is true or false, not ${String(oobScore)}`);
  if (oobScore && !bootstrap) fail('ML_PARAM', 'oobScore needs bootstrap: out-of-bag rows are the rows a bootstrap sample left out');
  if (p.seed !== undefined && !Number.isInteger(p.seed)) fail('ML_PARAM', `seed is a whole number, not ${p.seed}`);
  return { nEstimators, bootstrap, maxSamples, oobScore, seed: p.seed };
}

/** The trees' own parameters, from a forest's. */
function treeParams(p: ForestParams): DecisionTreeParams {
  const out: DecisionTreeParams = {};
  for (const k of ['maxDepth', 'minSamplesSplit', 'minSamplesLeaf', 'minWeightFractionLeaf', 'maxFeatures', 'maxLeafNodes', 'minImpurityDecrease', 'ccpAlpha'] as const) {
    if (p[k] !== undefined) (out as Record<string, unknown>)[k] = p[k];
  }
  return out;
}

/** Each row's weight, checked as a tree checks it (finite, 0 or more), or `null` where none was given. */
function rowWeights(sampleWeight: VectorInput | undefined, n: number): Float64Array | null {
  if (sampleWeight === undefined) return null;
  const w = Float64Array.from(toVector(sampleWeight, n));
  for (let i = 0; i < n; i++) if (!Number.isFinite(w[i]!) || w[i]! < 0) fail('ML_VALUE', `sampleWeight must be finite and 0 or more; row ${i} has ${w[i]}`);
  return w;
}

/** A grown tree, with the rows its bootstrap left out (for the out-of-bag score). */
interface Grown<T> {
  tree: T;
  outOfBag: Int32Array | null;
}

/**
 * Grows the trees, each with a seed for its column draws. With bootstrap, each tree's rows are a sample drawn with
 * replacement, each row with a chance in proportion to its weight (sampleWeight times its class's weight), and the tree
 * weighs each row by the times it was drawn; without, each tree weighs the rows by their weight. Both as scikit-learn
 * 1.9 does. `weights` is `null` where every row weighs 1.
 */
function growForest<T>(c: ReturnType<typeof checkForest>, n: number, weights: Float64Array | null, grow: (seed: number, w: Float64Array | undefined) => T): { trees: Grown<T>[]; seed: number } {
  let total = n;
  if (weights) {
    total = 0;
    for (const v of weights) total += v;
    if (!(total > 0)) fail('ML_VALUE', 'every row weighs 0 (sampleWeight, times classWeight): at least one row needs weight');
  }
  const seed = c.seed ?? Math.floor(Math.random() * 2 ** 31);
  const r = new Random(seed);
  // A whole maxSamples is that many draws; a share is of the weights' sum (the rows, where they weigh 1), at least 1.
  const draws = c.maxSamples === null ? n : Number.isInteger(c.maxSamples) ? c.maxSamples : Math.max(1, Math.floor(c.maxSamples * total));
  // The weights summed row by row: a draw is the first row whose running sum passes a uniform number below the total.
  let cumulative: Float64Array | null = null;
  if (c.bootstrap && weights) {
    cumulative = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) cumulative[i] = sum += weights[i]!;
  }
  const trees: Grown<T>[] = [];
  for (let t = 0; t < c.nEstimators; t++) {
    const treeSeed = r.uint32() >>> 1;
    if (!c.bootstrap) {
      trees.push({ tree: grow(treeSeed, weights ?? undefined), outOfBag: null });
      continue;
    }
    const counts = new Float64Array(n);
    if (cumulative) {
      const end = cumulative[n - 1]!;
      for (let d = 0; d < draws; d++) {
        const u = r.next() * end;
        let lo = 0;
        let hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (cumulative[mid]! > u) hi = mid;
          else lo = mid + 1;
        }
        counts[lo]! += 1;
      }
    } else for (let d = 0; d < draws; d++) counts[r.int(n)]! += 1;
    const out: number[] = [];
    for (let i = 0; i < n; i++) if (counts[i] === 0) out.push(i);
    trees.push({ tree: grow(treeSeed, counts), outOfBag: Int32Array.from(out) });
  }
  return { trees, seed };
}

/** The rows `rows` of X, as a matrix. */
function pickRows(m: Matrix, rows: Int32Array): Matrix {
  const n = m.rows;
  const k = rows.length;
  const data = new Float64Array(k * m.cols);
  for (let j = 0; j < m.cols; j++) for (let r = 0; r < k; r++) data[j * k + r] = m.data[j * n + rows[r]!]!;
  return new Matrix(k, m.cols, data, m.names);
}

const OOB_FEW = 'some rows were drawn by every tree, so no tree could score them out of bag: their out-of-bag values are 0, and they count in oobScore as such. Use more trees (nEstimators)';

interface ForestState<T> {
  trees: T[];
  seed: number;
  nFeaturesIn: number;
  featureNamesIn?: readonly string[];
  oobScore?: number;
}

/** What both forests share: parameters, the trees, importances, saving. */
abstract class Forest<T extends DecisionTreeRegressor | DecisionTreeClassifier> {
  abstract readonly kind: string;
  protected readonly checked: ReturnType<typeof checkForest>;
  protected readonly trees: DecisionTreeParams;
  protected state: ForestState<T> | null = null;

  constructor(params: ForestParams) {
    this.checked = checkForest(params);
    this.trees = treeParams(params);
    // The trees check their own parameters: a forest with ones they refuse fails here, not at fit.
    void new DecisionTreeRegressor(this.trees);
  }

  get fitted(): boolean {
    return this.state !== null;
  }
  /** The fitted trees, in the order they were grown. scikit-learn's `estimators_`. */
  get estimators(): readonly T[] {
    return this.fittedState().trees;
  }
  get nEstimators(): number {
    return this.checked.nEstimators;
  }
  get nFeaturesIn(): number {
    return this.fittedState().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.fittedState().featureNamesIn;
  }
  /** The seed the forest was grown with: the one given, or the one drawn. */
  get seed(): number {
    return this.fittedState().seed;
  }
  /** The trees' feature importances, averaged. scikit-learn's `feature_importances_`. */
  get featureImportances(): Float64Array {
    const s = this.fittedState();
    const out = new Float64Array(s.nFeaturesIn);
    for (const t of s.trees) {
      const imp = t.featureImportances;
      for (let j = 0; j < out.length; j++) out[j]! += imp[j]!;
    }
    for (let j = 0; j < out.length; j++) out[j]! /= s.trees.length;
    return out;
  }
  /** The out-of-bag score (R² or accuracy), with `oobScore: true`. scikit-learn's `oob_score_`. */
  get oobScore(): number {
    const s = this.fittedState();
    if (s.oobScore === undefined) fail('ML_PARAM', `${this.kind} was fitted without oobScore: true, so it has no out-of-bag score`);
    return s.oobScore;
  }

  protected forestParams(): ForestParams {
    const c = this.checked;
    return {
      nEstimators: c.nEstimators,
      bootstrap: c.bootstrap,
      ...(c.maxSamples !== null ? { maxSamples: c.maxSamples } : {}),
      ...(c.oobScore ? { oobScore: true } : {}),
      ...(c.seed !== undefined ? { seed: c.seed } : {}),
      ...this.trees,
    };
  }

  protected fittedState(): ForestState<T> {
    if (!this.state) return notFitted(this.kind);
    return this.state;
  }

  protected rows(X: MatrixInput): Matrix {
    const m = Matrix.from(X);
    checkColumns(m, this.fittedState().nFeaturesIn, this.kind);
    return m;
  }

  protected saved(): Record<string, unknown> | null {
    const s = this.state;
    if (!s) return null;
    return {
      seed: s.seed,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
      ...(s.oobScore !== undefined ? { oobScore: s.oobScore } : {}),
      trees: s.trees.map((t) => t.toJSON()),
    };
  }

  protected restoreBase(state: Record<string, unknown> | null | undefined, load: (doc: ModelDocument) => T): ForestState<T> | null {
    if (!state) return null;
    if (!Array.isArray(state.trees) || state.trees.length === 0) fail('ML_FORMAT', `${this.kind} needs its trees`);
    return {
      trees: (state.trees as ModelDocument[]).map(load),
      seed: Number(state.seed),
      nFeaturesIn: Number(state.nFeaturesIn),
      ...(Array.isArray(state.featureNamesIn) ? { featureNamesIn: state.featureNamesIn as string[] } : {}),
      ...(typeof state.oobScore === 'number' ? { oobScore: state.oobScore } : {}),
    };
  }
}

/**
 * A random forest of regression trees: each tree grown on a bootstrap sample of the rows and, at each split, a random
 * subset of the columns (`maxFeatures`, every column by default); predictions are the trees' mean. scikit-learn's
 * `RandomForestRegressor` (ADR 0039).
 */
export class RandomForestRegressor extends Forest<DecisionTreeRegressor> {
  readonly kind = 'RandomForestRegressor';
  readonly criterion: NonNullable<RandomForestRegressorParams['criterion']>;
  #oobPrediction: Float64Array | null = null;

  constructor(params: RandomForestRegressorParams = {}) {
    super(params);
    this.criterion = new DecisionTreeRegressor({ criterion: params.criterion }).criterion;
  }

  getParams(): RandomForestRegressorParams {
    return { criterion: this.criterion, ...this.forestParams() };
  }

  /** Each row's mean prediction by the trees that didn't draw it, with `oobScore: true`. scikit-learn's `oob_prediction_`. */
  get oobPrediction(): Float64Array {
    if (!this.#oobPrediction) return fail('ML_PARAM', 'RandomForestRegressor has no out-of-bag predictions: fit with oobScore: true (they are not saved)');
    return this.#oobPrediction;
  }

  async fit(X: MatrixInput, y: VectorInput, options: ForestFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: VectorInput, options: ForestFitOptions = {}): this {
    const m = Matrix.from(X);
    const target = Float64Array.from(toVector(y, m.rows));
    const weights = rowWeights(options.sampleWeight, m.rows);
    const params = { ...this.trees, criterion: this.criterion };
    const { trees, seed } = growForest(this.checked, m.rows, weights, (s, w) => new DecisionTreeRegressor({ ...params, seed: s }).fitSync(m, target, w ? { sampleWeight: w } : {}));
    let oobScore: number | undefined;
    this.#oobPrediction = null;
    if (this.checked.oobScore) {
      const sum = new Float64Array(m.rows);
      const count = new Float64Array(m.rows);
      for (const { tree, outOfBag } of trees) {
        if (!outOfBag || outOfBag.length === 0) continue;
        const p = tree.predictSync(pickRows(m, outOfBag));
        for (let k = 0; k < outOfBag.length; k++) {
          sum[outOfBag[k]!]! += p[k]!;
          count[outOfBag[k]!]! += 1;
        }
      }
      if (count.some((c) => c === 0)) warn('ML_OOB', OOB_FEW);
      const pred = Float64Array.from(sum, (s, i) => (count[i]! > 0 ? s / count[i]! : 0));
      this.#oobPrediction = pred;
      oobScore = r2(target, pred);
    }
    this.state = { trees: trees.map((t) => t.tree), seed, nFeaturesIn: m.cols, featureNamesIn: m.names, ...(oobScore !== undefined ? { oobScore } : {}) };
    return this;
  }

  async predict(X: MatrixInput): Promise<Float64Array> {
    return this.predictSync(X);
  }

  /** Each row's prediction: the trees' mean. */
  predictSync(X: MatrixInput): Float64Array {
    const s = this.fittedState();
    const m = this.rows(X);
    const out = new Float64Array(m.rows);
    for (const t of s.trees) {
      const p = t.predictSync(m);
      for (let i = 0; i < out.length; i++) out[i]! += p[i]!;
    }
    for (let i = 0; i < out.length; i++) out[i]! /= s.trees.length;
    return out;
  }

  toJSON(): ModelDocument {
    return document(this.kind, { ...this.getParams() }, this.saved());
  }

  static fromJSON(doc: ModelDocument): RandomForestRegressor {
    const d = checkDocument(doc, 'RandomForestRegressor');
    const f = new RandomForestRegressor(d.params as RandomForestRegressorParams);
    f.state = f.restoreBase(d.state, (t) => DecisionTreeRegressor.fromJSON(t));
    return f;
  }
}

/**
 * A random forest of classification trees: each tree grown on a bootstrap sample of the rows and, at each split, a
 * random subset of the columns (`maxFeatures`, √columns by default); class probabilities are the trees' mean, and
 * the prediction the most probable class. scikit-learn's `RandomForestClassifier` (ADR 0039).
 */
export class RandomForestClassifier extends Forest<DecisionTreeClassifier> {
  readonly kind = 'RandomForestClassifier';
  readonly criterion: NonNullable<RandomForestClassifierParams['criterion']>;
  readonly classWeight: RandomForestClassifierParams['classWeight'];
  #classes: Label[] = [];
  #oobDecision: Matrix | null = null;

  constructor(params: RandomForestClassifierParams = {}) {
    super({ maxFeatures: 'sqrt', ...params });
    if ((params.classWeight as unknown) === 'balanced_subsample') {
      fail('ML_PARAM', "classWeight 'balanced_subsample' isn't supported yet: use 'balanced', weights by label, or null");
    }
    const tree = new DecisionTreeClassifier({ criterion: params.criterion, classWeight: params.classWeight ?? null });
    this.criterion = tree.criterion;
    this.classWeight = tree.classWeight;
  }

  getParams(): RandomForestClassifierParams {
    return { criterion: this.criterion, ...(this.classWeight ? { classWeight: this.classWeight } : {}), ...this.forestParams() };
  }

  /** The labels seen in fit, sorted: the order of predictProba's columns. */
  get classes(): readonly Label[] {
    this.fittedState();
    return this.#classes;
  }

  /**
   * Each row's class probabilities by the trees that didn't draw it, with `oobScore: true`. scikit-learn's
   * `oob_decision_function_`.
   */
  get oobDecisionFunction(): Matrix {
    if (!this.#oobDecision) return fail('ML_PARAM', 'RandomForestClassifier has no out-of-bag probabilities: fit with oobScore: true (they are not saved)');
    return this.#oobDecision;
  }

  async fit(X: MatrixInput, y: Labels, options: ForestFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: Labels, options: ForestFitOptions = {}): this {
    const m = Matrix.from(X);
    const labels = toLabels(y, m.rows);
    const classes = uniqueLabels(labels);
    const weights = withClassWeights(rowWeights(options.sampleWeight, m.rows), labels, classes, this.classWeight ?? null);
    // The class weights are in the rows' weights (and so in the bootstrap's chances): the trees take none of their own.
    const params = { ...this.trees, criterion: this.criterion };
    const { trees, seed } = growForest(this.checked, m.rows, weights, (s, w) => new DecisionTreeClassifier({ ...params, seed: s }).fitSync(m, y, w ? { sampleWeight: w } : {}));
    const K = classes.length;
    let oobScore: number | undefined;
    this.#oobDecision = null;
    if (this.checked.oobScore) {
      const n = m.rows;
      const sum = new Float64Array(n * K);
      const count = new Float64Array(n);
      for (const { tree, outOfBag } of trees) {
        if (!outOfBag || outOfBag.length === 0) continue;
        const p = tree.predictProbaSync(pickRows(m, outOfBag));
        const k = outOfBag.length;
        for (let r = 0; r < k; r++) {
          const i = outOfBag[r]!;
          for (let c = 0; c < K; c++) sum[c * n + i]! += p.data[c * k + r]!;
          count[i]! += 1;
        }
      }
      if (count.some((c) => c === 0)) warn('ML_OOB', OOB_FEW);
      for (let i = 0; i < n; i++) if (count[i]! > 0) for (let c = 0; c < K; c++) sum[c * n + i]! /= count[i]!;
      this.#oobDecision = new Matrix(n, K, sum, classes.map(String));
      let right = 0;
      for (let i = 0; i < n; i++) {
        let best = 0;
        for (let c = 1; c < K; c++) if (sum[c * n + i]! > sum[best * n + i]!) best = c;
        if (classes[best] === labels[i]) right++;
      }
      oobScore = right / n;
    }
    this.#classes = classes;
    this.state = { trees: trees.map((t) => t.tree), seed, nFeaturesIn: m.cols, featureNamesIn: m.names, ...(oobScore !== undefined ? { oobScore } : {}) };
    return this;
  }

  async predict(X: MatrixInput): Promise<Label[]> {
    return this.predictSync(X);
  }

  /** Each row's most probable class, by the trees' mean probabilities (of equal ones, the first class). */
  predictSync(X: MatrixInput): Label[] {
    const p = this.predictProbaSync(X);
    const n = p.rows;
    const K = p.cols;
    return Array.from({ length: n }, (_, i) => {
      let best = 0;
      for (let c = 1; c < K; c++) if (p.data[c * n + i]! > p.data[best * n + i]!) best = c;
      return this.#classes[best]!;
    });
  }

  async predictProba(X: MatrixInput): Promise<Matrix> {
    return this.predictProbaSync(X);
  }

  /** Each row's class probabilities, the trees' mean: a column per class in the order of `classes`. */
  predictProbaSync(X: MatrixInput): Matrix {
    const s = this.fittedState();
    const m = this.rows(X);
    const n = m.rows;
    const K = this.#classes.length;
    const out = new Float64Array(n * K);
    for (const t of s.trees) {
      const p = t.predictProbaSync(m);
      for (let i = 0; i < out.length; i++) out[i]! += p.data[i]!;
    }
    for (let i = 0; i < out.length; i++) out[i]! /= s.trees.length;
    return new Matrix(n, K, out, this.#classes.map(String));
  }

  toJSON(): ModelDocument {
    const saved = this.saved();
    return document(this.kind, { ...this.getParams() }, saved && { ...saved, classes: [...this.#classes] });
  }

  static fromJSON(doc: ModelDocument): RandomForestClassifier {
    const d = checkDocument(doc, 'RandomForestClassifier');
    const f = new RandomForestClassifier(d.params as RandomForestClassifierParams);
    f.state = f.restoreBase(d.state, (t) => DecisionTreeClassifier.fromJSON(t));
    if (f.state) {
      const classes = (d.state as Record<string, unknown>).classes;
      if (!Array.isArray(classes)) fail('ML_FORMAT', 'RandomForestClassifier needs its classes');
      f.#classes = classes as Label[];
    }
    return f;
  }
}

/**
 * The rows' weights times their classes' weights, or `null` where both are none. `'balanced'` gives each class the
 * total weight over (classes × the class's weight), so every class weighs the same in all; with no sampleWeight, the
 * class's weight is its count of rows. As scikit-learn 1.9's forests do.
 */
function withClassWeights(weights: Float64Array | null, labels: readonly Label[], classes: readonly Label[], cw: RandomForestClassifierParams['classWeight']): Float64Array | null {
  if (!cw) return weights;
  const index = new Map(classes.map((c, k) => [c, k]));
  const K = classes.length;
  let perClass: Float64Array;
  if (cw === 'balanced') {
    const sums = new Float64Array(K);
    let total = 0;
    labels.forEach((l, i) => {
      const w = weights ? weights[i]! : 1;
      sums[index.get(l)!]! += w;
      total += w;
    });
    const empty = classes.find((_, k) => !(sums[k]! > 0));
    if (empty !== undefined) fail('ML_VALUE', `classWeight 'balanced': every row of class ${String(empty)} weighs 0, so its weight can't be found`);
    perClass = Float64Array.from(sums, (s) => total / (K * s));
  } else {
    for (const label of Object.keys(cw)) {
      if (!classes.some((c) => String(c) === label)) fail('ML_PARAM', `classWeight names ${label}, which isn't one of the classes in y (${classes.map(String).join(', ')})`);
    }
    perClass = Float64Array.from(classes, (c) => cw[String(c)] ?? 1);
  }
  return Float64Array.from(labels, (l, i) => (weights ? weights[i]! : 1) * perClass[index.get(l)!]!);
}

/** R² of `pred` against `y`: 1 − Σ(y − pred)² / Σ(y − ȳ)². */
function r2(y: Float64Array, pred: Float64Array): number {
  let mean = 0;
  for (const v of y) mean += v;
  mean /= y.length;
  let ss = 0;
  let tot = 0;
  for (let i = 0; i < y.length; i++) {
    ss += (y[i]! - pred[i]!) ** 2;
    tot += (y[i]! - mean) ** 2;
  }
  return tot === 0 ? (ss === 0 ? 1 : 0) : 1 - ss / tot;
}
