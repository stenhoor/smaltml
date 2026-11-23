/**
 * Gradient-boosted trees with XGBoost's semantics (ADR 0044): BoostedTreesRegressor and BoostedTreesClassifier,
 * XGBoost's `XGBRegressor` and `XGBClassifier`.
 * @module
 */
import { fail } from '../core/errors.js';
import { assertFinite, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, toLabels, uniqueLabels, type Label, type Labels, type ModelDocument } from '../core/model.js';
import { binColumns, growTree, MAX_BIN_LIMIT, treeValue, type BoostTree } from './hist.js';

export type { BoostTree } from './hist.js';

/** The parameters both boosted models take: XGBoost's scikit-learn interface's, in camelCase, with its defaults. */
export interface BoostedTreesParams {
  /** How many boosting rounds: one tree each. Default 100. */
  nEstimators?: number;
  /** Each tree's leaf values are multiplied by this (XGBoost's `eta`). Default 0.3. */
  learningRate?: number;
  /** The deepest a leaf may be (the root is depth 0); trees grow a level at a time. Default 6. */
  maxDepth?: number;
  /** The smallest Hessian sum (for squared error, the weight) each child keeps. Default 1. */
  minChildWeight?: number;
  /** The smallest gain a split must reach (XGBoost's `min_split_loss`). Default 0. */
  gamma?: number;
  /** L2 regularization of the leaf values (XGBoost's `lambda`). Default 1. */
  regLambda?: number;
  /** L1 regularization of the leaf values (XGBoost's `alpha`). Default 0. */
  regAlpha?: number;
  /** The most bins a column is cut into. A column with more distinct values isn't supported yet. Default 256. */
  maxBin?: number;
  /** The share of rows each tree samples. Only 1 so far. */
  subsample?: number;
  /** The shares of columns each tree, level and node samples. Only 1 so far. */
  colsampleBytree?: number;
  colsampleBylevel?: number;
  colsampleBynode?: number;
  /** The seed for the sampling, when it comes. Default: none. */
  seed?: number;
}

/** BoostedTreesRegressor's parameters: XGBoost's `XGBRegressor`'s, in camelCase, with its defaults. */
export interface BoostedTreesRegressorParams extends BoostedTreesParams {
  /** Every row's starting prediction. Default: fitted, the weighted mean of y. */
  baseScore?: number | null;
  /** The loss. Only `'reg:squarederror'` (the default) so far. */
  objective?: 'reg:squarederror';
}

/** BoostedTreesClassifier's parameters: XGBoost's `XGBClassifier`'s, in camelCase, with its defaults. */
export interface BoostedTreesClassifierParams extends BoostedTreesParams {
  /**
   * Two classes only: the positive class's starting probability, between 0 and 1. Default: fitted, the positive
   * class's weighted share. With more classes, each class's starting value is fitted (a given one isn't supported yet).
   */
  baseScore?: number | null;
  /**
   * The loss: `'binary:logistic'` (the default), which more than two classes turn into `'multi:softprob'`, as XGBoost's
   * does; `'multi:softprob'` needs more than two classes.
   */
  objective?: 'binary:logistic' | 'multi:softprob';
}

export interface BoostFitOptions {
  /** One weight per row, 0 or more: a row's gradient and Hessian are multiplied by it. */
  sampleWeight?: VectorInput;
}

interface State {
  trees: BoostTree[];
  baseScore: number;
  nFeaturesIn: number;
  featureNamesIn?: readonly string[];
}

const isWhole = (v: unknown, least: number) => typeof v === 'number' && Number.isInteger(v) && v >= least;
const atLeast = (v: unknown, least: number) => typeof v === 'number' && Number.isFinite(v) && v >= least;

function checkCommon(p: BoostedTreesParams) {
  const c = {
    nEstimators: p.nEstimators ?? 100,
    learningRate: p.learningRate ?? 0.3,
    maxDepth: p.maxDepth ?? 6,
    minChildWeight: p.minChildWeight ?? 1,
    gamma: p.gamma ?? 0,
    regLambda: p.regLambda ?? 1,
    regAlpha: p.regAlpha ?? 0,
    maxBin: p.maxBin ?? 256,
    subsample: p.subsample ?? 1,
    colsampleBytree: p.colsampleBytree ?? 1,
    colsampleBylevel: p.colsampleBylevel ?? 1,
    colsampleBynode: p.colsampleBynode ?? 1,
    seed: p.seed,
  };
  if (!isWhole(c.nEstimators, 1)) fail('ML_PARAM', `nEstimators is a whole number, 1 or more, not ${String(c.nEstimators)}`);
  if (!atLeast(c.learningRate, 0)) fail('ML_PARAM', `learningRate is 0 or more, not ${String(c.learningRate)}`);
  if (!isWhole(c.maxDepth, 1)) fail('ML_PARAM', `maxDepth is a whole number, 1 or more, not ${String(c.maxDepth)}`);
  for (const k of ['minChildWeight', 'gamma', 'regLambda', 'regAlpha'] as const) {
    if (!atLeast(c[k], 0)) fail('ML_PARAM', `${k} is 0 or more, not ${String(c[k])}`);
  }
  if (!isWhole(c.maxBin, 2) || c.maxBin > MAX_BIN_LIMIT) fail('ML_PARAM', `maxBin is a whole number from 2 to ${MAX_BIN_LIMIT}, not ${String(c.maxBin)}`);
  for (const k of ['subsample', 'colsampleBytree', 'colsampleBylevel', 'colsampleBynode'] as const) {
    if (c[k] !== 1) fail('ML_PARAM', `${k} other than 1 isn't supported yet: every tree sees every row and column`);
  }
  if (c.seed !== undefined && !Number.isInteger(c.seed)) fail('ML_PARAM', `seed is a whole number, not ${String(c.seed)}`);
  return c;
}

function checkParams(p: BoostedTreesRegressorParams) {
  const c = { ...checkCommon(p), baseScore: p.baseScore ?? null, objective: p.objective ?? 'reg:squarederror' };
  if (c.baseScore !== null && !Number.isFinite(c.baseScore)) fail('ML_PARAM', `baseScore is a finite number or null, not ${String(c.baseScore)}`);
  if (c.objective !== 'reg:squarederror') fail('ML_PARAM', `objective '${String(c.objective)}' isn't supported yet: only 'reg:squarederror'`);
  return c;
}

function checkClassifierParams(p: BoostedTreesClassifierParams) {
  const c = { ...checkCommon(p), baseScore: p.baseScore ?? null, objective: p.objective ?? 'binary:logistic' };
  if (c.baseScore !== null && !(typeof c.baseScore === 'number' && c.baseScore > 0 && c.baseScore < 1)) {
    fail('ML_PARAM', `baseScore is a probability between 0 and 1 (not either), or null, not ${String(c.baseScore)}`);
  }
  if (c.objective !== 'binary:logistic' && c.objective !== 'multi:softprob') {
    fail('ML_PARAM', `objective is 'binary:logistic' or 'multi:softprob', not '${String(c.objective)}'`);
  }
  return c;
}

/** One weight per row: sampleWeight, checked, or 1s. */
function rowWeights(sampleWeight: VectorInput | undefined, n: number): Float64Array {
  const w = new Float64Array(n).fill(1);
  if (sampleWeight !== undefined) {
    const sw = toVector(sampleWeight, n, 'sampleWeight');
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(sw[i]!) || sw[i]! < 0) fail('ML_VALUE', `sampleWeight must be finite and 0 or more; row ${i} has ${sw[i]}`);
      w[i] = sw[i]!;
    }
  }
  return w;
}

/**
 * Each column's mean gain over the splits on it in every tree, normalized to sum to 1: XGBoost's `feature_importances_`
 * (its importance type `gain`). All 0 where no tree splits.
 */
function gainImportances(trees: readonly BoostTree[], p: number): Float64Array {
  const sum = new Float64Array(p);
  const count = new Float64Array(p);
  for (const t of trees) {
    for (let k = 0; k < t.left.length; k++) {
      if (t.left[k] === -1) continue;
      sum[t.feature[k]!]! += t.gain[k]!;
      count[t.feature[k]!]! += 1;
    }
  }
  const mean = Float64Array.from(sum, (v, j) => (count[j]! > 0 ? v / count[j]! : 0));
  let total = 0;
  for (const v of mean) total += v;
  if (total > 0) for (let j = 0; j < mean.length; j++) mean[j]! /= total;
  return mean;
}

function treesToJSON(trees: readonly BoostTree[]) {
  return trees.map((t) => ({
    left: [...t.left],
    right: [...t.right],
    feature: [...t.feature],
    // A leaf's threshold is NaN, which JSON can't hold: saved as null.
    threshold: Array.from(t.threshold, (v) => (Number.isNaN(v) ? null : v)),
    value: numbers(t.value),
    gain: numbers(t.gain),
    cover: numbers(t.cover),
  }));
}

function treesFromJSON(trees: unknown, kind: string): BoostTree[] {
  if (!Array.isArray(trees)) fail('ML_FORMAT', `${kind} needs its trees`);
  const ints = (v: unknown, what: string) => Int32Array.from(floats(v, what));
  return (trees as Record<string, unknown>[]).map((t) => ({
    left: ints(t.left, 'a tree\'s left'),
    right: ints(t.right, 'a tree\'s right'),
    feature: ints(t.feature, 'a tree\'s feature'),
    threshold: Float64Array.from(t.threshold as (number | null)[], (v) => (v === null ? Number.NaN : v)),
    value: floats(t.value, 'a tree\'s value'),
    gain: floats(t.gain, 'a tree\'s gain'),
    cover: floats(t.cover, 'a tree\'s cover'),
  }));
}

/**
 * X as the trees read it, checked: XGBoost reads X as 32-bit floats and cuts between those values, so smaltml does too.
 * Missing values (NaN) aren't supported yet.
 */
function prepareX(X: MatrixInput): Matrix {
  const m = Matrix.from(X);
  for (let k = 0; k < m.data.length; k++) {
    if (Number.isNaN(m.data[k]!)) fail('ML_VALUE', `X column ${m.columnName(Math.floor(k / m.rows))} has a missing value (NaN): boosted trees don't take missing values yet; fill them first (SimpleImputer)`);
  }
  assertFinite(m);
  const data = new Float64Array(m.data.length);
  for (let k = 0; k < data.length; k++) data[k] = Math.fround(m.data[k]!);
  return new Matrix(m.rows, m.cols, data, m.names);
}

/**
 * Gradient-boosted regression trees, as XGBoost's `XGBRegressor` grows them (ADR 0044): each round, a tree fitted to
 * the loss's gradients and Hessians by XGBoost's histogram method, its leaf values times the learning rate added to the
 * predictions. Squared error so far.
 */
export class BoostedTreesRegressor {
  readonly kind = 'BoostedTreesRegressor';
  readonly #checked: ReturnType<typeof checkParams>;
  #state: State | null = null;

  constructor(params: BoostedTreesRegressorParams = {}) {
    this.#checked = checkParams(params);
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  getParams(): BoostedTreesRegressorParams {
    const c = this.#checked;
    const { seed, ...rest } = c;
    return { ...rest, ...(seed !== undefined ? { seed } : {}) };
  }

  /** The trees, one per round, in order. */
  get trees(): readonly BoostTree[] {
    return this.#fitted().trees;
  }
  /** Every row's starting prediction: the one given, or the weighted mean of y. XGBoost's `base_score`. */
  get baseScore(): number {
    return this.#fitted().baseScore;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  /**
   * Each column's mean gain over the splits on it, normalized to sum to 1: XGBoost's `feature_importances_` (its
   * importance type `gain`). All 0 where no tree splits.
   */
  get featureImportances(): Float64Array {
    const s = this.#fitted();
    return gainImportances(s.trees, s.nFeaturesIn);
  }

  async fit(X: MatrixInput, y: VectorInput, options: BoostFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: VectorInput, options: BoostFitOptions = {}): this {
    const c = this.#checked;
    const m = prepareX(X);
    const n = m.rows;
    if (n === 0) fail('ML_SHAPE', 'X has no rows');
    const target = toVector(y, n);
    for (let i = 0; i < n; i++) if (!Number.isFinite(target[i]!)) fail('ML_VALUE', `y[${i}] is ${target[i]}`);
    const w = rowWeights(options.sampleWeight, n);
    let wsum = 0;
    let wy = 0;
    for (let i = 0; i < n; i++) {
      wsum += w[i]!;
      wy += w[i]! * target[i]!;
    }
    if (!(wsum > 0)) fail('ML_VALUE', 'sampleWeight is 0 for every row: at least one row needs weight');
    // XGBoost's base score for squared error, fitted from the data: the weighted mean of y.
    const baseScore = c.baseScore ?? wy / wsum;
    const binned = binColumns(m, c.maxBin);
    const pred = new Float64Array(n).fill(baseScore);
    const g = new Float64Array(n);
    const h = new Float64Array(n);
    const trees: BoostTree[] = [];
    for (let round = 0; round < c.nEstimators; round++) {
      // Squared error, ½(prediction − y)²: gradient prediction − y, Hessian 1, each times the row's weight.
      for (let i = 0; i < n; i++) {
        g[i] = w[i]! * (pred[i]! - target[i]!);
        h[i] = w[i]!;
      }
      trees.push(growTree(binned, g, h, c, pred));
    }
    this.#state = { trees, baseScore, nFeaturesIn: m.cols, ...(m.names ? { featureNamesIn: m.names } : {}) };
    return this;
  }

  async predict(X: MatrixInput): Promise<Float64Array> {
    return this.predictSync(X);
  }

  /** Each row's prediction: the base score plus the value of the leaf it falls in, in each tree. */
  predictSync(X: MatrixInput): Float64Array {
    const s = this.#fitted();
    const m = prepareX(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = s.baseScore;
      for (const t of s.trees) v += treeValue(t, m.data, n, i);
      out[i] = v;
    }
    return out;
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      baseScore: s.baseScore,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
      trees: treesToJSON(s.trees),
    });
  }

  static fromJSON(doc: ModelDocument): BoostedTreesRegressor {
    const d = checkDocument(doc, 'BoostedTreesRegressor');
    const model = new BoostedTreesRegressor(d.params as BoostedTreesRegressorParams);
    const s = d.state as Record<string, unknown> | null;
    if (s) {
      model.#state = {
        baseScore: Number(s.baseScore),
        nFeaturesIn: Number(s.nFeaturesIn),
        ...(Array.isArray(s.featureNamesIn) ? { featureNamesIn: s.featureNamesIn as string[] } : {}),
        trees: treesFromJSON(s.trees, 'BoostedTreesRegressor'),
      };
    }
    return model;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}

interface ClassifierState {
  /** The trees, round by round; with more than two classes, one per class in each round. */
  trees: BoostTree[];
  /** Two classes: the positive class's starting probability. More: each class's starting margin. */
  baseScore: Float64Array;
  classes: Label[];
  objective: 'binary:logistic' | 'multi:softprob';
  nFeaturesIn: number;
  featureNamesIn?: readonly string[];
}

/** A probability's log odds: the margin a binary model starts from. */
const logit = (p: number) => Math.log(p / (1 - p));

/**
 * Gradient-boosted classification trees, as XGBoost's `XGBClassifier` grows them (ADR 0044). Two classes:
 * `binary:logistic`, one tree a round on the log odds of the second class. More: `multi:softprob`, a tree per class a
 * round on that class's margin, the probabilities their softmax.
 */
export class BoostedTreesClassifier {
  readonly kind = 'BoostedTreesClassifier';
  readonly #checked: ReturnType<typeof checkClassifierParams>;
  #state: ClassifierState | null = null;

  constructor(params: BoostedTreesClassifierParams = {}) {
    this.#checked = checkClassifierParams(params);
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  getParams(): BoostedTreesClassifierParams {
    const c = this.#checked;
    const { seed, ...rest } = c;
    return { ...rest, ...(seed !== undefined ? { seed } : {}) };
  }

  /** The labels seen in fit, sorted: the order of predictProba's columns. */
  get classes(): readonly Label[] {
    return this.#fitted().classes;
  }
  /** The loss the fit used: `'binary:logistic'` for two classes, `'multi:softprob'` for more. */
  get objective(): 'binary:logistic' | 'multi:softprob' {
    return this.#fitted().objective;
  }
  /**
   * The trees, round by round. With more than two classes each round has a tree per class, in the order of `classes`:
   * tree r · K + k is round r's for class k.
   */
  get trees(): readonly BoostTree[] {
    return this.#fitted().trees;
  }
  /**
   * XGBoost's `base_score`. Two classes: the second class's starting probability, the one given or its weighted share
   * (one value). More: each class's starting margin, log of its weighted share less the mean of those logs.
   */
  get baseScore(): Float64Array {
    return this.#fitted().baseScore.slice();
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  /** Each column's mean gain over the splits on it in every tree, normalized to sum to 1: XGBoost's `feature_importances_`. */
  get featureImportances(): Float64Array {
    const s = this.#fitted();
    return gainImportances(s.trees, s.nFeaturesIn);
  }

  async fit(X: MatrixInput, y: Labels, options: BoostFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: Labels, options: BoostFitOptions = {}): this {
    const c = this.#checked;
    const m = prepareX(X);
    const n = m.rows;
    if (n === 0) fail('ML_SHAPE', 'X has no rows');
    const labels = toLabels(y, n);
    const classes = uniqueLabels(labels);
    if (classes.length < 2) fail('ML_LABELS', `y has one class only (${String(classes[0])}): a classifier needs at least two`);
    const K = classes.length;
    const index = new Map(classes.map((l, k) => [l, k]));
    const target = Int32Array.from(labels, (l) => index.get(l)!);
    // XGBoost's XGBClassifier: binary:logistic (its default) becomes multi:softprob with more than two classes, and
    // multi:softprob with two is an error.
    if (K === 2 && c.objective === 'multi:softprob') fail('ML_PARAM', "objective 'multi:softprob' needs more than two classes: two use 'binary:logistic'");
    const objective = K === 2 ? 'binary:logistic' : 'multi:softprob';
    if (K > 2 && c.baseScore !== null) fail('ML_PARAM', 'baseScore given with more than two classes isn\'t supported yet: each class\'s is fitted');
    const w = rowWeights(options.sampleWeight, n);
    const share = new Float64Array(K);
    let wsum = 0;
    for (let i = 0; i < n; i++) {
      share[target[i]!]! += w[i]!;
      wsum += w[i]!;
    }
    if (!(wsum > 0)) fail('ML_VALUE', 'sampleWeight is 0 for every row: at least one row needs weight');
    for (let k = 0; k < K; k++) {
      if (!(share[k]! > 0)) fail('ML_VALUE', `class ${String(classes[k])} has no weight (its rows' sampleWeight is 0): such a class isn't supported yet`);
      share[k]! /= wsum;
    }
    const binned = binColumns(m, c.maxBin);
    const g = new Float64Array(n);
    const h = new Float64Array(n);
    const trees: BoostTree[] = [];
    let baseScore: Float64Array;
    if (K === 2) {
      // XGBoost's base score for binary:logistic, fitted from the data: the second class's weighted share, as a
      // probability; the trees start from its log odds.
      const base = c.baseScore ?? share[1]!;
      baseScore = Float64Array.of(base);
      const margin = new Float64Array(n).fill(logit(base));
      for (let round = 0; round < c.nEstimators; round++) {
        // Log loss on the margin: gradient p − y, Hessian p(1 − p), each times the row's weight.
        for (let i = 0; i < n; i++) {
          const p = 1 / (1 + Math.exp(-margin[i]!));
          g[i] = w[i]! * (p - target[i]!);
          h[i] = w[i]! * p * (1 - p);
        }
        trees.push(growTree(binned, g, h, c, margin));
      }
    } else {
      // XGBoost's base scores for multi:softprob: each class's log share, less their mean (so they sum to 0).
      const logs = Float64Array.from(share, Math.log);
      let mean = 0;
      for (const v of logs) mean += v;
      mean /= K;
      baseScore = Float64Array.from(logs, (v) => v - mean);
      const margins = Array.from({ length: K }, (_, k) => new Float64Array(n).fill(baseScore[k]!));
      const prob = Array.from({ length: K }, () => new Float64Array(n));
      for (let round = 0; round < c.nEstimators; round++) {
        // Every class's probabilities from the round's starting margins, before any of its trees grows.
        softmax(margins, prob, n);
        for (let k = 0; k < K; k++) {
          // Softmax log loss on class k's margin: gradient p − [y = k], Hessian 2p(1 − p) (XGBoost's, read from its
          // trees' covers), each times the row's weight.
          const pk = prob[k]!;
          for (let i = 0; i < n; i++) {
            const p = pk[i]!;
            g[i] = w[i]! * (p - (target[i] === k ? 1 : 0));
            h[i] = w[i]! * 2 * p * (1 - p);
          }
          trees.push(growTree(binned, g, h, c, margins[k]!));
        }
      }
    }
    this.#state = { trees, baseScore, classes, objective, nFeaturesIn: m.cols, ...(m.names ? { featureNamesIn: m.names } : {}) };
    return this;
  }

  async predict(X: MatrixInput): Promise<Label[]> {
    return this.predictSync(X);
  }

  /** Each row's most likely class (of equal probabilities, the first class). */
  predictSync(X: MatrixInput): Label[] {
    const s = this.#fitted();
    const P = this.predictProbaSync(X);
    const n = P.rows;
    const K = s.classes.length;
    const out: Label[] = new Array(n);
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let k = 1; k < K; k++) if (P.data[k * n + i]! > P.data[best * n + i]!) best = k;
      out[i] = s.classes[best]!;
    }
    return out;
  }

  async predictProba(X: MatrixInput): Promise<Matrix> {
    return this.predictProbaSync(X);
  }

  /** Each class's probability, a column per class in the order of `classes`: the margins' sigmoid, or softmax. */
  predictProbaSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = prepareX(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const K = s.classes.length;
    const out = new Float64Array(n * K);
    if (K === 2) {
      const start = logit(s.baseScore[0]!);
      for (let i = 0; i < n; i++) {
        let v = start;
        for (const t of s.trees) v += treeValue(t, m.data, n, i);
        const p = 1 / (1 + Math.exp(-v));
        out[i] = 1 - p;
        out[n + i] = p;
      }
    } else {
      const margins = Array.from({ length: K }, (_, k) => new Float64Array(n).fill(s.baseScore[k]!));
      for (let r = 0; r < s.trees.length; r++) {
        const t = s.trees[r]!;
        const mk = margins[r % K]!;
        for (let i = 0; i < n; i++) mk[i]! += treeValue(t, m.data, n, i);
      }
      const prob = Array.from({ length: K }, (_, k) => out.subarray(k * n, (k + 1) * n));
      softmax(margins, prob, n);
    }
    return new Matrix(n, K, out, s.classes.map(String));
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      baseScore: numbers(s.baseScore),
      classes: [...s.classes],
      objective: s.objective,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
      trees: treesToJSON(s.trees),
    });
  }

  static fromJSON(doc: ModelDocument): BoostedTreesClassifier {
    const d = checkDocument(doc, 'BoostedTreesClassifier');
    const model = new BoostedTreesClassifier(d.params as BoostedTreesClassifierParams);
    const s = d.state as Record<string, unknown> | null;
    if (s) {
      if (!Array.isArray(s.classes) || s.classes.length < 2) fail('ML_FORMAT', 'BoostedTreesClassifier needs its classes');
      model.#state = {
        baseScore: floats(s.baseScore, 'the base score'),
        classes: s.classes as Label[],
        objective: s.objective === 'multi:softprob' ? 'multi:softprob' : 'binary:logistic',
        nFeaturesIn: Number(s.nFeaturesIn),
        ...(Array.isArray(s.featureNamesIn) ? { featureNamesIn: s.featureNamesIn as string[] } : {}),
        trees: treesFromJSON(s.trees, 'BoostedTreesClassifier'),
      };
    }
    return model;
  }

  #fitted(): ClassifierState {
    return this.#state ?? notFitted(this.kind);
  }
}

/** prob[k][i] = the softmax of margins[·][i] at class k, less the largest margin first so exp can't overflow. */
function softmax(margins: readonly Float64Array[], prob: readonly Float64Array[], n: number): void {
  const K = margins.length;
  for (let i = 0; i < n; i++) {
    let top = -Infinity;
    for (let k = 0; k < K; k++) if (margins[k]![i]! > top) top = margins[k]![i]!;
    let sum = 0;
    for (let k = 0; k < K; k++) {
      const e = Math.exp(margins[k]![i]! - top);
      prob[k]![i] = e;
      sum += e;
    }
    for (let k = 0; k < K; k++) prob[k]![i]! /= sum;
  }
}
