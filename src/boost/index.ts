/**
 * Gradient-boosted trees with XGBoost's semantics (ADR 0044): BoostedTreesRegressor, XGBoost's `XGBRegressor`.
 * @module
 */
import { fail } from '../core/errors.js';
import { assertFinite, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, type ModelDocument } from '../core/model.js';
import { binColumns, growTree, MAX_BIN_LIMIT, treeValue, type BoostTree } from './hist.js';

export type { BoostTree } from './hist.js';

/** BoostedTreesRegressor's parameters: XGBoost's scikit-learn interface's, in camelCase, with its defaults. */
export interface BoostedTreesRegressorParams {
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
  /** Every row's starting prediction. Default: fitted, the weighted mean of y. */
  baseScore?: number | null;
  /** The loss. Only `'reg:squarederror'` (the default) so far. */
  objective?: 'reg:squarederror';
  /** The share of rows each tree samples. Only 1 so far. */
  subsample?: number;
  /** The shares of columns each tree, level and node samples. Only 1 so far. */
  colsampleBytree?: number;
  colsampleBylevel?: number;
  colsampleBynode?: number;
  /** The seed for the sampling, when it comes. Default: none. */
  seed?: number;
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

function checkParams(p: BoostedTreesRegressorParams) {
  const c = {
    nEstimators: p.nEstimators ?? 100,
    learningRate: p.learningRate ?? 0.3,
    maxDepth: p.maxDepth ?? 6,
    minChildWeight: p.minChildWeight ?? 1,
    gamma: p.gamma ?? 0,
    regLambda: p.regLambda ?? 1,
    regAlpha: p.regAlpha ?? 0,
    maxBin: p.maxBin ?? 256,
    baseScore: p.baseScore ?? null,
    objective: p.objective ?? 'reg:squarederror',
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
  if (c.baseScore !== null && !Number.isFinite(c.baseScore)) fail('ML_PARAM', `baseScore is a finite number or null, not ${String(c.baseScore)}`);
  if (c.objective !== 'reg:squarederror') fail('ML_PARAM', `objective '${String(c.objective)}' isn't supported yet: only 'reg:squarederror'`);
  for (const k of ['subsample', 'colsampleBytree', 'colsampleBylevel', 'colsampleBynode'] as const) {
    if (c[k] !== 1) fail('ML_PARAM', `${k} other than 1 isn't supported yet: every tree sees every row and column`);
  }
  if (c.seed !== undefined && !Number.isInteger(c.seed)) fail('ML_PARAM', `seed is a whole number, not ${String(c.seed)}`);
  return c;
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
    const sum = new Float64Array(s.nFeaturesIn);
    const count = new Float64Array(s.nFeaturesIn);
    for (const t of s.trees) {
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
    const w = new Float64Array(n).fill(1);
    if (options.sampleWeight !== undefined) {
      const sw = toVector(options.sampleWeight, n, 'sampleWeight');
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(sw[i]!) || sw[i]! < 0) fail('ML_VALUE', `sampleWeight must be finite and 0 or more; row ${i} has ${sw[i]}`);
        w[i] = sw[i]!;
      }
    }
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
      trees: s.trees.map((t) => ({
        left: [...t.left],
        right: [...t.right],
        feature: [...t.feature],
        // A leaf's threshold is NaN, which JSON can't hold: saved as null.
        threshold: Array.from(t.threshold, (v) => (Number.isNaN(v) ? null : v)),
        value: numbers(t.value),
        gain: numbers(t.gain),
        cover: numbers(t.cover),
      })),
    });
  }

  static fromJSON(doc: ModelDocument): BoostedTreesRegressor {
    const d = checkDocument(doc, 'BoostedTreesRegressor');
    const model = new BoostedTreesRegressor(d.params as BoostedTreesRegressorParams);
    const s = d.state as Record<string, unknown> | null;
    if (s) {
      if (!Array.isArray(s.trees)) fail('ML_FORMAT', 'BoostedTreesRegressor needs its trees');
      const ints = (v: unknown, what: string) => Int32Array.from(floats(v, what));
      model.#state = {
        baseScore: Number(s.baseScore),
        nFeaturesIn: Number(s.nFeaturesIn),
        ...(Array.isArray(s.featureNamesIn) ? { featureNamesIn: s.featureNamesIn as string[] } : {}),
        trees: (s.trees as Record<string, unknown>[]).map((t) => ({
          left: ints(t.left, 'a tree\'s left'),
          right: ints(t.right, 'a tree\'s right'),
          feature: ints(t.feature, 'a tree\'s feature'),
          threshold: Float64Array.from(t.threshold as (number | null)[], (v) => (v === null ? Number.NaN : v)),
          value: floats(t.value, 'a tree\'s value'),
          gain: floats(t.gain, 'a tree\'s gain'),
          cover: floats(t.cover, 'a tree\'s cover'),
        })),
      };
    }
    return model;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}
