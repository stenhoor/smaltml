/**
 * Decision trees: CART for classification and regression (ADR 0039). scikit-learn's `tree`.
 * @module
 */
import { fail } from '../core/errors.js';
import { assertFinite, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import {
  checkColumns,
  checkDocument,
  document,
  notFitted,
  numbers,
  toLabels,
  uniqueLabels,
  type Label,
  type Labels,
  type ModelDocument,
} from '../core/model.js';
import { applyTree, depthOf, featureImportances, grow, prepareX, type Criterion, type Targets, type TreeArrays } from './cart.js';

export { LEAF, UNDEFINED, type TreeArrays } from './cart.js';

/** The parameters both trees take, as scikit-learn's, in camelCase. */
export interface DecisionTreeParams {
  /** The deepest a leaf may be (the root is depth 0). Default: no limit. */
  maxDepth?: number | null;
  /** The fewest rows a node needs to be split: a whole number, 2 or more, or a share of the rows (above 0, at most 1). Default 2. */
  minSamplesSplit?: number;
  /** The fewest rows each leaf keeps: a whole number, 1 or more, or a share of the rows (above 0, below 1). Default 1. */
  minSamplesLeaf?: number;
  /** The smallest share of the total weight each leaf keeps, 0 to 0.5. Default 0. */
  minWeightFractionLeaf?: number;
  /**
   * How many columns each split looks at: a whole number, a share of the columns (above 0, below 1), `'sqrt'`,
   * `'log2'`, or every column (`null`, the default). Which columns, and in which order, is drawn from `seed`.
   */
  maxFeatures?: number | 'sqrt' | 'log2' | null;
  /** The most leaves: grown best first (the split that improves most next). Default: no limit, grown depth first. */
  maxLeafNodes?: number | null;
  /** A split must decrease the weighted impurity at least this much. Default 0. */
  minImpurityDecrease?: number;
  /** Cost-complexity pruning. Only 0 (no pruning) is supported so far. */
  ccpAlpha?: number;
  /**
   * How each split is chosen: `'best'` (the default), the best threshold of each column looked at; or `'random'`, one
   * threshold per column drawn uniformly between the node's smallest and largest value, the best of those kept (an
   * extra tree, as ExtraTreesRegressor and ExtraTreesClassifier grow them).
   */
  splitter?: 'best' | 'random';
  /**
   * The seed for the order the columns are looked at, which breaks ties between equal splits, and for the thresholds
   * of `splitter: 'random'`. Default: a new one.
   */
  seed?: number;
}

export interface DecisionTreeRegressorParams extends DecisionTreeParams {
  /** `'squared_error'` (the default) or `'friedman_mse'`. `'absolute_error'` and `'poisson'` aren't supported yet. */
  criterion?: 'squared_error' | 'friedman_mse' | 'absolute_error' | 'poisson';
}

export interface DecisionTreeClassifierParams extends DecisionTreeParams {
  /** `'gini'` (the default), or `'entropy'` and `'log_loss'` (the same: entropy in bits). */
  criterion?: 'gini' | 'entropy' | 'log_loss';
  /** Weights for the classes: `'balanced'` (n / (classes × the class's rows)) or an object by label. */
  classWeight?: 'balanced' | Record<string, number> | null;
}

export interface TreeFitOptions {
  /** One weight per row, 0 or more; a row with weight 2 counts as that row twice. */
  sampleWeight?: VectorInput;
}

interface State {
  tree: TreeArrays;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
  /** The classifier's labels, sorted. */
  classes?: Label[];
  /** The seed the tree was grown with: the one given, or the one drawn. */
  seed: number;
}

const isWhole = (v: number, least: number) => Number.isInteger(v) && v >= least;

/** The parameters, checked as scikit-learn checks them. */
function checkParams(p: DecisionTreeParams) {
  const maxDepth = p.maxDepth ?? null;
  if (maxDepth !== null && !isWhole(maxDepth, 1)) fail('ML_PARAM', `maxDepth is a whole number, 1 or more, or null; not ${maxDepth}`);
  const minSamplesSplit = p.minSamplesSplit ?? 2;
  if (!(isWhole(minSamplesSplit, 2) || (minSamplesSplit > 0 && minSamplesSplit <= 1))) {
    fail('ML_PARAM', `minSamplesSplit is a whole number, 2 or more, or a share above 0 and at most 1; not ${minSamplesSplit}`);
  }
  const minSamplesLeaf = p.minSamplesLeaf ?? 1;
  if (!(isWhole(minSamplesLeaf, 1) || (minSamplesLeaf > 0 && minSamplesLeaf < 1))) {
    fail('ML_PARAM', `minSamplesLeaf is a whole number, 1 or more, or a share between 0 and 1; not ${minSamplesLeaf}`);
  }
  const minWeightFractionLeaf = p.minWeightFractionLeaf ?? 0;
  if (!(minWeightFractionLeaf >= 0 && minWeightFractionLeaf <= 0.5)) fail('ML_PARAM', `minWeightFractionLeaf is from 0 to 0.5, not ${minWeightFractionLeaf}`);
  const maxFeatures = p.maxFeatures ?? null;
  if (maxFeatures !== null && maxFeatures !== 'sqrt' && maxFeatures !== 'log2' && !(isWhole(maxFeatures as number, 1) || ((maxFeatures as number) > 0 && (maxFeatures as number) < 1))) {
    fail('ML_PARAM', `maxFeatures is a whole number, a share between 0 and 1, 'sqrt', 'log2' or null; not ${String(maxFeatures)}`);
  }
  const maxLeafNodes = p.maxLeafNodes ?? null;
  if (maxLeafNodes !== null && !isWhole(maxLeafNodes, 2)) fail('ML_PARAM', `maxLeafNodes is a whole number, 2 or more, or null; not ${maxLeafNodes}`);
  const minImpurityDecrease = p.minImpurityDecrease ?? 0;
  if (!(minImpurityDecrease >= 0)) fail('ML_PARAM', `minImpurityDecrease is 0 or more, not ${minImpurityDecrease}`);
  const ccpAlpha = p.ccpAlpha ?? 0;
  if (ccpAlpha !== 0) fail('ML_PARAM', 'ccpAlpha (cost-complexity pruning) other than 0 isn\'t supported yet');
  const splitter = p.splitter ?? 'best';
  if (splitter !== 'best' && splitter !== 'random') fail('ML_PARAM', `splitter is 'best' or 'random', not ${String(splitter)}`);
  if (p.seed !== undefined && !Number.isInteger(p.seed)) fail('ML_PARAM', `seed is a whole number, not ${p.seed}`);
  return { maxDepth, minSamplesSplit, minSamplesLeaf, minWeightFractionLeaf, maxFeatures, splitter, maxLeafNodes, minImpurityDecrease, ccpAlpha, seed: p.seed };
}

/** What the tree's growth needs, from the parameters and the data's size. */
function growParams(c: ReturnType<typeof checkParams>, criterion: Criterion, n: number, p: number, total: number, seed: number) {
  const minSamplesLeaf = c.minSamplesLeaf >= 1 ? c.minSamplesLeaf : Math.ceil(c.minSamplesLeaf * n);
  let minSamplesSplit = isWhole(c.minSamplesSplit, 2) ? c.minSamplesSplit : Math.max(2, Math.ceil(c.minSamplesSplit * n));
  minSamplesSplit = Math.max(minSamplesSplit, 2 * minSamplesLeaf);
  const mf = c.maxFeatures;
  const maxFeatures =
    mf === null ? p
    : mf === 'sqrt' ? Math.max(1, Math.floor(Math.sqrt(p)))
    : mf === 'log2' ? Math.max(1, Math.floor(Math.log2(p)))
    : mf >= 1 ? Math.min(mf, p)
    : Math.max(1, Math.floor(mf * p));
  return {
    criterion,
    maxDepth: c.maxDepth ?? Infinity,
    minSamplesSplit,
    minSamplesLeaf,
    minWeightLeaf: c.minWeightFractionLeaf * total,
    maxFeatures,
    splitter: c.splitter,
    maxLeafNodes: c.maxLeafNodes,
    minImpurityDecrease: c.minImpurityDecrease,
    seed,
  };
}

function rowWeights(sampleWeight: VectorInput | undefined, n: number): Float64Array {
  if (sampleWeight === undefined) return new Float64Array(n).fill(1);
  let w: Float64Array;
  try {
    w = toVector(sampleWeight, n);
  } catch {
    return fail('ML_SHAPE', `sampleWeight needs one number per row of X (${n})`);
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = w[i]!;
    if (!Number.isFinite(v) || v < 0) fail('ML_VALUE', `sampleWeight must be finite and 0 or more; row ${i} has ${v}`);
    sum += v;
  }
  if (!(sum > 0)) fail('ML_VALUE', 'sampleWeight is 0 for every row: at least one row needs weight');
  return Float64Array.from(w);
}

function saveTree(t: TreeArrays) {
  return {
    left: [...t.left],
    right: [...t.right],
    feature: [...t.feature],
    threshold: numbers(t.threshold.map((v) => (v === -2 ? -2 : v))),
    impurity: numbers(t.impurity),
    nSamples: [...t.nSamples],
    weightedNSamples: numbers(t.weightedNSamples),
    value: numbers(t.value),
    width: t.width,
  };
}

function loadTree(s: unknown): TreeArrays {
  const t = s as Record<string, number[] | number>;
  if (!t || !Array.isArray(t.left) || !Array.isArray(t.value)) fail('ML_FORMAT', 'a tree needs its nodes');
  return {
    left: Int32Array.from(t.left as number[]),
    right: Int32Array.from(t.right as number[]),
    feature: Int32Array.from(t.feature as number[]),
    threshold: Float64Array.from(t.threshold as number[]),
    impurity: Float64Array.from(t.impurity as number[]),
    nSamples: Int32Array.from(t.nSamples as number[]),
    weightedNSamples: Float64Array.from(t.weightedNSamples as number[]),
    value: Float64Array.from(t.value as number[]),
    width: Number(t.width),
  };
}

/** What both trees share: growing, the tree's shape and importances, saving. */
abstract class DecisionTree {
  abstract readonly kind: string;
  protected readonly checked: ReturnType<typeof checkParams>;
  protected state: State | null = null;

  constructor(params: DecisionTreeParams) {
    this.checked = checkParams(params);
  }

  get fitted(): boolean {
    return this.state !== null;
  }
  /** The tree's nodes as arrays, as scikit-learn's `tree_`: children (−1 at a leaf), column and threshold (−2 at a leaf), impurity, rows, weight and value. */
  get tree(): TreeArrays {
    return this.fittedState().tree;
  }
  /** Each column's share of the total impurity decrease (sums to 1). scikit-learn's `feature_importances_`. */
  get featureImportances(): Float64Array {
    const s = this.fittedState();
    return featureImportances(s.tree, s.nFeaturesIn);
  }
  get nFeaturesIn(): number {
    return this.fittedState().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.fittedState().featureNamesIn;
  }
  /** The tree's depth: 0 for a single leaf. */
  getDepth(): number {
    return depthOf(this.fittedState().tree);
  }
  /** How many leaves the tree has. */
  getNLeaves(): number {
    return this.fittedState().tree.left.filter((v) => v === -1).length;
  }
  /** The leaf each row falls in, by node index. */
  apply(X: MatrixInput): Int32Array {
    const s = this.fittedState();
    const m = prepareX(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return applyTree(s.tree, m);
  }

  protected baseParams(): DecisionTreeParams {
    const c = this.checked;
    return {
      maxDepth: c.maxDepth,
      minSamplesSplit: c.minSamplesSplit,
      minSamplesLeaf: c.minSamplesLeaf,
      minWeightFractionLeaf: c.minWeightFractionLeaf,
      maxFeatures: c.maxFeatures,
      ...(c.splitter !== 'best' ? { splitter: c.splitter } : {}),
      maxLeafNodes: c.maxLeafNodes,
      minImpurityDecrease: c.minImpurityDecrease,
      ccpAlpha: c.ccpAlpha,
      ...(c.seed !== undefined ? { seed: c.seed } : {}),
    };
  }

  protected growOn(m: Matrix, targets: Targets, weights: Float64Array, criterion: Criterion): { tree: TreeArrays; seed: number } {
    let total = 0;
    for (const w of weights) total += w;
    const seed = this.checked.seed ?? Math.floor(Math.random() * 2 ** 31);
    // Rows of weight 0 take no part (grow leaves them out); shares of the rows (minSamplesSplit, minSamplesLeaf) stay
    // shares of all of X's.
    const params = growParams(this.checked, criterion, m.rows, m.cols, total, seed);
    return { tree: grow(m, targets, weights, params), seed };
  }

  protected saved(): Record<string, unknown> | null {
    const s = this.state;
    return s && {
      tree: saveTree(s.tree),
      seed: s.seed,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.classes ? { classes: s.classes } : {}),
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    };
  }

  protected restore(s: Record<string, unknown> | null): this {
    if (s) {
      this.state = {
        tree: loadTree(s.tree),
        seed: Number(s.seed),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
        ...(Array.isArray(s.classes) ? { classes: s.classes as Label[] } : {}),
      };
    }
    return this;
  }

  protected fittedState(): State {
    return this.state ?? notFitted(this.kind);
  }
}

/**
 * A regression tree: each leaf predicts the (weighted) mean of its training rows. scikit-learn's
 * `DecisionTreeRegressor`, grown by CART (ADR 0039).
 */
export class DecisionTreeRegressor extends DecisionTree {
  readonly kind = 'DecisionTreeRegressor';
  readonly criterion: 'squared_error' | 'friedman_mse';

  constructor(params: DecisionTreeRegressorParams = {}) {
    super(params);
    const c = params.criterion ?? 'squared_error';
    if (c === 'absolute_error' || c === 'poisson') fail('ML_PARAM', `criterion '${c}' isn't supported yet: use 'squared_error' or 'friedman_mse'`);
    if (c !== 'squared_error' && c !== 'friedman_mse') fail('ML_PARAM', `criterion is 'squared_error' or 'friedman_mse', not ${String(c)}`);
    this.criterion = c;
  }

  getParams(): DecisionTreeRegressorParams {
    return { criterion: this.criterion, ...this.baseParams() };
  }

  async fit(X: MatrixInput, y: VectorInput, options: TreeFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: VectorInput, options: TreeFitOptions = {}): this {
    const m = prepareX(X);
    const target = toVector(y, m.rows);
    for (let i = 0; i < target.length; i++) if (!Number.isFinite(target[i]!)) fail('ML_VALUE', `y[${i}] is ${target[i]}`);
    const weights = rowWeights(options.sampleWeight, m.rows);
    const { tree, seed } = this.growOn(m, { kind: 'regression', y: target }, weights, this.criterion);
    this.state = { tree, seed, nFeaturesIn: m.cols, featureNamesIn: m.names };
    return this;
  }

  async predict(X: MatrixInput): Promise<Float64Array> {
    return this.predictSync(X);
  }

  /** Each row's leaf value: the mean of the training rows in its leaf. */
  predictSync(X: MatrixInput): Float64Array {
    const t = this.fittedState().tree;
    return Float64Array.from(this.apply(X), (leaf) => t.value[leaf]!);
  }

  toJSON(): ModelDocument {
    return document(this.kind, { ...this.getParams() }, this.saved());
  }

  static fromJSON(doc: ModelDocument): DecisionTreeRegressor {
    const d = checkDocument(doc, 'DecisionTreeRegressor');
    return new DecisionTreeRegressor(d.params as DecisionTreeRegressorParams).restore(d.state);
  }
}

/**
 * A classification tree: each leaf gives its training rows' (weighted) class shares, and predicts the largest.
 * scikit-learn's `DecisionTreeClassifier`, grown by CART (ADR 0039).
 */
export class DecisionTreeClassifier extends DecisionTree {
  readonly kind = 'DecisionTreeClassifier';
  readonly criterion: 'gini' | 'entropy' | 'log_loss';
  readonly classWeight: 'balanced' | Record<string, number> | null;

  constructor(params: DecisionTreeClassifierParams = {}) {
    super(params);
    const c = params.criterion ?? 'gini';
    if (c !== 'gini' && c !== 'entropy' && c !== 'log_loss') fail('ML_PARAM', `criterion is 'gini', 'entropy' or 'log_loss', not ${String(c)}`);
    this.criterion = c;
    const cw = params.classWeight ?? null;
    if (cw !== null && cw !== 'balanced' && (typeof cw !== 'object' || Object.values(cw).some((v) => !(typeof v === 'number' && v >= 0 && Number.isFinite(v))))) {
      fail('ML_PARAM', "classWeight is 'balanced', an object of weights 0 or more by label, or null");
    }
    this.classWeight = cw;
  }

  /** The labels seen in fit, sorted: the order of predictProba's columns. */
  get classes(): readonly Label[] {
    return this.fittedState().classes!;
  }

  getParams(): DecisionTreeClassifierParams {
    return { criterion: this.criterion, ...(this.classWeight !== null ? { classWeight: this.classWeight } : {}), ...this.baseParams() };
  }

  async fit(X: MatrixInput, y: Labels, options: TreeFitOptions = {}): Promise<this> {
    return this.fitSync(X, y, options);
  }

  fitSync(X: MatrixInput, y: Labels, options: TreeFitOptions = {}): this {
    const m = prepareX(X);
    const labels = toLabels(y, m.rows);
    const classes = uniqueLabels(labels);
    const index = new Map(classes.map((c, k) => [c, k]));
    const target = Int32Array.from(labels, (l) => index.get(l)!);
    const weights = rowWeights(options.sampleWeight, m.rows);
    const cw = this.classWeight;
    if (cw !== null) {
      let perClass: Float64Array;
      if (cw === 'balanced') {
        // n / (classes × the class's rows), counting rows, not their sample weights, as scikit-learn's trees do.
        const counts = new Float64Array(classes.length);
        for (const t of target) counts[t]! += 1;
        perClass = Float64Array.from(counts, (c) => m.rows / (classes.length * c));
      } else {
        for (const label of Object.keys(cw)) {
          if (!classes.some((c) => String(c) === label)) fail('ML_PARAM', `classWeight names ${label}, which isn't one of the classes in y (${classes.map(String).join(', ')})`);
        }
        perClass = Float64Array.from(classes, (c) => cw[String(c)] ?? 1);
      }
      for (let i = 0; i < weights.length; i++) weights[i]! *= perClass[target[i]!]!;
      if (!weights.some((w) => w > 0)) fail('ML_VALUE', 'every row weighs 0 (sampleWeight and classWeight together): at least one row needs weight');
    }
    const criterion: Criterion = this.criterion === 'gini' ? 'gini' : 'entropy';
    const { tree, seed } = this.growOn(m, { kind: 'classes', y: target, K: classes.length }, weights, criterion);
    this.state = { tree, seed, nFeaturesIn: m.cols, featureNamesIn: m.names, classes };
    return this;
  }

  async predict(X: MatrixInput): Promise<Label[]> {
    return this.predictSync(X);
  }

  /** Each row's most likely class: the largest share in its leaf (of equal shares, the first class). */
  predictSync(X: MatrixInput): Label[] {
    const s = this.fittedState();
    const t = s.tree;
    const K = t.width;
    return Array.from(this.apply(X), (leaf) => {
      let best = 0;
      for (let k = 1; k < K; k++) if (t.value[leaf * K + k]! > t.value[leaf * K + best]!) best = k;
      return s.classes![best]!;
    });
  }

  async predictProba(X: MatrixInput): Promise<Matrix> {
    return this.predictProbaSync(X);
  }

  /** Each row's class shares in its leaf, a column per class in the order of `classes`. */
  predictProbaSync(X: MatrixInput): Matrix {
    const s = this.fittedState();
    const t = s.tree;
    const K = t.width;
    const leaves = this.apply(X);
    const n = leaves.length;
    const out = new Float64Array(n * K);
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) out[k * n + i] = t.value[leaves[i]! * K + k]!;
    return new Matrix(n, K, out, s.classes!.map(String));
  }

  toJSON(): ModelDocument {
    return document(this.kind, { ...this.getParams() }, this.saved());
  }

  static fromJSON(doc: ModelDocument): DecisionTreeClassifier {
    const d = checkDocument(doc, 'DecisionTreeClassifier');
    return new DecisionTreeClassifier(d.params as DecisionTreeClassifierParams).restore(d.state);
  }
}
