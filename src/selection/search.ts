import { fail } from '../core/errors.js';
import { withKernelCache } from '../linear/calls.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { checkDocument, document, isClassifier, isPredictor, unfitDocument, type Label, type Labels, type Model, type ModelDocument } from '../core/model.js';
import { numericMatrix, toTable, type TableInput } from '../core/table.js';
import { loadModel, registerLoader } from '../io/index.js';
import {
  accuracyScore,
  f1Score,
  logLoss,
  meanAbsoluteError,
  meanSquaredError,
  precisionScore,
  r2Score,
  recallScore,
  rocAucScore,
} from '../metrics/index.js';
import { KFold, StratifiedKFold, type Fold, type Splitter } from './folds.js';

/** A fitted model's score on rows: higher is better. */
export type Scorer = (model: Model, X: MatrixInput, y: Labels) => Promise<number>;

/** scikit-learn's scorer names. Error metrics are negated (`neg_…`), so that higher is better for every one. */
export type ScorerName =
  | 'accuracy'
  | 'r2'
  | 'neg_mean_squared_error'
  | 'neg_root_mean_squared_error'
  | 'neg_mean_absolute_error'
  | 'neg_log_loss'
  | 'roc_auc'
  | 'f1'
  | 'f1_macro'
  | 'f1_micro'
  | 'f1_weighted'
  | 'precision'
  | 'precision_macro'
  | 'recall'
  | 'recall_macro';

const predictions = async (m: Model, X: MatrixInput) => {
  if (!isPredictor(m)) fail('ML_PARAM', `${m.kind} doesn't predict, so it can't be scored`);
  return m.predict(X);
};
const proba = async (m: Model, X: MatrixInput): Promise<Matrix> => {
  const f = (m as Partial<{ predictProba(X: MatrixInput): Promise<Matrix> }>).predictProba;
  if (!f) fail('ML_PARAM', `${m.kind} doesn't give probabilities, which this scorer needs`);
  return f.call(m, X);
};
const classesOf = (m: Model) => (m as Partial<{ classes: readonly Label[] }>).classes;

const SCORERS: Record<ScorerName, Scorer> = {
  accuracy: async (m, X, y) => accuracyScore(y, (await predictions(m, X)) as Labels),
  r2: async (m, X, y) => r2Score(y as ArrayLike<number>, (await predictions(m, X)) as Float64Array),
  neg_mean_squared_error: async (m, X, y) => -meanSquaredError(y as ArrayLike<number>, (await predictions(m, X)) as Float64Array),
  neg_root_mean_squared_error: async (m, X, y) => -Math.sqrt(meanSquaredError(y as ArrayLike<number>, (await predictions(m, X)) as Float64Array)),
  neg_mean_absolute_error: async (m, X, y) => -meanAbsoluteError(y as ArrayLike<number>, (await predictions(m, X)) as Float64Array),
  neg_log_loss: async (m, X, y) => -logLoss(y, await proba(m, X), { labels: classesOf(m) }),
  roc_auc: async (m, X, y) => {
    const P = await proba(m, X);
    const labels = classesOf(m);
    return P.cols === 2 ? rocAucScore(y, P.column(1), { labels }) : rocAucScore(y, P, { labels, multiClass: 'ovr' });
  },
  f1: async (m, X, y) => f1Score(y, (await predictions(m, X)) as Labels),
  f1_macro: async (m, X, y) => f1Score(y, (await predictions(m, X)) as Labels, { average: 'macro' }),
  f1_micro: async (m, X, y) => f1Score(y, (await predictions(m, X)) as Labels, { average: 'micro' }),
  f1_weighted: async (m, X, y) => f1Score(y, (await predictions(m, X)) as Labels, { average: 'weighted' }),
  precision: async (m, X, y) => precisionScore(y, (await predictions(m, X)) as Labels),
  precision_macro: async (m, X, y) => precisionScore(y, (await predictions(m, X)) as Labels, { average: 'macro' }),
  recall: async (m, X, y) => recallScore(y, (await predictions(m, X)) as Labels),
  recall_macro: async (m, X, y) => recallScore(y, (await predictions(m, X)) as Labels, { average: 'macro' }),
};

/** A scorer from its name, or the function given; by default, accuracy for a classifier and R² for anything else. */
function scorerFor(model: Model, scoring: ScorerName | Scorer | undefined): Scorer {
  // Scoring takes one target so far; scikit-learn averages R² over several.
  if (model.kind === 'MultiOutputRegressor') fail('ML_PARAM', "crossValScore and GridSearchCV don't take several targets yet: fit a model per target, or MultiOutputRegressor on its own");
  if (typeof scoring === 'function') return scoring;
  if (scoring === undefined) return classifies(model) ? SCORERS.accuracy : SCORERS.r2;
  const s = SCORERS[scoring];
  if (!s) fail('ML_PARAM', `no scorer is named ${String(scoring)}: ${Object.keys(SCORERS).join(', ')}`);
  return s;
}

/** Whether a model predicts classes: a pipeline does when its last step does (it has predictProba either way). */
function classifies(m: Model): boolean {
  const last = (m as Partial<{ last: Model }>).last;
  return last ? classifies(last) : isClassifier(m);
}

/** The folds to use: a number of folds (stratified for a classifier, as scikit-learn chooses), a splitter, or folds. */
function foldsFor(model: Model, cv: number | Splitter | readonly Fold[] | undefined, n: number, y: Labels | undefined): Fold[] {
  if (Array.isArray(cv)) return cv as Fold[];
  if (cv && typeof cv === 'object') return (cv as Splitter).split(n, y);
  const k = (cv as number | undefined) ?? 5;
  return classifies(model) && y ? new StratifiedKFold({ nSplits: k }).split(n, y) : new KFold({ nSplits: k }).split(n);
}

/** The rows at `index` of X and y. */
function asMatrixIfNumeric(X: TableInput): TableInput {
  if (X instanceof Matrix) return X;
  try {
    return numericMatrix(X);
  } catch {
    return X;
  }
}

function rowsOf(X: TableInput, index: Uint32Array): TableInput {
  if (X instanceof Matrix) return X.take(index);
  const t = toTable(X);
  return Object.fromEntries(t.columns.map((c, j) => [t.named ? t.names[j]! : `x${j}`, Array.from(index, (i) => c[i]!)])) as TableInput;
}
function labelsOf(y: Labels, index: Uint32Array): Labels {
  return y instanceof Float64Array ? Float64Array.from(index, (i) => y[i]!) : (Array.from(index, (i) => (y as ArrayLike<Label>)[i]!) as Labels);
}

/**
 * A fresh, unfitted copy of a model with some parameters changed: `{ C: 10 }`, `{ step__param: v }` inside a pipeline or
 * a ColumnTransformer, or `{ estimator__param: v }` inside a MultiOutputRegressor.
 */
export function cloneModel<M extends Model>(model: M, params: Record<string, unknown> = {}): M {
  const doc = structuredClone(model.toJSON());
  for (const [key, value] of Object.entries(params)) setParam(doc, key, value);
  return loadModel(unfitDocument(doc)) as M;
}

function setParam(doc: ModelDocument, key: string, value: unknown): void {
  const cut = key.indexOf('__');
  if (cut < 0) {
    if (doc.model === 'Pipeline' || doc.model === 'ColumnTransformer') fail('ML_PARAM', `${doc.model} has no parameter ${key}: name a step, as step__${key}`);
    if (doc.model === 'MultiOutputRegressor') fail('ML_PARAM', `MultiOutputRegressor has no parameter ${key}: name its model's, as estimator__${key}`);
    if (!(key in doc.params) && !Object.keys(doc.params).length) fail('ML_PARAM', `${doc.model} has no parameter ${key}`);
    doc.params[key] = value;
    return;
  }
  const head = key.slice(0, cut);
  const rest = key.slice(cut + 2);
  if (doc.model === 'MultiOutputRegressor') {
    if (head !== 'estimator') fail('ML_PARAM', `MultiOutputRegressor has no part named ${head}: its model's parameters are estimator__name`);
    setParam(doc.params.estimator as ModelDocument, rest, value);
    return;
  }
  const names = doc.model === 'Pipeline' ? (doc.params.steps as string[]) : doc.model === 'ColumnTransformer' ? (doc.params.parts as { name: string }[]).map((p) => p.name) : null;
  if (!names) fail('ML_PARAM', `${doc.model} has no steps: ${key} names one`);
  const k = names.indexOf(head);
  if (k < 0) fail('ML_PARAM', `${doc.model} has no step named ${head}`);
  const subs = (doc.model === 'Pipeline' ? doc.state!.steps : doc.state!.parts) as ModelDocument[];
  if (typeof subs[k] === 'string') fail('ML_PARAM', `step ${head} is '${subs[k]}', which has no parameters`);
  setParam(subs[k]!, rest, value);
}

/** The document with every fitted state removed. */
export interface CrossValOptions {
  /** The folds: a number (default 5; stratified for a classifier), a splitter such as KFold, or a list of folds. */
  cv?: number | Splitter | readonly Fold[];
  /** How to score each fold: a scorer's name or a function. Default: accuracy for a classifier, R² otherwise. */
  scoring?: ScorerName | Scorer;
}

/**
 * Fits a fresh copy of the model on each fold's training rows and scores it on the fold's test rows: one score per
 * fold. scikit-learn's `cross_val_score`.
 */
export async function crossValScore(model: Model, X: TableInput, y: Labels, options: CrossValOptions = {}): Promise<Float64Array> {
  const n = toTable(X).rows;
  if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
  const scorer = scorerFor(model, options.scoring);
  const out: number[] = [];
  for (const { train, test } of foldsFor(model, options.cv, n, y)) {
    const m = cloneModel(model) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
    await m.fit(rowsOf(X, train), labelsOf(y, train));
    out.push(await scorer(m, rowsOf(X, test) as MatrixInput, labelsOf(y, test)));
  }
  return Float64Array.from(out);
}

/** A parameter grid: each name's values to try. A list of grids tries each grid's combinations. */
export type ParamGrid = Readonly<Record<string, readonly unknown[]>> | readonly Readonly<Record<string, readonly unknown[]>>[];

/** Every combination of a grid, in scikit-learn's order: names sorted, the last name changing fastest. */
export function parameterGrid(grid: ParamGrid): Record<string, unknown>[] {
  const grids = Array.isArray(grid) ? (grid as Record<string, readonly unknown[]>[]) : [grid as Record<string, readonly unknown[]>];
  return grids.flatMap((g) => {
    const keys = Object.keys(g).sort();
    for (const k of keys) if (!Array.isArray(g[k]) || g[k]!.length === 0) fail('ML_PARAM', `the grid's ${k} must be a non-empty list of values`);
    let combos: Record<string, unknown>[] = [{}];
    for (const k of keys) combos = combos.flatMap((c) => g[k]!.map((v) => ({ ...c, [k]: v })));
    return combos;
  });
}

export interface CvResults {
  params: Record<string, unknown>[];
  meanTestScore: Float64Array;
  stdTestScore: Float64Array;
  rankTestScore: Int32Array;
  /** Each fold's scores: splitTestScores[fold][candidate]. */
  splitTestScores: Float64Array[];
}

export interface GridSearchOptions extends CrossValOptions {
  /** Fit the best parameters on all the rows afterwards, as bestEstimator. Default true. */
  refit?: boolean;
}

/**
 * Tries every combination of a parameter grid by cross-validation and keeps the best. scikit-learn's `GridSearchCV`:
 * the same combinations, in the same order, the same mean and spread of fold scores, and the same choice (the
 * highest mean; the first among ties).
 */
export class GridSearchCV implements Model {
  readonly kind = 'GridSearchCV';
  readonly estimator: Model;
  readonly paramGrid: ParamGrid;
  readonly options: GridSearchOptions;
  #results: CvResults | null = null;
  #best: Model | null = null;

  constructor(estimator: Model, paramGrid: ParamGrid, options: GridSearchOptions = {}) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.options = options;
    parameterGrid(paramGrid);
  }

  get fitted(): boolean {
    return this.#results !== null;
  }
  /** Every candidate's parameters and scores. */
  get cvResults(): CvResults {
    return this.#results ?? fail('ML_NOT_FITTED', "this GridSearchCV isn't fitted yet: call fit() first");
  }
  /** The index of the best candidate. */
  get bestIndex(): number {
    return this.cvResults.rankTestScore.indexOf(1);
  }
  /** The best candidate's parameters. */
  get bestParams(): Record<string, unknown> {
    return this.cvResults.params[this.bestIndex]!;
  }
  /** The best candidate's mean score. */
  get bestScore(): number {
    return this.cvResults.meanTestScore[this.bestIndex]!;
  }
  /** The best parameters fitted on all the rows (with refit). */
  get bestEstimator(): Model {
    return this.#best ?? fail('ML_NOT_FITTED', 'no bestEstimator: fit with refit: true');
  }

  getParams(): Record<string, unknown> {
    return { estimator: this.estimator.getParams(), paramGrid: this.paramGrid, refit: this.options.refit ?? true };
  }

  async fit(X: TableInput, y: Labels): Promise<this> {
    // The candidates share each fold's rows: linear models work out XᵀX and Xᵀy once per fold.
    return withKernelCache(() => this.#fit(X, y));
  }

  async #fit(X: TableInput, y: Labels): Promise<this> {
    // Numbers become one Matrix up front, so each fold's rows are one matrix every candidate fits on.
    X = asMatrixIfNumeric(X);
    const n = toTable(X).rows;
    if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
    const candidates = parameterGrid(this.paramGrid);
    const scorer = scorerFor(this.estimator, this.options.scoring);
    const folds = foldsFor(this.estimator, this.options.cv, n, y);
    const split = folds.map(() => new Float64Array(candidates.length));
    for (const [f, { train, test }] of folds.entries()) {
      const Xtr = rowsOf(X, train);
      const ytr = labelsOf(y, train);
      const Xte = rowsOf(X, test) as MatrixInput;
      const yte = labelsOf(y, test);
      for (const [c, params] of candidates.entries()) {
        const m = cloneModel(this.estimator, params) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
        await m.fit(Xtr, ytr);
        split[f]![c] = await scorer(m, Xte, yte);
      }
    }
    const mean = Float64Array.from(candidates, (_, c) => split.reduce((s, row) => s + row[c]!, 0) / folds.length);
    const std = Float64Array.from(candidates, (_, c) => Math.sqrt(split.reduce((s, row) => s + (row[c]! - mean[c]!) ** 2, 0) / folds.length));
    // Rank 1 is the highest mean; ties share the lowest rank among them.
    const rank = Int32Array.from(mean, (v) => 1 + Array.from(mean).filter((w) => w > v).length);
    this.#results = { params: candidates, meanTestScore: mean, stdTestScore: std, rankTestScore: rank, splitTestScores: split };
    if (this.options.refit ?? true) {
      const best = cloneModel(this.estimator, candidates[this.bestIndex]!) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
      await best.fit(X, y);
      this.#best = best;
    }
    return this;
  }

  /** The best estimator's predictions. */
  async predict(X: TableInput): Promise<Float64Array | Label[]> {
    const b = this.bestEstimator;
    if (!isPredictor(b)) fail('ML_PARAM', `${b.kind} doesn't predict`);
    return b.predict(X as MatrixInput);
  }

  /** The best estimator's class probabilities. */
  async predictProba(X: TableInput): Promise<Matrix> {
    return proba(this.bestEstimator, X as MatrixInput);
  }

  /** The labels of the best estimator, when it is a classifier. */
  get classes(): readonly Label[] {
    return classesOf(this.bestEstimator) ?? fail('ML_PARAM', 'the best estimator is not a classifier');
  }

  toJSON(): ModelDocument {
    const r = this.#results;
    return document(
      this.kind,
      { paramGrid: this.paramGrid as unknown as Record<string, unknown>, refit: this.options.refit ?? true, ...(typeof this.options.scoring === 'string' ? { scoring: this.options.scoring } : {}) },
      {
        estimator: this.estimator.toJSON(),
        ...(r ? { cvResults: { params: r.params, meanTestScore: [...r.meanTestScore], stdTestScore: [...r.stdTestScore], rankTestScore: [...r.rankTestScore] } } : {}),
        ...(this.#best ? { bestEstimator: this.#best.toJSON() } : {}),
      },
    );
  }

  static fromJSON(doc: ModelDocument): GridSearchCV {
    const d = checkDocument(doc, 'GridSearchCV');
    const s = d.state ?? {};
    const g = new GridSearchCV(loadModel(s.estimator as ModelDocument), d.params.paramGrid as ParamGrid, {
      refit: d.params.refit as boolean,
      ...(d.params.scoring ? { scoring: d.params.scoring as ScorerName } : {}),
    });
    const r = s.cvResults as { params: Record<string, unknown>[]; meanTestScore: number[]; stdTestScore: number[]; rankTestScore: number[] } | undefined;
    if (r) {
      g.#results = {
        params: r.params,
        meanTestScore: Float64Array.from(r.meanTestScore),
        stdTestScore: Float64Array.from(r.stdTestScore),
        rankTestScore: Int32Array.from(r.rankTestScore),
        splitTestScores: [],
      };
    }
    if (s.bestEstimator) g.#best = loadModel(s.bestEstimator as ModelDocument);
    return g;
  }
}

/** The names of scikit-learn scorers smaltml has. */
export const SCORER_NAMES = Object.keys(SCORERS) as ScorerName[];

registerLoader('GridSearchCV', (d) => GridSearchCV.fromJSON(d));
