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
import { KFold, StratifiedKFold, type Fold, type Groups, type Splitter } from './folds.js';

/** A fitted model's score on rows: higher is better. */
export type Scorer = (model: Model, X: MatrixInput, y: Labels) => Promise<number>;

/** scikit-learn's scorer names. Error metrics are negated (`neg_…`), so that higher is better for every one. */
/**
 * Several scorers at once: a list of scorers' names, or names for scorers (`{ fit: 'r2', error: myScorer }`). The
 * results are under those names. scikit-learn's multi-metric `scoring`.
 */
export type Scorers = readonly ScorerName[] | Readonly<Record<string, ScorerName | Scorer>>;

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

/** Scoring takes one target so far: a y with a column per target fails, as scikit-learn would average R² over them. */
function oneTarget(y: Labels): void {
  const first = (y as ArrayLike<unknown>)[0];
  if ((y as unknown) instanceof Matrix || (y.length > 0 && typeof first === 'object' && first !== null) || (typeof y === 'object' && !('length' in y))) {
    fail('ML_PARAM', "crossValScore and GridSearchCV don't take a y with several targets yet: score each target on its own");
  }
}

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

/** The scorers by name, and whether they were given as several (a list or a record, even of one). */
function scorersFor(model: Model, scoring: ScorerName | Scorer | Scorers | undefined): { names: string[]; scorers: Scorer[]; several: boolean } {
  if (Array.isArray(scoring)) {
    const names = scoring as readonly ScorerName[];
    if (names.length === 0) fail('ML_PARAM', 'scoring is an empty list: name at least one scorer');
    if (new Set(names).size !== names.length) fail('ML_PARAM', `scoring names a scorer twice: ${names.join(', ')}`);
    return { names: [...names], scorers: names.map((n) => scorerFor(model, n)), several: true };
  }
  if (scoring && typeof scoring === 'object') {
    const entries = Object.entries(scoring as Record<string, ScorerName | Scorer>);
    if (entries.length === 0) fail('ML_PARAM', 'scoring is an empty record: name at least one scorer');
    return { names: entries.map(([k]) => k), scorers: entries.map(([, v]) => scorerFor(model, v)), several: true };
  }
  return { names: ['score'], scorers: [scorerFor(model, scoring as ScorerName | Scorer | undefined)], several: false };
}

/** Whether a model predicts classes: a pipeline does when its last step does (it has predictProba either way). */
function classifies(m: Model): boolean {
  const last = (m as Partial<{ last: Model }>).last;
  return last ? classifies(last) : isClassifier(m);
}

/** The folds to use: a number of folds (stratified for a classifier, as scikit-learn chooses), a splitter, or folds. */
function foldsFor(model: Model, cv: number | Splitter | readonly Fold[] | undefined, n: number, y: Labels | undefined, groups?: Groups): Fold[] {
  if (groups !== undefined) {
    // scikit-learn warns and ignores groups where the splitter doesn't use them; a split that ignores them isn't the
    // one asked for, so it fails here.
    if (!cv || typeof cv !== 'object' || Array.isArray(cv) || !(cv as Splitter).usesGroups) {
      fail('ML_PARAM', 'groups need a splitter that keeps groups together, such as cv: new GroupKFold()');
    }
  }
  if (Array.isArray(cv)) return cv as Fold[];
  if (cv && typeof cv === 'object') return (cv as Splitter).split(n, y, groups);
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

export interface CrossValOptions {
  /** The folds: a number (default 5; stratified for a classifier), a splitter such as KFold, or a list of folds. */
  cv?: number | Splitter | readonly Fold[];
  /** How to score each fold: a scorer's name or a function. Default: accuracy for a classifier, R² otherwise. */
  scoring?: ScorerName | Scorer;
  /** Each row's group, for a splitter that keeps groups together (GroupKFold): scikit-learn's `groups`. */
  groups?: Groups;
}

/**
 * Fits a fresh copy of the model on each fold's training rows and scores it on the fold's test rows: one score per
 * fold. scikit-learn's `cross_val_score`.
 */
export async function crossValScore(model: Model, X: TableInput, y: Labels, options: CrossValOptions = {}): Promise<Float64Array> {
  if (options.scoring !== undefined && typeof options.scoring === 'object') {
    fail('ML_PARAM', 'crossValScore takes one scorer: for several, use crossValidate');
  }
  oneTarget(y);
  const n = toTable(X).rows;
  if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
  const scorer = scorerFor(model, options.scoring);
  const out: number[] = [];
  for (const { train, test } of foldsFor(model, options.cv, n, y, options.groups)) {
    const m = cloneModel(model) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
    await m.fit(rowsOf(X, train), labelsOf(y, train));
    out.push(await scorer(m, rowsOf(X, test) as MatrixInput, labelsOf(y, test)));
  }
  return Float64Array.from(out);
}

export interface CrossValidateOptions extends Omit<CrossValOptions, 'scoring'> {
  /** One scorer (its results are named `score`), or several (Scorers). Default: accuracy for a classifier, R² otherwise. */
  scoring?: ScorerName | Scorer | Scorers;
  /** Score each fold's training rows too, as `train`. Default false: it costs a prediction on them. */
  returnTrainScore?: boolean;
}

export interface CrossValidateResult {
  /** Each scorer's score per fold, on the fold's test rows; one scorer is named `score`. */
  test: Record<string, Float64Array>;
  /** The same on the fold's training rows, with returnTrainScore. */
  train?: Record<string, Float64Array>;
  /** Each fold's fit, and its scoring, in seconds. */
  fitTime: Float64Array;
  scoreTime: Float64Array;
}

/**
 * crossValScore with several scorers, train scores and times: fits a fresh copy of the model on each fold's training
 * rows and scores it on the fold's test rows (and training rows, with returnTrainScore). scikit-learn's
 * `cross_validate`, its results as `{ test: { r2: … }, train, fitTime, scoreTime }` where scikit-learn's are
 * `test_r2`, `train_r2`, `fit_time` and `score_time`.
 */
export async function crossValidate(model: Model, X: TableInput, y: Labels, options: CrossValidateOptions = {}): Promise<CrossValidateResult> {
  oneTarget(y);
  const n = toTable(X).rows;
  if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
  const { names, scorers } = scorersFor(model, options.scoring);
  const folds = foldsFor(model, options.cv, n, y, options.groups);
  const test = names.map(() => new Float64Array(folds.length));
  const train = options.returnTrainScore ? names.map(() => new Float64Array(folds.length)) : null;
  const fitTime = new Float64Array(folds.length);
  const scoreTime = new Float64Array(folds.length);
  for (const [f, fold] of folds.entries()) {
    const m = cloneModel(model) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
    const Xtr = rowsOf(X, fold.train);
    const ytr = labelsOf(y, fold.train);
    let t = now();
    await m.fit(Xtr, ytr);
    fitTime[f] = (now() - t) / 1000;
    t = now();
    const Xte = rowsOf(X, fold.test) as MatrixInput;
    const yte = labelsOf(y, fold.test);
    for (const [k, scorer] of scorers.entries()) test[k]![f] = await scorer(m, Xte, yte);
    scoreTime[f] = (now() - t) / 1000;
    if (train) for (const [k, scorer] of scorers.entries()) train[k]![f] = await scorer(m, Xtr as MatrixInput, ytr);
  }
  const byName = (rows: Float64Array[]) => Object.fromEntries(names.map((nm, k) => [nm, rows[k]!]));
  return { test: byName(test), ...(train ? { train: byName(train) } : {}), fitTime, scoreTime };
}

/** Milliseconds, from the clock pages, workers and Node share. */
function now(): number {
  return globalThis.performance?.now() ?? Date.now();
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

/** One scorer's results for every candidate. */
export interface ScorerResults {
  meanTestScore: Float64Array;
  stdTestScore: Float64Array;
  rankTestScore: Int32Array;
  /** Each fold's scores: splitTestScores[fold][candidate]. */
  splitTestScores: Float64Array[];
  /** The same on each fold's training rows, with returnTrainScore. */
  meanTrainScore?: Float64Array;
  stdTrainScore?: Float64Array;
  splitTrainScores?: Float64Array[];
}

/** Every candidate's parameters and its scores, for one scorer. */
export interface CvResults extends ScorerResults {
  params: Record<string, unknown>[];
}

export interface GridSearchOptions extends Omit<CrossValOptions, 'scoring'> {
  /** One scorer, or several (Scorers). Default: accuracy for a classifier, R² otherwise. */
  scoring?: ScorerName | Scorer | Scorers;
  /**
   * Fit the best parameters on all the rows afterwards, as bestEstimator. With several scorers, name the one that
   * chooses the best (true fails, as in scikit-learn), or false. Default true.
   */
  refit?: boolean | string;
  /** Score each fold's training rows too: the `…TrainScore` results. Default false. */
  returnTrainScore?: boolean;
}

/** The mean, spread and rank of each candidate's fold scores: rank 1 is the highest mean, ties share the lowest. */
function summarize(split: Float64Array[], candidates: number): { mean: Float64Array; std: Float64Array; rank: Int32Array } {
  const folds = split.length;
  const mean = Float64Array.from({ length: candidates }, (_, c) => split.reduce((s, row) => s + row[c]!, 0) / folds);
  const std = Float64Array.from({ length: candidates }, (_, c) => Math.sqrt(split.reduce((s, row) => s + (row[c]! - mean[c]!) ** 2, 0) / folds));
  const rank = Int32Array.from(mean, (v) => 1 + Array.from(mean).filter((w) => w > v).length);
  return { mean, std, rank };
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
  #results: { params: Record<string, unknown>[]; scores: Record<string, ScorerResults>; several: boolean } | null = null;
  #best: Model | null = null;

  constructor(estimator: Model, paramGrid: ParamGrid, options: GridSearchOptions = {}) {
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.options = options;
    parameterGrid(paramGrid);
    const several = options.scoring !== undefined && typeof options.scoring === 'object';
    const refit = options.refit ?? true;
    if (several && refit === true) {
      fail('ML_PARAM', 'with several scorers, refit names the one that chooses the best (or is false), as in scikit-learn');
    }
    if (several && typeof refit === 'string') {
      const names = Array.isArray(options.scoring) ? (options.scoring as string[]) : Object.keys(options.scoring as object);
      if (!names.includes(refit)) fail('ML_PARAM', `refit names ${refit}, which isn't one of the scorers: ${names.join(', ')}`);
    }
    if (typeof refit !== 'boolean' && typeof refit !== 'string') fail('ML_PARAM', 'refit is true, false, or a scorer\'s name');
  }

  get fitted(): boolean {
    return this.#results !== null;
  }
  /** Every candidate's parameters and scores, for one scorer. With several, read cvResultsByScorer. */
  get cvResults(): CvResults {
    const r = this.#fittedResults();
    if (r.several) fail('ML_PARAM', 'this search has several scorers: read cvResultsByScorer');
    return { params: r.params, ...r.scores.score! };
  }
  /** Every candidate's parameters, and each scorer's results under its name (one scorer's under `score`). */
  get cvResultsByScorer(): { params: Record<string, unknown>[]; scores: Record<string, ScorerResults> } {
    const r = this.#fittedResults();
    return { params: r.params, scores: r.scores };
  }
  /** The index of the best candidate: by the scorer refit names, with several. */
  get bestIndex(): number {
    return this.#choosing().rankTestScore.indexOf(1);
  }
  /** The best candidate's parameters. */
  get bestParams(): Record<string, unknown> {
    return this.#fittedResults().params[this.bestIndex]!;
  }
  /** The best candidate's mean score. */
  get bestScore(): number {
    return this.#choosing().meanTestScore[this.bestIndex]!;
  }

  #fittedResults() {
    return this.#results ?? fail('ML_NOT_FITTED', "this GridSearchCV isn't fitted yet: call fit() first");
  }

  /** The scorer that chooses the best: the one, or the one refit names. */
  #choosing(): ScorerResults {
    const r = this.#fittedResults();
    if (!r.several) return r.scores.score!;
    const refit = this.options.refit;
    if (typeof refit !== 'string') fail('ML_PARAM', 'with several scorers there is no best unless refit names a scorer, as in scikit-learn');
    return r.scores[refit]!;
  }
  /** The best parameters fitted on all the rows (with refit). */
  get bestEstimator(): Model {
    return this.#best ?? fail('ML_NOT_FITTED', 'no bestEstimator: fit with refit: true');
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator.getParams(),
      paramGrid: this.paramGrid,
      refit: this.options.refit ?? true,
      ...(this.options.returnTrainScore ? { returnTrainScore: true } : {}),
    };
  }

  /** Fits every candidate on every fold, then (with refit) the best on all of X. `groups` go to the splitter. */
  async fit(X: TableInput, y: Labels, options: { groups?: Groups } = {}): Promise<this> {
    // The candidates share each fold's rows: linear models work out XᵀX and Xᵀy once per fold.
    return withKernelCache(() => this.#fit(X, y, options.groups));
  }

  async #fit(X: TableInput, y: Labels, groups: Groups | undefined): Promise<this> {
    oneTarget(y);
    // Numbers become one Matrix up front, so each fold's rows are one matrix every candidate fits on.
    X = asMatrixIfNumeric(X);
    const n = toTable(X).rows;
    if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
    const candidates = parameterGrid(this.paramGrid);
    const { names, scorers, several } = scorersFor(this.estimator, this.options.scoring);
    const folds = foldsFor(this.estimator, this.options.cv, n, y, groups);
    const withTrain = this.options.returnTrainScore ?? false;
    // split[scorer][fold][candidate]
    const split = names.map(() => folds.map(() => new Float64Array(candidates.length)));
    const splitTrain = withTrain ? names.map(() => folds.map(() => new Float64Array(candidates.length))) : null;
    for (const [f, { train, test }] of folds.entries()) {
      const Xtr = rowsOf(X, train);
      const ytr = labelsOf(y, train);
      const Xte = rowsOf(X, test) as MatrixInput;
      const yte = labelsOf(y, test);
      for (const [c, params] of candidates.entries()) {
        const m = cloneModel(this.estimator, params) as Model & { fit(X: unknown, y: unknown): Promise<unknown> };
        await m.fit(Xtr, ytr);
        for (const [k, scorer] of scorers.entries()) split[k]![f]![c] = await scorer(m, Xte, yte);
        if (splitTrain) for (const [k, scorer] of scorers.entries()) splitTrain[k]![f]![c] = await scorer(m, Xtr as MatrixInput, ytr);
      }
    }
    const scores: Record<string, ScorerResults> = {};
    names.forEach((name, k) => {
      const t = summarize(split[k]!, candidates.length);
      const r: ScorerResults = { meanTestScore: t.mean, stdTestScore: t.std, rankTestScore: t.rank, splitTestScores: split[k]! };
      if (splitTrain) {
        const tr = summarize(splitTrain[k]!, candidates.length);
        Object.assign(r, { meanTrainScore: tr.mean, stdTrainScore: tr.std, splitTrainScores: splitTrain[k]! });
      }
      scores[name] = r;
    });
    this.#results = { params: candidates, scores, several };
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
    const scoring = this.options.scoring;
    // Scorers save by name; a function can't be saved, and a search scored by one loads without it.
    const savedScoring =
      typeof scoring === 'string' ? scoring
      : Array.isArray(scoring) ? [...scoring]
      : scoring && typeof scoring === 'object' && Object.values(scoring).every((v) => typeof v === 'string') ? { ...scoring }
      : undefined;
    const results = (x: ScorerResults) => ({
      meanTestScore: [...x.meanTestScore],
      stdTestScore: [...x.stdTestScore],
      rankTestScore: [...x.rankTestScore],
      ...(x.meanTrainScore ? { meanTrainScore: [...x.meanTrainScore], stdTrainScore: [...x.stdTrainScore!] } : {}),
    });
    return document(
      this.kind,
      {
        paramGrid: this.paramGrid as unknown as Record<string, unknown>,
        refit: this.options.refit ?? true,
        ...(savedScoring !== undefined ? { scoring: savedScoring } : {}),
        ...(this.options.returnTrainScore ? { returnTrainScore: true } : {}),
      },
      {
        estimator: this.estimator.toJSON(),
        ...(r
          ? {
              cvResults: r.several
                ? { params: r.params, scores: Object.fromEntries(Object.entries(r.scores).map(([k, v]) => [k, results(v)])) }
                : { params: r.params, ...results(r.scores.score!) },
            }
          : {}),
        ...(this.#best ? { bestEstimator: this.#best.toJSON() } : {}),
      },
    );
  }

  static fromJSON(doc: ModelDocument): GridSearchCV {
    const d = checkDocument(doc, 'GridSearchCV');
    const s = d.state ?? {};
    const g = new GridSearchCV(loadModel(s.estimator as ModelDocument), d.params.paramGrid as ParamGrid, {
      refit: d.params.refit as boolean | string,
      ...(d.params.scoring ? { scoring: d.params.scoring as ScorerName | Scorers } : {}),
      ...(d.params.returnTrainScore ? { returnTrainScore: true } : {}),
    });
    type Saved = { meanTestScore: number[]; stdTestScore: number[]; rankTestScore: number[]; meanTrainScore?: number[]; stdTrainScore?: number[] };
    const loaded = (x: Saved): ScorerResults => ({
      meanTestScore: Float64Array.from(x.meanTestScore),
      stdTestScore: Float64Array.from(x.stdTestScore),
      rankTestScore: Int32Array.from(x.rankTestScore),
      splitTestScores: [],
      ...(x.meanTrainScore ? { meanTrainScore: Float64Array.from(x.meanTrainScore), stdTrainScore: Float64Array.from(x.stdTrainScore!), splitTrainScores: [] } : {}),
    });
    const r = s.cvResults as ({ params: Record<string, unknown>[]; scores?: Record<string, Saved> } & Partial<Saved>) | undefined;
    if (r) {
      g.#results = r.scores
        ? { params: r.params, scores: Object.fromEntries(Object.entries(r.scores).map(([k, v]) => [k, loaded(v)])), several: true }
        : { params: r.params, scores: { score: loaded(r as Saved) }, several: false };
    }
    if (s.bestEstimator) g.#best = loadModel(s.bestEstimator as ModelDocument);
    return g;
  }
}

/** The names of scikit-learn scorers smaltml has. */
export const SCORER_NAMES = Object.keys(SCORERS) as ScorerName[];

registerLoader('GridSearchCV', (d) => GridSearchCV.fromJSON(d));
