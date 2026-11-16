/**
 * Several targets at once: one regressor per column of Y. scikit-learn's `multioutput`.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { checkDocument, document, isClassifier, isPredictor, notFitted, unfitDocument, type Model, type ModelDocument } from '../core/model.js';
import { loadModel, registerLoader } from '../io/index.js';

/** A regressor the wrapper fits once per target: a model whose predict() gives one number per row. */
type Regressor = Model & {
  fit(X: MatrixInput, y: ArrayLike<number>, options?: object): Promise<unknown>;
  fitSync(X: MatrixInput, y: ArrayLike<number>, options?: object): unknown;
  predict(X: MatrixInput, options?: object): Promise<Float64Array>;
  predictSync(X: MatrixInput): Float64Array;
};

export interface MultiOutputFitOptions {
  /** Passed to each target's fit (as for that model): `threads`, `device`. */
  threads?: number | 'auto';
  /** One weight per row, passed to each target's fit; the model must take `sampleWeight`. */
  sampleWeight?: ArrayLike<number>;
}

interface State {
  estimators: Regressor[];
  targetNames: readonly string[] | undefined;
}

/**
 * Fits one copy of a regressor to each column of Y, and predicts a column for each. scikit-learn's
 * `MultiOutputRegressor`. For LinearRegression, Ridge, Lasso and ElasticNet, whose targets don't interact, this is
 * what scikit-learn fits for a two-dimensional y as well.
 */
export class MultiOutputRegressor implements Model {
  readonly kind = 'MultiOutputRegressor';
  /** The regressor each target gets a fresh copy of; it isn't fitted itself. */
  readonly estimator: Model;
  #state: State | null = null;

  constructor(estimator: Model) {
    if (!estimator || typeof estimator.toJSON !== 'function') fail('ML_PARAM', 'MultiOutputRegressor needs a model to fit per target');
    if (!isPredictor(estimator) || isClassifier(estimator)) {
      fail('ML_PARAM', `MultiOutputRegressor fits a regressor per target; ${estimator.kind} isn't one`);
    }
    this.estimator = estimator;
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  /** The fitted regressors, one per column of Y, in its order. */
  get estimators(): readonly Model[] {
    return this.#fitted().estimators;
  }

  /** Y's column names, when it had them: the names of predict()'s columns. */
  get targetNames(): readonly string[] | undefined {
    return this.#fitted().targetNames;
  }

  get nFeaturesIn(): number {
    return (this.#fitted().estimators[0] as unknown as { nFeaturesIn: number }).nFeaturesIn;
  }

  getParams(): { estimator: Model } {
    return { estimator: this.estimator };
  }

  /** Fits a fresh copy of the estimator to each column of Y, in turn. */
  async fit(X: MatrixInput, Y: MatrixInput, options: MultiOutputFitOptions = {}): Promise<this> {
    const { m, targets } = this.#prepare(X, Y);
    const estimators: Regressor[] = [];
    for (let k = 0; k < targets.cols; k++) {
      const e = this.#copy();
      await e.fit(m, targets.column(k).slice(), passed(options));
      estimators.push(e);
    }
    this.#state = { estimators, targetNames: targets.names };
    return this;
  }

  fitSync(X: MatrixInput, Y: MatrixInput, options: Pick<MultiOutputFitOptions, 'sampleWeight'> = {}): this {
    const { m, targets } = this.#prepare(X, Y);
    const estimators: Regressor[] = [];
    for (let k = 0; k < targets.cols; k++) {
      const e = this.#copy();
      e.fitSync(m, targets.column(k).slice(), passed(options));
      estimators.push(e);
    }
    this.#state = { estimators, targetNames: targets.names };
    return this;
  }

  /** A column per target, in Y's order and with its names. `device` goes to each target's predict(). */
  async predict(X: MatrixInput, options: { device?: 'auto' | 'cpu' | 'gpu' } = {}): Promise<Matrix> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    const columns: Float64Array[] = [];
    for (const e of s.estimators) columns.push(await e.predict(m, options));
    return Matrix.fromColumns(columns, s.targetNames);
  }

  predictSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    return Matrix.fromColumns(s.estimators.map((e) => e.predictSync(m)), s.targetNames);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { estimator: unfitDocument(this.estimator.toJSON()) }, s && {
      estimators: s.estimators.map((e) => e.toJSON()),
      ...(s.targetNames ? { targetNames: [...s.targetNames] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): MultiOutputRegressor {
    const d = checkDocument(doc, 'MultiOutputRegressor');
    const model = new MultiOutputRegressor(loadModel(d.params.estimator as ModelDocument));
    const s = d.state;
    if (s) {
      if (!Array.isArray(s.estimators) || s.estimators.length === 0) fail('ML_FORMAT', 'a fitted MultiOutputRegressor document needs its estimators');
      model.#state = {
        estimators: s.estimators.map((e) => loadModel(e as ModelDocument) as Regressor),
        targetNames: Array.isArray(s.targetNames) ? (s.targetNames as string[]) : undefined,
      };
    }
    return model;
  }

  #prepare(X: MatrixInput, Y: MatrixInput) {
    const m = Matrix.from(X);
    let targets: Matrix;
    try {
      targets = Matrix.from(Y);
    } catch {
      return fail('ML_SHAPE', 'Y needs two dimensions, a column per target: rows (an array of arrays) or named columns');
    }
    if (targets.rows !== m.rows) fail('ML_SHAPE', `Y has ${targets.rows} rows; X has ${m.rows}`);
    if (targets.cols === 0) fail('ML_SHAPE', 'Y has no columns: it needs a column per target');
    return { m, targets };
  }

  /** A fresh, unfitted copy of the estimator, from its saved form. */
  #copy(): Regressor {
    return loadModel(unfitDocument(this.estimator.toJSON())) as Regressor;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}

/** The options a target's fit takes: only those given, so a model that takes no weights isn't handed any. */
function passed(options: MultiOutputFitOptions): object {
  const out: Record<string, unknown> = {};
  if (options.threads !== undefined) out.threads = options.threads;
  if (options.sampleWeight !== undefined) out.sampleWeight = options.sampleWeight;
  return out;
}

registerLoader('MultiOutputRegressor', (d) => MultiOutputRegressor.fromJSON(d));
