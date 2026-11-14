import { fail } from '../core/errors.js';
import { leastSquares, matVec } from '../core/linalg.js';
import { assertFinite, assertFiniteVector, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, type ModelDocument, type Predictor } from '../core/model.js';

export interface LinearRegressionParams {
  /** Fit an intercept. With false, the line passes through the origin. Default true. */
  fitIntercept?: boolean;
}

interface State {
  coef: Float64Array;
  intercept: number;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Ordinary least squares: the coefficients w and intercept b that minimize the sum of (y − X·w − b)².
 * scikit-learn's `LinearRegression`, for one target. Solved by QR on centred data, on the CPU in 64-bit floats.
 */
export class LinearRegression implements Predictor {
  readonly kind = 'LinearRegression';
  readonly fitIntercept: boolean;
  #state: State | null = null;

  constructor(params: LinearRegressionParams = {}) {
    this.fitIntercept = params.fitIntercept ?? true;
    if (typeof this.fitIntercept !== 'boolean') fail('ML_PARAM', 'fitIntercept is true or false');
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  /** One coefficient per column of X. */
  get coef(): Float64Array {
    return this.#fitted().coef;
  }

  /** The intercept (0 with fitIntercept false). */
  get intercept(): number {
    return this.#fitted().intercept;
  }

  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }

  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): LinearRegressionParams {
    return { fitIntercept: this.fitIntercept };
  }

  async fit(X: MatrixInput, y: VectorInput): Promise<this> {
    return this.fitSync(X, y);
  }

  fitSync(X: MatrixInput, y: VectorInput): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    assertFinite(m);
    const target = toVector(y, m.rows);
    assertFiniteVector(target);
    const n = m.rows;
    const p = m.cols;
    let A = m;
    let b = target;
    const xMean = new Float64Array(p);
    let yMean = 0;
    if (this.fitIntercept) {
      // Centre X and y; the intercept is then ȳ − x̄·w.
      const data = new Float64Array(m.data.length);
      for (let j = 0; j < p; j++) {
        const col = m.column(j);
        let s = 0;
        for (let i = 0; i < n; i++) s += col[i]!;
        const mu = s / n;
        xMean[j] = mu;
        for (let i = 0; i < n; i++) data[j * n + i] = col[i]! - mu;
      }
      A = new Matrix(n, p, data, m.names);
      for (let i = 0; i < n; i++) yMean += target[i]!;
      yMean /= n;
      b = target.map((v) => v - yMean);
    }
    const coef = p === 0 ? new Float64Array(0) : leastSquares(A, b);
    let intercept = 0;
    if (this.fitIntercept) {
      intercept = yMean;
      for (let j = 0; j < p; j++) intercept -= xMean[j]! * coef[j]!;
    }
    this.#state = { coef, intercept, nFeaturesIn: p, featureNamesIn: m.names };
    return this;
  }

  async predict(X: MatrixInput): Promise<Float64Array> {
    return this.predictSync(X);
  }

  /** X·w + b, one prediction per row. */
  predictSync(X: MatrixInput): Float64Array {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return matVec(m, s.coef, s.intercept);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      coef: numbers(s.coef),
      intercept: s.intercept,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): LinearRegression {
    const d = checkDocument(doc, 'LinearRegression');
    const model = new LinearRegression(d.params as LinearRegressionParams);
    const s = d.state;
    if (s) {
      model.#state = {
        coef: floats(s.coef, 'coef'),
        intercept: Number(s.intercept),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return model;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}
