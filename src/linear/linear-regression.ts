import { fail } from '../core/errors.js';
import { checkedMeans, cholesky, choleskySolve, xw } from '../core/kernels.js';
import { threadPool, type ThreadOptions } from '../parallel/index.js';
import { keptMeans, runHere, runOn, type Solver } from './calls.js';
import { leastSquares } from '../core/linalg.js';
import { assertFinite, assertFiniteVector, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { matMul, predictOnGpu, type DeviceOptions } from '../gpu/index.js';
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

  /**
   * Fits the model. With `threads` (default `'auto'`), the heavy products run on several cores where shared memory
   * allows and X is large enough, with the same result as fitSync() to the last bit.
   */
  async fit(X: MatrixInput, y: VectorInput, options: ThreadOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    const pool = await threadPool(options.threads, m.rows * m.cols);
    if (!pool) return this.fitSync(m, y);
    const job = this.#prepare(m, y);
    const coef = job.p === 0 ? new Float64Array(0) : ((await runOn(pool, seminormal(m, job.xMean, job.b))) ?? householder(m, job.xMean, job.b));
    return this.#finish(job, coef);
  }

  fitSync(X: MatrixInput, y: VectorInput): this {
    const m = Matrix.from(X);
    const job = this.#prepare(m, y);
    const coef = job.p === 0 ? new Float64Array(0) : (runHere(seminormal(m, job.xMean, job.b)) ?? householder(m, job.xMean, job.b));
    return this.#finish(job, coef);
  }

  #prepare(m: Matrix, y: VectorInput) {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    // With an intercept, the column means check X on the way; without one, check it here.
    if (!this.fitIntercept) assertFinite(m);
    const target = toVector(y, m.rows);
    assertFiniteVector(target);
    const n = m.rows;
    const p = m.cols;
    let xMean: Float64Array | undefined;
    let yMean = 0;
    let b = target;
    if (this.fitIntercept) {
      // Centre: the intercept is then ȳ − x̄·w. X isn't copied; the kernels subtract its means as they read it.
      xMean = keptMeans(m, () => checkedMeans(m, assertFinite));
      for (let i = 0; i < n; i++) yMean += target[i]!;
      yMean /= n;
      b = target.map((v) => v - yMean);
    }
    return { m, p, xMean, yMean, b };
  }

  #finish(job: { m: Matrix; p: number; xMean: Float64Array | undefined; yMean: number }, coef: Float64Array): this {
    const { m, p, xMean, yMean } = job;
    let intercept = 0;
    if (xMean) {
      intercept = yMean;
      for (let j = 0; j < p; j++) intercept -= xMean[j]! * coef[j]!;
    }
    this.#state = { coef, intercept, nFeaturesIn: p, featureNamesIn: m.names };
    return this;
  }

  /** X·w + b, one prediction per row; on the GPU only with `device: 'gpu'` (32-bit floats there). */
  async predict(X: MatrixInput, options: DeviceOptions = {}): Promise<Float64Array> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    // On the CPU, exactly predictSync()'s result.
    if (!(await predictOnGpu(options.device))) return this.predictSync(m);
    const product = await matMul(m, new Matrix(s.coef.length, 1, s.coef), { device: 'gpu' });
    const out = product.data;
    for (let i = 0; i < out.length; i++) out[i]! += s.intercept;
    return out;
  }

  /** X·w + b, one prediction per row. */
  predictSync(X: MatrixInput): Float64Array {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return xw(m, s.coef, s.intercept);
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

/**
 * Least squares by the corrected semi-normal equations (Björck): Cholesky of XᵀX, then up to two steps of iterative
 * refinement on the residual. Fast, as XᵀX is one blocked pass over X; as accurate as QR when the columns aren't
 * nearly dependent. Returns null when they are (Cholesky fails, or the refinement doesn't settle), and the caller
 * uses QR instead.
 */
function* seminormal(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Float64Array | null> {
  const p = X.cols;
  if (X.rows < p) return null;
  const L = cholesky(yield { k: 'gram', X, mean }, p);
  if (!L) return null;
  const w = choleskySolve(L, p, yield { k: 'xtv', X, v: y, mean });
  for (let step = 0; step < 3; step++) {
    const fitted: Float64Array = yield { k: 'xw', X, w: w.slice(), b: 0, mean };
    const r = new Float64Array(X.rows);
    for (let i = 0; i < r.length; i++) r[i] = y[i]! - fitted[i]!;
    const dw = choleskySolve(L, p, yield { k: 'xtv', X, v: r, mean });
    let dn = 0;
    let wn = 0;
    for (let j = 0; j < p; j++) {
      w[j]! += dw[j]!;
      dn = Math.max(dn, Math.abs(dw[j]!));
      wn = Math.max(wn, Math.abs(w[j]!));
    }
    // Settled: the correction is at rounding level.
    if (dn <= 1e-12 * wn || wn === 0) return w;
  }
  return null;
}

/** Least squares by Householder QR on a centred copy: slower, and right for nearly dependent columns too. */
function householder(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Float64Array {
  if (!mean) return leastSquares(X, y);
  const n = X.rows;
  const data = new Float64Array(X.data.length);
  for (let j = 0; j < X.cols; j++) {
    const mu = mean[j]!;
    for (let i = 0; i < n; i++) data[j * n + i] = X.data[j * n + i]! - mu;
  }
  return leastSquares(new Matrix(n, X.cols, data, X.names), y);
}
