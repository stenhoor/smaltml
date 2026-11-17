import { fail } from '../core/errors.js';
import { checkedMeans, cholesky, choleskySolve, xw } from '../core/kernels.js';
import { threadPool, type ThreadOptions } from '../parallel/index.js';
import { keptMeans, runHere, runOn, withKernelCache, withKernelCacheSync, type Solver } from './calls.js';
import { nonNegativeLeastSquares } from './nnls.js';
import { checkedWeights, type SampleWeightOptions } from './weights.js';
import { coefColumns, coefsMatrix, interceptOf, oneTargetOnly, planFit, savedTargets, type FitPlan, type Targets } from './targets.js';
import { leastSquares } from '../core/linalg.js';
import { assertFinite, Matrix, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { matMul, predictOnGpu, type DeviceOptions } from '../gpu/index.js';
import { checkColumns, checkDocument, document, notFitted, numbers, type ModelDocument, type Predictor } from '../core/model.js';

export interface LinearRegressionParams {
  /** Fit an intercept. With false, the line passes through the origin. Default true. */
  fitIntercept?: boolean;
  /** Hold every coefficient at 0 or more: non-negative least squares. Default false. */
  positive?: boolean;
}

interface State {
  /** One coefficient vector per target (one for a one-dimensional y). */
  coefs: Float64Array[];
  intercepts: Float64Array;
  /** Whether y had a column per target (scikit-learn's 2-D y), even one. */
  several: boolean;
  targetNames: readonly string[] | undefined;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Ordinary least squares: the coefficients w and intercept b that minimize the sum of (y − X·w − b)².
 * scikit-learn's `LinearRegression`. Solved by QR on centred data, on the CPU in 64-bit floats; with
 * `positive`, by non-negative least squares (Lawson and Hanson's active set). y can have a column per target
 * (scikit-learn's 2-D y): each is fitted on its own, sharing XᵀX, and read through coefs, intercepts and
 * predictTargets() (ADR 0037).
 */
export class LinearRegression implements Predictor {
  readonly kind = 'LinearRegression';
  readonly fitIntercept: boolean;
  readonly positive: boolean;
  #state: State | null = null;

  constructor(params: LinearRegressionParams = {}) {
    this.fitIntercept = params.fitIntercept ?? true;
    if (typeof this.fitIntercept !== 'boolean') fail('ML_PARAM', 'fitIntercept is true or false');
    this.positive = params.positive ?? false;
    if (typeof this.positive !== 'boolean') fail('ML_PARAM', 'positive is true or false');
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  /** One coefficient per column of X. For one target: with a y that has columns, read coefs. */
  get coef(): Float64Array {
    return this.#one('coef').coefs[0]!;
  }

  /** The intercept (0 with fitIntercept false). For one target: with a y that has columns, read intercepts. */
  get intercept(): number {
    return this.#one('intercept').intercepts[0]!;
  }

  /** The coefficients, a row per target (one row for a one-dimensional y) and a column per feature. */
  get coefs(): Matrix {
    const s = this.#fitted();
    return coefsMatrix(s.coefs, s.nFeaturesIn, s.featureNamesIn);
  }

  /** One intercept per target. */
  get intercepts(): Float64Array {
    return this.#fitted().intercepts.slice();
  }

  /** y's column names, when it had them: the names of predictTargets()'s columns. */
  get targetNames(): readonly string[] | undefined {
    return this.#fitted().targetNames;
  }

  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }

  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): LinearRegressionParams {
    return { fitIntercept: this.fitIntercept, ...(this.positive ? { positive: true } : {}) };
  }

  /**
   * Fits the model. With `threads` (default `'auto'`), the heavy products run on several cores where shared memory
   * allows and X is large enough, with the same result as fitSync() to the last bit. With `sampleWeight`, each row's
   * squared error counts that many times. y is one target, or a column per target.
   */
  async fit(X: MatrixInput, y: Targets, options: ThreadOptions & SampleWeightOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    const pool = await threadPool(options.threads, m.rows * m.cols);
    if (!pool) return this.fitSync(m, y, options);
    const plan = this.#plan(m, y, options.sampleWeight);
    const coefs = await withKernelCache(async () => {
      const out: Float64Array[] = [];
      for (const t of plan.targets) {
        out.push(
          m.cols === 0 ? new Float64Array(0)
          : this.positive ? (await runOn(pool, nonNegativeLeastSquares(plan.design, plan.kernelMean, t.b, 0))).coef
          : ((await runOn(pool, seminormal(plan.design, plan.kernelMean, t.b))) ?? householder(plan.design, plan.kernelMean, t.b)),
        );
      }
      return out;
    }, plan.targets.length > 1);
    return this.#finish(m, plan, coefs);
  }

  fitSync(X: MatrixInput, y: Targets, options: SampleWeightOptions = {}): this {
    const m = Matrix.from(X);
    const plan = this.#plan(m, y, options.sampleWeight);
    const coefs = withKernelCacheSync(() =>
      plan.targets.map((t) =>
        m.cols === 0 ? new Float64Array(0)
        : this.positive ? runHere(nonNegativeLeastSquares(plan.design, plan.kernelMean, t.b, 0)).coef
        : (runHere(seminormal(plan.design, plan.kernelMean, t.b)) ?? householder(plan.design, plan.kernelMean, t.b)),
      ),
      plan.targets.length > 1,
    );
    return this.#finish(m, plan, coefs);
  }

  #plan(m: Matrix, y: Targets, sampleWeight: VectorInput | undefined): FitPlan {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const weights = checkedWeights(sampleWeight, m.rows);
    return planFit(m, y, weights, this.fitIntercept, () => keptMeans(m, () => checkedMeans(m, assertFinite)));
  }

  #finish(m: Matrix, plan: FitPlan, coefs: Float64Array[]): this {
    const intercepts = Float64Array.from(coefs, (c, t) => interceptOf(plan.xMean, plan.targets[t]!.yMean, c));
    this.#state = { coefs, intercepts, several: plan.several, targetNames: plan.targetNames, nFeaturesIn: m.cols, featureNamesIn: m.names };
    return this;
  }

  /** X·w + b, one prediction per row; on the GPU only with `device: 'gpu'` (32-bit floats there). For one target. */
  async predict(X: MatrixInput, options: DeviceOptions = {}): Promise<Float64Array> {
    const s = this.#one('predict()');
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    // On the CPU, exactly predictSync()'s result.
    if (!(await predictOnGpu(options.device))) return this.predictSync(m);
    const product = await matMul(m, new Matrix(s.coefs[0]!.length, 1, s.coefs[0]!), { device: 'gpu' });
    const out = product.data;
    for (let i = 0; i < out.length; i++) out[i]! += s.intercepts[0]!;
    return out;
  }

  /** X·w + b, one prediction per row. For one target: with a y that has columns, use predictTargetsSync(). */
  predictSync(X: MatrixInput): Float64Array {
    const s = this.#one('predictSync()');
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return xw(m, s.coefs[0]!, s.intercepts[0]!);
  }

  /** A column per target, X·wₜ + bₜ, with y's column names; on the GPU only with `device: 'gpu'`. */
  async predictTargets(X: MatrixInput, options: DeviceOptions = {}): Promise<Matrix> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    if (!(await predictOnGpu(options.device))) return this.predictTargetsSync(m);
    const out = await matMul(m, coefColumns(s.coefs, s.nFeaturesIn), { device: 'gpu' });
    s.intercepts.forEach((b, t) => {
      const col = out.column(t);
      for (let i = 0; i < col.length; i++) col[i]! += b;
    });
    return new Matrix(out.rows, out.cols, out.data, s.targetNames);
  }

  /** A column per target, X·wₜ + bₜ, with y's column names. */
  predictTargetsSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return Matrix.fromColumns(s.coefs.map((c, t) => xw(m, c, s.intercepts[t]!)), s.targetNames);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      ...(s.several
        ? { coefs: s.coefs.map(numbers), intercepts: numbers(s.intercepts), ...(s.targetNames ? { targetNames: [...s.targetNames] } : {}) }
        : { coef: numbers(s.coefs[0]!), intercept: s.intercepts[0]! }),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): LinearRegression {
    const d = checkDocument(doc, 'LinearRegression');
    const model = new LinearRegression(d.params as LinearRegressionParams);
    const s = d.state;
    if (s) model.#state = { ...savedTargets(s), nFeaturesIn: Number(s.nFeaturesIn), featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined };
    return model;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }

  /** The state, for an accessor of one target's results. */
  #one(what: string): State {
    const s = this.#fitted();
    if (s.several) oneTargetOnly(this.kind, what, s.coefs.length);
    return s;
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
