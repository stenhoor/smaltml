import { fail, warn } from '../core/errors.js';
import { checkedMeans, cholesky, choleskySolve, xw } from '../core/kernels.js';
import { threadPool, type ThreadOptions } from '../parallel/index.js';
import { keptMeans, runHere, runOn, type Solver } from './calls.js';
import { assertFinite, assertFiniteVector, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, type ModelDocument, type Predictor } from '../core/model.js';
import { matMul, predictOnGpu, type DeviceOptions } from '../gpu/index.js';

interface State {
  coef: Float64Array;
  intercept: number;
  nIter: number;
  /** Whether the solver converged; null for a model saved before smaltml recorded it (format version 1). */
  converged: boolean | null;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/** A solver's answer. Coordinate descent also gives its last duality gap and the tolerance it had to reach. */
interface Solution {
  coef: Float64Array;
  nIter: number;
  converged: boolean;
  gap?: number;
  gapTol?: number;
}

/**
 * The column means and y's mean (when there is an intercept), and y centred. X itself isn't copied: the kernels
 * subtract the means as they read it.
 */
function centring(m: Matrix, y: Float64Array, intercept: boolean) {
  // The column means check X on the way; without an intercept, check it here.
  const xMean = intercept ? keptMeans(m, () => checkedMeans(m, assertFinite)) : (assertFinite(m), undefined);
  let yMean = 0;
  if (intercept) {
    for (let i = 0; i < y.length; i++) yMean += y[i]!;
    yMean /= y.length;
  }
  return { xMean, yMean, y: intercept ? y.map((v) => v - yMean) : y };
}

/** A linear model with a penalty: what Ridge, Lasso and ElasticNet share. */
abstract class PenalizedLinear implements Predictor {
  abstract readonly kind: string;
  readonly alpha: number;
  readonly fitIntercept: boolean;
  #state: State | null = null;

  constructor(alpha: number, fitIntercept: boolean) {
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    if (!(alpha >= 0) || !Number.isFinite(alpha)) fail('ML_PARAM', `alpha must be 0 or more, not ${alpha}`);
    if (typeof fitIntercept !== 'boolean') fail('ML_PARAM', 'fitIntercept is true or false');
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
  /** The solver's iterations (0 for a closed-form solution). */
  get nIter(): number {
    return this.#fitted().nIter;
  }
  /**
   * Whether the solver converged within `maxIter` (always true for Ridge, which solves in closed form). Where it didn't,
   * fit() warned (`ML_CONVERGENCE`), as scikit-learn does. null for a model saved by smaltml before it recorded this.
   */
  get converged(): boolean | null {
    return this.#fitted().converged;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  abstract getParams(): object;

  /** The coefficients for X less `mean` (when there is an intercept) and centred y, and the iterations it took. */
  protected abstract solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution>;

  /**
   * Fits the model. With `threads` (default `'auto'`), the Gram matrix and Xᵀy are computed on several cores where
   * shared memory allows and X is large enough, with the same result as fitSync() to the last bit.
   */
  async fit(X: MatrixInput, y: VectorInput, options: ThreadOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    const pool = await threadPool(options.threads, m.rows * m.cols);
    if (!pool) return this.fitSync(m, y);
    const c = this.#prepare(m, y);
    return this.#finish(m, c, m.cols === 0 ? { coef: new Float64Array(0), nIter: 0, converged: true } : await runOn(pool, this.solve(m, c.xMean, c.y)));
  }

  fitSync(X: MatrixInput, y: VectorInput): this {
    const m = Matrix.from(X);
    const c = this.#prepare(m, y);
    return this.#finish(m, c, m.cols === 0 ? { coef: new Float64Array(0), nIter: 0, converged: true } : runHere(this.solve(m, c.xMean, c.y)));
  }

  #prepare(m: Matrix, y: VectorInput) {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const target = toVector(y, m.rows);
    assertFiniteVector(target);
    return centring(m, target, this.fitIntercept);
  }

  #finish(m: Matrix, c: { xMean: Float64Array | undefined; yMean: number }, { coef, nIter, converged, gap, gapTol }: Solution): this {
    let intercept = 0;
    if (c.xMean) {
      intercept = c.yMean;
      for (let j = 0; j < m.cols; j++) intercept -= c.xMean[j]! * coef[j]!;
    }
    this.#state = { coef, intercept, nIter, converged, nFeaturesIn: m.cols, featureNamesIn: m.names };
    if (!converged) {
      const why = gap === undefined || gapTol === undefined ? '' : `: the duality gap is ${gap.toPrecision(3)}, above the tolerance ${gapTol.toPrecision(3)}`;
      warn('ML_CONVERGENCE', `${this.kind} didn't converge in ${nIter} iterations (maxIter)${why}. Raise maxIter, scale the columns (StandardScaler), or raise alpha`);
    }
    return this;
  }

  /** X·w + b, one prediction per row; on the GPU only with `device: 'gpu'` (32-bit floats there). */
  async predict(X: MatrixInput, options: DeviceOptions = {}): Promise<Float64Array> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    if (!(await predictOnGpu(options.device))) return this.predictSync(m);
    const out = (await matMul(m, new Matrix(s.coef.length, 1, s.coef), { device: 'gpu' })).data;
    for (let i = 0; i < out.length; i++) out[i]! += s.intercept;
    return out;
  }

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
      nIter: s.nIter,
      converged: s.converged,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  protected restore(s: Record<string, unknown> | null): this {
    if (s) {
      this.#state = {
        coef: floats(s.coef, 'coef'),
        intercept: Number(s.intercept),
        nIter: Number(s.nIter ?? 0),
        converged: typeof s.converged === 'boolean' ? s.converged : null,
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return this;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}

export interface RidgeParams {
  /** The L2 penalty's strength. Default 1. */
  alpha?: number;
  /** Fit an intercept (not penalized). Default true. */
  fitIntercept?: boolean;
}

/**
 * Linear regression with an L2 penalty: minimizes ‖y − X·w − b‖² + alpha · ‖w‖². scikit-learn's `Ridge`, for one
 * target. Solved in closed form, (XᵀX + alpha·I) w = Xᵀy on centred data, by Cholesky in 64-bit floats.
 */
export class Ridge extends PenalizedLinear {
  readonly kind = 'Ridge';

  constructor(params: RidgeParams = {}) {
    super(params.alpha ?? 1, params.fitIntercept ?? true);
  }

  getParams(): Required<RidgeParams> {
    return { alpha: this.alpha, fitIntercept: this.fitIntercept };
  }

  protected *solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution> {
    const p = X.cols;
    // (XᵀX + alpha·I) w = Xᵀy, on X less its means.
    const A: Float64Array = yield { k: 'gram', X, mean };
    for (let j = 0; j < p; j++) A[j * p + j]! += this.alpha;
    const L = cholesky(A, p);
    if (!L) fail('ML_SINGULAR', "the columns don't determine one answer (one is a combination of others): use alpha > 0");
    return { coef: choleskySolve(L, p, yield { k: 'xtv', X, v: y, mean }), nIter: 0, converged: true };
  }

  static fromJSON(doc: ModelDocument): Ridge {
    const d = checkDocument(doc, 'Ridge');
    return new Ridge(d.params as RidgeParams).restore(d.state);
  }
}

export interface ElasticNetParams {
  /** The penalty's strength. Default 1. */
  alpha?: number;
  /** The L1 share of the penalty, from 0 (all L2) to 1 (all L1, the lasso). Default 0.5. */
  l1Ratio?: number;
  /** Fit an intercept (not penalized). Default true. */
  fitIntercept?: boolean;
  /** The most passes over the coefficients. Default 1000. */
  maxIter?: number;
  /** The stopping tolerance on the coefficient updates and the duality gap. Default 1e-4. */
  tol?: number;
}

/**
 * Linear regression with an elastic-net penalty: minimizes ‖y − X·w − b‖² / (2n) + alpha · l1Ratio · ‖w‖₁ +
 * alpha · (1 − l1Ratio) / 2 · ‖w‖². scikit-learn's `ElasticNet`, for one target, by cyclic coordinate descent with
 * its documented stopping rule: when the largest update is at most tol times the largest coefficient, and the
 * duality gap is at most tol · ‖y‖² / n.
 */
export class ElasticNet extends PenalizedLinear {
  readonly kind: string = 'ElasticNet';
  readonly l1Ratio: number;
  readonly maxIter: number;
  readonly tol: number;

  constructor(params: ElasticNetParams = {}) {
    super(params.alpha ?? 1, params.fitIntercept ?? true);
    this.l1Ratio = params.l1Ratio ?? 0.5;
    this.maxIter = params.maxIter ?? 1000;
    this.tol = params.tol ?? 1e-4;
    if (!(this.l1Ratio >= 0 && this.l1Ratio <= 1)) fail('ML_PARAM', `l1Ratio is from 0 to 1, not ${this.l1Ratio}`);
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) fail('ML_PARAM', `maxIter must be a whole number of at least 1, not ${this.maxIter}`);
    if (!(this.tol >= 0)) fail('ML_PARAM', `tol must be 0 or more, not ${this.tol}`);
  }

  getParams(): ElasticNetParams {
    return { alpha: this.alpha, l1Ratio: this.l1Ratio, fitIntercept: this.fitIntercept, maxIter: this.maxIter, tol: this.tol };
  }

  protected *solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution> {
    const l1 = this.alpha * this.l1Ratio * X.rows;
    const l2 = this.alpha * (1 - this.l1Ratio) * X.rows;
    // With more rows than columns, descend on the Gram matrix: a pass costs p², not a read of X. (Its blocked kernel
    // costs about three passes over X, and descent on X needs four or more with its gap check: Gram first measured faster.)
    if (X.rows >= X.cols && X.cols <= GRAM_MAX) {
      const G: Float64Array = yield { k: 'gram', X, mean };
      const q: Float64Array = yield { k: 'xtv', X, v: y, mean };
      return gramDescent(G, q, dotSelf(y), X.cols, l1, l2, this.maxIter, this.tol);
    }
    return coordinateDescent(X, mean, y, l1, l2, this.maxIter, this.tol);
  }

  static fromJSON(doc: ModelDocument): ElasticNet {
    const d = checkDocument(doc, 'ElasticNet');
    return new ElasticNet(d.params as ElasticNetParams).restore(d.state);
  }
}

export type LassoParams = Omit<ElasticNetParams, 'l1Ratio'>;

/**
 * Linear regression with an L1 penalty, which sets some coefficients to exactly 0: minimizes
 * ‖y − X·w − b‖² / (2n) + alpha · ‖w‖₁. scikit-learn's `Lasso`: ElasticNet with l1Ratio 1.
 */
export class Lasso extends ElasticNet {
  override readonly kind: string = 'Lasso';

  constructor(params: LassoParams = {}) {
    super({ ...params, l1Ratio: 1 });
  }

  override getParams(): LassoParams {
    const { l1Ratio: _, ...rest } = super.getParams();
    return rest;
  }

  static override fromJSON(doc: ModelDocument): Lasso {
    const d = checkDocument(doc, 'Lasso');
    return new Lasso(d.params as LassoParams).restore(d.state);
  }
}

/**
 * Cyclic coordinate descent for ½‖y − Xw‖² + l1 ‖w‖₁ + (l2 / 2) ‖w‖² (the elastic-net objective times n), with the
 * duality gap of the lasso on X augmented by √l2 · I, which this objective is. X less `mean` is the design: column j is
 * read as xⱼ − μⱼ, with no copy of X, which matters on wide data.
 */
function coordinateDescent(X: Matrix, mean: Float64Array | undefined, y: Float64Array, l1: number, l2: number, maxIter: number, tol: number) {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  const w = new Float64Array(p);
  const r = y.slice();
  const norms = new Float64Array(p);
  // sums[j] = Σᵢ (xᵢⱼ − μⱼ): about 0, kept so Σr stays exact as r changes.
  const sums = new Float64Array(p);
  const mu = (j: number) => (mean ? mean[j]! : 0);
  for (let j = 0; j < p; j++) {
    const o = j * n;
    const m = mu(j);
    let s = 0;
    let t = 0;
    for (let i = 0; i < n; i++) {
      const v = d[o + i]! - m;
      s += v * v;
      t += v;
    }
    norms[j] = s;
    sums[j] = t;
  }
  let rsum = 0;
  for (let i = 0; i < n; i++) rsum += r[i]!;
  const gapTol = tol * dotSelf(y);
  let iter = 0;
  let converged = false;
  let gap = Infinity;
  for (; iter < maxIter; iter++) {
    let maxStep = 0;
    let maxW = 0;
    for (let j = 0; j < p; j++) {
      if (norms[j] === 0) continue;
      const o = j * n;
      const m = mu(j);
      const old = w[j]!;
      // ρ = (xⱼ − μⱼ)ᵀ(r + (xⱼ − μⱼ) wⱼ): the fit of column j to the residual without it.
      let rho = 0;
      for (let i = 0; i < n; i++) rho += d[o + i]! * r[i]!;
      rho += norms[j]! * old - m * rsum;
      const next = Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0) / (norms[j]! + l2);
      if (next !== old) {
        const step = next - old;
        const shift = m * step;
        for (let i = 0; i < n; i++) r[i]! -= d[o + i]! * step - shift;
        rsum -= sums[j]! * step;
        w[j] = next;
      }
      maxStep = Math.max(maxStep, Math.abs(next - old));
      maxW = Math.max(maxW, Math.abs(next));
    }
    if (maxW === 0 || maxStep / maxW <= tol || iter === maxIter - 1) {
      gap = dualityGap(X, mean, y, w, r, rsum, l1, l2);
      if (gap <= gapTol) {
        converged = true;
        iter++;
        break;
      }
    }
  }
  return { coef: w, nIter: Math.min(iter, maxIter), converged, gap, gapTol };
}

/** The duality gap of the elastic net, as the lasso on [X; √l2 I] with target [y; 0]; X less `mean`, as above. */
function dualityGap(X: Matrix, mean: Float64Array | undefined, y: Float64Array, w: Float64Array, r: Float64Array, rsum: number, l1: number, l2: number): number {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  let rr = 0;
  let ry = 0;
  for (let i = 0; i < n; i++) {
    rr += r[i]! * r[i]!;
    ry += r[i]! * y[i]!;
  }
  let ww = 0;
  let w1 = 0;
  let dualNorm = 0;
  let xtr2 = 0;
  for (let j = 0; j < p; j++) {
    const o = j * n;
    let g = 0;
    for (let i = 0; i < n; i++) g += d[o + i]! * r[i]!;
    if (mean) g -= mean[j]! * rsum;
    dualNorm = Math.max(dualNorm, Math.abs(g - l2 * w[j]!));
    xtr2 += g * g;
    ww += w[j]! * w[j]!;
    w1 += Math.abs(w[j]!);
  }
  if (l1 === 0 && l2 > 0) return ridgeGap(rr, ry, ww, xtr2, l2);
  // The dual point θ = s · [r; −√l2 w], scaled into the dual's feasible set ‖X̃ᵀθ‖∞ ≤ l1.
  const s = dualNorm > l1 ? l1 / dualNorm : 1;
  const augmented = rr + l2 * ww;
  const primal = 0.5 * augmented + l1 * w1;
  const dual = s * ry - 0.5 * s * s * augmented;
  return primal - dual;
}

/**
 * The duality gap with no L1 term (l1 = 0, l2 > 0), where the scaling above gives s = 0 and a gap that never closes.
 * With l2 > 0 the dual, θᵀy − ½‖θ‖² − Σⱼ (|xⱼᵀθ| − l1)₊² / (2·l2), needs no feasible set, so θ = r gives the gap:
 * ½‖r‖² + (l2/2)‖w‖² − (rᵀy − ½‖r‖² − ‖Xᵀr‖² / (2·l2)). It is 0 at the optimum, where Xᵀr = l2·w.
 */
function ridgeGap(rr: number, ry: number, ww: number, xtr2: number, l2: number): number {
  return 0.5 * rr + 0.5 * l2 * ww - (ry - 0.5 * rr - xtr2 / (2 * l2));
}

/** Above this many columns, the p × p Gram matrix is too big to be worth it, and descent reads X instead. */
const GRAM_MAX = 2048;

function dotSelf(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return s;
}

/**
 * coordinateDescent() on the Gram matrix G = XᵀX, with q = Xᵀy and yy = yᵀy: the same updates in the same order,
 * with Xᵀr kept as c = q − G w, and ‖r‖² and rᵀy from G, q and yy for the duality gap.
 */
function gramDescent(G: Float64Array, q: Float64Array, yy: number, p: number, l1: number, l2: number, maxIter: number, tol: number) {
  const w = new Float64Array(p);
  const c = q.slice();
  const gapTol = tol * yy;
  let iter = 0;
  let converged = false;
  let gap = Infinity;
  for (; iter < maxIter; iter++) {
    let maxStep = 0;
    let maxW = 0;
    for (let j = 0; j < p; j++) {
      const gjj = G[j * p + j]!;
      if (gjj === 0) continue;
      const old = w[j]!;
      const rho = c[j]! + gjj * old;
      const next = Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0) / (gjj + l2);
      if (next !== old) {
        const d = next - old;
        const row = j * p;
        for (let k = 0; k < p; k++) c[k]! -= G[row + k]! * d;
        w[j] = next;
      }
      maxStep = Math.max(maxStep, Math.abs(next - old));
      maxW = Math.max(maxW, Math.abs(next));
    }
    if (maxW === 0 || maxStep / maxW <= tol || iter === maxIter - 1) {
      gap = gramGap(G, q, yy, w, c, l1, l2);
      if (gap <= gapTol) {
        converged = true;
        iter++;
        break;
      }
    }
  }
  return { coef: w, nIter: Math.min(iter, maxIter), converged, gap, gapTol };
}

/** dualityGap() from the Gram matrix: ‖r‖² = yy − 2 wᵀq + wᵀGw, rᵀy = yy − wᵀq, Xᵀr = c. */
function gramGap(G: Float64Array, q: Float64Array, yy: number, w: Float64Array, c: Float64Array, l1: number, l2: number): number {
  const p = w.length;
  let wq = 0;
  let wGw = 0;
  let ww = 0;
  let w1 = 0;
  let dualNorm = 0;
  let xtr2 = 0;
  for (let j = 0; j < p; j++) {
    wq += w[j]! * q[j]!;
    // wᵀGw = wᵀ(q − c).
    wGw += w[j]! * (q[j]! - c[j]!);
    ww += w[j]! * w[j]!;
    w1 += Math.abs(w[j]!);
    dualNorm = Math.max(dualNorm, Math.abs(c[j]! - l2 * w[j]!));
    xtr2 += c[j]! * c[j]!;
  }
  const rr = Math.max(yy - 2 * wq + wGw, 0);
  const ry = yy - wq;
  if (l1 === 0 && l2 > 0) return ridgeGap(rr, ry, ww, xtr2, l2);
  const s = dualNorm > l1 ? l1 / dualNorm : 1;
  const augmented = rr + l2 * ww;
  return 0.5 * augmented + l1 * w1 - (s * ry - 0.5 * s * s * augmented);
}
