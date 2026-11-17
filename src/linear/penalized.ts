import { fail, warn } from '../core/errors.js';
import { checkedMeans, cholesky, choleskySolve, xw } from '../core/kernels.js';
import { threadPool, type ThreadOptions } from '../parallel/index.js';
import { keptMeans, runHere, runOn, withKernelCache, withKernelCacheSync, type Solver } from './calls.js';
import { nonNegativeLeastSquares } from './nnls.js';
import { checkedWeights, weightsSummingToRows, type SampleWeightOptions } from './weights.js';
import { coefColumns, coefsMatrix, interceptOf, oneTargetOnly, planFit, savedTargets, type FitPlan, type Targets } from './targets.js';
import { assertFinite, Matrix, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, notFitted, numbers, type ModelDocument, type Predictor } from '../core/model.js';
import { matMul, predictOnGpu, type DeviceOptions } from '../gpu/index.js';

interface State {
  /** One coefficient vector per target (one for a one-dimensional y). */
  coefs: Float64Array[];
  intercepts: Float64Array;
  /** Each target's iterations, and whether it converged (null for a model saved before smaltml recorded it). */
  nIters: number[];
  converged: (boolean | null)[];
  /** Whether y had a column per target (scikit-learn's 2-D y), even one. */
  several: boolean;
  targetNames: readonly string[] | undefined;
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

/** A linear model with a penalty: what Ridge, Lasso and ElasticNet share. */
abstract class PenalizedLinear implements Predictor {
  abstract readonly kind: string;
  readonly alpha: number;
  readonly fitIntercept: boolean;
  /** Hold every coefficient at 0 or more. */
  readonly positive: boolean;
  #state: State | null = null;

  constructor(alpha: number, fitIntercept: boolean, positive: boolean) {
    this.alpha = alpha;
    this.fitIntercept = fitIntercept;
    this.positive = positive;
    if (typeof positive !== 'boolean') fail('ML_PARAM', 'positive is true or false');
    if (!(alpha >= 0) || !Number.isFinite(alpha)) fail('ML_PARAM', `alpha must be 0 or more, not ${alpha}`);
    if (typeof fitIntercept !== 'boolean') fail('ML_PARAM', 'fitIntercept is true or false');
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
  /** The solver's iterations (0 for a closed-form solution); with several targets, the most any took. */
  get nIter(): number {
    return Math.max(...this.#fitted().nIters);
  }
  /**
   * Whether the solver converged within `maxIter` (always true for Ridge, which solves in closed form); with several
   * targets, whether every one did. Where it didn't, fit() warned (`ML_CONVERGENCE`), as scikit-learn does. null for
   * a model saved by smaltml before it recorded this.
   */
  get converged(): boolean | null {
    const c = this.#fitted().converged;
    return c.includes(null) ? null : c.every(Boolean);
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  abstract getParams(): object;

  /**
   * Whether weights are scaled to sum to the number of rows before the fit, as scikit-learn's ElasticNet and Lasso do
   * (their objective divides by n). Ridge's objective doesn't, so its weights are used as given.
   */
  protected readonly weightsSumToRows: boolean = false;

  /** The coefficients for X less `mean` (when there is an intercept) and centred y, and the iterations it took. */
  protected abstract solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution>;

  /**
   * Fits the model. With `threads` (default `'auto'`), the Gram matrix and Xᵀy are computed on several cores where
   * shared memory allows and X is large enough, with the same result as fitSync() to the last bit. With
   * `sampleWeight`, each row's squared error counts that many times. y is one target, or a column per target, each
   * fitted on its own (sharing XᵀX).
   */
  async fit(X: MatrixInput, y: Targets, options: ThreadOptions & SampleWeightOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    const pool = await threadPool(options.threads, m.rows * m.cols);
    if (!pool) return this.fitSync(m, y, options);
    const plan = this.#plan(m, y, options.sampleWeight);
    const solutions = await withKernelCache(async () => {
      const out: Solution[] = [];
      for (const t of plan.targets) out.push(m.cols === 0 ? EMPTY : await runOn(pool, this.solve(plan.design, plan.kernelMean, t.b)));
      return out;
    }, plan.targets.length > 1);
    return this.#finish(m, plan, solutions);
  }

  fitSync(X: MatrixInput, y: Targets, options: SampleWeightOptions = {}): this {
    const m = Matrix.from(X);
    const plan = this.#plan(m, y, options.sampleWeight);
    const solutions = withKernelCacheSync(
      () => plan.targets.map((t) => (m.cols === 0 ? EMPTY : runHere(this.solve(plan.design, plan.kernelMean, t.b)))),
      plan.targets.length > 1,
    );
    return this.#finish(m, plan, solutions);
  }

  #plan(m: Matrix, y: Targets, sampleWeight: VectorInput | undefined): FitPlan {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    let weights = checkedWeights(sampleWeight, m.rows);
    if (weights && this.weightsSumToRows) weights = weightsSummingToRows(weights);
    return planFit(m, y, weights, this.fitIntercept, () => keptMeans(m, () => checkedMeans(m, assertFinite)));
  }

  #finish(m: Matrix, plan: FitPlan, solutions: Solution[]): this {
    this.#state = {
      coefs: solutions.map((x) => x.coef),
      intercepts: Float64Array.from(solutions, (x, t) => interceptOf(plan.xMean, plan.targets[t]!.yMean, x.coef)),
      nIters: solutions.map((x) => x.nIter),
      converged: solutions.map((x) => x.converged),
      several: plan.several,
      targetNames: plan.targetNames,
      nFeaturesIn: m.cols,
      featureNamesIn: m.names,
    };
    solutions.forEach(({ nIter, converged, gap, gapTol }, t) => {
      if (converged) return;
      const which = plan.several ? ` on target ${plan.targetNames?.[t] ?? t}` : '';
      const why = gap === undefined || gapTol === undefined ? '' : `: the duality gap is ${gap.toPrecision(3)}, above the tolerance ${gapTol.toPrecision(3)}`;
      warn('ML_CONVERGENCE', `${this.kind} didn't converge${which} in ${nIter} iterations (maxIter)${why}. Raise maxIter, scale the columns (StandardScaler), or raise alpha`);
    });
    return this;
  }

  /** X·w + b, one prediction per row; on the GPU only with `device: 'gpu'` (32-bit floats there). For one target. */
  async predict(X: MatrixInput, options: DeviceOptions = {}): Promise<Float64Array> {
    const s = this.#one('predict()');
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    if (!(await predictOnGpu(options.device))) return this.predictSync(m);
    const out = (await matMul(m, new Matrix(s.coefs[0]!.length, 1, s.coefs[0]!), { device: 'gpu' })).data;
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
        ? {
            coefs: s.coefs.map(numbers),
            intercepts: numbers(s.intercepts),
            nIters: [...s.nIters],
            convergedEach: [...s.converged],
            ...(s.targetNames ? { targetNames: [...s.targetNames] } : {}),
          }
        : { coef: numbers(s.coefs[0]!), intercept: s.intercepts[0]!, nIter: s.nIters[0]!, converged: s.converged[0]! }),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  protected restore(s: Record<string, unknown> | null): this {
    if (s) {
      const t = savedTargets(s);
      const several = t.several;
      const iters = several && Array.isArray(s.nIters) ? (s.nIters as unknown[]).map(Number) : [Number(s.nIter ?? 0)];
      const conv = several && Array.isArray(s.convergedEach) ? (s.convergedEach as unknown[]) : [s.converged];
      this.#state = {
        ...t,
        nIters: t.coefs.map((_, k) => iters[k] ?? 0),
        converged: t.coefs.map((_, k) => (typeof conv[k] === 'boolean' ? (conv[k] as boolean) : null)),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return this;
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

/** The fit on no columns. */
const EMPTY: Solution = { coef: new Float64Array(0), nIter: 0, converged: true };

export interface RidgeParams {
  /** The L2 penalty's strength. Default 1. */
  alpha?: number;
  /** Fit an intercept (not penalized). Default true. */
  fitIntercept?: boolean;
  /** Hold every coefficient at 0 or more. Default false. */
  positive?: boolean;
}

/**
 * Linear regression with an L2 penalty: minimizes ‖y − X·w − b‖² + alpha · ‖w‖². scikit-learn's `Ridge`, for one
 * target. Solved in closed form, (XᵀX + alpha·I) w = Xᵀy on centred data, by Cholesky in 64-bit floats; with
 * `positive`, by non-negative least squares on the same equations (Lawson and Hanson's active set).
 */
export class Ridge extends PenalizedLinear {
  readonly kind = 'Ridge';

  constructor(params: RidgeParams = {}) {
    super(params.alpha ?? 1, params.fitIntercept ?? true, params.positive ?? false);
  }

  getParams(): RidgeParams {
    return { alpha: this.alpha, fitIntercept: this.fitIntercept, ...(this.positive ? { positive: true } : {}) };
  }

  protected *solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution> {
    if (this.positive) {
      const { coef } = yield* nonNegativeLeastSquares(X, mean, y, this.alpha);
      return { coef, nIter: 0, converged: true };
    }
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
  /** Hold every coefficient at 0 or more. Default false. */
  positive?: boolean;
}

/**
 * Linear regression with an elastic-net penalty: minimizes ‖y − X·w − b‖² / (2n) + alpha · l1Ratio · ‖w‖₁ +
 * alpha · (1 − l1Ratio) / 2 · ‖w‖². scikit-learn's `ElasticNet`, for one target, by cyclic coordinate descent with
 * its documented stopping rule: when the largest update is at most tol times the largest coefficient, and the
 * duality gap is at most tol · ‖y‖² / n.
 */
export class ElasticNet extends PenalizedLinear {
  readonly kind: string = 'ElasticNet';
  protected override readonly weightsSumToRows: boolean = true;
  readonly l1Ratio: number;
  readonly maxIter: number;
  readonly tol: number;

  constructor(params: ElasticNetParams = {}) {
    super(params.alpha ?? 1, params.fitIntercept ?? true, params.positive ?? false);
    this.l1Ratio = params.l1Ratio ?? 0.5;
    this.maxIter = params.maxIter ?? 1000;
    this.tol = params.tol ?? 1e-4;
    if (!(this.l1Ratio >= 0 && this.l1Ratio <= 1)) fail('ML_PARAM', `l1Ratio is from 0 to 1, not ${this.l1Ratio}`);
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) fail('ML_PARAM', `maxIter must be a whole number of at least 1, not ${this.maxIter}`);
    if (!(this.tol >= 0)) fail('ML_PARAM', `tol must be 0 or more, not ${this.tol}`);
  }

  getParams(): ElasticNetParams {
    return {
      alpha: this.alpha,
      l1Ratio: this.l1Ratio,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
      ...(this.positive ? { positive: true } : {}),
    };
  }

  protected *solve(X: Matrix, mean: Float64Array | undefined, y: Float64Array): Solver<Solution> {
    const l1 = this.alpha * this.l1Ratio * X.rows;
    const l2 = this.alpha * (1 - this.l1Ratio) * X.rows;
    // With more rows than columns, descend on the Gram matrix: a pass costs p², not a read of X. (Its blocked kernel
    // costs about three passes over X, and descent on X needs four or more with its gap check: Gram first measured faster.)
    if (X.rows >= X.cols && X.cols <= GRAM_MAX) {
      const G: Float64Array = yield { k: 'gram', X, mean };
      const q: Float64Array = yield { k: 'xtv', X, v: y, mean };
      return gramDescent(G, q, dotSelf(y), X.cols, l1, l2, this.maxIter, this.tol, this.positive);
    }
    return coordinateDescent(X, mean, y, l1, l2, this.maxIter, this.tol, this.positive);
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
function coordinateDescent(X: Matrix, mean: Float64Array | undefined, y: Float64Array, l1: number, l2: number, maxIter: number, tol: number, positive: boolean) {
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
      const next = shrink(rho, l1, positive) / (norms[j]! + l2);
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
      gap = dualityGap(X, mean, y, w, r, rsum, l1, l2, positive);
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
function dualityGap(
  X: Matrix,
  mean: Float64Array | undefined,
  y: Float64Array,
  w: Float64Array,
  r: Float64Array,
  rsum: number,
  l1: number,
  l2: number,
  positive: boolean,
): number {
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
    dualNorm = Math.max(dualNorm, positive ? g - l2 * w[j]! : Math.abs(g - l2 * w[j]!));
    xtr2 += positive ? Math.max(g, 0) ** 2 : g * g;
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

/**
 * The coordinate update's soft-threshold: ρ moved l1 towards 0, and 0 within l1 of it. With `positive`, a coefficient
 * the update would make negative is 0, as the constraint w ≥ 0 asks.
 */
function shrink(rho: number, l1: number, positive: boolean): number {
  if (positive) return Math.max(rho - l1, 0);
  return Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0);
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
function gramDescent(G: Float64Array, q: Float64Array, yy: number, p: number, l1: number, l2: number, maxIter: number, tol: number, positive: boolean) {
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
      const next = shrink(rho, l1, positive) / (gjj + l2);
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
      gap = gramGap(G, q, yy, w, c, l1, l2, positive);
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
function gramGap(G: Float64Array, q: Float64Array, yy: number, w: Float64Array, c: Float64Array, l1: number, l2: number, positive: boolean): number {
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
    dualNorm = Math.max(dualNorm, positive ? c[j]! - l2 * w[j]! : Math.abs(c[j]! - l2 * w[j]!));
    xtr2 += positive ? Math.max(c[j]!, 0) ** 2 : c[j]! * c[j]!;
  }
  const rr = Math.max(yy - 2 * wq + wGw, 0);
  const ry = yy - wq;
  if (l1 === 0 && l2 > 0) return ridgeGap(rr, ry, ww, xtr2, l2);
  const s = dualNorm > l1 ? l1 / dualNorm : 1;
  const augmented = rr + l2 * ww;
  return 0.5 * augmented + l1 * w1 - (s * ry - 0.5 * s * s * augmented);
}
