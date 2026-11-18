import { fail, warn } from '../core/errors.js';
import { binaryLogistic, cholesky, choleskySolve, finishLogistic, multinomialLogistic, prepareLogistic, weightedGram, xw, xwEach } from '../core/kernels.js';
import { logisticOnGpu } from '../gpu/logistic.js';
import { minimize, minimizeAsync, type MinimizeResult } from '../core/linalg.js';
import { logisticParallel, threadPool, weightedGramParallel, xwParallel, type ThreadOptions } from '../parallel/index.js';
import { assertFinite, Matrix, type MatrixInput } from '../core/matrix.js';
import { checkedWeights, type SampleWeightOptions } from './weights.js';
import { saga } from './saga.js';
import { gpuAvailable, matMul, predictOnGpu, type DeviceOptions } from '../gpu/index.js';
import {
  checkColumns,
  checkDocument,
  document,
  floats,
  notFitted,
  numbers,
  toLabels,
  uniqueLabels,
  type Classifier,
  type Label,
  type Labels,
  type ModelDocument,
} from '../core/model.js';

export interface LogisticRegressionParams {
  /** The inverse of the penalty's strength: smaller is a stronger L2 penalty. `Infinity` turns the penalty off. Default 1. */
  C?: number;
  /**
   * The L1 share of the penalty, from 0 (all L2, the default) to 1 (all L1, which sets some coefficients to exactly 0);
   * between, the elastic net. Above 0 it needs solver `'saga'`, as in scikit-learn.
   */
  l1Ratio?: number;
  /** Stop when the largest gradient component is at most this. Default 1e-4. */
  tol?: number;
  /** Fit an intercept (not penalized). Default true. */
  fitIntercept?: boolean;
  /** The most iterations of the solver. Default 100. */
  maxIter?: number;
  /**
   * How the fit is solved. `'lbfgs'` (the default) works from the gradient alone. `'newton-cholesky'` takes Newton
   * steps with the exact Hessian, solved by Cholesky: few, costly iterations, the better choice with many rows and
   * not many columns (up to a few hundred), and with few classes: its Hessian grows as the classes squared. It runs
   * on the CPU. `'saga'` takes one row at a time (stochastic), and is the solver for an L1 or elastic-net penalty
   * (`l1Ratio` above 0); it converges fastest on scaled columns.
   */
  solver?: 'lbfgs' | 'newton-cholesky' | 'saga';
  /** The seed for `'saga'`'s order of rows: the same seed gives the same fit. Default: a new order each fit. scikit-learn's `random_state`. */
  seed?: number;
  /**
   * Weights for the classes, multiplying each row's weight by its class's: `'balanced'` gives each class
   * n / (classes × its rows), its rows counted by their sample weights, so every class counts as much in all; an object gives a class's weight by its label
   * (`{ yes: 5 }`); classes it leaves out weigh 1. scikit-learn's `class_weight`. Default: every class 1.
   */
  classWeight?: 'balanced' | Readonly<Record<string, number>>;
}

const SOLVERS = ['lbfgs', 'newton-cholesky', 'saga'] as const;

/** Where a fit runs: `'gpu'` asks for the GPU; `'auto'` (the default) and `'cpu'` fit on the CPU. */
export interface FitDeviceOptions {
  device?: 'auto' | 'cpu' | 'gpu';
}

/** A fit's prepared inputs. */
interface Job {
  m: Matrix;
  classes: Label[];
  target: Int32Array;
  l2: number;
  /** Each row's weight (sample and class weights together), or undefined when every row weighs 1. */
  weights: Float64Array | undefined;
  /** The rows' total weight: n when unweighted. */
  total: number;
  k: number;
  width: number;
  x0: Float64Array;
  options: { gtol: number; maxIter: number; ftolFactor: number };
}

interface State {
  classes: Label[];
  /** One row per class (one row in all for two classes), one column per feature. */
  coef: Matrix;
  intercept: Float64Array;
  nIter: number;
  /** Whether the solver converged; null for a model saved before smaltml recorded it (format version 1). */
  converged: boolean | null;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/** What the Newton solver asks for: the objective and its gradient at x, the scores X·w + b, or XᵀWX. */
type NewtonCall =
  | { k: 'objective'; x: Float64Array; grad: Float64Array }
  | { k: 'scores'; w: Float64Array; b: number }
  | { k: 'hessian'; weights: Float64Array };

/**
 * Newton's method for the binary objective: each step solves H d = −g, with H = X̃ᵀ diag(p(1 − p)) X̃ / n + l2 (the
 * intercept unpenalized) by Cholesky, then backtracks along d until the objective falls enough (Armijo). It stops, as
 * scikit-learn's newton-cholesky does, when the largest gradient component and half the Newton decrement dᵀHd are
 * both at most tol. A generator, like the linear solvers, so fitSync() and fit() (the kernels on a thread pool) share it.
 */
function* newtonBinary(n: number, p: number, l2: number, intercept: boolean, tol: number, maxIter: number, rowWeights?: Float64Array, total = n): Generator<NewtonCall, MinimizeResult, Float64Array | number> {
  const q = p + (intercept ? 1 : 0);
  let x = new Float64Array(q);
  let g = new Float64Array(q);
  let f = (yield { k: 'objective', x, grad: g }) as number;
  const weights = new Float64Array(n);
  let iterations = 0;
  let decrement = Infinity;
  let converged = false;
  for (; iterations < maxIter; iterations++) {
    let gmax = 0;
    for (let j = 0; j < q; j++) gmax = Math.max(gmax, Math.abs(g[j]!));
    if (gmax <= tol && 0.5 * decrement <= tol) {
      converged = true;
      break;
    }
    // The Hessian at x: each row's weight is w·p(1 − p) / Σw, with p the probability of the second class and w the
    // row's own weight (1 without weights, when Σw is n).
    const z = (yield { k: 'scores', w: x.slice(0, p), b: intercept ? x[p]! : 0 }) as Float64Array;
    for (let i = 0; i < n; i++) {
      const pi = 1 / (1 + Math.exp(-z[i]!));
      weights[i] = ((rowWeights ? rowWeights[i]! : 1) * pi * (1 - pi)) / total;
    }
    const H = (yield { k: 'hessian', weights }) as Float64Array;
    for (let j = 0; j < p; j++) H[j * q + j]! += l2;
    const L = cholesky(H, q);
    // Where the Hessian is singular to working precision (separable classes, no penalty), stop at the best point so far.
    if (!L) break;
    const d = choleskySolve(L, q, Float64Array.from(g, (v) => -v));
    let slope = 0;
    for (let j = 0; j < q; j++) slope += g[j]! * d[j]!;
    decrement = -slope;
    // Backtracking: the full step first, halved until the objective falls by at least 1e-4 of the slope.
    let t = 1;
    let ft = Infinity;
    let trial = new Float64Array(q);
    let tg = new Float64Array(q);
    for (let tries = 0; tries < 40; tries++, t /= 2) {
      trial = Float64Array.from(x, (v, j) => v + t * d[j]!);
      tg = new Float64Array(q);
      ft = (yield { k: 'objective', x: trial, grad: tg }) as number;
      if (ft <= f + 1e-4 * t * slope) break;
    }
    if (!(ft <= f)) break;
    x = trial;
    g = tg;
    f = ft;
  }
  return { x, value: f, iterations, converged };
}

/**
 * Newton's method for the multinomial objective, K classes, x as K rows of (w, then b). The Hessian has a q × q block
 * for each pair of classes a, b: X̃ᵀ diag(w·p_a(δ_ab − p_b)) X̃ / Σw, plus l2 on the coefficients of the diagonal
 * blocks. Adding the same vector to every class's row changes no probability, so the full Hessian is singular in that
 * direction: each step is solved in the subspace where the K rows of the step sum to 0 (the last row is minus the sum
 * of the others), the subspace L-BFGS keeps to from its start at 0, where the penalized optimum also lies. The line
 * search and stopping rule are newtonBinary()'s.
 */
function* newtonMultinomial(n: number, p: number, K: number, l2: number, intercept: boolean, tol: number, maxIter: number, rowWeights?: Float64Array, total = n): Generator<NewtonCall, MinimizeResult, Float64Array | number> {
  const q = p + (intercept ? 1 : 0);
  const size = K * q;
  const r = (K - 1) * q;
  let x = new Float64Array(size);
  let g = new Float64Array(size);
  let f = (yield { k: 'objective', x, grad: g }) as number;
  let iterations = 0;
  let decrement = Infinity;
  let converged = false;
  for (; iterations < maxIter; iterations++) {
    let gmax = 0;
    for (let j = 0; j < size; j++) gmax = Math.max(gmax, Math.abs(g[j]!));
    if (gmax <= tol && 0.5 * decrement <= tol) {
      converged = true;
      break;
    }
    // Each class's scores, then each row's probabilities.
    const z: Float64Array[] = [];
    for (let c = 0; c < K; c++) z.push((yield { k: 'scores', w: x.slice(c * q, c * q + p), b: intercept ? x[c * q + p]! : 0 }) as Float64Array);
    const prob = z.map(() => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      let max = -Infinity;
      for (let c = 0; c < K; c++) max = Math.max(max, z[c]![i]!);
      let sum = 0;
      for (let c = 0; c < K; c++) sum += prob[c]![i] = Math.exp(z[c]![i]! - max);
      for (let c = 0; c < K; c++) prob[c]![i]! /= sum;
    }
    // The blocks, a ≤ b: weights w·p_a(1 − p_a) on the diagonal, −w·p_a·p_b off it (as +w·p_a·p_b, then negated, as
    // the kernel takes weights of 0 or more).
    const block: Float64Array[][] = Array.from({ length: K }, () => new Array<Float64Array>(K));
    for (let a = 0; a < K; a++) {
      for (let b = a; b < K; b++) {
        const pa = prob[a]!;
        const pb = prob[b]!;
        const wts = Float64Array.from(pa, (v, i) => ((rowWeights ? rowWeights[i]! : 1) * v * (a === b ? 1 - v : pb[i]!)) / total);
        const B = (yield { k: 'hessian', weights: wts }) as Float64Array;
        if (a === b) {
          for (let j = 0; j < p; j++) B[j * q + j]! += l2;
        } else {
          for (let t = 0; t < B.length; t++) B[t] = -B[t]!;
        }
        block[a]![b] = B;
        block[b]![a] = B;
      }
    }
    // The step in the subspace: d_c = u_c for c < K − 1, d_{K−1} = −Σ u_c. Its Hessian is TᵀHT, its gradient Tᵀg:
    // H_ab − H_a,last − H_last,b + H_last,last, and g_c − g_last.
    const last = K - 1;
    const Hr = new Float64Array(r * r);
    for (let a = 0; a < last; a++) {
      for (let b = 0; b < last; b++) {
        const Bab = block[a]![b]!;
        const Bal = block[a]![last]!;
        const Blb = block[last]![b]!;
        const Bll = block[last]![last]!;
        for (let i = 0; i < q; i++) {
          // Every block is a symmetric matrix (a weighted Gram matrix), so block (a, b) is block (b, a).
          for (let j = 0; j < q; j++) {
            Hr[(a * q + i) * r + b * q + j] = Bab[i * q + j]! - Bal[i * q + j]! - Blb[i * q + j]! + Bll[i * q + j]!;
          }
        }
      }
    }
    const gr = new Float64Array(r);
    for (let c = 0; c < last; c++) for (let j = 0; j < q; j++) gr[c * q + j] = -(g[c * q + j]! - g[last * q + j]!);
    const L = cholesky(Hr, r);
    if (!L) break;
    const u = choleskySolve(L, r, gr);
    const d = new Float64Array(size);
    for (let c = 0; c < last; c++) {
      for (let j = 0; j < q; j++) {
        d[c * q + j] = u[c * q + j]!;
        d[last * q + j]! -= u[c * q + j]!;
      }
    }
    let slope = 0;
    for (let j = 0; j < size; j++) slope += g[j]! * d[j]!;
    decrement = -slope;
    let t = 1;
    let ft = Infinity;
    let trial = new Float64Array(size);
    let tg = new Float64Array(size);
    for (let tries = 0; tries < 40; tries++, t /= 2) {
      trial = Float64Array.from(x, (v, j) => v + t * d[j]!);
      tg = new Float64Array(size);
      ft = (yield { k: 'objective', x: trial, grad: tg }) as number;
      if (ft <= f + 1e-4 * t * slope) break;
    }
    if (!(ft <= f)) break;
    x = trial;
    g = tg;
    f = ft;
  }
  return { x, value: f, iterations, converged };
}

/**
 * Logistic regression: a linear model of each class's log-odds, fitted by L-BFGS to minimize the log loss plus an L2
 * penalty, ‖w‖² / (2 C n). scikit-learn's `LogisticRegression` with its default solver, `lbfgs`: two classes take one
 * row of coefficients, more classes the multinomial (softmax) model with a row per class.
 */
export class LogisticRegression implements Classifier {
  readonly kind = 'LogisticRegression';
  readonly C: number;
  readonly l1Ratio: number;
  readonly tol: number;
  readonly fitIntercept: boolean;
  readonly maxIter: number;
  readonly solver: 'lbfgs' | 'newton-cholesky' | 'saga';
  readonly seed: number | undefined;
  readonly classWeight: 'balanced' | Readonly<Record<string, number>> | undefined;
  #state: State | null = null;

  constructor(params: LogisticRegressionParams = {}) {
    this.C = params.C ?? 1;
    this.l1Ratio = params.l1Ratio ?? 0;
    this.tol = params.tol ?? 1e-4;
    this.fitIntercept = params.fitIntercept ?? true;
    this.maxIter = params.maxIter ?? 100;
    this.solver = params.solver ?? 'lbfgs';
    if (!SOLVERS.includes(this.solver)) fail('ML_PARAM', `solver is 'lbfgs', 'newton-cholesky' or 'saga', not ${String(this.solver)}`);
    this.seed = params.seed;
    if (this.seed !== undefined && !Number.isInteger(this.seed)) fail('ML_PARAM', `seed must be a whole number, not ${this.seed}`);
    this.classWeight = params.classWeight;
    const cw = this.classWeight;
    if (cw !== undefined && cw !== 'balanced') {
      if (typeof cw !== 'object' || cw === null || Array.isArray(cw)) fail('ML_PARAM', "classWeight is 'balanced' or an object of a weight per class");
      for (const [label, w] of Object.entries(cw)) {
        if (typeof w !== 'number' || !Number.isFinite(w) || w < 0) fail('ML_PARAM', `classWeight for ${label} must be a finite number, 0 or more, not ${String(w)}`);
      }
    }
    if (!(this.C > 0)) fail('ML_PARAM', `C must be greater than 0, not ${this.C}`);
    if (!(this.l1Ratio >= 0 && this.l1Ratio <= 1)) fail('ML_PARAM', `l1Ratio is from 0 to 1, not ${this.l1Ratio}`);
    if (this.l1Ratio > 0 && this.solver !== 'saga') {
      fail('ML_PARAM', `solver '${this.solver}' takes an L2 penalty only, not l1Ratio ${this.l1Ratio}: use solver 'saga' for an L1 or elastic-net penalty`);
    }
    if (!(this.tol > 0)) fail('ML_PARAM', `tol must be greater than 0, not ${this.tol}`);
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) fail('ML_PARAM', `maxIter must be a whole number of at least 1, not ${this.maxIter}`);
    if (typeof this.fitIntercept !== 'boolean') fail('ML_PARAM', 'fitIntercept is true or false');
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  get classes(): readonly Label[] {
    return this.#fitted().classes;
  }

  /** The coefficients: one row for two classes (the second class's log-odds), else one row per class. */
  get coef(): Matrix {
    return this.#fitted().coef;
  }

  /** The intercepts: one for two classes, else one per class. */
  get intercept(): Float64Array {
    return this.#fitted().intercept;
  }

  /** The solver's iterations. */
  get nIter(): number {
    return this.#fitted().nIter;
  }

  /**
   * Whether the solver converged within `maxIter`. Where it didn't, fit() warned (`ML_CONVERGENCE`), as scikit-learn
   * does. null for a model saved by smaltml before it recorded this (format version 1).
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

  getParams(): LogisticRegressionParams {
    return {
      C: this.C,
      l1Ratio: this.l1Ratio,
      tol: this.tol,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      solver: this.solver,
      ...(this.seed === undefined ? {} : { seed: this.seed }),
      ...(this.classWeight === undefined ? {} : { classWeight: this.classWeight === 'balanced' ? 'balanced' : { ...this.classWeight } }),
    };
  }

  /**
   * Fits the model. With `threads` (default `'auto'`), the objective runs on several cores where shared memory allows
   * and X is large enough, with the same result as fitSync() to the last bit. With `device: 'gpu'`, it runs on the GPU
   * instead, in 32-bit floats; `'auto'` (the default) keeps fits on the CPU.
   */
  async fit(X: MatrixInput, y: Labels, options: ThreadOptions & FitDeviceOptions & SampleWeightOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    const sync = { sampleWeight: options.sampleWeight };
    if (this.solver === 'saga') {
      // One row at a time: on this thread, on the CPU.
      if (options.device === 'gpu') fail('ML_PARAM', "solver 'saga' runs on the CPU: use solver 'lbfgs' on the GPU");
      return this.fitSync(m, y, sync);
    }
    if (this.solver === 'newton-cholesky') {
      if (options.device === 'gpu') fail('ML_PARAM', "solver 'newton-cholesky' runs on the CPU: use solver 'lbfgs' on the GPU");
      const pool = await threadPool(options.threads, m.rows * m.cols);
      if (!pool) return this.fitSync(m, y, sync);
      const job = this.#prepare(m, y, options.sampleWeight);
      const objective = logisticParallel(pool, m, job.target, job.k, job.l2, this.fitIntercept, job.weights, job.total);
      const solver =
        job.k === 1
          ? newtonBinary(m.rows, m.cols, job.l2, this.fitIntercept, this.tol, this.maxIter, job.weights, job.total)
          : newtonMultinomial(m.rows, m.cols, job.k, job.l2, this.fitIntercept, this.tol, this.maxIter, job.weights, job.total);
      let r = solver.next();
      while (!r.done) {
        const c = r.value;
        r = solver.next(
          c.k === 'objective'
            ? await objective(c.x, c.grad)
            : c.k === 'scores'
              ? await xwParallel(pool, m, c.w, c.b)
              : await weightedGramParallel(pool, m, c.weights, this.fitIntercept),
        );
      }
      return this.#finish(job, r.value);
    }
    if (options.device === 'gpu') {
      // The objective on the GPU, in 32-bit floats: the result is close to the CPU's, not equal to it.
      if (!(await gpuAvailable())) fail('ML_PARAM', "device 'gpu' needs WebGPU, which isn't available here: leave device out, or use 'cpu'");
      const job = this.#prepare(m, y, options.sampleWeight);
      if (job.weights) fail('ML_PARAM', "a fit on the GPU doesn't take weights yet (sampleWeight, classWeight): leave device out, or use 'cpu'");
      const gpu = await logisticOnGpu(m, job.target, job.k, this.fitIntercept);
      if (gpu) {
        try {
          const intercept = this.fitIntercept;
          const objective = async (x: Float64Array, g: Float64Array) => finishLogistic(await gpu.sums(x, g), m, job.l2, intercept, x, g, job.k);
          return this.#finish(job, await minimizeAsync(objective, job.x0, job.options));
        } finally {
          gpu.destroy();
        }
      }
      // More classes, or a larger X, than the GPU takes: the CPU.
    } else if (options.device !== undefined && options.device !== 'cpu' && options.device !== 'auto') {
      fail('ML_PARAM', `device is 'auto', 'cpu' or 'gpu', not ${String(options.device)}`);
    }
    const pool = await threadPool(options.threads, m.rows * m.cols);
    if (!pool) return this.fitSync(m, y, sync);
    const job = this.#prepare(m, y, options.sampleWeight);
    const objective = logisticParallel(pool, m, job.target, job.k, job.l2, this.fitIntercept, job.weights, job.total);
    return this.#finish(job, await minimizeAsync(objective, job.x0, job.options));
  }

  /** fit() on this thread; with `sampleWeight`, each row's loss counts that many times. */
  fitSync(X: MatrixInput, y: Labels, options: SampleWeightOptions = {}): this {
    const m = Matrix.from(X);
    const job = this.#prepare(m, y, options.sampleWeight);
    const { target, k, l2, weights, total } = job;
    const withIntercept = this.fitIntercept;
    if (this.solver === 'saga') {
      // The penalty split by l1Ratio, against the weighted mean loss: λ2 = (1 − r)/(C·Σw), λ1 = r/(C·Σw).
      const strength = Number.isFinite(this.C) ? 1 / (this.C * total) : 0;
      const result = saga({
        X: m,
        y: target,
        K: k,
        intercept: withIntercept,
        weights,
        total,
        l2: strength * (1 - this.l1Ratio),
        l1: strength * this.l1Ratio,
        tol: this.tol,
        maxIter: this.maxIter,
        seed: this.seed ?? Date.now(),
      });
      return this.#finish(job, result);
    }
    // The products kernel, where one is in use, keeps X for this fit; released when the fit ends.
    const products = prepareLogistic(m);
    try {
      return this.#optimize(job, m, target, k, l2, withIntercept, weights, total, products);
    } finally {
      products?.release();
    }
  }

  #optimize(job: Job, m: Matrix, target: Int32Array, k: number, l2: number, withIntercept: boolean, weights: Float64Array | undefined, total: number, products: ReturnType<typeof prepareLogistic>): this {
    if (this.solver === 'newton-cholesky') {
      const solver =
        k === 1
          ? newtonBinary(m.rows, m.cols, l2, withIntercept, this.tol, this.maxIter, weights, total)
          : newtonMultinomial(m.rows, m.cols, k, l2, withIntercept, this.tol, this.maxIter, weights, total);
      let r = solver.next();
      while (!r.done) {
        const c = r.value;
        r = solver.next(
          c.k === 'objective'
            ? k === 1
              ? binaryLogistic(m, target, l2, withIntercept, c.x, c.grad, weights, total, products)
              : multinomialLogistic(m, target, k, l2, withIntercept, c.x, c.grad, weights, total, products)
            : c.k === 'scores'
              ? xw(m, c.w, c.b)
              : weightedGram(m, c.weights, withIntercept),
        );
      }
      return this.#finish(job, r.value);
    }
    const objective =
      k === 1
        ? (x: Float64Array, g: Float64Array) => binaryLogistic(m, target, l2, withIntercept, x, g, weights, total, products)
        : (x: Float64Array, g: Float64Array) => multinomialLogistic(m, target, k, l2, withIntercept, x, g, weights, total, products);
    return this.#finish(job, minimize(objective, job.x0, job.options));
  }

  /** What a fit needs before the optimizer: the checked data, class indexes, the penalty and the starting point. */
  #prepare(m: Matrix, y: Labels, sampleWeight?: SampleWeightOptions['sampleWeight']): Job {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    assertFinite(m);
    const labels = toLabels(y, m.rows);
    const classes = uniqueLabels(labels);
    if (classes.length < 2) fail('ML_LABELS', `y has one class only (${String(classes[0])}): a classifier needs at least two`);
    const index = new Map(classes.map((c, k) => [c, k]));
    const target = Int32Array.from(labels, (l) => index.get(l)!);
    const weights = this.#rowWeights(sampleWeight, target, classes);
    let total = m.rows;
    if (weights) {
      total = 0;
      for (const v of weights) total += v;
      if (!(total > 0)) fail('ML_VALUE', 'every row weighs 0 (sampleWeight and classWeight together): at least one row needs weight');
    }
    // The penalty is ‖w‖² / (2 C Σw), against the weighted mean of the loss, as scikit-learn's.
    const l2 = Number.isFinite(this.C) ? 1 / (this.C * total) : 0;
    const k = classes.length === 2 ? 1 : classes.length;
    const width = m.cols + (this.fitIntercept ? 1 : 0);
    return { m, classes, target, l2, weights, total, k, width, x0: new Float64Array(k * width), options: { gtol: this.tol, maxIter: this.maxIter, ftolFactor: 64 } };
  }

  /** Each row's weight: its sampleWeight times its class's classWeight. Undefined when there are neither. */
  #rowWeights(sampleWeight: SampleWeightOptions['sampleWeight'], target: Int32Array, classes: readonly Label[]): Float64Array | undefined {
    const n = target.length;
    const sample = checkedWeights(sampleWeight, n);
    const cw = this.classWeight;
    if (!cw) return sample;
    let perClass: Float64Array;
    if (cw === 'balanced') {
      // Σw / (classes × the class's Σw), counting rows by their sample weights (n / (classes × its rows) without
      // them): every class weighs as much in all.
      const counts = new Float64Array(classes.length);
      let all = 0;
      for (let i = 0; i < n; i++) {
        const w = sample ? sample[i]! : 1;
        counts[target[i]!]! += w;
        all += w;
      }
      perClass = Float64Array.from(counts, (c) => (c > 0 ? all / (classes.length * c) : 0));
    } else {
      const known = new Map(classes.map((c, k) => [String(c), k]));
      for (const label of Object.keys(cw)) {
        if (!known.has(label)) fail('ML_PARAM', `classWeight names ${label}, which isn't one of the classes in y (${classes.map(String).join(', ')})`);
      }
      perClass = Float64Array.from(classes, (c) => cw[String(c)] ?? 1);
    }
    return Float64Array.from(target, (t, i) => perClass[t]! * (sample ? sample[i]! : 1));
  }

  #finish(job: Job, result: MinimizeResult): this {
    const { m, classes, k, width } = job;
    const p = m.cols;
    const coef = new Matrix(k, p);
    const intercept = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < p; j++) coef.data[j * k + c] = result.x[c * width + j]!;
      intercept[c] = this.fitIntercept ? result.x[c * width + p]! : 0;
    }
    this.#state = { classes, coef, intercept, nIter: result.iterations, converged: result.converged, nFeaturesIn: p, featureNamesIn: m.names };
    if (!result.converged) {
      warn('ML_CONVERGENCE', `LogisticRegression (solver '${this.solver}') stopped after ${result.iterations} of at most ${this.maxIter} iterations without converging: raise maxIter, or scale the columns (StandardScaler)`);
    }
    return this;
  }

  /** The most probable class for each row; the scores on the GPU only with `device: 'gpu'`. */
  async predict(X: MatrixInput, options: DeviceOptions = {}): Promise<Label[]> {
    return this.#labels(await this.decisionFunction(X, options));
  }

  /** The most probable class for each row. */
  predictSync(X: MatrixInput): Label[] {
    return this.#labels(this.decisionFunctionSync(X));
  }

  #labels(scores: Matrix): Label[] {
    const s = this.#fitted();
    const n = scores.rows;
    const out: Label[] = new Array(n);
    if (scores.cols === 1) {
      for (let i = 0; i < n; i++) out[i] = s.classes[scores.data[i]! > 0 ? 1 : 0]!;
      return out;
    }
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let c = 1; c < scores.cols; c++) if (scores.data[c * n + i]! > scores.data[best * n + i]!) best = c;
      out[i] = s.classes[best]!;
    }
    return out;
  }

  /** Each class's probability; the scores on the GPU only with `device: 'gpu'`. */
  async predictProba(X: MatrixInput, options: DeviceOptions = {}): Promise<Matrix> {
    return this.#proba(await this.decisionFunction(X, options));
  }

  /** Each class's probability, one column per class in the order of `classes`. */
  predictProbaSync(X: MatrixInput): Matrix {
    return this.#proba(this.decisionFunctionSync(X));
  }

  #proba(scores: Matrix): Matrix {
    const s = this.#fitted();
    const n = scores.rows;
    const K = s.classes.length;
    const out = new Float64Array(n * K);
    if (scores.cols === 1) {
      for (let i = 0; i < n; i++) {
        const p1 = sigmoid(scores.data[i]!);
        out[i] = 1 - p1;
        out[n + i] = p1;
      }
    } else {
      for (let i = 0; i < n; i++) {
        let max = -Infinity;
        for (let c = 0; c < K; c++) max = Math.max(max, scores.data[c * n + i]!);
        let sum = 0;
        for (let c = 0; c < K; c++) sum += out[c * n + i] = Math.exp(scores.data[c * n + i]! - max);
        for (let c = 0; c < K; c++) out[c * n + i]! /= sum;
      }
    }
    return new Matrix(n, K, out, s.classes.map(String));
  }

  /** Each row's score, X·wᵀ + b, as decisionFunctionSync(); the product on the GPU only with `device: 'gpu'`. */
  async decisionFunction(X: MatrixInput, options: DeviceOptions = {}): Promise<Matrix> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const k = s.coef.rows;
    // On the CPU, exactly decisionFunctionSync()'s result.
    if (!(await predictOnGpu(options.device))) return this.decisionFunctionSync(m);
    const p = s.coef.cols;
    // coefᵀ, p × k: column c is coef's row c.
    const wT = new Float64Array(p * k);
    for (let c = 0; c < k; c++) for (let j = 0; j < p; j++) wT[c * p + j] = s.coef.data[j * k + c]!;
    const scores = await matMul(m, new Matrix(p, k, wT), { device: 'gpu' });
    for (let c = 0; c < k; c++) {
      const b = s.intercept[c]!;
      for (let i = 0; i < m.rows; i++) scores.data[c * m.rows + i]! += b;
    }
    return scores;
  }

  /** Each row's score: X·wᵀ + b, one column for two classes (the second class's log-odds), else one per class. */
  decisionFunctionSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const k = s.coef.rows;
    const out = new Float64Array(n * k);
    const ws = Array.from({ length: k }, (_, c) => Float64Array.from({ length: m.cols }, (_, j) => s.coef.data[j * k + c]!));
    const outs = Array.from({ length: k }, (_, c) => out.subarray(c * n, (c + 1) * n));
    // Three classes' scores in one pass over X (xwEach), equal to xw()'s class by class.
    xwEach(m, ws, Array.from(s.intercept), outs);
    return new Matrix(n, k, out);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams(), C: Number.isFinite(this.C) ? this.C : 'Infinity' }, s && {
      classes: [...s.classes],
      coef: s.coef.toRows(),
      intercept: numbers(s.intercept),
      nIter: s.nIter,
      converged: s.converged,
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): LogisticRegression {
    const d = checkDocument(doc, 'LogisticRegression');
    const { C, ...rest } = d.params as Omit<LogisticRegressionParams, 'C'> & { C?: number | string };
    const model = new LogisticRegression({ ...rest, C: C === 'Infinity' ? Infinity : (C as number | undefined) });
    const s = d.state;
    if (s) {
      if (!Array.isArray(s.classes) || !Array.isArray(s.coef)) fail('ML_FORMAT', 'a LogisticRegression state needs classes and coef');
      model.#state = {
        classes: s.classes as Label[],
        coef: Matrix.fromRows(s.coef as number[][]),
        intercept: floats(s.intercept, 'intercept'),
        nIter: Number(s.nIter),
        converged: typeof s.converged === 'boolean' ? s.converged : null,
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

function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}
