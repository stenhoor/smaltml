import { fail } from '../core/errors.js';
import { binaryLogistic, finishLogistic, multinomialLogistic, xw } from '../core/kernels.js';
import { logisticOnGpu } from '../gpu/logistic.js';
import { minimize, minimizeAsync, type MinimizeResult } from '../core/linalg.js';
import { logisticParallel, threadPool, type ThreadOptions } from '../parallel/index.js';
import { assertFinite, Matrix, type MatrixInput } from '../core/matrix.js';
import { gpuAvailable, matMul, useGpu, type DeviceOptions } from '../gpu/index.js';
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
  /** The mix of L1 in the penalty. Only 0, an L2 penalty, is supported so far. Default 0. */
  l1Ratio?: number;
  /** Stop when the largest gradient component is at most this. Default 1e-4. */
  tol?: number;
  /** Fit an intercept (not penalized). Default true. */
  fitIntercept?: boolean;
  /** The most L-BFGS iterations. Default 100. */
  maxIter?: number;
}

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
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
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
  #state: State | null = null;

  constructor(params: LogisticRegressionParams = {}) {
    this.C = params.C ?? 1;
    this.l1Ratio = params.l1Ratio ?? 0;
    this.tol = params.tol ?? 1e-4;
    this.fitIntercept = params.fitIntercept ?? true;
    this.maxIter = params.maxIter ?? 100;
    if (!(this.C > 0)) fail('ML_PARAM', `C must be greater than 0, not ${this.C}`);
    if (this.l1Ratio !== 0) fail('ML_PARAM', `l1Ratio ${this.l1Ratio} isn't supported yet: only 0 (an L2 penalty)`);
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

  /** The L-BFGS iterations the fit took. */
  get nIter(): number {
    return this.#fitted().nIter;
  }

  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }

  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): Required<LogisticRegressionParams> {
    return { C: this.C, l1Ratio: this.l1Ratio, tol: this.tol, fitIntercept: this.fitIntercept, maxIter: this.maxIter };
  }

  /**
   * Fits the model. With `threads` (default `'auto'`), the objective runs on several cores where shared memory allows
   * and X is large enough, with the same result as fitSync() to the last bit. With `device: 'gpu'`, it runs on the GPU
   * instead, in 32-bit floats; `'auto'` (the default) keeps fits on the CPU.
   */
  async fit(X: MatrixInput, y: Labels, options: ThreadOptions & FitDeviceOptions = {}): Promise<this> {
    const m = Matrix.from(X);
    if (options.device === 'gpu') {
      // The objective on the GPU, in 32-bit floats: the result is close to the CPU's, not equal to it.
      if (!(await gpuAvailable())) fail('ML_PARAM', "device 'gpu' needs WebGPU, which isn't available here: leave device out, or use 'cpu'");
      const job = this.#prepare(m, y);
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
    if (!pool) return this.fitSync(m, y);
    const job = this.#prepare(m, y);
    const objective = logisticParallel(pool, m, job.target, job.k, job.l2, this.fitIntercept);
    return this.#finish(job, await minimizeAsync(objective, job.x0, job.options));
  }

  fitSync(X: MatrixInput, y: Labels): this {
    const m = Matrix.from(X);
    const job = this.#prepare(m, y);
    const { target, k, l2 } = job;
    const withIntercept = this.fitIntercept;
    const objective =
      k === 1
        ? (x: Float64Array, g: Float64Array) => binaryLogistic(m, target, l2, withIntercept, x, g)
        : (x: Float64Array, g: Float64Array) => multinomialLogistic(m, target, k, l2, withIntercept, x, g);
    return this.#finish(job, minimize(objective, job.x0, job.options));
  }

  /** What a fit needs before the optimizer: the checked data, class indexes, the penalty and the starting point. */
  #prepare(m: Matrix, y: Labels): Job {
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    assertFinite(m);
    const labels = toLabels(y, m.rows);
    const classes = uniqueLabels(labels);
    if (classes.length < 2) fail('ML_LABELS', `y has one class only (${String(classes[0])}): a classifier needs at least two`);
    const index = new Map(classes.map((c, k) => [c, k]));
    const target = Int32Array.from(labels, (l) => index.get(l)!);
    const l2 = Number.isFinite(this.C) ? 1 / (this.C * m.rows) : 0;
    const k = classes.length === 2 ? 1 : classes.length;
    const width = m.cols + (this.fitIntercept ? 1 : 0);
    return { m, classes, target, l2, k, width, x0: new Float64Array(k * width), options: { gtol: this.tol, maxIter: this.maxIter, ftolFactor: 64 } };
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
    this.#state = { classes, coef, intercept, nIter: result.iterations, nFeaturesIn: p, featureNamesIn: m.names };
    return this;
  }

  /** The most probable class for each row; the scores on the GPU when `device` chooses it. */
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

  /** Each class's probability; the scores on the GPU when `device` chooses it. */
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

  /** Each row's score, X·wᵀ + b, as decisionFunctionSync(); the product on the GPU when `device` chooses it. */
  async decisionFunction(X: MatrixInput, options: DeviceOptions = {}): Promise<Matrix> {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const k = s.coef.rows;
    // On the CPU, exactly decisionFunctionSync()'s result.
    if (!(await useGpu(options.device, m.rows * m.cols * k))) return this.decisionFunctionSync(m);
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
    for (let c = 0; c < k; c++) {
      const w = Float64Array.from({ length: m.cols }, (_, j) => s.coef.data[j * k + c]!);
      xw(m, w, s.intercept[c]!, undefined, out.subarray(c * n, (c + 1) * n));
    }
    return new Matrix(n, k, out);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams(), C: Number.isFinite(this.C) ? this.C : 'Infinity' }, s && {
      classes: [...s.classes],
      coef: s.coef.toRows(),
      intercept: numbers(s.intercept),
      nIter: s.nIter,
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
