import { fail } from '../core/errors.js';
import { minimize } from '../core/linalg.js';
import { assertFinite, Matrix, type MatrixInput } from '../core/matrix.js';
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

  async fit(X: MatrixInput, y: Labels): Promise<this> {
    return this.fitSync(X, y);
  }

  fitSync(X: MatrixInput, y: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    assertFinite(m);
    const labels = toLabels(y, m.rows);
    const classes = uniqueLabels(labels);
    if (classes.length < 2) fail('ML_LABELS', `y has one class only (${String(classes[0])}): a classifier needs at least two`);
    const index = new Map(classes.map((c, k) => [c, k]));
    const target = Int32Array.from(labels, (l) => index.get(l)!);
    const n = m.rows;
    const p = m.cols;
    const l2 = Number.isFinite(this.C) ? 1 / (this.C * n) : 0;
    const k = classes.length === 2 ? 1 : classes.length;
    const width = p + (this.fitIntercept ? 1 : 0);
    const objective = k === 1 ? binaryLoss(m, target, l2, this.fitIntercept) : multinomialLoss(m, target, k, l2, this.fitIntercept);
    const result = minimize(objective, new Float64Array(k * width), { gtol: this.tol, maxIter: this.maxIter, ftolFactor: 64 });
    const coef = new Matrix(k, p);
    const intercept = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < p; j++) coef.data[j * k + c] = result.x[c * width + j]!;
      intercept[c] = this.fitIntercept ? result.x[c * width + p]! : 0;
    }
    this.#state = { classes, coef, intercept, nIter: result.iterations, nFeaturesIn: p, featureNamesIn: m.names };
    return this;
  }

  async predict(X: MatrixInput): Promise<Label[]> {
    return this.predictSync(X);
  }

  /** The most probable class for each row. */
  predictSync(X: MatrixInput): Label[] {
    const s = this.#fitted();
    const scores = this.decisionFunctionSync(X);
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

  async predictProba(X: MatrixInput): Promise<Matrix> {
    return this.predictProbaSync(X);
  }

  /** Each class's probability, one column per class in the order of `classes`. */
  predictProbaSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const scores = this.decisionFunctionSync(X);
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

  /** Each row's score: X·wᵀ + b, one column for two classes (the second class's log-odds), else one per class. */
  decisionFunctionSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const k = s.coef.rows;
    const out = new Float64Array(n * k);
    for (let c = 0; c < k; c++) {
      const off = c * n;
      out.fill(s.intercept[c]!, off, off + n);
      for (let j = 0; j < m.cols; j++) {
        const w = s.coef.data[j * k + c]!;
        const col = j * n;
        for (let i = 0; i < n; i++) out[off + i]! += m.data[col + i]! * w;
      }
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

/** log(1 + e^z), without overflow. */
function log1pExp(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/**
 * The binary objective: the mean log loss of y ∈ {0, 1} under sigmoid(X·w + b), plus l2 / 2 · ‖w‖².
 * x holds w, then b when there is an intercept.
 */
function binaryLoss(X: Matrix, y: Int32Array, l2: number, intercept: boolean) {
  const n = X.rows;
  const p = X.cols;
  const z = new Float64Array(n);
  const r = new Float64Array(n);
  return (x: Float64Array, grad: Float64Array): number => {
    z.fill(intercept ? x[p]! : 0);
    for (let j = 0; j < p; j++) {
      const w = x[j]!;
      if (w === 0) continue;
      const off = j * n;
      for (let i = 0; i < n; i++) z[i]! += X.data[off + i]! * w;
    }
    let loss = 0;
    let rsum = 0;
    for (let i = 0; i < n; i++) {
      const zi = z[i]!;
      // −log p(yᵢ) = log(1 + e^z) − y z
      loss += log1pExp(zi) - y[i]! * zi;
      const ri = sigmoid(zi) - y[i]!;
      r[i] = ri;
      rsum += ri;
    }
    let reg = 0;
    for (let j = 0; j < p; j++) {
      const off = j * n;
      let g = 0;
      for (let i = 0; i < n; i++) g += X.data[off + i]! * r[i]!;
      const w = x[j]!;
      grad[j] = g / n + l2 * w;
      reg += w * w;
    }
    if (intercept) grad[p] = rsum / n;
    return loss / n + (l2 / 2) * reg;
  };
}

/**
 * The multinomial objective: the mean log loss under softmax(X·Wᵀ + b), plus l2 / 2 · ‖W‖². x holds K rows of
 * (w, then b when there is an intercept).
 */
function multinomialLoss(X: Matrix, y: Int32Array, K: number, l2: number, intercept: boolean) {
  const n = X.rows;
  const p = X.cols;
  const width = p + (intercept ? 1 : 0);
  const z = new Float64Array(n * K);
  return (x: Float64Array, grad: Float64Array): number => {
    for (let c = 0; c < K; c++) {
      const off = c * n;
      z.fill(intercept ? x[c * width + p]! : 0, off, off + n);
      for (let j = 0; j < p; j++) {
        const w = x[c * width + j]!;
        if (w === 0) continue;
        const col = j * n;
        for (let i = 0; i < n; i++) z[off + i]! += X.data[col + i]! * w;
      }
    }
    let loss = 0;
    // z becomes the residuals softmax − onehot, row by row.
    for (let i = 0; i < n; i++) {
      let max = -Infinity;
      for (let c = 0; c < K; c++) max = Math.max(max, z[c * n + i]!);
      let sum = 0;
      for (let c = 0; c < K; c++) sum += Math.exp(z[c * n + i]! - max);
      const lse = max + Math.log(sum);
      loss += lse - z[y[i]! * n + i]!;
      for (let c = 0; c < K; c++) z[c * n + i] = Math.exp(z[c * n + i]! - lse) - (c === y[i] ? 1 : 0);
    }
    let reg = 0;
    for (let c = 0; c < K; c++) {
      const off = c * n;
      for (let j = 0; j < p; j++) {
        const col = j * n;
        let g = 0;
        for (let i = 0; i < n; i++) g += X.data[col + i]! * z[off + i]!;
        const w = x[c * width + j]!;
        grad[c * width + j] = g / n + l2 * w;
        reg += w * w;
      }
      if (intercept) {
        let g = 0;
        for (let i = 0; i < n; i++) g += z[off + i]!;
        grad[c * width + p] = g / n;
      }
    }
    return loss / n + (l2 / 2) * reg;
  };
}
