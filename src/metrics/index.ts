/**
 * Metrics: how good predictions are. scikit-learn's `metrics`, by the same definitions.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { toLabels, uniqueLabels, type Label, type Labels } from '../core/model.js';

/** The mean of the squared errors: Σ (yᵢ − ŷᵢ)² / n. scikit-learn's `mean_squared_error`. */
export function meanSquaredError(yTrue: VectorInput, yPred: VectorInput): number {
  const [t, p] = pair(yTrue, yPred);
  let s = 0;
  for (let i = 0; i < t.length; i++) {
    const d = t[i]! - p[i]!;
    s += d * d;
  }
  return s / t.length;
}

/**
 * The coefficient of determination: 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)². 1 is a perfect fit; 0 is no better than
 * predicting the mean, and worse is negative. When y is constant it is 1 for a perfect prediction, else 0.
 * scikit-learn's `r2_score`.
 */
export function r2Score(yTrue: VectorInput, yPred: VectorInput): number {
  const [t, p] = pair(yTrue, yPred);
  let mean = 0;
  for (let i = 0; i < t.length; i++) mean += t[i]!;
  mean /= t.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < t.length; i++) {
    ssRes += (t[i]! - p[i]!) ** 2;
    ssTot += (t[i]! - mean) ** 2;
  }
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}

/** The share of rows whose predicted label equals the true one, from 0 to 1. scikit-learn's `accuracy_score`. */
export function accuracyScore(yTrue: Labels, yPred: Labels): number {
  const t = toLabels(yTrue, yTrue?.length ?? 0);
  const p = toLabels(yPred, t.length);
  if (t.length === 0) fail('ML_SHAPE', 'yTrue is empty');
  let right = 0;
  for (let i = 0; i < t.length; i++) if (t[i] === p[i]) right++;
  return right / t.length;
}

export interface LogLossOptions {
  /** The classes, in the order of yProba's columns. Default: yTrue's labels, sorted. */
  labels?: readonly Label[];
}

/**
 * The mean negative log-likelihood of the true labels: −Σ log pᵢ(yᵢ) / n. `yProba` has a column per class (as
 * predictProba() gives), or, for two classes, the second class's probability alone. Probabilities are clipped to
 * [ε, 1 − ε], ε the float64 epsilon. scikit-learn's `log_loss`.
 */
export function logLoss(yTrue: Labels, yProba: MatrixInput | ArrayLike<number>, options: LogLossOptions = {}): number {
  const t = toLabels(yTrue, yTrue?.length ?? 0);
  const n = t.length;
  if (n === 0) fail('ML_SHAPE', 'yTrue is empty');
  const classes = options.labels ? [...options.labels] : uniqueLabels(t);
  if (classes.length < 2) {
    fail('ML_LABELS', `yTrue has one class only (${String(classes[0])}): pass every class in options.labels`);
  }
  let P: Matrix;
  if (isFlat(yProba)) {
    const p1 = toVector(yProba, n, 'yProba');
    P = new Matrix(n, 2, Float64Array.from([...p1].map((v) => 1 - v).concat([...p1])));
  } else {
    P = Matrix.from(yProba as MatrixInput);
  }
  if (P.rows !== n) fail('ML_SHAPE', `yProba has ${P.rows} rows; yTrue has ${n}`);
  if (P.cols !== classes.length) fail('ML_SHAPE', `yProba has ${P.cols} columns for ${classes.length} classes`);
  const index = new Map(classes.map((c, k) => [c, k]));
  const eps = Number.EPSILON;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const k = index.get(t[i]!);
    if (k === undefined) fail('ML_LABELS', `yTrue[${i}] is ${String(t[i])}, which isn't one of the labels`);
    const p = Math.min(Math.max(P.data[k * n + i]!, eps), 1 - eps);
    s -= Math.log(p);
  }
  return s / n;
}

function pair(yTrue: VectorInput, yPred: VectorInput): [Float64Array, Float64Array] {
  const t = toVector(yTrue, yTrue?.length ?? 0, 'yTrue');
  if (t.length === 0) fail('ML_SHAPE', 'yTrue is empty');
  return [t, toVector(yPred, t.length, 'yPred')];
}

function isFlat(v: unknown): v is ArrayLike<number> {
  return ArrayBuffer.isView(v) || (Array.isArray(v) && (v.length === 0 || typeof v[0] === 'number'));
}
