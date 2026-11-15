import { fail } from '../core/errors.js';
import { assertFinite, Matrix, toVector, type VectorInput } from '../core/matrix.js';

/** Per-row weights for a fit: `sampleWeight`, scikit-learn's `sample_weight`. */
export interface SampleWeightOptions {
  /** One weight per row of X, 0 or more; a row with weight 2 counts as that row twice. Default: every row 1. */
  sampleWeight?: VectorInput;
}

/** The weights as a vector, checked: one per row, finite, none negative, not all 0. Undefined when none are given. */
export function checkedWeights(sampleWeight: VectorInput | undefined, n: number): Float64Array | undefined {
  if (sampleWeight === undefined) return undefined;
  let w: Float64Array;
  try {
    w = toVector(sampleWeight, n);
  } catch {
    return fail('ML_SHAPE', `sampleWeight needs one number per row of X (${n})`);
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = w[i]!;
    if (!Number.isFinite(v) || v < 0) fail('ML_VALUE', `sampleWeight must be finite and 0 or more; row ${i} has ${v}`);
    sum += v;
  }
  if (!(sum > 0)) fail('ML_VALUE', 'sampleWeight is 0 for every row: at least one row needs weight');
  return w;
}

/**
 * Least squares with weights, as one without: the columns and y are centred on their weighted means (with an
 * intercept), and each row is multiplied by √w. Minimizing Σ (√w·r)² is minimizing Σ w·r², so the unweighted solvers
 * give the weighted answer on the result. This makes a copy of X, which the kernels then read as it is (no means).
 */
export function weightedLeastSquares(m: Matrix, y: Float64Array, w: Float64Array, intercept: boolean) {
  assertFinite(m);
  const n = m.rows;
  const p = m.cols;
  let total = 0;
  for (let i = 0; i < n; i++) total += w[i]!;
  const root = Float64Array.from(w, Math.sqrt);
  let xMean: Float64Array | undefined;
  let yMean = 0;
  if (intercept) {
    xMean = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const o = j * n;
      let s = 0;
      for (let i = 0; i < n; i++) s += w[i]! * m.data[o + i]!;
      xMean[j] = s / total;
    }
    for (let i = 0; i < n; i++) yMean += w[i]! * y[i]!;
    yMean /= total;
  }
  const data = new Float64Array(n * p);
  for (let j = 0; j < p; j++) {
    const o = j * n;
    const mu = xMean ? xMean[j]! : 0;
    for (let i = 0; i < n; i++) data[o + i] = (m.data[o + i]! - mu) * root[i]!;
  }
  const target = Float64Array.from(y, (v, i) => (v - yMean) * root[i]!);
  return { X: new Matrix(n, p, data, m.names), y: target, xMean, yMean };
}

/** The weights scaled to sum to n, as scikit-learn's ElasticNet and Lasso scale them before they fit. */
export function weightsSummingToRows(w: Float64Array): Float64Array {
  let total = 0;
  for (const v of w) total += v;
  const k = w.length / total;
  return Float64Array.from(w, (v) => v * k);
}
