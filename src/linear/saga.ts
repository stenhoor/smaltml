import type { MinimizeResult } from '../core/linalg.js';
import type { Matrix } from '../core/matrix.js';
import { Random } from '../core/random.js';

/** What SAGA is given: the data, the classes, the penalty and when to stop. */
export interface SagaProblem {
  X: Matrix;
  /** Each row's class index. */
  y: Int32Array;
  /** 1 for two classes (one score, the second class's log-odds), else the number of classes (softmax). */
  K: number;
  intercept: boolean;
  /** Each row's weight, or undefined for 1 each. */
  weights: Float64Array | undefined;
  /** The rows' total weight (n without weights). */
  total: number;
  /** The L2 and L1 penalties against the weighted mean loss: λ2/2·‖w‖² + λ1·‖w‖₁, never on the intercepts. */
  l2: number;
  l1: number;
  tol: number;
  maxIter: number;
  seed: number;
}

/**
 * SAGA (Defazio, Bach and Lacoste-Julien, 2014), the proximal stochastic average gradient, for the logistic objective
 * with an elastic-net penalty: the weighted mean log loss + λ2/2·‖w‖² + λ1·‖w‖₁. Each step takes one row at random,
 * replaces that row's stored gradient with its gradient at the current point, and moves by the stored gradients'
 * mean (corrected by the change in that row's), then the L2 part, then the L1 part's proximal step (soft-thresholding).
 * The table of gradients is kept as each row's residuals (K numbers a row), as a linear model's gradient is a residual
 * times the row. The step is 1/(3L), with L the loss's largest smoothness constant over the rows plus λ2 (the paper's
 * choice). It stops, as scikit-learn's sag and saga solvers do, after an epoch (n steps) in which the largest change in
 * a coefficient or intercept is at most tol times the largest of them.
 */
export function saga(problem: SagaProblem): MinimizeResult {
  const { X, y, K, intercept, weights, total, l2, l1, tol, maxIter } = problem;
  const n = X.rows;
  const p = X.cols;
  const width = p + (intercept ? 1 : 0);
  // The rows row-major, for one row at a time.
  const rows = new Float64Array(n * p);
  for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) rows[i * p + j] = X.data[j * n + i]!;
  // Each row's weight in the mean: w·n / Σw, 1 without weights.
  const scale = Float64Array.from({ length: n }, (_, i) => (weights ? (weights[i]! * n) / total : 1));
  let maxSq = 0;
  for (let i = 0; i < n; i++) {
    let s = intercept ? 1 : 0;
    for (let j = 0; j < p; j++) s += rows[i * p + j]! ** 2;
    maxSq = Math.max(maxSq, scale[i]! * s);
  }
  // The log loss's second derivative is at most 1/4 for two classes, 1/2 for the softmax.
  const L = (K === 1 ? 0.25 : 0.5) * maxSq + l2;
  const step = 1 / (3 * L);
  const x = new Float64Array(K * width);
  const table = new Float64Array(n * K);
  const mean = new Float64Array(K * width);
  const z = new Float64Array(K);
  const r = new Float64Array(K);
  // The residuals at the start (x = 0) fill the table, so the mean starts as the full gradient there.
  for (let i = 0; i < n; i++) {
    residuals(i, r);
    for (let c = 0; c < K; c++) {
      const d = scale[i]! * r[c]!;
      table[i * K + c] = r[c]!;
      for (let j = 0; j < p; j++) mean[c * width + j]! += (d * rows[i * p + j]!) / n;
      if (intercept) mean[c * width + p]! += d / n;
    }
  }
  const rng = new Random(problem.seed);
  const before = new Float64Array(K * width);
  const shrink = step * l1;
  let iterations = 0;
  let converged = false;
  for (; iterations < maxIter; iterations++) {
    before.set(x);
    for (let t = 0; t < n; t++) {
      const i = rng.int(n);
      residuals(i, r);
      for (let c = 0; c < K; c++) {
        const old = table[i * K + c]!;
        const delta = scale[i]! * (r[c]! - old);
        table[i * K + c] = r[c]!;
        const o = c * width;
        for (let j = 0; j < p; j++) {
          const xij = rows[i * p + j]!;
          let w = x[o + j]! - step * (delta * xij + mean[o + j]! + l2 * x[o + j]!);
          if (shrink > 0) w = w > shrink ? w - shrink : w < -shrink ? w + shrink : 0;
          x[o + j] = w;
          mean[o + j]! += (delta * xij) / n;
        }
        if (intercept) {
          x[o + p]! -= step * (delta + mean[o + p]!);
          mean[o + p]! += delta / n;
        }
      }
    }
    let change = 0;
    let largest = 0;
    for (let k = 0; k < x.length; k++) {
      change = Math.max(change, Math.abs(x[k]! - before[k]!));
      largest = Math.max(largest, Math.abs(x[k]!));
    }
    if (change <= tol * largest || largest === 0) {
      iterations++;
      converged = true;
      break;
    }
  }
  return { x, value: NaN, iterations, converged };

  /** Row i's residuals at x: σ(z) − y for two classes, softmax − one-hot for more. */
  function residuals(i: number, out: Float64Array): void {
    for (let c = 0; c < K; c++) {
      const o = c * width;
      let s = intercept ? x[o + p]! : 0;
      for (let j = 0; j < p; j++) s += x[o + j]! * rows[i * p + j]!;
      z[c] = s;
    }
    const yi = y[i]!;
    if (K === 1) {
      const zi = z[0]!;
      out[0] = (zi >= 0 ? 1 / (1 + Math.exp(-zi)) : Math.exp(zi) / (1 + Math.exp(zi))) - yi;
      return;
    }
    let max = -Infinity;
    for (let c = 0; c < K; c++) max = Math.max(max, z[c]!);
    let sum = 0;
    for (let c = 0; c < K; c++) sum += out[c] = Math.exp(z[c]! - max);
    for (let c = 0; c < K; c++) out[c] = out[c]! / sum - (c === yi ? 1 : 0);
  }
}
