import { fail } from '../core/errors.js';
import { cholesky, choleskySolve } from '../core/kernels.js';
import type { Matrix } from '../core/matrix.js';
import type { Solver } from './calls.js';

/**
 * Least squares with every coefficient 0 or more: minimizes ‖y − X·w‖² + ridge·‖w‖² subject to w ≥ 0, for X less
 * `mean` (when there is an intercept). The active-set method of Lawson and Hanson (Solving Least Squares Problems,
 * 1974, chapter 23), on the normal equations: A = XᵀX + ridge·I and b = Xᵀy, so X is read once (on several threads
 * where the caller runs it so) and each step solves a small system by Cholesky.
 *
 * The set P holds the coefficients free to move; the rest are held at 0. Each outer step frees the held coefficient
 * whose gradient most wants it above 0; the inner steps solve on P, and where that solution goes below 0, step back
 * along the way to the first coefficient that reaches 0 and hold it there. It stops when no held coefficient's
 * gradient points up: the Karush–Kuhn–Tucker conditions.
 */
export function* nonNegativeLeastSquares(X: Matrix, mean: Float64Array | undefined, y: Float64Array, ridge: number): Solver<{ coef: Float64Array; nIter: number }> {
  const p = X.cols;
  const A: Float64Array = yield { k: 'gram', X, mean };
  for (let j = 0; j < p; j++) A[j * p + j]! += ridge;
  const b: Float64Array = yield { k: 'xtv', X, v: y, mean };
  const w = new Float64Array(p);
  const free = new Uint8Array(p);
  // The gradient's scale, for the stopping test: a component this small is rounding.
  let scale = 0;
  for (let j = 0; j < p; j++) scale = Math.max(scale, Math.abs(b[j]!), A[j * p + j]!);
  const tol = 1e-12 * scale;
  const maxIter = 3 * p + 10;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    // g = b − A·w: minus the gradient of ½wᵀAw − bᵀw.
    let best = -1;
    let most = tol;
    for (let j = 0; j < p; j++) {
      if (free[j]) continue;
      let g = b[j]!;
      for (let k = 0; k < p; k++) g -= A[j * p + k]! * w[k]!;
      if (g > most) {
        most = g;
        best = j;
      }
    }
    if (best < 0) break;
    free[best] = 1;
    for (;;) {
      const s = solveFree(A, b, free, p);
      let step = 1;
      let block = -1;
      for (let j = 0; j < p; j++) {
        if (!free[j] || s[j]! > 0) continue;
        const t = w[j]! / (w[j]! - s[j]!);
        if (block < 0 || t < step) {
          step = t;
          block = j;
        }
      }
      if (block < 0) {
        w.set(s);
        break;
      }
      for (let j = 0; j < p; j++) if (free[j]) w[j] = w[j]! + step * (s[j]! - w[j]!);
      // The coefficient that reached 0, and any that rounding took to 0 or below, are held there.
      w[block] = 0;
      for (let j = 0; j < p; j++) {
        if (free[j] && w[j]! <= 0) {
          w[j] = 0;
          free[j] = 0;
        }
      }
    }
  }
  if (iter === maxIter) fail('ML_SINGULAR', "the non-negative least-squares fit didn't settle: the columns are nearly dependent");
  return { coef: w, nIter: iter };
}

/** The solution on the free coefficients, A_PP s_P = b_P, by Cholesky; 0 elsewhere. */
function solveFree(A: Float64Array, b: Float64Array, free: Uint8Array, p: number): Float64Array {
  const idx: number[] = [];
  for (let j = 0; j < p; j++) if (free[j]) idx.push(j);
  const k = idx.length;
  const sub = new Float64Array(k * k);
  const rhs = new Float64Array(k);
  for (let a = 0; a < k; a++) {
    rhs[a] = b[idx[a]!]!;
    for (let c = 0; c < k; c++) sub[a * k + c] = A[idx[a]! * p + idx[c]!]!;
  }
  const L = cholesky(sub, k);
  if (!L) fail('ML_SINGULAR', "the columns don't determine one answer (one is a combination of others)");
  const x = choleskySolve(L, k, rhs);
  const s = new Float64Array(p);
  for (let a = 0; a < k; a++) s[idx[a]!] = x[a]!;
  return s;
}
