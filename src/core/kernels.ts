/**
 * The numerical kernels the models spend their time in, written for the JavaScript engine: rows in blocks small enough
 * to stay in cache, several columns per loop, and column means subtracted while a block is read rather than in a
 * centred copy of X. Results equal the plain loops' within rounding (the tests check them against those).
 */
import type { Matrix } from './matrix.js';

/** Rows per block: 512 rows of up to a few hundred columns stay within a CPU's second-level cache. */
export const BLOCK = 512;

/**
 * (X − mean)ᵀ (X − mean): the p × p Gram matrix, row-major, both triangles filled. `mean` (one per column) is
 * subtracted as each block is read; leave it out for X as it is. `from` and `to` limit it to those rows, so parts
 * computed in parallel add up to the whole.
 */
export function gramPart(X: Matrix, mean: ArrayLike<number> | undefined, from: number, to: number): Float64Array {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  const A = new Float64Array(p * p);
  const B = BLOCK;
  const buf = new Float64Array(B * p);
  for (let r0 = from; r0 < to; r0 += B) {
    const m = Math.min(B, to - r0);
    for (let j = 0; j < p; j++) {
      const off = j * n + r0;
      const mu = mean ? mean[j]! : 0;
      const bj = j * B;
      for (let i = 0; i < m; i++) buf[bj + i] = d[off + i]! - mu;
    }
    let j = 0;
    // Two rows of A against four of its columns at a time: eight sums in registers per pass over the block.
    for (; j + 1 < p; j += 2) {
      const bj0 = j * B;
      const bj1 = bj0 + B;
      let k = 0;
      for (; k + 3 <= j; k += 4) {
        const bk0 = k * B;
        const bk1 = bk0 + B;
        const bk2 = bk1 + B;
        const bk3 = bk2 + B;
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (let i = 0; i < m; i++) {
          const x0 = buf[bj0 + i]!;
          const x1 = buf[bj1 + i]!;
          const y0 = buf[bk0 + i]!;
          const y1 = buf[bk1 + i]!;
          const y2 = buf[bk2 + i]!;
          const y3 = buf[bk3 + i]!;
          a0 += x0 * y0; a1 += x0 * y1; a2 += x0 * y2; a3 += x0 * y3;
          c0 += x1 * y0; c1 += x1 * y1; c2 += x1 * y2; c3 += x1 * y3;
        }
        const r = j * p + k;
        A[r]! += a0; A[r + 1]! += a1; A[r + 2]! += a2; A[r + 3]! += a3;
        A[r + p]! += c0; A[r + p + 1]! += c1; A[r + p + 2]! += c2; A[r + p + 3]! += c3;
      }
      for (; k <= j + 1; k++) {
        const bk = k * B;
        let s0 = 0;
        let s1 = 0;
        for (let i = 0; i < m; i++) {
          const y = buf[bk + i]!;
          s0 += buf[bj0 + i]! * y;
          s1 += buf[bj1 + i]! * y;
        }
        if (k <= j) A[j * p + k]! += s0;
        A[(j + 1) * p + k]! += s1;
      }
    }
    for (; j < p; j++) {
      const bj = j * B;
      for (let k = 0; k <= j; k++) {
        const bk = k * B;
        let s = 0;
        for (let i = 0; i < m; i++) s += buf[bj + i]! * buf[bk + i]!;
        A[j * p + k]! += s;
      }
    }
  }
  return A;
}

/**
 * Rows per chunk: whole-matrix kernels sum their parts over chunks of this many rows, in order, whether one thread
 * computes them or several workers do, so the result is the same to the last bit however many threads ran it.
 */
export const CHUNK = 16384;

/** The chunks [from, to) of n rows. */
export function chunks(n: number): [number, number][] {
  const out: [number, number][] = [];
  for (let r = 0; r < n; r += CHUNK) out.push([r, Math.min(r + CHUNK, n)]);
  return out;
}

/** Adds part into total, element by element. */
export function addInto(total: Float64Array, part: ArrayLike<number>): void {
  for (let i = 0; i < total.length; i++) total[i]! += part[i]!;
}

/** Fills the upper triangle of a p × p matrix from its lower one. */
export function symmetrize(A: Float64Array, p: number): Float64Array {
  for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) A[k * p + j] = A[j * p + k]!;
  return A;
}

/** (X − mean)ᵀ (X − mean) over every row: gramPart() summed over the chunks, both triangles filled. */
export function gram(X: Matrix, mean?: ArrayLike<number>): Float64Array {
  const A = new Float64Array(X.cols * X.cols);
  for (const [a, b] of chunks(X.rows)) addInto(A, gramPart(X, mean, a, b));
  return symmetrize(A, X.cols);
}

/** (X − mean)ᵀ v over every row: xtvPart() summed over the chunks. */
export function xtv(X: Matrix, v: ArrayLike<number>, mean?: ArrayLike<number>): Float64Array {
  const out = new Float64Array(X.cols);
  for (const [a, b] of chunks(X.rows)) addInto(out, xtvPart(X, v, mean, a, b));
  return out;
}

/** (X − mean)ᵀ v over rows `from` to `to`: one value per column. */
export function xtvPart(X: Matrix, v: ArrayLike<number>, mean: ArrayLike<number> | undefined, from: number, to: number, out = new Float64Array(X.cols)): Float64Array {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  out.fill(0);
  let vsum = 0;
  if (mean) for (let i = from; i < to; i++) vsum += v[i]!;
  for (let r0 = from; r0 < to; r0 += BLOCK) {
    const m = Math.min(BLOCK, to - r0);
    let j = 0;
    for (; j + 3 < p; j += 4) {
      const o0 = j * n + r0;
      const o1 = o0 + n;
      const o2 = o1 + n;
      const o3 = o2 + n;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < m; i++) {
        const r = v[r0 + i]!;
        s0 += d[o0 + i]! * r; s1 += d[o1 + i]! * r; s2 += d[o2 + i]! * r; s3 += d[o3 + i]! * r;
      }
      out[j]! += s0; out[j + 1]! += s1; out[j + 2]! += s2; out[j + 3]! += s3;
    }
    for (; j < p; j++) {
      const o = j * n + r0;
      let s = 0;
      for (let i = 0; i < m; i++) s += d[o + i]! * v[r0 + i]!;
      out[j]! += s;
    }
  }
  // Σ (xᵢ − μ) vᵢ = Σ xᵢ vᵢ − μ Σ vᵢ. Where it is used (the residual of a centred fit) Σ vᵢ is close to 0.
  if (mean) for (let j = 0; j < p; j++) out[j]! -= mean[j]! * vsum;
  return out;
}

/** (X − mean) w + b: one value per row (rows `from` to `to` of `out`). */
export function xw(X: Matrix, w: ArrayLike<number>, b = 0, mean?: ArrayLike<number>, out = new Float64Array(X.rows), from = 0, to = X.rows): Float64Array {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  let shift = b;
  if (mean) for (let j = 0; j < p; j++) shift -= mean[j]! * w[j]!;
  out.fill(shift, from, to);
  for (let r0 = from; r0 < to; r0 += BLOCK) {
    const m = Math.min(BLOCK, to - r0);
    let j = 0;
    // Eight columns per pass over the block's slice of out.
    for (; j + 7 < p; j += 8) {
      const o = j * n + r0;
      const w0 = w[j]!, w1 = w[j + 1]!, w2 = w[j + 2]!, w3 = w[j + 3]!, w4 = w[j + 4]!, w5 = w[j + 5]!, w6 = w[j + 6]!, w7 = w[j + 7]!;
      for (let i = 0; i < m; i++) {
        out[r0 + i]! +=
          d[o + i]! * w0 + d[o + n + i]! * w1 + d[o + 2 * n + i]! * w2 + d[o + 3 * n + i]! * w3 +
          d[o + 4 * n + i]! * w4 + d[o + 5 * n + i]! * w5 + d[o + 6 * n + i]! * w6 + d[o + 7 * n + i]! * w7;
      }
    }
    for (; j + 3 < p; j += 4) {
      const o0 = j * n + r0;
      const o1 = o0 + n;
      const o2 = o1 + n;
      const o3 = o2 + n;
      const w0 = w[j]!, w1 = w[j + 1]!, w2 = w[j + 2]!, w3 = w[j + 3]!;
      for (let i = 0; i < m; i++) out[r0 + i]! += d[o0 + i]! * w0 + d[o1 + i]! * w1 + d[o2 + i]! * w2 + d[o3 + i]! * w3;
    }
    for (; j < p; j++) {
      const o = j * n + r0;
      const wj = w[j]!;
      for (let i = 0; i < m; i++) out[r0 + i]! += d[o + i]! * wj;
    }
  }
  return out;
}

function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function log1pExp(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/**
 * The binary logistic objective and its gradient in one pass over X: for each block of rows, the scores, then the
 * residuals, then their contribution to the gradient, while the block is still in cache. x holds w, then b when
 * there is an intercept. Returns the mean log loss plus l2 / 2 · ‖w‖².
 */
export function binaryLogistic(X: Matrix, y: Int32Array, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array): number {
  return finishLogistic(sumParts(X.rows, grad, (g, a, b) => binaryLogisticPart(X, y, intercept, x, g, a, b)), X, l2, intercept, x, grad, 1);
}

/** Sums a logistic part over the chunks, in order: the loss returned, the gradient into `grad`. */
function sumParts(n: number, grad: Float64Array, part: (g: Float64Array, from: number, to: number) => number): number {
  grad.fill(0);
  const g = new Float64Array(grad.length);
  let loss = 0;
  for (const [a, b] of chunks(n)) {
    loss += part(g, a, b);
    addInto(grad, g);
  }
  return loss;
}

/**
 * binaryLogistic()'s sums over rows `from` to `to`: the log loss (returned) and, in `grad`, Xᵀr, then Σr when there
 * is an intercept. Parts over disjoint rows add up; finishLogistic() turns the total into the objective.
 */
export function binaryLogisticPart(X: Matrix, y: Int32Array, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number): number {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  const z = new Float64Array(BLOCK);
  grad.fill(0);
  const b = intercept ? x[p]! : 0;
  let loss = 0;
  let rsum = 0;
  for (let r0 = from; r0 < to; r0 += BLOCK) {
    const m = Math.min(BLOCK, to - r0);
    z.fill(b);
    let j = 0;
    for (; j + 3 < p; j += 4) {
      const o0 = j * n + r0;
      const o1 = o0 + n;
      const o2 = o1 + n;
      const o3 = o2 + n;
      const w0 = x[j]!, w1 = x[j + 1]!, w2 = x[j + 2]!, w3 = x[j + 3]!;
      for (let i = 0; i < m; i++) z[i]! += d[o0 + i]! * w0 + d[o1 + i]! * w1 + d[o2 + i]! * w2 + d[o3 + i]! * w3;
    }
    for (; j < p; j++) {
      const o = j * n + r0;
      const wj = x[j]!;
      for (let i = 0; i < m; i++) z[i]! += d[o + i]! * wj;
    }
    for (let i = 0; i < m; i++) {
      const zi = z[i]!;
      const yi = y[r0 + i]!;
      // −log p(yᵢ) = log(1 + e^z) − y z; the residual σ(z) − y replaces z.
      loss += log1pExp(zi) - yi * zi;
      const ri = sigmoid(zi) - yi;
      z[i] = ri;
      rsum += ri;
    }
    j = 0;
    for (; j + 3 < p; j += 4) {
      const o0 = j * n + r0;
      const o1 = o0 + n;
      const o2 = o1 + n;
      const o3 = o2 + n;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < m; i++) {
        const r = z[i]!;
        s0 += d[o0 + i]! * r; s1 += d[o1 + i]! * r; s2 += d[o2 + i]! * r; s3 += d[o3 + i]! * r;
      }
      grad[j]! += s0; grad[j + 1]! += s1; grad[j + 2]! += s2; grad[j + 3]! += s3;
    }
    for (; j < p; j++) {
      const o = j * n + r0;
      let s = 0;
      for (let i = 0; i < m; i++) s += d[o + i]! * z[i]!;
      grad[j]! += s;
    }
  }
  if (intercept) grad[p] = rsum;
  return loss;
}

/**
 * The multinomial logistic objective and its gradient in one pass over X, as binaryLogistic(): x holds K rows of
 * (w, then b when there is an intercept).
 */
export function multinomialLogistic(X: Matrix, y: Int32Array, K: number, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array): number {
  return finishLogistic(sumParts(X.rows, grad, (g, a, b) => multinomialLogisticPart(X, y, K, intercept, x, g, a, b)), X, l2, intercept, x, grad, K);
}

/** multinomialLogistic()'s sums over rows `from` to `to`, as binaryLogisticPart() gives the binary ones. */
export function multinomialLogisticPart(X: Matrix, y: Int32Array, K: number, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number): number {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  const width = p + (intercept ? 1 : 0);
  const z = new Float64Array(BLOCK * K);
  grad.fill(0);
  let loss = 0;
  for (let r0 = from; r0 < to; r0 += BLOCK) {
    const m = Math.min(BLOCK, to - r0);
    for (let c = 0; c < K; c++) {
      const zc = c * BLOCK;
      z.fill(intercept ? x[c * width + p]! : 0, zc, zc + m);
      const wc = c * width;
      let j = 0;
      for (; j + 3 < p; j += 4) {
        const o0 = j * n + r0;
        const o1 = o0 + n;
        const o2 = o1 + n;
        const o3 = o2 + n;
        const w0 = x[wc + j]!, w1 = x[wc + j + 1]!, w2 = x[wc + j + 2]!, w3 = x[wc + j + 3]!;
        for (let i = 0; i < m; i++) z[zc + i]! += d[o0 + i]! * w0 + d[o1 + i]! * w1 + d[o2 + i]! * w2 + d[o3 + i]! * w3;
      }
      for (; j < p; j++) {
        const o = j * n + r0;
        const wj = x[wc + j]!;
        for (let i = 0; i < m; i++) z[zc + i]! += d[o + i]! * wj;
      }
    }
    for (let i = 0; i < m; i++) {
      let max = -Infinity;
      for (let c = 0; c < K; c++) max = Math.max(max, z[c * BLOCK + i]!);
      let sum = 0;
      for (let c = 0; c < K; c++) sum += Math.exp(z[c * BLOCK + i]! - max);
      const lse = max + Math.log(sum);
      const yi = y[r0 + i]!;
      loss += lse - z[yi * BLOCK + i]!;
      // The residuals softmax − one-hot replace the scores.
      for (let c = 0; c < K; c++) z[c * BLOCK + i] = Math.exp(z[c * BLOCK + i]! - lse) - (c === yi ? 1 : 0);
    }
    for (let c = 0; c < K; c++) {
      const zc = c * BLOCK;
      const gc = c * width;
      let j = 0;
      for (; j + 3 < p; j += 4) {
        const o0 = j * n + r0;
        const o1 = o0 + n;
        const o2 = o1 + n;
        const o3 = o2 + n;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let i = 0; i < m; i++) {
          const r = z[zc + i]!;
          s0 += d[o0 + i]! * r; s1 += d[o1 + i]! * r; s2 += d[o2 + i]! * r; s3 += d[o3 + i]! * r;
        }
        grad[gc + j]! += s0; grad[gc + j + 1]! += s1; grad[gc + j + 2]! += s2; grad[gc + j + 3]! += s3;
      }
      for (; j < p; j++) {
        const o = j * n + r0;
        let s = 0;
        for (let i = 0; i < m; i++) s += d[o + i]! * z[zc + i]!;
        grad[gc + j]! += s;
      }
      if (intercept) {
        let s = 0;
        for (let i = 0; i < m; i++) s += z[zc + i]!;
        grad[gc + p]! += s;
      }
    }
  }
  return loss;
}

/**
 * The objective from a logistic loss summed over every row, with `grad` holding the summed gradient: both divided by
 * n, plus the L2 penalty l2 / 2 · ‖w‖² (never on the intercepts) and its gradient.
 */
export function finishLogistic(loss: number, X: Matrix, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array, K: number): number {
  const n = X.rows;
  const p = X.cols;
  const width = p + (intercept ? 1 : 0);
  let reg = 0;
  for (let c = 0; c < K; c++) {
    for (let j = 0; j < p; j++) {
      const w = x[c * width + j]!;
      grad[c * width + j] = grad[c * width + j]! / n + l2 * w;
      reg += w * w;
    }
    if (intercept) grad[c * width + p] = grad[c * width + p]! / n;
  }
  return loss / n + (l2 / 2) * reg;
}

/**
 * Solves A x = b for a symmetric positive definite p × p A (row-major), by Cholesky. Returns null when A isn't
 * positive definite to working precision, so the caller can fall back or fail as it should.
 */
export function cholesky(A: Float64Array, p: number): Float64Array | null {
  const L = new Float64Array(p * p);
  let scale = 0;
  for (let j = 0; j < p; j++) scale = Math.max(scale, Math.abs(A[j * p + j]!));
  for (let j = 0; j < p; j++) {
    let s = A[j * p + j]!;
    for (let k = 0; k < j; k++) s -= L[j * p + k]! ** 2;
    if (!(s > p * Number.EPSILON * scale)) return null;
    const ljj = Math.sqrt(s);
    L[j * p + j] = ljj;
    for (let i = j + 1; i < p; i++) {
      let t = A[i * p + j]!;
      for (let k = 0; k < j; k++) t -= L[i * p + k]! * L[j * p + k]!;
      L[i * p + j] = t / ljj;
    }
  }
  return L;
}

/** Solves L Lᵀ x = b, with L from cholesky(). */
export function choleskySolve(L: Float64Array, p: number, b: ArrayLike<number>): Float64Array {
  const z = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= L[i * p + k]! * z[k]!;
    z[i] = s / L[i * p + i]!;
  }
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) {
    let s = z[i]!;
    for (let k = i + 1; k < p; k++) s -= L[k * p + i]! * x[k]!;
    x[i] = s / L[i * p + i]!;
  }
  return x;
}

/**
 * Each column's mean, checking on the way that X holds no missing or infinite value: one pass where a check and the
 * means would take two. A NaN or ±Infinity makes its column's sum non-finite; only then is X searched, to name the
 * value. (A sum can also overflow to Infinity from huge finite values; the search then finds nothing, and the means
 * are what they are.)
 */
export function checkedMeans(X: Matrix, check: (X: Matrix) => void): Float64Array {
  const n = X.rows;
  const d = X.data;
  const out = new Float64Array(X.cols);
  let bad = false;
  for (let j = 0; j < X.cols; j++) {
    const o = j * n;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let i = 0;
    for (; i + 3 < n; i += 4) {
      s0 += d[o + i]!; s1 += d[o + i + 1]!; s2 += d[o + i + 2]!; s3 += d[o + i + 3]!;
    }
    for (; i < n; i++) s0 += d[o + i]!;
    const s = s0 + s1 + s2 + s3;
    if (!Number.isFinite(s)) bad = true;
    out[j] = s / n;
  }
  if (bad) check(X);
  return out;
}
