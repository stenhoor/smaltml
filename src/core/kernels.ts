/**
 * The numerical kernels the models spend their time in, written for the JavaScript engine: rows in blocks small enough
 * to stay in cache, several columns per loop, and column means subtracted while a block is read rather than in a
 * centred copy of X. Results equal the plain loops' within rounding (the tests check them against those).
 */
import type { Matrix } from './matrix.js';

/**
 * The kernels another set can replace, such as smaltml's SIMD kernels (useKernels(), in kernel-set.ts), with the same signatures as smaltml's own. Each
 * one, called from here, runs the replacement when there is one and smaltml's own otherwise.
 */
export interface Kernels {
  /** The Gram matrix over a range of rows: {@link ownGramPart}'s signature and result. */
  gramPart?: (X: Matrix, mean: ArrayLike<number> | undefined, from: number, to: number, weights?: ArrayLike<number>, ones?: boolean) => Float64Array;
  /** The matrix product of A (n × p) and B (p × k), both column-major: {@link ownMatMul}'s signature and result. */
  matMul?: (n: number, p: number, k: number, a: Float64Array, b: Float64Array) => Float64Array;
  /** The binary logistic objective's row step: {@link ownBinaryLogisticRows}'s signature and result. */
  binaryLogisticRows?: (z: Float64Array, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;
  /** The multinomial logistic objective's row step: {@link ownMultinomialLogisticRows}'s signature and result. */
  multinomialLogisticRows?: (z: Float64Array, stride: number, K: number, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;
  /**
   * The logistic objectives' products over X (X·w and Xᵀr) for one fit: X kept by the kernel for the fit's length,
   * read on every evaluation; `null` where it won't take this X. See {@link LogisticProducts}.
   */
  logisticProducts?: (X: Matrix) => LogisticProducts | null;
}

/** The binary row step's signature: {@link binaryLogisticRows}. */
export type BinaryRows = (z: Float64Array, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;
/** The multinomial row step's signature: {@link multinomialLogisticRows}. */
export type MultinomialRows = (z: Float64Array, stride: number, K: number, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;

/**
 * One fit's logistic products, from the `logisticProducts` kernel: binaryLogisticPart() and multinomialLogisticPart()
 * on the X it was given, with their results, and the row step passed in (`rows`, a block of `BLOCK` rows at a time,
 * scores at stride `BLOCK`), so the row step stays smaltml's or its own kernel. release() frees X's copy.
 */
export interface LogisticProducts {
  binaryPart(y: Int32Array, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number, weights: Float64Array | undefined, rows: BinaryRows): number;
  multinomialPart(y: Int32Array, K: number, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number, weights: Float64Array | undefined, rows: MultinomialRows): number;
  release(): void;
}

/** The `logisticProducts` kernel's products over X for one fit, or `null` (smaltml's own then): release() them after. */
export function prepareLogistic(X: Matrix): LogisticProducts | null {
  const k = replaced.logisticProducts;
  return k ? k(X) : null;
}

/** The replacements in use: set only by useKernels(). */
export const replaced: Kernels = {};

/** Rows per block: 512 rows of up to a few hundred columns stay within a CPU's second-level cache. */
export const BLOCK = 512;

/**
 * (X − mean)ᵀ (X − mean): the p × p Gram matrix, row-major, the lower triangle filled. `mean` (one per column) is
 * subtracted as each block is read; leave it out for X as it is. `from` and `to` limit it to those rows, so parts
 * computed in parallel add up to the whole. With `weights` (one per row, 0 or more), it is the weighted Gram matrix
 * Xᵀ diag(weights) X, and with `ones`, X gains a last column of ones: (p + 1) × (p + 1), as a Newton step needs.
 */
export function gramPart(X: Matrix, mean: ArrayLike<number> | undefined, from: number, to: number, weights?: ArrayLike<number>, ones = false): Float64Array {
  const k = replaced.gramPart;
  return k ? k(X, mean, from, to, weights, ones) : ownGramPart(X, mean, from, to, weights, ones);
}

/** smaltml's own gramPart(), in TypeScript: the one that runs unless useKernels() replaced it. */
export function ownGramPart(X: Matrix, mean: ArrayLike<number> | undefined, from: number, to: number, weights?: ArrayLike<number>, ones = false): Float64Array {
  const n = X.rows;
  const cols = X.cols;
  const p = cols + (ones ? 1 : 0);
  const d = X.data;
  const A = new Float64Array(p * p);
  const B = BLOCK;
  const buf = new Float64Array(B * p);
  // √weight of each row of the block, once per row.
  const root = new Float64Array(weights ? B : 0);
  for (let r0 = from; r0 < to; r0 += B) {
    const m = Math.min(B, to - r0);
    if (weights) for (let i = 0; i < m; i++) root[i] = Math.sqrt(weights[r0 + i]!);
    // Each row times √weight, so the plain sums of products below are the weighted ones.
    for (let j = 0; j < cols; j++) {
      const off = j * n + r0;
      const mu = mean ? mean[j]! : 0;
      const bj = j * B;
      if (weights) for (let i = 0; i < m; i++) buf[bj + i] = (d[off + i]! - mu) * root[i]!;
      else for (let i = 0; i < m; i++) buf[bj + i] = d[off + i]! - mu;
    }
    if (ones) {
      const bj = cols * B;
      if (weights) buf.set(root.subarray(0, m), bj);
      else buf.fill(1, bj, bj + m);
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

/** Xᵀ diag(weights) X over every row, with a last column of ones when `ones`: gramPart() summed over the chunks, both triangles filled. */
export function weightedGram(X: Matrix, weights: ArrayLike<number>, ones: boolean): Float64Array {
  const q = X.cols + (ones ? 1 : 0);
  const A = new Float64Array(q * q);
  for (const [a, b] of chunks(X.rows)) addInto(A, gramPart(X, undefined, a, b, weights, ones));
  return symmetrize(A, q);
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

/**
 * xw() for several weight vectors at once: outs[c][i] = ws[c]·X[i] + bs[c] over rows `from` to `to`. Three at a time
 * share one pass over each block of X, so each value of X is read once for the three; each sum adds the same products
 * in the same groups (eight columns, then four, then one) as xw(), so the result equals xw()'s bit for bit. The rest
 * go through xw().
 */
export function xwEach(X: Matrix, ws: readonly ArrayLike<number>[], bs: readonly number[], outs: readonly Float64Array<ArrayBuffer>[], from = 0, to = X.rows): void {
  let c = 0;
  for (; c + 2 < ws.length; c += 3) xw3(X, ws[c]!, ws[c + 1]!, ws[c + 2]!, bs[c]!, bs[c + 1]!, bs[c + 2]!, outs[c]!, outs[c + 1]!, outs[c + 2]!, from, to);
  for (; c < ws.length; c++) xw(X, ws[c]!, bs[c]!, undefined, outs[c]!, from, to);
}

function xw3(
  X: Matrix,
  u: ArrayLike<number>, v: ArrayLike<number>, z: ArrayLike<number>,
  bu: number, bv: number, bz: number,
  ou: Float64Array, ov: Float64Array, oz: Float64Array,
  from: number, to: number,
): void {
  const n = X.rows;
  const p = X.cols;
  const d = X.data;
  ou.fill(bu, from, to);
  ov.fill(bv, from, to);
  oz.fill(bz, from, to);
  for (let r0 = from; r0 < to; r0 += BLOCK) {
    const m = Math.min(BLOCK, to - r0);
    let j = 0;
    for (; j + 7 < p; j += 8) {
      const o = j * n + r0;
      const u0 = u[j]!, u1 = u[j + 1]!, u2 = u[j + 2]!, u3 = u[j + 3]!, u4 = u[j + 4]!, u5 = u[j + 5]!, u6 = u[j + 6]!, u7 = u[j + 7]!;
      const v0 = v[j]!, v1 = v[j + 1]!, v2 = v[j + 2]!, v3 = v[j + 3]!, v4 = v[j + 4]!, v5 = v[j + 5]!, v6 = v[j + 6]!, v7 = v[j + 7]!;
      const z0 = z[j]!, z1 = z[j + 1]!, z2 = z[j + 2]!, z3 = z[j + 3]!, z4 = z[j + 4]!, z5 = z[j + 5]!, z6 = z[j + 6]!, z7 = z[j + 7]!;
      for (let i = 0; i < m; i++) {
        const x0 = d[o + i]!, x1 = d[o + n + i]!, x2 = d[o + 2 * n + i]!, x3 = d[o + 3 * n + i]!;
        const x4 = d[o + 4 * n + i]!, x5 = d[o + 5 * n + i]!, x6 = d[o + 6 * n + i]!, x7 = d[o + 7 * n + i]!;
        ou[r0 + i]! += x0 * u0 + x1 * u1 + x2 * u2 + x3 * u3 + x4 * u4 + x5 * u5 + x6 * u6 + x7 * u7;
        ov[r0 + i]! += x0 * v0 + x1 * v1 + x2 * v2 + x3 * v3 + x4 * v4 + x5 * v5 + x6 * v6 + x7 * v7;
        oz[r0 + i]! += x0 * z0 + x1 * z1 + x2 * z2 + x3 * z3 + x4 * z4 + x5 * z5 + x6 * z6 + x7 * z7;
      }
    }
    for (; j + 3 < p; j += 4) {
      const o = j * n + r0;
      const u0 = u[j]!, u1 = u[j + 1]!, u2 = u[j + 2]!, u3 = u[j + 3]!;
      const v0 = v[j]!, v1 = v[j + 1]!, v2 = v[j + 2]!, v3 = v[j + 3]!;
      const z0 = z[j]!, z1 = z[j + 1]!, z2 = z[j + 2]!, z3 = z[j + 3]!;
      for (let i = 0; i < m; i++) {
        const x0 = d[o + i]!, x1 = d[o + n + i]!, x2 = d[o + 2 * n + i]!, x3 = d[o + 3 * n + i]!;
        ou[r0 + i]! += x0 * u0 + x1 * u1 + x2 * u2 + x3 * u3;
        ov[r0 + i]! += x0 * v0 + x1 * v1 + x2 * v2 + x3 * v3;
        oz[r0 + i]! += x0 * z0 + x1 * z1 + x2 * z2 + x3 * z3;
      }
    }
    for (; j < p; j++) {
      const o = j * n + r0;
      const uj = u[j]!, vj = v[j]!, zj = z[j]!;
      for (let i = 0; i < m; i++) {
        const x = d[o + i]!;
        ou[r0 + i]! += x * uj;
        ov[r0 + i]! += x * vj;
        oz[r0 + i]! += x * zj;
      }
    }
  }
}

function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function log1pExp(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/**
 * The binary logistic objective's row step, for rows `at` to `at + m − 1` of y and weights, whose scores are z[0] to
 * z[m − 1]: returns `loss` plus their loss, Σ w (log(1 + e^z) − y z), added row by row, and replaces each score by its
 * residual w (σ(z) − y), which the gradient Xᵀr needs. The running loss goes in, so the sum over every block keeps one
 * order. Another kernel set can replace it (useKernels()).
 */
export function binaryLogisticRows(z: Float64Array, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss = 0): number {
  const k = replaced.binaryLogisticRows;
  return k ? k(z, y, weights, at, m, loss) : ownBinaryLogisticRows(z, y, weights, at, m, loss);
}

/** smaltml's own binaryLogisticRows(), in TypeScript: the one that runs unless useKernels() replaced it. */
export function ownBinaryLogisticRows(z: Float64Array, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss = 0): number {
  for (let i = 0; i < m; i++) {
    const zi = z[i]!;
    const yi = y[at + i]!;
    // −log p(yᵢ) = log(1 + e^z) − y z; the residual σ(z) − y replaces z, times the row's weight.
    const wi = weights ? weights[at + i]! : 1;
    // One exponential for both: e = e^(−|z|). log(1 + e^z) = max(z, 0) + log1p(e), and σ(z) = 1 / (1 + e) for z ≥ 0,
    // e / (1 + e) below: the same operations sigmoid() and log1pExp() do, so the same results to the last bit.
    const e = Math.exp(zi > 0 ? -zi : zi);
    loss += wi * ((zi > 0 ? zi + Math.log1p(e) : Math.log1p(e)) - yi * zi);
    z[i] = wi * ((zi >= 0 ? 1 / (1 + e) : e / (1 + e)) - yi);
  }
  return loss;
}

/**
 * The multinomial logistic objective's row step, for rows `at` to `at + m − 1` of y and weights, whose K classes'
 * scores are z[c · stride + i]: returns `loss` plus their loss, Σ w (max + log Σ e^(z − max) − z_y), added row by row,
 * and replaces each score by its residual w (softmax − one-hot). Another kernel set can replace it (useKernels()).
 */
export function multinomialLogisticRows(z: Float64Array, stride: number, K: number, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss = 0): number {
  const k = replaced.multinomialLogisticRows;
  return k ? k(z, stride, K, y, weights, at, m, loss) : ownMultinomialLogisticRows(z, stride, K, y, weights, at, m, loss);
}

/** smaltml's own multinomialLogisticRows(), in TypeScript: the one that runs unless useKernels() replaced it. */
export function ownMultinomialLogisticRows(z: Float64Array, stride: number, K: number, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss = 0): number {
  for (let i = 0; i < m; i++) {
    let max = -Infinity;
    for (let c = 0; c < K; c++) max = Math.max(max, z[c * stride + i]!);
    const yi = y[at + i]!;
    const zy = z[yi * stride + i]!;
    // One exp per class: e = exp(z − max) replaces the score, then the residual e / Σe − one-hot replaces e.
    let sum = 0;
    for (let c = 0; c < K; c++) {
      const e = Math.exp(z[c * stride + i]! - max);
      z[c * stride + i] = e;
      sum += e;
    }
    // The row's weight scales its loss and its residuals.
    const wi = weights ? weights[at + i]! : 1;
    loss += wi * (max + Math.log(sum) - zy);
    const inv = 1 / sum;
    for (let c = 0; c < K; c++) z[c * stride + i] = wi * (z[c * stride + i]! * inv - (c === yi ? 1 : 0));
  }
  return loss;
}

/**
 * The binary logistic objective and its gradient in one pass over X: for each block of rows, the scores, then the
 * residuals, then their contribution to the gradient, while the block is still in cache. x holds w, then b when
 * there is an intercept. Returns the mean log loss plus l2 / 2 · ‖w‖².
 */
export function binaryLogistic(X: Matrix, y: Int32Array, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array, weights?: Float64Array, total = X.rows, products?: LogisticProducts | null): number {
  const part = products
    ? (g: Float64Array, a: number, b: number) => products.binaryPart(y, intercept, x, g, a, b, weights, binaryLogisticRows)
    : (g: Float64Array, a: number, b: number) => binaryLogisticPart(X, y, intercept, x, g, a, b, weights);
  return finishLogistic(sumParts(X.rows, grad, part), X, l2, intercept, x, grad, 1, total);
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
 * is an intercept. Parts over disjoint rows add up; finishLogistic() turns the total into the objective. With
 * `weights`, each row's loss and residual count that many times.
 */
export function binaryLogisticPart(X: Matrix, y: Int32Array, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number, weights?: Float64Array): number {
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
    loss = binaryLogisticRows(z, y, weights, r0, m, loss);
    for (let i = 0; i < m; i++) rsum += z[i]!;
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
export function multinomialLogistic(X: Matrix, y: Int32Array, K: number, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array, weights?: Float64Array, total = X.rows, products?: LogisticProducts | null): number {
  const part = products
    ? (g: Float64Array, a: number, b: number) => products.multinomialPart(y, K, intercept, x, g, a, b, weights, multinomialLogisticRows)
    : (g: Float64Array, a: number, b: number) => multinomialLogisticPart(X, y, K, intercept, x, g, a, b, weights);
  return finishLogistic(sumParts(X.rows, grad, part), X, l2, intercept, x, grad, K, total);
}

/** multinomialLogistic()'s sums over rows `from` to `to`, as binaryLogisticPart() gives the binary ones. */
export function multinomialLogisticPart(X: Matrix, y: Int32Array, K: number, intercept: boolean, x: Float64Array, grad: Float64Array, from: number, to: number, weights?: Float64Array): number {
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
    loss = multinomialLogisticRows(z, BLOCK, K, y, weights, r0, m, loss);
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
 * n (the rows' total weight, `total`, when they are weighted), plus the L2 penalty l2 / 2 · ‖w‖² (never on the intercepts) and its gradient.
 */
export function finishLogistic(loss: number, X: Matrix, l2: number, intercept: boolean, x: Float64Array, grad: Float64Array, K: number, total = X.rows): number {
  // The rows' total weight: n without weights.
  const n = total;
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

/** A · B, both column-major (A is n × p, B is p × k): the n × k product, column-major. */
export function matMulKernel(n: number, p: number, k: number, a: Float64Array, b: Float64Array): Float64Array {
  const f = replaced.matMul;
  return f ? f(n, p, k, a, b) : ownMatMul(n, p, k, a, b);
}

/** smaltml's own matMulKernel(), in TypeScript: the one that runs unless useKernels() replaced it. */
export function ownMatMul(n: number, p: number, k: number, a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(n * k);
  for (let j = 0; j < k; j++) {
    const o = j * n;
    for (let l = 0; l < p; l++) {
      const w = b[j * p + l]!;
      if (w === 0) continue;
      const c = l * n;
      for (let i = 0; i < n; i++) out[o + i]! += a[c + i]! * w;
    }
  }
  return out;
}
