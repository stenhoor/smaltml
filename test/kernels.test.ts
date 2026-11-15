import { describe, expect, test } from 'vitest';
import { binaryLogistic, gram, multinomialLogistic, xtv, xw } from '../src/core/kernels.js';
import { leastSquares } from '../src/core/linalg.js';
import { Matrix } from '../src/core/matrix.js';
import { Random } from '../src/core/random.js';
import { LinearRegression } from '../src/index.js';

/** A matrix of uneven size, so the kernels' blocks and four-column steps all have remainders. */
function data(n: number, p: number, seed: number) {
  const r = new Random(seed);
  const X = new Matrix(n, p, Float64Array.from({ length: n * p }, () => r.next() * 10 - 3));
  const v = Float64Array.from({ length: n }, () => r.next() - 0.5);
  const mean = Float64Array.from({ length: p }, () => r.next() * 4);
  return { X, v, mean };
}

const near = (a: ArrayLike<number>, b: ArrayLike<number>, rel = 1e-11) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i]! - b[i]!)).toBeLessThanOrEqual(rel * (1 + Math.abs(b[i]!)));
};

describe('the kernels equal plain loops', () => {
  for (const [n, p] of [[1, 1], [7, 3], [513, 5], [1300, 11], [2049, 9]] as const) {
    const { X, v, mean } = data(n, p, n + p);
    const at = (i: number, j: number, m?: Float64Array) => X.get(i, j) - (m ? m[j]! : 0);

    test(`gram, xtv and xw on ${n} × ${p}`, () => {
      for (const m of [undefined, mean]) {
        const G = gram(X, m);
        const plain = new Float64Array(p * p);
        for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) for (let i = 0; i < n; i++) plain[j * p + k]! += at(i, j, m) * at(i, k, m);
        near(G, plain);
        const g = xtv(X, v, m);
        near(g, Float64Array.from({ length: p }, (_, j) => { let s = 0; for (let i = 0; i < n; i++) s += at(i, j, m) * v[i]!; return s; }), 1e-9);
        const w = Float64Array.from({ length: p }, (_, j) => Math.sin(j + 1));
        near(xw(X, w, 0.5, m), Float64Array.from({ length: n }, (_, i) => { let s = 0.5; for (let j = 0; j < p; j++) s += at(i, j, m) * w[j]!; return s; }));
      }
    });

    test(`the logistic objectives and gradients on ${n} × ${p}`, () => {
      const y2 = Int32Array.from({ length: n }, (_, i) => (v[i]! > 0 ? 1 : 0));
      const x = Float64Array.from({ length: p + 1 }, (_, j) => 0.1 * Math.cos(j));
      const g = new Float64Array(p + 1);
      const f = binaryLogistic(X, y2, 0.01, true, x, g);
      // Plain: mean log loss + l2/2 ‖w‖², and its gradient.
      let loss = 0;
      const pg = new Float64Array(p + 1);
      for (let i = 0; i < n; i++) {
        let z = x[p]!;
        for (let j = 0; j < p; j++) z += X.get(i, j) * x[j]!;
        const prob = 1 / (1 + Math.exp(-z));
        loss += -(y2[i]! * Math.log(prob) + (1 - y2[i]!) * Math.log(1 - prob));
        for (let j = 0; j < p; j++) pg[j]! += (prob - y2[i]!) * X.get(i, j);
        pg[p]! += prob - y2[i]!;
      }
      let reg = 0;
      for (let j = 0; j < p; j++) {
        pg[j] = pg[j]! / n + 0.01 * x[j]!;
        reg += x[j]! ** 2;
      }
      pg[p] = pg[p]! / n;
      expect(f).toBeCloseTo(loss / n + 0.005 * reg, 10);
      near(g, pg, 1e-9);

      const K = 3;
      const y3 = Int32Array.from({ length: n }, (_, i) => i % K);
      const xm = Float64Array.from({ length: K * (p + 1) }, (_, j) => 0.05 * Math.sin(j));
      const gm = new Float64Array(K * (p + 1));
      const fm = multinomialLogistic(X, y3, K, 0, true, xm, gm);
      let lm = 0;
      const pgm = new Float64Array(K * (p + 1));
      for (let i = 0; i < n; i++) {
        const z = Array.from({ length: K }, (_, c) => { let s = xm[c * (p + 1) + p]!; for (let j = 0; j < p; j++) s += X.get(i, j) * xm[c * (p + 1) + j]!; return s; });
        const mx = Math.max(...z);
        const e = z.map((t) => Math.exp(t - mx));
        const sum = e.reduce((a, b) => a + b, 0);
        lm += -Math.log(e[y3[i]!]! / sum);
        for (let c = 0; c < K; c++) {
          const r = e[c]! / sum - (c === y3[i] ? 1 : 0);
          for (let j = 0; j < p; j++) pgm[c * (p + 1) + j]! += (r * X.get(i, j)) / n;
          pgm[c * (p + 1) + p]! += r / n;
        }
      }
      expect(fm).toBeCloseTo(lm / n, 10);
      near(gm, pgm, 1e-9);
    });
  }
});

describe('least squares by the semi-normal equations', () => {
  test('agrees with Householder QR, and falls back to it for nearly dependent columns', () => {
    const r = new Random(5);
    const n = 3000;
    const a = Float64Array.from({ length: n }, () => r.next() * 100 + 1000);
    // b is a plus a tiny wiggle: nearly dependent, condition number around 1e9.
    const b = Float64Array.from(a, (v) => v + 1e-6 * (r.next() - 0.5));
    const c = Float64Array.from({ length: n }, () => r.next());
    const y = Float64Array.from({ length: n }, (_, i) => 2 * a[i]! - b[i]! + 3 * c[i]! + 0.01 * (r.next() - 0.5));
    for (const cols of [{ a, c }, { a, b, c }] as Record<string, Float64Array>[]) {
      const fit = new LinearRegression().fitSync(cols, y);
      // The reference: QR on the centred columns.
      const names = Object.keys(cols);
      const means = names.map((k) => (cols as Record<string, Float64Array>)[k]!.reduce((s, v) => s + v, 0) / n);
      const centred = Matrix.fromColumns(names.map((k, j) => (cols as Record<string, Float64Array>)[k]!.map((v) => v - means[j]!)));
      const ym = y.reduce((s, v) => s + v, 0) / n;
      const ref = leastSquares(centred, y.map((v) => v - ym));
      near(fit.coef, ref, names.length === 3 ? 1e-4 : 1e-10);
    }
  });
});
