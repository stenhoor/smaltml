import { fail } from './errors.js';
import { Matrix } from './matrix.js';

/** X · w (+ b), one value per row: column by column, which suits X's column-major storage. */
export function matVec(X: Matrix, w: ArrayLike<number>, b = 0, out = new Float64Array(X.rows)): Float64Array {
  const n = X.rows;
  out.fill(b);
  const d = X.data;
  for (let j = 0; j < X.cols; j++) {
    const wj = w[j]!;
    if (wj === 0) continue;
    const off = j * n;
    for (let i = 0; i < n; i++) out[i]! += d[off + i]! * wj;
  }
  return out;
}

/** Xᵀ · v, one value per column. */
export function matTVec(X: Matrix, v: ArrayLike<number>, out = new Float64Array(X.cols)): Float64Array {
  const n = X.rows;
  const d = X.data;
  for (let j = 0; j < X.cols; j++) {
    const off = j * n;
    let s = 0;
    for (let i = 0; i < n; i++) s += d[off + i]! * v[i]!;
    out[j] = s;
  }
  return out;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/**
 * The least-squares solution of A · x ≈ b by Householder QR. A is n × p, n ≥ p, and must have full column rank:
 * otherwise many solutions fit equally well, and this fails with ML_SINGULAR rather than pick one.
 * A and b are not changed.
 */
export function leastSquares(A: Matrix, b: Float64Array): Float64Array {
  const n = A.rows;
  const p = A.cols;
  if (n < p) {
    fail('ML_SINGULAR', `${n} rows can't determine ${p} coefficients: least squares needs at least as many rows as columns`);
  }
  const R = A.data.slice();
  const y = b.slice();
  const diag = new Float64Array(p);
  let maxNorm = 0;
  for (let j = 0; j < p; j++) maxNorm = Math.max(maxNorm, Math.hypot(...A.column(j)));
  const tol = Math.max(n, p) * Number.EPSILON * maxNorm;
  for (let k = 0; k < p; k++) {
    const ck = k * n;
    // The norm of column k below the diagonal, scaled to avoid overflow.
    let scale = 0;
    for (let i = k; i < n; i++) scale = Math.max(scale, Math.abs(R[ck + i]!));
    let norm = 0;
    if (scale > 0) {
      let s = 0;
      for (let i = k; i < n; i++) {
        const v = R[ck + i]! / scale;
        s += v * v;
      }
      norm = scale * Math.sqrt(s);
    }
    if (norm <= tol) {
      fail('ML_SINGULAR', `column ${A.columnName(k)} is a combination of the columns before it, so the coefficients aren't determined: drop it or use a penalized model`);
    }
    const alpha = R[ck + k]! > 0 ? -norm : norm;
    // v = x − alpha e1, stored in place of the column; R[k][k] becomes alpha.
    R[ck + k]! -= alpha;
    let vnorm2 = 0;
    for (let i = k; i < n; i++) vnorm2 += R[ck + i]! * R[ck + i]!;
    // Apply H = I − 2 v vᵀ / (vᵀ v) to the later columns and to y.
    for (let j = k + 1; j < p; j++) {
      const cj = j * n;
      let s = 0;
      for (let i = k; i < n; i++) s += R[ck + i]! * R[cj + i]!;
      const f = (2 * s) / vnorm2;
      for (let i = k; i < n; i++) R[cj + i]! -= f * R[ck + i]!;
    }
    let s = 0;
    for (let i = k; i < n; i++) s += R[ck + i]! * y[i]!;
    const f = (2 * s) / vnorm2;
    for (let i = k; i < n; i++) y[i]! -= f * R[ck + i]!;
    diag[k] = alpha;
  }
  // Back substitution with R's upper triangle (its diagonal kept apart).
  const x = new Float64Array(p);
  for (let k = p - 1; k >= 0; k--) {
    let s = y[k]!;
    for (let j = k + 1; j < p; j++) s -= R[j * n + k]! * x[j]!;
    x[k] = s / diag[k]!;
  }
  return x;
}

/** An objective for minimize(): its value at x, with its gradient written into `grad`. */
export type Objective = (x: Float64Array, grad: Float64Array) => number;

export interface MinimizeOptions {
  /** Stop when the largest gradient component is at most this (scipy's `pgtol`). */
  gtol: number;
  /** Stop when the objective's relative decrease in a step is at most this times machine epsilon (scipy's `ftol` / eps). */
  ftolFactor?: number;
  /** The most iterations. */
  maxIter: number;
  /** Corrections kept (scipy's `maxcor`, default 10). */
  memory?: number;
}

export interface MinimizeResult {
  x: Float64Array;
  value: number;
  iterations: number;
  converged: boolean;
}

/**
 * Unconstrained L-BFGS with a strong-Wolfe line search (Nocedal and Wright, algorithms 7.4, 3.5 and 3.6). It stops as
 * scipy's L-BFGS-B does without bounds, which scikit-learn calls: on the largest gradient component, or on a relative
 * decrease of the objective below `ftolFactor` × epsilon (scikit-learn passes 64).
 */
export function minimize(f: Objective, x0: Float64Array, options: MinimizeOptions): MinimizeResult {
  const m = options.memory ?? 10;
  const ftol = (options.ftolFactor ?? 64) * Number.EPSILON;
  const p = x0.length;
  let x: Float64Array = x0.slice();
  let g: Float64Array = new Float64Array(p);
  let fx = f(x, g);
  const S: Float64Array[] = [];
  const Y: Float64Array[] = [];
  const rho: number[] = [];
  const d = new Float64Array(p);
  const alpha = new Float64Array(m);
  let iter = 0;
  if (maxAbs(g) <= options.gtol) return { x, value: fx, iterations: 0, converged: true };
  while (iter < options.maxIter) {
    // Two-loop recursion: d = −H g.
    d.set(g);
    for (let k = S.length - 1; k >= 0; k--) {
      const a = rho[k]! * dot(S[k]!, d);
      alpha[k] = a;
      axpy(-a, Y[k]!, d);
    }
    if (S.length > 0) {
      const last = S.length - 1;
      const gamma = dot(S[last]!, Y[last]!) / dot(Y[last]!, Y[last]!);
      for (let i = 0; i < p; i++) d[i]! *= gamma;
    } else {
      // The first step: a unit step along −g, scaled to the gradient's size.
      const gn = Math.sqrt(dot(g, g));
      for (let i = 0; i < p; i++) d[i]! /= gn;
    }
    for (let k = 0; k < S.length; k++) {
      const b = rho[k]! * dot(Y[k]!, d);
      axpy(alpha[k]! - b, S[k]!, d);
    }
    for (let i = 0; i < p; i++) d[i] = -d[i]!;
    let slope = dot(g, d);
    if (slope >= 0) {
      // Not a descent direction: start over from steepest descent.
      S.length = 0;
      Y.length = 0;
      rho.length = 0;
      const gn = Math.sqrt(dot(g, g));
      for (let i = 0; i < p; i++) d[i] = -g[i]! / gn;
      slope = dot(g, d);
    }
    const step = lineSearch(f, x, fx, g, d, slope);
    iter++;
    if (!step) break;
    const s = new Float64Array(p);
    const y = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      s[i] = step.x[i]! - x[i]!;
      y[i] = step.g[i]! - g[i]!;
    }
    const fPrev = fx;
    x = step.x;
    g = step.g;
    fx = step.f;
    const sy = dot(s, y);
    if (sy > 1e-10 * dot(y, y)) {
      if (S.length === m) {
        S.shift();
        Y.shift();
        rho.shift();
      }
      S.push(s);
      Y.push(y);
      rho.push(1 / sy);
    }
    if (maxAbs(g) <= options.gtol) return { x, value: fx, iterations: iter, converged: true };
    if ((fPrev - fx) / Math.max(Math.abs(fPrev), Math.abs(fx), 1) <= ftol) {
      return { x, value: fx, iterations: iter, converged: true };
    }
  }
  return { x, value: fx, iterations: iter, converged: maxAbs(g) <= options.gtol };
}

interface Step {
  x: Float64Array;
  f: number;
  g: Float64Array;
}

function lineSearch(f: Objective, x: Float64Array, f0: number, g0: Float64Array, d: Float64Array, slope0: number): Step | null {
  const c1 = 1e-4;
  const c2 = 0.9;
  const p = x.length;
  const at = (a: number): Step & { slope: number } => {
    const xa = new Float64Array(p);
    for (let i = 0; i < p; i++) xa[i] = x[i]! + a * d[i]!;
    const ga = new Float64Array(p);
    const fa = f(xa, ga);
    return { x: xa, f: fa, g: ga, slope: dot(ga, d) };
  };
  const zoom = (lo: number, hi: number, flo: number, slo: number, fhi: number): Step | null => {
    for (let k = 0; k < 40; k++) {
      // Quadratic interpolation in the bracket, kept away from its ends.
      let a = quadraticMin(lo, flo, slo, hi, fhi);
      const lowEnd = Math.min(lo, hi);
      const width = Math.abs(hi - lo);
      if (!Number.isFinite(a) || a < lowEnd + 0.1 * width || a > lowEnd + 0.9 * width) a = (lo + hi) / 2;
      const s = at(a);
      if (s.f > f0 + c1 * a * slope0 || s.f >= flo) {
        hi = a;
        fhi = s.f;
      } else {
        if (Math.abs(s.slope) <= -c2 * slope0) return s;
        if (s.slope * (hi - lo) >= 0) {
          hi = lo;
          fhi = flo;
        }
        lo = a;
        flo = s.f;
        slo = s.slope;
      }
      if (Math.abs(hi - lo) < 1e-16) return lo > 0 ? at(lo) : null;
    }
    return lo > 0 ? at(lo) : null;
  };
  let aPrev = 0;
  let fPrev = f0;
  let sPrev = slope0;
  let a = 1;
  for (let k = 0; k < 40; k++) {
    const s = at(a);
    if (!Number.isFinite(s.f)) {
      a = (aPrev + a) / 2;
      continue;
    }
    if (s.f > f0 + c1 * a * slope0 || (k > 0 && s.f >= fPrev)) return zoom(aPrev, a, fPrev, sPrev, s.f);
    if (Math.abs(s.slope) <= -c2 * slope0) return s;
    if (s.slope >= 0) return zoom(a, aPrev, s.f, s.slope, fPrev);
    aPrev = a;
    fPrev = s.f;
    sPrev = s.slope;
    a *= 2;
  }
  return null;
}

/** The minimizer of the quadratic through (a, fa) with slope sa, and (b, fb). */
function quadraticMin(a: number, fa: number, sa: number, b: number, fb: number): number {
  // With only one slope, fit the quadratic f(a) + sa t + c t², t = x − a.
  const t = b - a;
  const c = (fb - fa - sa * t) / (t * t);
  if (c <= 0) return NaN;
  return a - sa / (2 * c);
}

function axpy(a: number, x: Float64Array, y: Float64Array): void {
  for (let i = 0; i < x.length; i++) y[i]! += a * x[i]!;
}

function maxAbs(v: Float64Array): number {
  let m = 0;
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]!));
  return m;
}
