import { fail } from '../core/errors.js';
import { floats } from '../core/model.js';
import { assertFinite, assertFiniteVector, Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { weightedDesign, weightedTarget } from './weights.js';

/** y for a regression: one target (a vector), or several (rows, named columns or a Matrix: scikit-learn's 2-D y). */
export type Targets = VectorInput | MatrixInput;

/** Whether y has a column per target: a Matrix, rows of arrays, or named columns. */
export function severalTargets(y: Targets): boolean {
  if (y instanceof Matrix) return true;
  if (Array.isArray(y)) return y.length > 0 && typeof y[0] === 'object' && y[0] !== null;
  return !ArrayBuffer.isView(y) && typeof y === 'object' && y !== null && !('length' in y);
}

/** y's columns, checked: one per target, each as long as X has rows, finite. */
export function targetColumns(y: Targets, rows: number): { columns: Float64Array[]; names: readonly string[] | undefined } {
  let t: Matrix;
  try {
    t = Matrix.from(y as MatrixInput);
  } catch {
    return fail('ML_SHAPE', 'y with several targets needs a column per target: rows (an array of arrays) or named columns');
  }
  if (t.rows !== rows) fail('ML_SHAPE', `y has ${t.rows} rows; X has ${rows}`);
  if (t.cols === 0) fail('ML_SHAPE', 'y has no columns: it needs a column per target');
  const columns = Array.from({ length: t.cols }, (_, k) => t.column(k).slice());
  for (const c of columns) assertFiniteVector(c);
  return { columns, names: t.names };
}

/** The error for a single-target accessor on a model fitted to several targets. */
export function oneTargetOnly(kind: string, what: string, targets: number): never {
  return fail(
    'ML_SHAPE',
    `${kind} was fitted on ${targets} target${targets === 1 ? '' : 's'} (a y with columns): ${what} is for one target; read coefs, intercepts and predictTargets()`,
  );
}

/** What a least-squares fit solves: the design the kernels read, the means they subtract, and each target centred. */
export interface FitPlan {
  design: Matrix;
  /** The means the kernels subtract from `design` as they read it (unweighted, with an intercept). */
  kernelMean: Float64Array | undefined;
  /** The means the intercept comes from. */
  xMean: Float64Array | undefined;
  targets: { b: Float64Array; yMean: number }[];
  several: boolean;
  targetNames: readonly string[] | undefined;
}

/**
 * The plan for X and y (one target or several), with checked weights or none. Without weights, X isn't copied: with an
 * intercept the kernels subtract its column means (`means`, which checks X on the way) as they read it. With weights,
 * X is centred and scaled once (weightedDesign), and each target on its own.
 */
export function planFit(
  m: Matrix,
  y: Targets,
  weights: Float64Array | undefined,
  intercept: boolean,
  means: () => Float64Array,
): FitPlan {
  const n = m.rows;
  const several = severalTargets(y);
  let columns: Float64Array[];
  let targetNames: readonly string[] | undefined;
  if (several) ({ columns, names: targetNames } = targetColumns(y, n));
  else {
    const v = toVector(y as VectorInput, n);
    assertFiniteVector(v);
    columns = [v];
  }
  if (weights) {
    const d = weightedDesign(m, weights, intercept);
    const targets = columns.map((c) => {
      const t = weightedTarget(d, c);
      return { b: t.y, yMean: t.yMean };
    });
    return { design: d.X, kernelMean: undefined, xMean: d.xMean, targets, several, targetNames };
  }
  if (!intercept) assertFinite(m);
  const xMean = intercept ? means() : undefined;
  const targets = columns.map((c) => {
    if (!intercept) return { b: c, yMean: 0 };
    let yMean = 0;
    for (let i = 0; i < c.length; i++) yMean += c[i]!;
    yMean /= c.length;
    return { b: c.map((v) => v - yMean), yMean };
  });
  return { design: m, kernelMean: xMean, xMean, targets, several, targetNames };
}

/** The intercept that goes with these coefficients: ȳ − x̄·w, or 0 without one. */
export function interceptOf(xMean: Float64Array | undefined, yMean: number, coef: Float64Array): number {
  if (!xMean) return 0;
  let b = yMean;
  for (let j = 0; j < coef.length; j++) b -= xMean[j]! * coef[j]!;
  return b;
}

/** The coefficients as a Matrix, a row per target and a column per feature. */
export function coefsMatrix(coefs: readonly Float64Array[], p: number, names: readonly string[] | undefined): Matrix {
  const k = coefs.length;
  const data = new Float64Array(k * p);
  for (let t = 0; t < k; t++) for (let j = 0; j < p; j++) data[j * k + t] = coefs[t]![j]!;
  return new Matrix(k, p, data, names);
}

/** X·Wᵀ + b, a column per target: X's values times a p × k matrix whose column t is target t's coefficients. */
export function coefColumns(coefs: readonly Float64Array[], p: number): Matrix {
  const data = new Float64Array(p * coefs.length);
  coefs.forEach((c, t) => data.set(c, t * p));
  return new Matrix(p, coefs.length, data);
}

/**
 * A saved model's coefficients and intercepts: `coef` and `intercept` for one target (every format version), or
 * `coefs`, `intercepts` and `targetNames` for a y with columns (format version 3).
 */
export function savedTargets(s: Record<string, unknown>): {
  coefs: Float64Array[];
  intercepts: Float64Array;
  several: boolean;
  targetNames: readonly string[] | undefined;
} {
  if (Array.isArray(s.coefs)) {
    const coefs = s.coefs.map((c, t) => floats(c, `coefs[${t}]`));
    const intercepts = floats(s.intercepts, 'intercepts');
    if (intercepts.length !== coefs.length) fail('ML_FORMAT', 'a model fitted on several targets needs an intercept per target');
    return { coefs, intercepts, several: true, targetNames: Array.isArray(s.targetNames) ? (s.targetNames as string[]) : undefined };
  }
  return { coefs: [floats(s.coef, 'coef')], intercepts: Float64Array.of(Number(s.intercept)), several: false, targetNames: undefined };
}
