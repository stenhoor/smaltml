import { gram, xtv, xw } from '../core/kernels.js';
import type { Matrix } from '../core/matrix.js';
import { gramParallel, xtvParallel, xwParallel, type ThreadPool } from '../parallel/index.js';

/**
 * A kernel a solver needs: solvers are generators that yield these and get the results back, so one solver serves
 * fitSync() (the kernels on this thread) and fit() (the same kernels across a thread pool, with the same result).
 */
export type KernelCall =
  | { k: 'gram'; X: Matrix; mean: Float64Array | undefined }
  | { k: 'xtv'; X: Matrix; v: Float64Array; mean: Float64Array | undefined }
  | { k: 'xw'; X: Matrix; w: Float64Array; b: number; mean: Float64Array | undefined };

export type Solver<T> = Generator<KernelCall, T, Float64Array>;

/** Runs a solver with the kernels on this thread. */
export function runHere<T>(solver: Solver<T>): T {
  let r = solver.next();
  while (!r.done) {
    const c = r.value;
    r = solver.next(recall(c) ?? keep(c, c.k === 'gram' ? gram(c.X, c.mean) : c.k === 'xtv' ? xtv(c.X, c.v, c.mean) : xw(c.X, c.w, c.b, c.mean)));
  }
  return r.value;
}

/** Runs a solver with the kernels across a thread pool. */
export async function runOn<T>(pool: ThreadPool, solver: Solver<T>): Promise<T> {
  let r = solver.next();
  while (!r.done) {
    const c = r.value;
    r = solver.next(
      recall(c) ??
        keep(
          c,
          c.k === 'gram'
            ? await gramParallel(pool, c.X, c.mean)
            : c.k === 'xtv'
              ? await xtvParallel(pool, c.X, c.v, c.mean)
              : await xwParallel(pool, c.X, c.w, c.b, c.mean),
        ),
    );
  }
  return r.value;
}

/**
 * Results kept while a search runs (withKernelCache()): GridSearchCV fits every candidate on the same fold's rows, so
 * XᵀX and Xᵀy are worked out once per fold, not once per candidate. Keyed by the matrix itself and checked against
 * the means and the vector, so a hit gives exactly what the kernel would. Parallel and local kernels agree to the last
 * bit, so either may fill it.
 */
interface Kept {
  means?: Float64Array;
  gram: { mean: Float64Array | undefined; out: Float64Array }[];
  xtv: { mean: Float64Array | undefined; v: Float64Array; out: Float64Array }[];
}
let kept: WeakMap<Matrix, Kept> | null = null;
let scopes = 0;

/** Runs f with XᵀX and Xᵀy kept for the matrices it fits on, and forgets them after. */
export async function withKernelCache<T>(f: () => Promise<T>): Promise<T> {
  if (scopes++ === 0) kept = new WeakMap();
  try {
    return await f();
  } finally {
    if (--scopes === 0) kept = null;
  }
}

/** X's column means (checked on the way), from compute() or, in a search, kept from an earlier fit on X. */
export function keptMeans(X: Matrix, compute: () => Float64Array): Float64Array {
  if (!kept) return compute();
  let k = kept.get(X);
  if (!k) kept.set(X, (k = { gram: [], xtv: [] }));
  return (k.means ??= compute()).slice();
}

const same = (a: Float64Array | undefined, b: Float64Array | undefined) => {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
};

/** A copy of a kept result for this call (solvers may change what they get), or undefined. */
function recall(c: KernelCall): Float64Array | undefined {
  const k = kept?.get(c.X);
  if (!k) return undefined;
  if (c.k === 'gram') return k.gram.find((e) => same(e.mean, c.mean))?.out.slice();
  if (c.k === 'xtv') return k.xtv.find((e) => same(e.mean, c.mean) && same(e.v, c.v))?.out.slice();
  return undefined;
}

function keep(c: KernelCall, out: Float64Array): Float64Array {
  if (!kept || c.k === 'xw') return out;
  let k = kept.get(c.X);
  if (!k) kept.set(c.X, (k = { gram: [], xtv: [] }));
  if (c.k === 'gram') k.gram.push({ mean: c.mean?.slice(), out: out.slice() });
  else k.xtv.push({ mean: c.mean?.slice(), v: c.v.slice(), out: out.slice() });
  return out;
}
