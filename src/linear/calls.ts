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
    r = solver.next(c.k === 'gram' ? gram(c.X, c.mean) : c.k === 'xtv' ? xtv(c.X, c.v, c.mean) : xw(c.X, c.w, c.b, c.mean));
  }
  return r.value;
}

/** Runs a solver with the kernels across a thread pool. */
export async function runOn<T>(pool: ThreadPool, solver: Solver<T>): Promise<T> {
  let r = solver.next();
  while (!r.done) {
    const c = r.value;
    r = solver.next(
      c.k === 'gram'
        ? await gramParallel(pool, c.X, c.mean)
        : c.k === 'xtv'
          ? await xtvParallel(pool, c.X, c.v, c.mean)
          : await xwParallel(pool, c.X, c.w, c.b, c.mean),
    );
  }
  return r.value;
}
