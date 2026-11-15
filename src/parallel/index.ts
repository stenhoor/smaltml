/**
 * Threads: the heavy kernels on several cores at once, in workers sharing X's memory (ADR 0024, section 8). Each
 * worker takes a run of chunks of rows; the main thread adds the chunks' parts up in order, as the single-threaded
 * kernels do, so a fit gives the same result to the last bit with any number of threads. It needs shared memory:
 * Node always has it; a browser only on a cross-origin isolated page. Without it, fits run on one thread.
 * @module
 */
import { fail } from '../core/errors.js';
import { addInto, chunks, finishLogistic, symmetrize } from '../core/kernels.js';
import { Matrix } from '../core/matrix.js';
import type { Request, Response, SharedMatrix, Task } from './protocol.js';

/** How many threads a fit may use: `'auto'` (every core, where shared memory allows), or a number (1 is one thread). */
export type Threads = number | 'auto';

export interface ThreadOptions {
  threads?: Threads;
}

/** Work below this many values of X stays on one thread: messages cost more than the threads save. */
export const PARALLEL_MIN_VALUES = 1 << 20;

/**
 * A Node built-in module, imported at run time by a name bundlers don't follow, so a browser build never includes or
 * resolves it. Only called where Node is running.
 */
export function nodeModule(name: string): Promise<unknown> {
  return import(/* @vite-ignore */ name);
}

let scriptUrl: string | URL | undefined;
let maxThreads: number | undefined;

/**
 * Sets where the workers' script is (default: `smaltml-threads.js` beside this module, where the package builds it),
 * and the most threads 'auto' uses (default: every core).
 */
export function configureThreads(options: { script?: string | URL; max?: number }): void {
  if (options.script !== undefined) scriptUrl = options.script;
  if (options.max !== undefined) {
    if (!Number.isInteger(options.max) || options.max < 1) fail('ML_PARAM', `max must be a whole number of at least 1, not ${options.max}`);
    maxThreads = options.max;
  }
  pools.clear();
}

/** Whether this environment can share memory with workers: Node, or a cross-origin isolated page. */
export function sharedMemoryAvailable(): boolean {
  const g = globalThis as { SharedArrayBuffer?: unknown; crossOriginIsolated?: boolean; process?: { versions?: { node?: string } } };
  if (typeof g.SharedArrayBuffer !== 'function') return false;
  if (g.process?.versions?.node) return true;
  return g.crossOriginIsolated === true;
}

async function cores(): Promise<number> {
  const g = globalThis as { navigator?: { hardwareConcurrency?: number }; process?: { versions?: { node?: string } } };
  if (g.process?.versions?.node) {
    const os = (await nodeModule('node:os')) as { availableParallelism?: () => number; cpus(): unknown[] };
    return os.availableParallelism?.() ?? os.cpus().length;
  }
  return g.navigator?.hardwareConcurrency ?? 1;
}

interface WorkerLike {
  post(req: Request): void;
  onMessage(listener: (res: Response) => void): void;
  terminate(): void;
}

async function spawn(url: string | URL): Promise<WorkerLike> {
  const g = globalThis as { Worker?: new (u: string | URL, o?: object) => Worker; process?: { versions?: { node?: string } } };
  if (g.process?.versions?.node) {
    const wt = (await nodeModule('node:worker_threads')) as {
      Worker: new (u: string | URL) => { postMessage(m: unknown): void; on(e: 'message', l: (d: Response) => void): void; unref(): void; terminate(): void };
    };
    const w = new wt.Worker(url);
    // An idle pool doesn't keep Node running.
    w.unref();
    return { post: (r) => w.postMessage(r), onMessage: (l) => w.on('message', l), terminate: () => void w.terminate() };
  }
  if (!g.Worker) fail('ML_PARAM', 'Web Workers aren\'t available here');
  const w = new g.Worker(url, { type: 'module', name: 'smaltml-threads' });
  return {
    post: (r) => w.postMessage(r),
    onMessage: (l) => w.addEventListener('message', (e: MessageEvent) => l(e.data as Response)),
    terminate: () => w.terminate(),
  };
}

/** A set of workers that run tasks. */
export class ThreadPool {
  readonly size: number;
  #workers: WorkerLike[];
  #next = 1;
  #waiting = new Map<number, { resolve(r: Response & { ok: true }): void; reject(e: Error): void }>();

  private constructor(workers: WorkerLike[]) {
    this.size = workers.length;
    this.#workers = workers;
    for (const w of workers) {
      w.onMessage((res) => {
        const p = this.#waiting.get(res.id);
        if (!p) return;
        this.#waiting.delete(res.id);
        if (res.ok) p.resolve(res);
        else p.reject(new Error(`a smaltml thread failed: ${res.message}`));
      });
    }
  }

  static async create(size: number, url: string | URL): Promise<ThreadPool> {
    return new ThreadPool(await Promise.all(Array.from({ length: size }, () => spawn(url))));
  }

  /** Runs one task on worker i. */
  run(i: number, task: Task): Promise<Response & { ok: true }> {
    const id = this.#next++;
    return new Promise((resolve, reject) => {
      this.#waiting.set(id, { resolve, reject });
      this.#workers[i]!.post({ id, task });
    });
  }

  terminate(): void {
    for (const w of this.#workers) w.terminate();
    for (const p of this.#waiting.values()) p.reject(new Error('the smaltml thread pool was stopped'));
    this.#waiting.clear();
  }
}

const pools = new Map<number, Promise<ThreadPool | null>>();

/**
 * The pool for this many threads, or null where a fit should run on one thread: one thread asked for, no shared
 * memory, too little data (`values`, X's size), or workers that can't start.
 */
export async function threadPool(threads: Threads | undefined, values: number): Promise<ThreadPool | null> {
  const want = threads ?? 'auto';
  if (want !== 'auto' && (!Number.isInteger(want) || want < 1)) fail('ML_PARAM', `threads is 'auto' or a whole number of at least 1, not ${String(want)}`);
  if (want === 1 || !sharedMemoryAvailable()) return null;
  if (want === 'auto' && values < PARALLEL_MIN_VALUES) return null;
  const size = want === 'auto' ? Math.min(await cores(), maxThreads ?? Infinity) : want;
  if (size < 2) return null;
  let pool = pools.get(size);
  if (!pool) {
    const url = scriptUrl ?? new URL('./smaltml-threads.js', import.meta.url);
    pool = ThreadPool.create(size, url).catch(() => null);
    pools.set(size, pool);
  }
  return pool;
}

/** Stops every pool's workers (they are also unreferenced in Node, so an idle pool doesn't keep a process alive). */
export async function stopThreads(): Promise<void> {
  const all = await Promise.all(pools.values());
  pools.clear();
  for (const p of all) p?.terminate();
}

// ---------------------------------------------------------------------------------------------------------------------
// The kernels, in parallel

const shared = new WeakMap<Float64Array, SharedArrayBuffer>();

/** X in shared memory: X's own buffer when it is shared already, else a copy, kept for X's next fit (a grid search). */
export function share(X: Matrix): SharedMatrix {
  const d = X.data;
  if (d.buffer instanceof SharedArrayBuffer && d.byteOffset === 0 && d.length === d.buffer.byteLength / 8) {
    return { buffer: d.buffer, rows: X.rows, cols: X.cols };
  }
  let buf = shared.get(d);
  if (!buf) {
    buf = new SharedArrayBuffer(d.byteLength);
    new Float64Array(buf).set(d);
    shared.set(d, buf);
  }
  return { buffer: buf, rows: X.rows, cols: X.cols };
}

function sharedCopy(v: ArrayLike<number>, type: 'f64' | 'i32'): SharedArrayBuffer {
  const buf = new SharedArrayBuffer(v.length * (type === 'f64' ? 8 : 4));
  (type === 'f64' ? new Float64Array(buf) : new Int32Array(buf)).set(v);
  return buf;
}

/** The chunks of n rows, split into the pool's runs: contiguous, as even as whole chunks allow. */
function runs(n: number, size: number): [number, number][][] {
  const all = chunks(n);
  const out: [number, number][][] = [];
  const per = Math.ceil(all.length / size);
  for (let i = 0; i < all.length; i += per) out.push(all.slice(i, i + per));
  return out;
}

/** Each run's parts, all of them in chunk order. */
async function parts(pool: ThreadPool, n: number, task: (chunks: [number, number][]) => Task) {
  const results = await Promise.all(runs(n, pool.size).map((c, i) => pool.run(i, task(c))));
  return results.flatMap((r) => r.parts);
}

/** gram() in parallel: the same chunks, summed in the same order. */
export async function gramParallel(pool: ThreadPool, X: Matrix, mean?: Float64Array): Promise<Float64Array> {
  const S = share(X);
  const A = new Float64Array(X.cols * X.cols);
  for (const part of await parts(pool, X.rows, (c) => ({ op: 'gram', X: S, mean: mean ?? null, weights: null, ones: false, chunks: c }))) addInto(A, part.values);
  return symmetrize(A, X.cols);
}

/** weightedGram() in parallel. */
export async function weightedGramParallel(pool: ThreadPool, X: Matrix, weights: ArrayLike<number>, ones: boolean): Promise<Float64Array> {
  const S = share(X);
  const W = sharedCopy(weights, 'f64');
  const q = X.cols + (ones ? 1 : 0);
  const A = new Float64Array(q * q);
  for (const part of await parts(pool, X.rows, (c) => ({ op: 'gram', X: S, mean: null, weights: W, ones, chunks: c }))) addInto(A, part.values);
  return symmetrize(A, q);
}

/** xtv() in parallel. */
export async function xtvParallel(pool: ThreadPool, X: Matrix, v: ArrayLike<number>, mean?: Float64Array): Promise<Float64Array> {
  const S = share(X);
  const V = sharedCopy(v, 'f64');
  const out = new Float64Array(X.cols);
  for (const part of await parts(pool, X.rows, (c) => ({ op: 'xtv', X: S, v: V, mean: mean ?? null, chunks: c }))) addInto(out, part.values);
  return out;
}

/** xw() in parallel: each worker writes its rows. */
export async function xwParallel(pool: ThreadPool, X: Matrix, w: ArrayLike<number>, b: number, mean?: Float64Array): Promise<Float64Array> {
  const S = share(X);
  const out = new SharedArrayBuffer(X.rows * 8);
  await parts(pool, X.rows, (c) => ({ op: 'xw', X: S, w: Float64Array.from(w), b, mean: mean ?? null, out, chunks: c }));
  return new Float64Array(out).slice();
}

/**
 * The logistic objective in parallel, for minimizeAsync(): the per-chunk losses and gradients summed in chunk order,
 * then finished as the single-threaded one is. `y` holds the class indexes; K is 1 for two classes.
 */
export function logisticParallel(pool: ThreadPool, X: Matrix, y: Int32Array, K: number, l2: number, intercept: boolean) {
  const S = share(X);
  const Y = sharedCopy(y, 'i32');
  return async (x: Float64Array, grad: Float64Array): Promise<number> => {
    grad.fill(0);
    let loss = 0;
    for (const part of await parts(pool, X.rows, (c) => ({ op: 'logistic', X: S, y: Y, K, intercept, x: x.slice(), chunks: c }))) {
      loss += part.loss!;
      addInto(grad, part.values);
    }
    return finishLogistic(loss, X, l2, intercept, x, grad, K);
  };
}
