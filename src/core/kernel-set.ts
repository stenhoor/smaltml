/**
 * Replaceable kernels (ADR 0035): a kernel set can swap the CPU kernels smaltml already has for its own, with the same
 * signatures. The models and solvers don't change. smaltml's SIMD kernels (`@smaltworks/smaltml/simd`, ADR 0040) are
 * one such set, which kernelsReady() turns on where the engine has WebAssembly SIMD; a plugin could give another.
 */
import { fail } from './errors.js';
import { binaryLogisticPart, multinomialLogisticPart, ownBinaryLogisticRows, ownGramPart, ownMatMul, ownMultinomialLogisticRows, replaced, type Kernels } from './kernels.js';
import { Matrix } from './matrix.js';
import { Random } from './random.js';

export type { Kernels } from './kernels.js';

/**
 * The version of the kernel contract: the kernels' names, signatures and results. A set written for another version
 * isn't used. Adding a kernel doesn't change it; changing what one takes or gives does.
 */
export const KERNELS_VERSION = 1;

/** The names of the kernels a set can replace. */
export type KernelName = keyof Kernels;
const NAMES: readonly KernelName[] = ['gramPart', 'matMul', 'binaryLogisticRows', 'multinomialLogisticRows', 'logisticProducts'];

/** A set of kernels to use in place of smaltml's own: what smaltml's SIMD entry, or a plugin, gives useKernels(). */
export interface KernelSet {
  /** A name for the set, as kernelsInUse() reports it, such as `'smaltml-simd'`. */
  name: string;
  /** The contract version the set was written for: {@link KERNELS_VERSION}. */
  version: number;
  /** The kernels it replaces. A kernel it leaves out stays smaltml's. */
  kernels: Kernels;
  /**
   * How the thread pool's workers get the same kernels: a module each worker imports, whose default export takes
   * `options` and returns (or resolves to) the same set. Without it, fits run on one thread while the set is in use,
   * since workers running smaltml's own kernels would give other last bits than `fitSync`.
   */
  threads?: { module: string | URL; options?: unknown };
}

/** What useKernels() set: the set's name (`'smaltml'` for smaltml's own), and which kernels it replaced and didn't. */
export interface KernelsInUse {
  name: string;
  /** The kernels replaced. */
  kernels: KernelName[];
  /** The kernels the set gave that failed the check at start, and stay smaltml's. */
  rejected: KernelName[];
}

let inUse: KernelsInUse = { name: 'smaltml', kernels: [], rejected: [] };
let threads: KernelSet['threads'];
let generation = 0;
/** Whether useKernels() was called: a choice kernelsReady() doesn't override. */
let chosen = false;
let ready: Promise<KernelsInUse> | null = null;

/** smaltml's SIMD kernels, its own entry point (ADR 0040): imported by name, so the core doesn't load them. */
const SIMD = '@smaltworks/smaltml/simd';

/**
 * Turns smaltml's SIMD kernels (`@smaltworks/smaltml/simd`) on once, where the engine has WebAssembly SIMD; resolves
 * to the kernels in use then. smaltml's asynchronous fits and `matMul` wait for it, so they use them from the first
 * call. The `*Sync` calls don't wait: to have them use them from the start too, `await kernelsReady()` first. A
 * `useKernels()` call made before the look-up finishes is kept: `useKernels(null)` keeps smaltml's own kernels.
 * Where the entry can't load (a bundled app that doesn't import it), smaltml's own kernels stay: such an app imports
 * `@smaltworks/smaltml/simd` and calls `useSimd()` itself.
 */
export function kernelsReady(): Promise<KernelsInUse> {
  return (ready ??= lookForPlugin());
}

async function lookForPlugin(): Promise<KernelsInUse> {
  if (chosen) return kernelsInUse();
  let simd: { useSimd?: () => Promise<boolean> };
  try {
    // By a variable, so bundlers leave the import alone, and an app that doesn't import the entry builds without it.
    const name = SIMD;
    simd = (await import(/* @vite-ignore */ /* webpackIgnore: true */ name)) as typeof simd;
  } catch {
    return kernelsInUse();
  }
  if (!chosen && typeof simd.useSimd === 'function') {
    try {
      await simd.useSimd();
    } catch {
      // The SIMD kernels failed to start: smaltml's own kernels stay.
    }
  }
  return kernelsInUse();
}

/**
 * Uses a set of kernels in place of smaltml's own, or smaltml's own again with `null`. Each kernel the set gives is
 * checked first, on a small product against smaltml's own; one that throws, or differs by more than rounding, isn't
 * used. It affects every model fitted after it, in this thread and in the thread pool's workers.
 */
export function useKernels(set: KernelSet | null): KernelsInUse {
  chosen = true;
  for (const name of NAMES) delete replaced[name];
  threads = undefined;
  generation++;
  if (set === null) {
    inUse = { name: 'smaltml', kernels: [], rejected: [] };
    return kernelsInUse();
  }
  if (typeof set !== 'object' || typeof set.name !== 'string' || typeof set.kernels !== 'object' || set.kernels === null) {
    fail('ML_PARAM', 'useKernels takes a kernel set ({ name, version, kernels }) or null');
  }
  if (set.version !== KERNELS_VERSION) {
    fail('ML_PARAM', `the kernel set '${set.name}' is for kernel contract ${String(set.version)}; this smaltml has contract ${KERNELS_VERSION}`);
  }
  const unknown = Object.keys(set.kernels).filter((k) => !NAMES.includes(k as KernelName));
  if (unknown.length) fail('ML_PARAM', `the kernel set '${set.name}' gives kernels smaltml doesn't have: ${unknown.join(', ')}`);
  const kernels: KernelName[] = [];
  const rejected: KernelName[] = [];
  for (const name of NAMES) {
    const k = set.kernels[name];
    if (k === undefined) continue;
    if (typeof k === 'function' && passes(name, k)) {
      (replaced as Record<string, unknown>)[name] = k;
      kernels.push(name);
    } else rejected.push(name);
  }
  threads = set.threads;
  inUse = { name: set.name, kernels, rejected };
  return kernelsInUse();
}

/** The kernel set in use, and which kernels it replaced. */
export function kernelsInUse(): KernelsInUse {
  return { name: inUse.name, kernels: [...inUse.kernels], rejected: [...inUse.rejected] };
}

/**
 * For the thread pool: a number that changes with every useKernels() call (pools started before it are stale), and
 * how workers get the set (`null` when smaltml's own kernels are in use; `undefined` when a set is in use that
 * workers can't load, so fits stay on one thread).
 */
export function kernelsForThreads(): { generation: number; threads: KernelSet['threads'] | null; kernels: KernelName[] } {
  return { generation, threads: inUse.kernels.length ? threads : null, kernels: [...inUse.kernels] };
}

// ---------------------------------------------------------------------------------------------------------------------
// The check at start

/** Relative to each entry's scale: far wider than rounding in another order, far narrower than a wrong result. */
const TOLERANCE = 1e-12;

function passes(name: KernelName, k: NonNullable<Kernels[KernelName]>): boolean {
  try {
    switch (name) {
      case 'gramPart':
        return gramPasses(k as NonNullable<Kernels['gramPart']>);
      case 'matMul':
        return matMulPasses(k as NonNullable<Kernels['matMul']>);
      case 'binaryLogisticRows':
        return binaryRowsPass(k as NonNullable<Kernels['binaryLogisticRows']>);
      case 'multinomialLogisticRows':
        return multinomialRowsPass(k as NonNullable<Kernels['multinomialLogisticRows']>);
      case 'logisticProducts':
        return productsPass(k as NonNullable<Kernels['logisticProducts']>);
    }
  } catch {
    return false;
  }
}

function values(r: Random, length: number, scale: number): Float64Array {
  return Float64Array.from({ length }, () => (r.next() - 0.5) * scale);
}

/** gramPart on 37 rows of 9 columns (odd sizes, so a kernel's remainders run), plain, centred, weighted and with ones. */
function gramPasses(k: NonNullable<Kernels['gramPart']>): boolean {
  const r = new Random(35);
  const n = 37;
  const cols = 9;
  const X = new Matrix(n, cols, values(r, n * cols, 8));
  const mean = values(r, cols, 2);
  const weights = Float64Array.from({ length: n }, () => r.next() * 3);
  const cases: [ArrayLike<number> | undefined, number, number, ArrayLike<number> | undefined, boolean][] = [
    [undefined, 0, n, undefined, false],
    [mean, 0, n, undefined, false],
    [mean, 5, 30, undefined, false],
    [undefined, 0, n, weights, true],
    [undefined, 3, 36, weights, false],
    [undefined, 0, n, undefined, true],
  ];
  const before = X.data.slice();
  for (const [m, from, to, w, ones] of cases) {
    const want = ownGramPart(X, m, from, to, w, ones);
    const got = k(X, m, from, to, w, ones);
    // A kernel reads X; it never writes it.
    if (X.data.some((v, i) => v !== before[i])) return false;
    const p = cols + (ones ? 1 : 0);
    if (!(got instanceof Float64Array) || got.length !== p * p) return false;
    // The lower triangle, relative to √(A_jj · A_kk); the upper one is never read.
    for (let j = 0; j < p; j++) {
      for (let c = 0; c <= j; c++) {
        const scale = Math.sqrt(want[j * p + j]! * want[c * p + c]!) || 1;
        if (!(Math.abs(got[j * p + c]! - want[j * p + c]!) <= TOLERANCE * scale)) return false;
      }
    }
  }
  return true;
}

/** matMul on 13 × 7 by 7 × 3, and on a product with an empty inner size. */
function matMulPasses(k: NonNullable<Kernels['matMul']>): boolean {
  const r = new Random(36);
  for (const [n, p, q] of [[13, 7, 3], [5, 1, 4], [4, 0, 2]] as const) {
    const a = values(r, n * p, 6);
    const b = values(r, p * q, 6);
    const want = ownMatMul(n, p, q, a, b);
    const got = k(n, p, q, a.slice(), b.slice());
    if (!(got instanceof Float64Array) || got.length !== n * q) return false;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < q; j++) {
        let scale = 0;
        for (let l = 0; l < p; l++) scale += Math.abs(a[l * n + i]! * b[j * p + l]!);
        if (!(Math.abs(got[j * n + i]! - want[j * n + i]!) <= TOLERANCE * (scale || 1))) return false;
      }
    }
  }
  return true;
}

/** Within the tolerance of each value's scale: a residual is at most its row's weight in size. */
function near(got: ArrayLike<number>, want: ArrayLike<number>, scale: (i: number) => number): boolean {
  if (got.length !== want.length) return false;
  for (let i = 0; i < want.length; i++) if (!(Math.abs(got[i]! - want[i]!) <= TOLERANCE * scale(i))) return false;
  return true;
}

/** The binary row step on 41 rows from row 6 (an odd count, an offset), scores from −30 to 30, with and without weights. */
function binaryRowsPass(k: NonNullable<Kernels['binaryLogisticRows']>): boolean {
  const r = new Random(37);
  const n = 50;
  const y = Int32Array.from({ length: n }, () => (r.next() < 0.5 ? 1 : 0));
  const weights = Float64Array.from({ length: n }, () => r.next() * 2);
  const z = values(r, 43, 60);
  for (const w of [undefined, weights]) {
    const [want, got] = [z.slice(), z.slice()];
    const [y0, w0] = [y.slice(), weights.slice()];
    const lossWant = ownBinaryLogisticRows(want, y, w, 6, 41, 1.5);
    const lossGot = k(got, y, w, 6, 41, 1.5);
    if (!(Math.abs(lossGot - lossWant) <= TOLERANCE * Math.abs(lossWant))) return false;
    if (!near(got, want, (i) => (w && i < 41 ? Math.max(w[6 + i]!, 1) : 1))) return false;
    if (y.some((v, i) => v !== y0[i]) || weights.some((v, i) => v !== w0[i])) return false;
  }
  return true;
}

/** The multinomial row step on 4 classes of 37 rows from row 3, with a stride past them, with and without weights. */
function multinomialRowsPass(k: NonNullable<Kernels['multinomialLogisticRows']>): boolean {
  const r = new Random(38);
  const n = 45;
  const K = 4;
  const stride = 40;
  const y = Int32Array.from({ length: n }, () => Math.min(K - 1, Math.floor(r.next() * K)));
  const weights = Float64Array.from({ length: n }, () => r.next() * 2);
  const z = values(r, K * stride, 60);
  for (const w of [undefined, weights]) {
    const [want, got] = [z.slice(), z.slice()];
    const lossWant = ownMultinomialLogisticRows(want, stride, K, y, w, 3, 37, 2.5);
    const lossGot = k(got, stride, K, y, w, 3, 37, 2.5);
    if (!(Math.abs(lossGot - lossWant) <= TOLERANCE * Math.abs(lossWant))) return false;
    if (!near(got, want, (i) => (w && i % stride < 37 ? Math.max(w[3 + (i % stride)]!, 1) : 1))) return false;
  }
  return true;
}

/**
 * The logistic products on 1,100 rows of 9 columns (blocks of 512 and a remainder, odd column count), from row 0 and
 * from row 37, binary and with 3 classes, with and without weights and an intercept: against smaltml's own parts, both
 * with smaltml's own row steps.
 */
function productsPass(k: NonNullable<Kernels['logisticProducts']>): boolean {
  const r = new Random(39);
  const n = 1100;
  const p = 9;
  const X = new Matrix(n, p, values(r, n * p, 4));
  const before = X.data.slice();
  const prepared = k(X);
  if (prepared === null) return true; // it takes no X this small: nothing to check, and it is never used on one
  try {
    const y2 = Int32Array.from({ length: n }, () => (r.next() < 0.5 ? 1 : 0));
    const y3 = Int32Array.from({ length: n }, () => Math.min(2, Math.floor(r.next() * 3)));
    const weights = Float64Array.from({ length: n }, () => r.next() * 2);
    for (const [from, to] of [[0, n], [37, 1001]] as const) {
      for (const w of [undefined, weights]) {
        for (const intercept of [false, true]) {
          for (const K of [1, 3]) {
            const width = p + (intercept ? 1 : 0);
            const x = values(r, K * width, 0.6);
            const want = new Float64Array(K * width);
            const got = new Float64Array(K * width).fill(7); // a part fills grad, whatever it held
            const lossWant = K === 1 ? binaryLogisticPart(X, y2, intercept, x, want, from, to, w) : multinomialLogisticPart(X, y3, K, intercept, x, want, from, to, w);
            const lossGot = K === 1 ? prepared.binaryPart(y2, intercept, x, got, from, to, w, ownBinaryLogisticRows) : prepared.multinomialPart(y3, K, intercept, x, got, from, to, w, ownMultinomialLogisticRows);
            if (!(Math.abs(lossGot - lossWant) <= TOLERANCE * Math.abs(lossWant))) return false;
            // Each gradient entry is a sum of ~1,000 terms of size up to |x| · weight: relative to the sum of their sizes.
            const scale = (to - from) * 4 * 2;
            if (!near(got, want, () => scale)) return false;
          }
        }
      }
    }
    return X.data.every((v, i) => v === before[i]);
  } finally {
    prepared.release();
  }
}
