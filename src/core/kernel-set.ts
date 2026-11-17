/**
 * Replaceable kernels (ADR 0035): a plugin, such as smaltml-simd, can swap the CPU kernels smaltml already has for its
 * own, with the same signatures. The models and solvers don't change; nothing here looks for a plugin, and nothing
 * here runs code other than the functions it is given.
 */
import { fail } from './errors.js';
import { ownGramPart, ownMatMul, replaced, type Kernels } from './kernels.js';
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
const NAMES: readonly KernelName[] = ['gramPart', 'matMul'];

/** A set of kernels to use in place of smaltml's own: what a plugin gives useKernels(). */
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

/**
 * Uses a set of kernels in place of smaltml's own, or smaltml's own again with `null`. Each kernel the set gives is
 * checked first, on a small product against smaltml's own; one that throws, or differs by more than rounding, isn't
 * used. It affects every model fitted after it, in this thread and in the thread pool's workers.
 */
export function useKernels(set: KernelSet | null): KernelsInUse {
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
    return name === 'gramPart' ? gramPasses(k as NonNullable<Kernels['gramPart']>) : matMulPasses(k as NonNullable<Kernels['matMul']>);
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
