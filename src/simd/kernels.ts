/**
 * The SIMD kernel set, for useKernels(). This module is also what the thread pool's workers import, built on its own
 * as `smaltml-simd-kernels.js` (scripts/build.mjs): its default export gives them the same set, so it imports nothing
 * of smaltml's but types.
 */
import type { KernelSet } from '../core/kernel-set.js';
import { columnsModule, SimdColumns } from './columns.js';
import { gramModule, SimdGram } from './gram.js';
import { logisticModule, SimdLogistic } from './logistic.js';
import { matMulModule, SimdMatMul } from './matmul.js';
import { productsModule, SimdProducts } from './products.js';
import { simdSupport } from 'smaltkit/simd';

/** How the SIMD kernels run. */
export interface SimdOptions {
  /**
   * Fused multiply-add (relaxed SIMD's `f64x2.relaxed_madd`), where the engine has it: faster, with one rounding per
   * multiply-add where the processor fuses them, so the last bits can differ between processors. Off by default.
   */
  fusedMultiplyAdd?: boolean;
  /**
   * The logistic objectives' row steps too, for LogisticRegression's lbfgs fits, on smaltkit's f64x2 exp, log1p and
   * log, within a few rounding steps of Math's. On by default (the owner, 2026-11-18: lbfgs was 0.72–0.91 times
   * scikit-learn's time with them, 0.94–1.20 without, at 200k × 50); `false` keeps Math's.
   */
  logistic?: boolean;
  /**
   * Where the thread pool's workers load the kernels from: by default `smaltml-simd-kernels.js`, beside smaltml's
   * own files. Where a bundler folds smaltml into an app's bundle, copy `dist/smaltml-simd-kernels.js` (the kernels
   * and all they need, in one file) beside the app, and give its URL here; otherwise the workers can't load the
   * kernels, and fits run on one thread.
   */
  kernelsModule?: string;
}

/** The kernel contract the set is written for: smaltml's KERNELS_VERSION. */
export const KERNELS_VERSION = 1;

/** The set's name, as smaltml's kernelsInUse() reports it. */
export function setName(fused: boolean): string {
  return fused ? 'smaltml-simd, fused multiply-add' : 'smaltml-simd';
}

/**
 * The kernel set: gramPart (XᵀX), matMul (A · B), xtvPart (Xᵀv), columnSums, the logistic products and, unless `logistic` is false, the
 * logistic objectives' row steps, on WebAssembly SIMD, compiled here. Fused multiply-add only when asked for and the engine
 * has relaxed SIMD. Fails where the engine has no WebAssembly SIMD; useSimd() checks that first.
 */
export default async function kernelSet(options: SimdOptions = {}): Promise<KernelSet> {
  const support = simdSupport();
  if (!support.simd) throw new Error("this engine's WebAssembly has no SIMD");
  const fused = !!options.fusedMultiplyAdd && support.relaxed;
  const [gramMod, matMulMod, productsMod, columnsMod] = await Promise.all([gramModule, matMulModule, productsModule, columnsModule].map((m) => WebAssembly.compile(m(fused))));
  const gram = new SimdGram(gramMod!);
  const columns = new SimdColumns(columnsMod!);
  const product = new SimdMatMul(matMulMod!);
  const products = new SimdProducts(productsMod!);
  const logistic = (options.logistic ?? true) ? new SimdLogistic(await WebAssembly.compile(logisticModule(fused))) : undefined;
  return {
    name: setName(fused),
    version: KERNELS_VERSION,
    kernels: {
      gramPart: (X, mean, from, to, weights, ones) => gram.gramPart(X, mean, from, to, weights, ones),
      matMul: (n, p, k, a, b) => product.matMul(n, p, k, a, b),
      logisticProducts: (X) => products.prepare(X),
      xtvPart: (X, v, mean, from, to) => columns.xtvPart(X, v, mean, from, to),
      columnSums: (X) => columns.columnSums(X),
      ...(logistic && {
        binaryLogisticRows: (z, y, weights, at, m, loss) => loss + logistic.binaryRows(z, y, weights, at, m),
        multinomialLogisticRows: (z, stride, K, y, weights, at, m, loss) => loss + logistic.multinomialRows(z, stride, K, y, weights, at, m),
      }),
    } satisfies KernelSet['kernels'],
    threads: { module: options.kernelsModule ?? new URL('./smaltml-simd-kernels.js', import.meta.url).href, options: { fusedMultiplyAdd: fused, logistic: !!logistic } },
  };
}
