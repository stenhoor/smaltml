/**
 * WebAssembly SIMD kernels for smaltml's heaviest CPU work (ADR 0040): XᵀX, the matrix product, and the logistic
 * objectives' products and row steps. Optional: smaltml's own TypeScript kernels run wherever these don't, and smaltml
 * turns these on by itself where the engine has WebAssembly SIMD (kernelsReady()). Written from TypeScript at run time
 * with smaltkit's WebAssembly writer, which the build folds in: no `.wasm` file and no dependency.
 * @module
 */
import { simdSupport as kitSupport } from 'smaltkit/simd';
import { useKernels } from '../core/kernel-set.js';
import kernelSet, { type SimdOptions } from './kernels.js';

export { type SimdOptions } from './kernels.js';

/** What this engine's WebAssembly supports. */
export interface SimdSupport {
  /** 128-bit SIMD: the kernels can run. */
  simd: boolean;
  /** Relaxed SIMD: `fusedMultiplyAdd` can be turned on. */
  relaxed: boolean;
}

/** Whether this engine has WebAssembly SIMD, and relaxed SIMD. Compiles and runs nothing. */
export function simdSupport(): SimdSupport {
  return kitSupport();
}

/**
 * Turns the SIMD kernels on: fits use them from now on, in this thread and in the thread pool. Resolves to `true`
 * when they do, and to `false` where WebAssembly SIMD isn't available or a kernel failed the check at start: smaltml's
 * own kernels then stay in use. `useKernels(null)` turns them off.
 */
export async function useSimd(options: SimdOptions = {}): Promise<boolean> {
  if (!simdSupport().simd) return false;
  const used = useKernels(await kernelSet(options));
  return used.kernels.length > 0;
}
