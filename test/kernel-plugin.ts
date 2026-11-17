// A stand-in for a kernel plugin such as smaltml-simd, for test/kernel-set.test.ts: smaltml's own kernels, with each
// result moved by a few rounding steps, so a fit that used them can be told apart, to the last bit, from one that
// didn't. The tests bundle it for the thread pool's workers, which import it and call its default export.
import { ownBinaryLogisticRows, ownGramPart, ownMatMul, ownMultinomialLogisticRows } from '../src/core/kernels.js';
import type { KernelSet } from '../src/core/kernel-set.js';

/** x moved by `steps` relative rounding steps: well within the check at start's tolerance. */
const nudge = (out: Float64Array, steps: number) => out.map((v) => v * (1 + steps * 2 ** -52));

export const calls = { gramPart: 0, matMul: 0, binaryLogisticRows: 0, multinomialLogisticRows: 0 };

export default function plugin(options: { steps?: number; module?: string } = {}): KernelSet {
  const steps = options.steps ?? 3;
  return {
    name: 'nudged',
    version: 1,
    kernels: {
      gramPart(X, mean, from, to, weights, ones) {
        calls.gramPart++;
        return nudge(ownGramPart(X, mean, from, to, weights, ones), steps);
      },
      matMul(n, p, k, a, b) {
        calls.matMul++;
        return nudge(ownMatMul(n, p, k, a, b), steps);
      },
      binaryLogisticRows(z, y, weights, at, m, loss) {
        calls.binaryLogisticRows++;
        const out = ownBinaryLogisticRows(z, y, weights, at, m, loss);
        z.set(nudge(z.subarray(0, m), steps));
        return out * (1 + steps * 2 ** -52);
      },
      multinomialLogisticRows(z, stride, K, y, weights, at, m, loss) {
        calls.multinomialLogisticRows++;
        const out = ownMultinomialLogisticRows(z, stride, K, y, weights, at, m, loss);
        for (let c = 0; c < K; c++) z.set(nudge(z.subarray(c * stride, c * stride + m), steps), c * stride);
        return out * (1 + steps * 2 ** -52);
      },
    },
    threads: options.module ? { module: options.module, options: { steps } } : undefined,
  };
}
