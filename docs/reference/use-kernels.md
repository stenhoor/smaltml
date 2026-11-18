---
description: 'useKernels: other kernels, such as smaltml''s SIMD kernels, in place of its own TypeScript ones; kernelsInUse says which run.'
---

# useKernels

<!-- docs:skip -->
```ts
function useKernels(set: KernelSet | null): KernelsInUse;
function kernelsInUse(): KernelsInUse;
function kernelsReady(): Promise<KernelsInUse>;
const KERNELS_VERSION: number;

interface KernelSet {
  name: string;
  version: number;
  kernels: {
    gramPart?: (X: Matrix, mean: ArrayLike<number> | undefined, from: number, to: number, weights?: ArrayLike<number>, ones?: boolean) => Float64Array;
    matMul?: (n: number, p: number, k: number, a: Float64Array, b: Float64Array) => Float64Array;
    binaryLogisticRows?: (z: Float64Array, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;
    multinomialLogisticRows?: (z: Float64Array, stride: number, K: number, y: Int32Array, weights: ArrayLike<number> | undefined, at: number, m: number, loss: number) => number;
  };
  threads?: { module: string | URL; options?: unknown };
}

type KernelName = 'gramPart' | 'matMul' | 'binaryLogisticRows' | 'multinomialLogisticRows';

interface KernelsInUse {
  name: string;
  kernels: KernelName[];
  rejected: KernelName[];
}
```

Uses a set of kernels in place of smaltml's own TypeScript CPU kernels, or smaltml's own again with `null`
([ADR 0035](../adr/0035-simd-kernels-in-a-plugin), [ADR 0040](../adr/0040-simd-kernels-in-smaltml)). From
`@smaltworks/smaltml/core`, and the main entry point.

An app normally doesn't call it, except as `useKernels(null)`. smaltml's SIMD kernels use it: `kernelsReady()`
imports `@smaltworks/smaltml/simd` and calls its [`useSimd()`](./simd/use-simd), which calls `useKernels`, where the
engine has WebAssembly SIMD. smaltml's asynchronous fits and `matMul` wait for `kernelsReady()`; the `*Sync` calls
don't, so code that calls them first awaits it. A `useKernels()` call made before the look-up finishes is kept. The
models and solvers don't change; only the kernels they call do.

## Parameters

- `set`: the kernels to use, or `null` for smaltml's own.
  - `name`: the set's name, as `kernelsInUse()` reports it.
  - `version`: the kernel contract it was written for: `KERNELS_VERSION` (1). Adding a kernel to the contract doesn't
    change it; changing what one takes or gives does.
  - `kernels`: the kernels it replaces, with the signatures and results of smaltml's own. One it leaves out stays
    smaltml's.
    - `gramPart`: (X − mean)ᵀ diag(weights) (X − mean) over rows `from` to `to`, row-major, p × p (p + 1 with `ones`,
      a last column of ones), the lower triangle filled. It is behind LinearRegression, Ridge, Lasso, ElasticNet and
      `newton-cholesky`.
    - `matMul`: A · B, A n × p and B p × k, both column by column: n × k, column by column. It is behind `matMulSync`
      and `matMul` on the CPU.
    - `binaryLogisticRows`: the binary logistic objective's step over rows `at` to `at + m − 1`, whose scores are
      `z[0]` to `z[m − 1]`. It returns `loss` plus their loss, Σ w (log(1 + eᶻ) − y z), and replaces each score by its
      residual w (σ(z) − y). It is behind LogisticRegression's `lbfgs` fits with two classes.
    - `multinomialLogisticRows`: the same for K classes, whose scores are `z[c · stride + i]`. It returns `loss` plus
      Σ w (max + log Σ e^(z − max) − z_y), and replaces each score by its residual w (softmax − one-hot). It is behind
      `lbfgs` fits with more classes.
    - `logisticProducts`: the logistic objectives' products over X (X·w and Xᵀr) for one fit. Given X, it returns
      `null` (smaltml's own products then) or an object whose `binaryPart` and `multinomialPart` give the results of
      smaltml's own parts on that X, with the row step passed in, and whose `release()` frees what it kept. It is
      behind LogisticRegression's `lbfgs` and `newton-cholesky` fits.
  - `threads`: how the thread pool's workers load the same kernels: each imports `module`, and calls its default
    export with `options` for the set. Without it, fits run on one thread while the set is in use.

## Returns

What is in use: the set's name (`'smaltml'` for smaltml's own), the kernels it replaced, and the kernels it gave that
failed the check at start (`rejected`), which stay smaltml's. `kernelsInUse()` returns the same, at any time.
`kernelsReady()` resolves to the same once smaltml has turned its SIMD kernels on, where it can; it does so once.

## Errors

- `ML_PARAM`: a set for another `version` of the contract, a kernel smaltml doesn't have, or something other than a
  set or `null`. smaltml's own kernels stay in use.

## Notes

- **The check at start.** Each kernel runs on a small product first, against smaltml's own. One that throws, writes
  to its input, or differs by more than 10⁻¹² of the entries' scale isn't used, and is listed in `rejected`.
- **Close, not equal.** A replaced kernel sums in another order, so results move in the last bits, about 10⁻¹⁴
  relative with the SIMD kernels on the benchmark data. Agreement with scikit-learn holds within every compare case's tolerance.
- **Threads.** With `threads`, every worker of the pool loads the set, so a fit on any number of threads still gives
  exactly what `fitSync` gives. A worker that can't load it, or ends up with other kernels, keeps the fit on one
  thread. Each call to `useKernels` restarts the pools at the next fit.
- **A Web Worker** started by [`startWorker`](./worker/start-worker) has kernels of its own: smaltml's, unless the
  worker's script turns the SIMD kernels on (`useSimd()`) before it calls `serve`.

## Examples

A kernel that fails the check at start isn't used:

```ts
import { useKernels, KERNELS_VERSION } from '@smaltworks/smaltml';

const wrong = useKernels({
  name: 'zeros',
  version: KERNELS_VERSION,
  kernels: { gramPart: (X, mean, from, to, weights, ones) => new Float64Array((X.cols + (ones ? 1 : 0)) ** 2) },
});
wrong.rejected
```

| value |
| --- |
| gramPart |

A set for another contract fails:

<!-- docs:error ML_PARAM -->
```ts
useKernels({ name: 'future', version: KERNELS_VERSION + 1, kernels: {} })
```

## Working demo

```ts
import { useKernels, kernelsInUse } from '@smaltworks/smaltml';

useKernels(null);
kernelsInUse().name
```

| value |
| --- |
| smaltml |

## See also

[`configureThreads`](./parallel/configure-threads), [`matMul`](./gpu/mat-mul),
[`useSimd`](./simd/use-simd), [The CPU, WebGPU and workers](../guide/hardware#simd-kernels),
[ADR 0040](../adr/0040-simd-kernels-in-smaltml)
