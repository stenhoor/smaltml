---
description: 'Run on the hardware you have: the CPU everywhere, on several cores where memory can be shared, WebGPU where the browser has it, and a Web Worker so long fits leave the page free.'
---

# The CPU, WebGPU and workers

Every model runs on the CPU, in 64-bit floats, in every browser and in Node. Large fits use several cores where memory
can be shared. Where the browser has WebGPU, smaltml can run the heavy steps on the GPU, and a Web Worker keeps long
fits off the page's main thread.

## On several cores

The asynchronous `fit` of the linear models takes a `threads` option. The heavy steps (XᵀX, Xᵀy, the logistic loss
and its gradient) then run in workers that share X's memory, each on its own rows:

- `'auto'` (the default) uses every core when X holds at least `PARALLEL_MIN_VALUES` numbers (about a million).
  Smaller fits stay on one thread, where messages to the workers would cost more than they save.
- A number asks for that many threads; `1` keeps the fit on this thread.

Each worker sums its rows in the same chunks, and the parts are added up in the same order as on one thread. A fit
on any number of threads therefore gives exactly what `fitSync` gives, to the last bit. A pipeline passes `threads`
to its last step.

The random forests' `fit` takes `threads` too, and splits the work differently: each worker grows whole trees. Each
tree draws its sample of rows and its columns from its own stream, which the forest's `seed` gives, so the forest is
the same on any number of threads, to the last bit. `'auto'` counts X's numbers once per tree against
`PARALLEL_MIN_VALUES`, so a forest of 100 trees uses the cores from about 10,000 numbers in X.

<!-- docs:skip -->
```ts
import { LogisticRegression } from '@smaltworks/smaltml';

const model = await new LogisticRegression().fit(X, y, { threads: 'auto' }); // 2 to 3 times faster on 4 cores
```

Threads need shared memory. Node always has it. A browser has it only on a page that is cross-origin isolated (served
with the headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`).
Elsewhere, fits run on one thread with the same result. `sharedMemoryAvailable()` says which.
[`configureThreads`](../reference/parallel/configure-threads) sets where the workers' script is, for bundlers that
move it, and the most threads `'auto'` uses.

## SIMD kernels

smaltml's kernels are TypeScript, and JavaScript has no SIMD. smaltml also has the heaviest of them written as
WebAssembly SIMD ([ADR 0040](../adr/0040-simd-kernels-in-smaltml)): XᵀX, behind the linear models and
`newton-cholesky`; the matrix product, behind `matMulSync`; and the logistic objectives' products over X, behind
LogisticRegression's `lbfgs` and `newton-cholesky`, with their row steps (the exp and log between the products). They are in their own entry point, `@smaltworks/smaltml/simd`,
written from TypeScript at run time: no `.wasm` file and no dependency.

smaltml turns them on by itself where the engine has WebAssembly SIMD (Chrome 91, Firefox 89, Safari 16.4, Node,
and later): the first asynchronous fit or `matMul` does. They go through [`useKernels`](../reference/use-kernels):
each kernel is checked first against smaltml's own, and the thread pool's workers load the same kernels, so a fit on
several threads still gives exactly what `fitSync` gives. Results move in the last bits, since SIMD sums in another
order: close to smaltml's own, not equal, and within every compare case's tolerance against scikit-learn.
`kernelsInUse()` says which kernels run.

- **Synchronous code first:** the `*Sync` calls don't turn them on. `await kernelsReady()` first, and they use them
  from the start; otherwise they use smaltml's own kernels until the first asynchronous call.
- **Bundled apps:** a bundler doesn't follow smaltml's look-up, so an app that bundles smaltml imports
  `@smaltworks/smaltml/simd` and calls [`useSimd()`](../reference/simd/use-simd) itself, and copies
  `smaltml-simd-kernels.js` beside its bundle for the thread pool's workers.
- **Options:** `useSimd({ fusedMultiplyAdd: true })` where the engine has relaxed SIMD, a little faster, with last bits
  that can differ between processors (off by default); `useSimd({ logistic: false })` keeps the logistic row steps
  (the exp and log of lbfgs fits) on `Math`'s functions, which are on SIMD by default.
- **Turning them off:** [`useKernels(null)`](../reference/use-kernels), before the first asynchronous fit or after: an
  app's own choice is kept.

[`simdSupport()`](../reference/simd/simd-support) says whether this engine can run them, and fused multiply-add:

```ts
import { simdSupport } from '@smaltworks/smaltml/simd';

simdSupport()
```

| simd | relaxed |
| --- | --- |
| true | true |

## Choosing the device

The asynchronous calls that can use the GPU take a `device` option:

- `'auto'` (the default) runs predictions and fits on the CPU: in the benchmarks, the CPU was faster for every
  prediction measured ([ADR 0034](../adr/0034-where-auto-runs)). `matMul` uses the GPU when the browser has WebGPU and
  the product is at least `GPU_MIN_WORK` multiply-adds.
- `'cpu'` always runs on the CPU, and gives exactly what the `*Sync` call gives; so does `'auto'` for predictions.
- `'gpu'` always runs on the GPU, and fails with `ML_PARAM` where there is none.

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const model = await new LinearRegression().fit([[1, 2], [2, 1], [3, 4], [4, 3]], [5, 4, 11, 10]);
await model.predict([[5, 5], [6, 2]], { device: 'auto' })
```

| value |
| --- |
| 15 |
| 10 |

With `'gpu'`: `predict` and `predictProba` of the linear models and of pipelines that end in one, `matMul`, and
`LogisticRegression` fits.

### Fits on the GPU

`LogisticRegression.fit` takes `device: 'gpu'`. The loss and its gradient, the step that reads all of X, then run on
the GPU at each iteration; the optimizer stays on the CPU. Each workgroup sums its rows' gradient in 32-bit floats,
and the CPU adds the workgroups' sums in 64-bit floats. The fit lands close to the CPU's, not on it: in the browser
tests, its probabilities are within about 1e-6 of a CPU fit's (the tests allow 2e-3).

<!-- docs:skip -->
```ts
const model = await new LogisticRegression().fit(X, y, { device: 'gpu' });
```

A fit asks for the GPU only with `'gpu'`: `'auto'`, the default, keeps fits on the CPU, where they give exactly what
`fitSync` gives. More than 8 classes, or an X larger than the GPU's largest buffer, fits on the CPU. Weights
(`sampleWeight`, `classWeight`) aren't on the GPU yet: a weighted fit with `'gpu'` fails with `ML_PARAM`.
Without WebGPU, `'gpu'` fails with `ML_PARAM`.

## What the GPU computes

The GPU computes in 32-bit floats, the CPU in 64-bit. A GPU result agrees with the CPU's within about 1e-6 of the size
of the terms it adds up, not bit for bit. The browser tests check every GPU path against the CPU within that
tolerance. A step where 32-bit error matters stays on the CPU.

`matMul` is the matrix product the models use, and is there for your own code:

```ts
import { matMul } from '@smaltworks/smaltml';

await matMul([[1, 2], [3, 4]], [[5], [6]])
```

| 0 |
| --- |
| 17 |
| 39 |

## In a Web Worker

A fit on the page's main thread holds up the page until it finishes. `startWorker` runs smaltml in a worker of its
own: the worker fits a copy of your model and hands back an ordinary fitted model, and predictions run there too.

<!-- docs:skip -->
```ts
import { startWorker } from '@smaltworks/smaltml/worker';
import { LogisticRegression } from '@smaltworks/smaltml';

const worker = startWorker();
const model = await worker.fit(new LogisticRegression(), XTrain, yTrain); // fitted in the worker
const labels = await worker.predict(model, XTest, { device: 'auto' });      // WebGPU works in workers too
worker.terminate();
```

Code that already runs off the main thread, such as a notebook kernel or Node, can call the models directly. A fit in
the worker is the same code as a fit on the page, so it gives the same result, bit for bit.

## See also

- [`matMul`](../reference/gpu/mat-mul), [`gpuAvailable`](../reference/gpu/gpu-available),
  [`configureThreads`](../reference/parallel/configure-threads), [`useKernels`](../reference/use-kernels),
  [`useSimd`](../reference/simd/use-simd), [`startWorker`](../reference/worker/start-worker), [`connect`](../reference/worker/connect)
