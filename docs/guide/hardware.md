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

## Choosing the device

The asynchronous calls that can use the GPU take a `device` option:

- `'auto'` (the default) uses the GPU when the browser has WebGPU and the work is large enough to gain from it
  (`GPU_MIN_WORK` multiply-adds). Small work stays on the CPU, where moving data to the GPU would cost more than it
  saves.
- `'cpu'` always runs on the CPU, and gives exactly what the `*Sync` call gives.
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

This release runs predictions on the GPU: `predict` and `predictProba` of the linear models and of pipelines that end
in one, and `matMul`. Fits run on the CPU.

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
  [`configureThreads`](../reference/parallel/configure-threads),
  [`startWorker`](../reference/worker/start-worker), [`connect`](../reference/worker/connect)
