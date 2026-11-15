---
description: 'startWorker: smaltml in a Web Worker of its own, so long fits leave the page free.'
---

# startWorker

<!-- docs:skip -->
```ts
function startWorker(url?: string | URL): MlWorker;
interface MlWorker {
  fit<M extends Model>(model: M, X: MatrixInput, y?: Labels, options?: { sampleWeight?: ArrayLike<number> }): Promise<M>;
  predict(model: Model, X: MatrixInput, options?: { device?: Device }): Promise<Float64Array | Label[]>;
  predictProba(model: Model, X: MatrixInput, options?: { device?: Device }): Promise<Matrix>;
  transform(model: Model, X: MatrixInput): Promise<Matrix>;
  terminate(): void;
}
```

Starts smaltml's worker script in a Web Worker and connects to it. From `@smaltworks/smaltml/worker`.

## Parameters

- `url`: the worker script. Default: `smaltml-worker.js` beside the module, where the package builds it. With a bundler
  that moves files, pass its URL for `@smaltworks/smaltml/worker-script`.

## Returns

An `MlWorker`:

- `fit` sends the model's parameters and the data to the worker, fits a copy there, and returns the fitted copy, of
  the same class. The model you pass is unchanged.
- `predict`, `predictProba` and `transform` run a fitted model in the worker. `device` chooses the GPU there.
- `terminate` stops the worker; requests still waiting fail.

## Errors

- `ML_PARAM`: no Web Workers here (Node). Call the models directly.
- Errors from the worker come back as `SmaltmlError`s with their codes.

## Notes

Data is copied to the worker, so your arrays stay usable. Requests run one at a time, in order. A fit in the worker
is the same code as a fit on the page and gives the same result, bit for bit.

## Examples

<!-- docs:skip -->
```ts
import { startWorker } from '@smaltworks/smaltml/worker';
import { LogisticRegression } from '@smaltworks/smaltml';

const worker = startWorker();
const model = await worker.fit(new LogisticRegression({ C: 0.5 }), X, y);
const proba = await worker.predictProba(model, X);
worker.terminate();
```

## See also

[`connect`](./connect), [The CPU, WebGPU and workers](../../guide/hardware#in-a-web-worker)
