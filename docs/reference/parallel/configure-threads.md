---
description: 'configureThreads: where the thread pool finds its workers script, and how many threads fits may use.'
---

# configureThreads

<!-- docs:skip -->
```ts
function configureThreads(options: { script?: string | URL; max?: number }): void;
function stopThreads(): Promise<void>;
function sharedMemoryAvailable(): boolean;
const PARALLEL_MIN_VALUES: number;
```

Settings for the thread pool that runs large fits on several cores
([The CPU, WebGPU and workers](../../guide/hardware#on-several-cores)). From `@smaltworks/smaltml/parallel`, and the
main entry point.

## Parameters

- `script`: the URL of the workers' script, `smaltml-threads.js`. By default the pool loads it from beside smaltml's
  own module, where the package ships it (`@smaltworks/smaltml/threads-script`). Set it when a bundler moves files.
- `max`: the most threads `threads: 'auto'` uses. By default it uses every core. A whole number of at least 1.

Either setting stops the pools already started; the next fit starts new ones.

## Returns

`configureThreads` returns nothing. `stopThreads` stops every pool's workers; the next threaded fit starts them again.
`sharedMemoryAvailable` is `true` in Node and on a cross-origin isolated page, where threads can run. Fits with
`threads: 'auto'` use one thread below `PARALLEL_MIN_VALUES` numbers in X.

## Errors

- `ML_PARAM`: `max` that isn't a whole number of at least 1. A fit given `threads` that isn't `'auto'` or a whole
  number of at least 1 fails with it too.

## Notes

Where the workers can't start (no shared memory, a script that doesn't load), fits run on one thread and give the
same result. In Node an idle pool doesn't keep the process running, so `stopThreads` is needed only to free the
workers sooner.

## Examples

<!-- docs:skip -->
```ts
import { configureThreads, LinearRegression } from '@smaltworks/smaltml';

configureThreads({ script: '/assets/smaltml-threads.js', max: 4 });
const model = await new LinearRegression().fit(X, y); // up to 4 threads, when X is large
```

## Working demo

```ts
import { PARALLEL_MIN_VALUES } from '@smaltworks/smaltml';

PARALLEL_MIN_VALUES
```

| value |
| --- |
| 1048576 |

## See also

[`gpuAvailable`](../gpu/gpu-available), [`startWorker`](../worker/start-worker),
[The CPU, WebGPU and workers](../../guide/hardware)
