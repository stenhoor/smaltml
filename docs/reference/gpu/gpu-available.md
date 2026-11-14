---
description: 'gpuAvailable: whether this browser gives smaltml a WebGPU device.'
---

# gpuAvailable

<!-- docs:skip -->
```ts
function gpuAvailable(): Promise<boolean>;
function gpuDevice(): Promise<GPUDevice | null>;
const GPU_MIN_WORK: number;
```

Whether this browser gives smaltml a WebGPU device. `gpuDevice` is the device itself, or `null`; smaltml asks for it
once and shares it. From `@smaltworks/smaltml/gpu`.

## Returns

`true` where WebGPU works (most Chrome and Edge, Firefox on Windows, Safari in part), `false` elsewhere, in Node and in
browsers without it. `GPU_MIN_WORK` is the number of multiply-adds below which `device: 'auto'` stays on the CPU.

## Errors

None: a browser without WebGPU, or an adapter that refuses, gives `false`.

## Notes

The result can differ from one visitor to the next, so code that uses it should work both ways. A lost device (after a
driver reset, for example) is asked for again on the next call.

## Examples

<!-- docs:skip -->
```ts
import { gpuAvailable } from '@smaltworks/smaltml';

const where = (await gpuAvailable()) ? 'the GPU' : 'the CPU';
```

## Working demo

```ts
import { GPU_MIN_WORK } from '@smaltworks/smaltml';

GPU_MIN_WORK
```

| value |
| --- |
| 4194304 |

## See also

[`matMul`](./mat-mul), [The CPU, WebGPU and workers](../../guide/hardware)
