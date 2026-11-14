---
description: 'matMul: the matrix product A · B, on the CPU in 64-bit floats or on the GPU in 32-bit floats.'
---

# matMul

<!-- docs:skip -->
```ts
function matMul(A: MatrixInput, B: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
function matMulSync(A: MatrixInput, B: MatrixInput): Matrix;
```

The matrix product `A · B`: `A` is n × p, `B` is p × k, the result n × k. `matMul` chooses the device; `matMulSync`
always runs on the CPU. From `@smaltworks/smaltml/gpu`.

## Parameters

- `A`, `B`: matrices, as rows, named columns or `Matrix`.
- `device` (default `'auto'`): where it runs ([Choosing the device](../../guide/hardware#choosing-the-device)).

## Returns

A Matrix, n × k. On the GPU, each value agrees with the CPU's within about 1e-6 of the sum of the absolute terms
(32-bit floats).

## Errors

- `ML_SHAPE`: `A`'s columns differ from `B`'s rows, or `B` is larger than the GPU's buffers.
- `ML_PARAM`: `device: 'gpu'` where WebGPU isn't available, or a device that isn't `'auto'`, `'cpu'` or `'gpu'`.

## Notes

On the GPU, `A` is processed in chunks of rows that fit the device's buffer limits, so it can be larger than one
buffer. Each chunk is a 16 × 16-tiled compute shader.

## Examples

```ts
import { matMul, matMulSync } from '@smaltworks/smaltml';

matMulSync([[1, 0], [0, 1], [1, 1]], [[2, 3], [4, 5]])
```

| 0 | 1 |
| --- | --- |
| 2 | 3 |
| 4 | 5 |
| 6 | 8 |

## Working demo

```ts
const scores = await matMul({ maths: [70, 85, 60], art: [90, 55, 75] }, [[0.6], [0.4]]);
scores
```

| 0 |
| --- |
| 78 |
| 73 |
| 66 |

## See also

[`gpuAvailable`](./gpu-available), [The CPU, WebGPU and workers](../../guide/hardware)
