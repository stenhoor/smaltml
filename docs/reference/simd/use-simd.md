---
description: 'useSimd: turns smaltml''s WebAssembly SIMD kernels on, for XᵀX, the matrix product and the logistic objectives.'
---

# useSimd

<!-- docs:skip -->
```ts
function useSimd(options?: SimdOptions): Promise<boolean>;

interface SimdOptions {
  fusedMultiplyAdd?: boolean;
  logistic?: boolean;
  kernelsModule?: string;
}
```

Turns smaltml's SIMD kernels on: XᵀX (`gramPart`), the matrix product (`matMul`) and the logistic objectives' products
over X (`logisticProducts`), written as WebAssembly SIMD ([ADR 0040](../../adr/0040-simd-kernels-in-smaltml)). From
`@smaltworks/smaltml/simd`. smaltml calls it by itself at the first asynchronous fit or `matMul`, where the engine has
WebAssembly SIMD ([The CPU, WebGPU and workers](../../guide/hardware#simd-kernels)); call it yourself in a bundled
app, or to choose its options.

## Parameters

- `fusedMultiplyAdd`: relaxed SIMD's fused multiply-add, where the engine has it (`simdSupport().relaxed`): a little
  faster, with last bits that can differ between processors. Off by default.
- `logistic`: the logistic objectives' row steps (the exp and log between X·w and Xᵀr) on SIMD too, with our own exp,
  log1p and log, within a few rounding steps of `Math`'s. Off by default.
- `kernelsModule`: where the thread pool's workers load the kernels from. By default `smaltml-simd-kernels.js`, beside
  smaltml's own files (`@smaltworks/smaltml/simd-kernels-script`); set it when a bundler moves files.

## Returns

`true` when the SIMD kernels are in use, in this thread and in the thread pool; `false` where the engine has no
WebAssembly SIMD, or a kernel failed the check at start. Then smaltml's own kernels stay in use.

## Errors

None: where the SIMD kernels can't run, it resolves to `false`.

## Notes

- **Close, not equal.** SIMD sums in another order, so fits move in the last bits: close to smaltml's own, within
  every compare case's tolerance against scikit-learn. A fit on several threads still equals `fitSync`, bit for bit:
  the workers load the same kernels.
- **The `*Sync` calls don't wait.** Code that calls them first, and wants SIMD from the start, awaits
  [`kernelsReady()`](../use-kernels) or `useSimd()` first.
- **Turning them off:** [`useKernels(null)`](../use-kernels). An app's own choice, made before smaltml turns them on,
  is kept.
- While a logistic fit runs, `logisticProducts` keeps a copy of X in WebAssembly memory (one per worker on several
  threads), and frees it when the fit ends.

## Examples

<!-- docs:skip -->
```ts
import { kernelsInUse, Ridge } from '@smaltworks/smaltml';
import { useSimd } from '@smaltworks/smaltml/simd';

await useSimd({ fusedMultiplyAdd: true });
const model = new Ridge().fitSync(X, y); // XᵀX on SIMD, fused
kernelsInUse(); // { name: 'smaltml-simd, fused multiply-add', kernels: ['gramPart', 'matMul', 'logisticProducts'], … }
```

## Working demo

```ts
import { kernelsInUse, useKernels } from '@smaltworks/smaltml';
import { useSimd } from '@smaltworks/smaltml/simd';

const on = await useSimd();
const used = kernelsInUse();
useKernels(null);
({ on, name: used.name, kernels: used.kernels.join(', ') })
```

## See also

[`simdSupport`](./simd-support), [`useKernels`](../use-kernels), [The CPU, WebGPU and workers](../../guide/hardware)
