---
description: 'simdSupport: whether this engine has WebAssembly SIMD, and relaxed SIMD, for smaltml''s SIMD kernels.'
---

# simdSupport

<!-- docs:skip -->
```ts
function simdSupport(): SimdSupport;

interface SimdSupport {
  simd: boolean;
  relaxed: boolean;
}
```

Whether this engine's WebAssembly has 128-bit SIMD, which smaltml's SIMD kernels need, and relaxed SIMD, which
[`useSimd`](./use-simd)'s `fusedMultiplyAdd` needs. From `@smaltworks/smaltml/simd`.

## Parameters

None.

## Returns

- `simd`: the SIMD kernels can run. Chrome 91, Firefox 89, Safari 16.4 and Node have it, and later versions.
- `relaxed`: fused multiply-add can be turned on (Chrome 114 and later, and Node; not every browser).

## Errors

None. Where there is no WebAssembly at all, both are `false`.

## Notes

It compiles and runs nothing: it asks the engine to validate two modules of a few bytes, each using one instruction,
`v128.const` for SIMD and `f64x2.relaxed_madd` for relaxed SIMD. `useSimd()` calls it first.

## Examples

<!-- docs:skip -->
```ts
import { simdSupport, useSimd } from '@smaltworks/smaltml/simd';

if (simdSupport().relaxed) await useSimd({ fusedMultiplyAdd: true });
```

## Working demo

```ts
import { simdSupport } from '@smaltworks/smaltml/simd';

simdSupport()
```

| simd | relaxed |
| --- | --- |
| true | true |

## See also

[`useSimd`](./use-simd), [The CPU, WebGPU and workers](../../guide/hardware)
