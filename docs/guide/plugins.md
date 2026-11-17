---
description: 'Plugins for smaltml: what they add, how to turn them on, and what they change. smaltml SIMD runs the heaviest kernels on WebAssembly SIMD.'
---

# Plugins

smaltml is TypeScript, with no dependencies, and it runs the same everywhere without any plugin. A plugin adds
something the core leaves out on purpose, and an app turns it on by name. Nothing in smaltml looks for one, and an app
that doesn't install one runs exactly as without it.

This page lists smaltml's plugins. An Official or Verified plugin badge links to its entry here.

## smaltml SIMD

<a href="https://github.com/stenhoor/smaltml-simd"><picture><source media="(prefers-color-scheme: dark)" srcset="/official-smaltml-on-dark.svg"><img alt="smaltml official plugin" src="/official-smaltml.svg" height="28"></picture></a>

`@smaltworks/smaltml-simd` runs two of smaltml's CPU kernels on WebAssembly SIMD. One is XᵀX, behind
LinearRegression, Ridge, Lasso, ElasticNet and LogisticRegression's `newton-cholesky`. The other is the matrix product,
behind [`matMulSync`](../reference/gpu/mat-mul), and `matMul` when it runs on the CPU. It is made and maintained by
smaltworks, at [stenhoor/smaltml-simd](https://github.com/stenhoor/smaltml-simd)
([ADR 0035](../adr/0035-simd-kernels-in-a-plugin)).

Install it beside smaltml (`npm install @smaltworks/smaltml-simd`), and smaltml uses it: the first asynchronous fit
or `matMul` looks for it and turns it on, where the engine has WebAssembly SIMD.

<!-- docs:skip -->
```ts
import { Ridge, kernelsInUse } from '@smaltworks/smaltml';

const model = await new Ridge().fit(X, y); // XᵀX on SIMD, here and on smaltml's threads
kernelsInUse(); // { name: 'smaltml-simd', kernels: ['gramPart', 'matMul'], rejected: [] }
```

- **Synchronous code first:** the `*Sync` calls don't look for the plugin. Call `await kernelsReady()` first, and
  they use it from the start; otherwise they use smaltml's own kernels until the first asynchronous call.
- **Bundled apps:** a bundler doesn't follow smaltml's look-up, so an app that bundles smaltml imports the plugin and
  calls `await useSimd()` from `@smaltworks/smaltml-simd` itself.

- **Why a plugin:** JavaScript has no SIMD, and WebAssembly is the platform's only route to it. smaltml itself stays
  TypeScript. The plugin's kernels are TypeScript too: functions that write the WebAssembly bytes when they run, with
  no toolchain, no `.wasm` file, and nothing loaded from another site.
- **What it changes:** speed, on the fits built on XᵀX and on the matrix product, and the last bits of those fits.
  SIMD sums XᵀX in another order, so fits are close to smaltml's own but not equal. The matrix product, under plain
  SIMD, is smaltml's own bit for bit.
- **Speed, measured:** on one 4-core machine, fits were 1.2 to 4.7 times as fast on one thread, and 1.1 to 3.0 times
  as fast on 4 threads. The gain grows with the number of columns. `matMulSync` was 3.3 to 6.3 times as fast with 3 or
  more columns of B, and 1.1 times with one. The method and every row are in its
  [benchmark](https://github.com/stenhoor/smaltml-simd/blob/main/bench/README.md).
- **What it doesn't change:** the models, the API and saved models. Agreement with scikit-learn holds: the plugin's CI
  runs smaltml's compare tests with it on. A fit on several threads still equals `fitSync`, because the thread pool's
  workers load the same kernels.
- **Fused multiply-add:** `useSimd({ fusedMultiplyAdd: true })`, where the browser has relaxed SIMD (Chrome 114 and
  later), is a little faster still. Its last bits can differ from one processor to another. It is off by default.
- **Where it runs:** WebAssembly SIMD is in Chrome 91, Firefox 89, Safari 16.4 and Node, and in later versions. Where
  it is missing, smaltml's own kernels stay in use (and `useSimd()` resolves to `false`).
- **Turning it off:** [`useKernels(null)`](../reference/use-kernels), before the first asynchronous fit or after: an
  app's own choice is kept.

## Writing a plugin

A plugin replaces kernels through [`useKernels`](../reference/use-kernels), with the signatures and results of
smaltml's own, for one version of the kernel contract (`KERNELS_VERSION`). smaltml checks each kernel against its own
before using it. A plugin that isn't smaltworks' own is named `smaltml-plugin-<name>`; the `@smaltworks` scope and names
that start with `smalt` are reserved.

## See also

[The CPU, WebGPU and workers](./hardware#plugin-kernels), [`useKernels`](../reference/use-kernels),
[ADR 0035](../adr/0035-simd-kernels-in-a-plugin)
