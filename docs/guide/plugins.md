---
description: 'Plugins for smaltml: what one may replace and how to write one. smaltml has none at present; its SIMD kernels are part of smaltml.'
---

# Plugins

smaltml is TypeScript, with no dependencies, and it runs the same everywhere without any plugin. A plugin adds
something the core leaves out on purpose, and an app turns it on by name.

smaltml has no plugins at present. Its WebAssembly SIMD kernels were a plugin, smaltml SIMD
(`@smaltworks/smaltml-simd`, [ADR 0035](../adr/0035-simd-kernels-in-a-plugin)), until 2026-11-18; they are now part of
smaltml, as `@smaltworks/smaltml/simd` ([ADR 0040](../adr/0040-simd-kernels-in-smaltml)), and smaltml turns them on
by itself ([SIMD kernels](./hardware#simd-kernels)).

## Writing a plugin

A plugin replaces kernels through [`useKernels`](../reference/use-kernels), with the signatures and results of
smaltml's own, for one version of the kernel contract (`KERNELS_VERSION`). smaltml checks each kernel against its own
before using it, and the thread pool's workers load the same set, from the module it names, so a fit on several
threads still equals `fitSync`. A plugin that isn't smaltworks' own is named `smaltml-plugin-<name>`; the
`@smaltworks` scope and names that start with `smalt` are reserved.

## See also

[The CPU, WebGPU and workers](./hardware#simd-kernels), [`useKernels`](../reference/use-kernels),
[ADR 0040](../adr/0040-simd-kernels-in-smaltml)
