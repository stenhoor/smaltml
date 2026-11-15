# ADR 0034: where `device: 'auto'` runs

- **Status:** accepted, 2026-11-15 (the owner's decisions of that day).
- **Date:** 2026-11-15
- **Amends:** [ADR 0024](./0024-smaltml-scope-and-architecture), section 3 ("Choosing").

## Context

ADR 0024 says `device: 'auto'` "uses the GPU when one is present and the data is large enough to gain". smaltml drew
that line at `GPU_MIN_WORK`, 2²² multiply-adds, a guess until benchmarks set it.

The first benchmark on a real GPU (`npm run bench:gpu`, pull request #18) ran on an RTX 3080 Laptop GPU and an
i9-11900H, in Chromium 157, with GPU calls timed end to end:

- **Predictions:** the CPU was faster in every case measured. On 200,000 rows × 50 columns,
  `LinearRegression.predict` took 7 ms on the CPU and 76 ms on the GPU, and `predictProba` with 3 classes took 30 ms
  and 61 ms. `'auto'` sent both to the GPU. A prediction reads X once, and each call converts X to 32-bit floats
  and uploads it; most likely that cost outweighs the GPU's speed (it wasn't timed on its own).
- **`matMul`:** the GPU was faster only with 8 output columns and at least about 8 million multiply-adds. `'auto'`
  picked the slower device in 4 of the 30 shapes measured.
- **Fits** (`LogisticRegression`, which `'auto'` keeps on the CPU). A fit keeps X on the GPU for every iteration, so
  its upload is paid once. With the default parameters, on columns of one scale, against the CPU on every core (16
  threads):
  - 2 classes: the GPU was at most 1.2× faster.
  - 3 classes: 2× or more in 1 of 14 sizes (10,000 to 1,000,000 rows, 10 to 100 columns).
  - 8 classes: 2× or more in 10 of 14 sizes, up to 6.8×.

  On the unscaled benchmark data, 200,000 × 50, the GPU was 2.0× (2 classes) and 3.0× (3 classes) faster than 4
  threads.

## Decision

1. **Predictions:** `'auto'`, the default, runs `predict`, `predictProba` and `decisionFunction` on the CPU. `'gpu'`
   still runs them on the GPU, as an explicit choice. On the CPU, an asynchronous prediction then gives exactly its
   `*Sync` twin's result, with or without WebGPU.
2. **Fits:** `'auto'` moves a fit to the GPU only where the benchmark measures it **at least 2× faster** than the CPU
   with all its threads, at that size of data. Until a follow-up sets that line from more measurements, `'auto'`
   fits stay on the CPU, as today.
3. **`matMul`** keeps `GPU_MIN_WORK` for now. It isn't decided here.
4. **One machine isn't enough.** The lines above come from one laptop. Issue #17 asks for testers with other GPUs and
   browsers; the lines change if their results disagree.

## Consequences

- Predictions get faster by default in browsers with WebGPU (up to 10× on the cases measured), and the same in Node and
  browsers without it.
- The browser tests check that an `'auto'` prediction above `GPU_MIN_WORK` equals the `*Sync` result bit for bit.
- The GPU prediction code stays, for `'gpu'` and for later work that may make it pay.
- Measured so far, 8-class fits mostly reach 2× against every core, and 2- and 3-class fits mostly don't: no single
  data size separates them. Until the follow-up sets the line, `'auto'` fits stay on the CPU.
