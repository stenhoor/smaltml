---
description: 'How fast smaltml trains and predicts, timed beside scikit-learn on the same data, the same parameters and the same machine, with the method.'
---

# Benchmarks

How long smaltml takes to train and predict, timed beside scikit-learn: the same data, the same parameters, the same
machine. ADR 0024 sets a goal (section 8): on the CPU, without the GPU, to train and predict at least as fast as
scikit-learn running natively in Python. It is a goal, not a release gate, and this page says where each model
stands.

## Method

- **Data:** two sizes, 10,000 rows × 20 columns and 200,000 rows × 50, made by a seeded generator
  (`bench/data.mjs`): columns of different scales and offsets, a linear target with noise, and two- and three-class
  labels from the same linear score. Both libraries read the same bytes.
- **Cases:** each library's defaults, except Lasso's `alpha` (0.1) and Ridge's (1). Fits on unscaled columns, so
  logistic regression stops at its 100-iteration default on the larger data; the iteration counts below show both
  libraries doing the same work.
- **Timing:** the median of 7 runs (3 for the larger data), after a warm-up run, in one process.
- **smaltml:** the built package (`dist/`), in Node: on one thread, calling the synchronous methods, and with fits on
  up to 4 threads (`fit(X, y, { threads: 'auto' })`, `configureThreads({ max: 4 })`), which leaves fits below about a
  million values on one thread. Predictions and `StandardScaler` run on one thread in both.
- **scikit-learn:** on one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and on every core, with its default
  BLAS.
- **Rerun:** `npm run bench` (smaltml), `npm run bench:python` (scikit-learn; needs `compare/requirements.txt`), then
  `npm run bench:report`, which writes the tables below from `bench/results/`.

`smaltml ÷ …` is how many times longer smaltml takes than scikit-learn on the same number of threads: below 1 is
faster.

## Results

<!-- bench:results -->

### 10k × 20

| Operation | smaltml, 1 thread | smaltml, 4 threads | scikit-learn, 1 thread | scikit-learn, 4 threads | smaltml ÷ scikit-learn, 1 thread | smaltml ÷ scikit-learn, 4 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StandardScaler fit + transform | 1.85 | 1.71 | 1.96 | 1.81 | 0.9× | 0.9× |
| LinearRegression fit | 3.90 | 3.71 | 4.52 | 4.63 | 0.9× | 0.8× |
| LinearRegression predict | 0.22 | 0.18 | 0.36 | 0.26 | 0.6× | 0.7× |
| Ridge fit | 2.74 | 2.44 | 1.89 | 1.55 | 1.5× | 1.6× |
| Lasso fit (alpha 0.1) | 2.51 | 2.39 | 1.80 | 1.60 | 1.4× | 1.5× |
| LogisticRegression fit, 2 classes | 79.7 | 115 | 60.7 | 66.2 | 1.3× | 1.7× |
| LogisticRegression fit, 2 classes, newton-cholesky | 21.9 | 37.6 | 14.0 | 24.0 | 1.6× | 1.6× |
| LogisticRegression fit, 3 classes | 187 | 203 | 133 | 138 | 1.4× | 1.5× |
| LogisticRegression predictProba, 3 classes | 1.35 | 1.85 | 1.44 | 1.31 | 0.9× | 1.4× |

Iterations: Lasso 5 (scikit-learn 5), logistic regression 100 and 100 (scikit-learn 100 and 100).

### 200k × 50

| Operation | smaltml, 1 thread | smaltml, 4 threads | scikit-learn, 1 thread | scikit-learn, 4 threads | smaltml ÷ scikit-learn, 1 thread | smaltml ÷ scikit-learn, 4 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StandardScaler fit + transform | 114 | 206 | 119 | 123 | 1.0× | 1.7× |
| LinearRegression fit | 293 | 159 | 501 | 265 | 0.6× | 0.6× |
| LinearRegression predict | 11.9 | 12.7 | 13.4 | 9.87 | 0.9× | 1.3× |
| Ridge fit | 202 | 100 | 74.9 | 75.4 | 2.7× | 1.3× |
| Lasso fit (alpha 0.1) | 276 | 113 | 119 | 64.1 | 2.3× | 1.8× |
| LogisticRegression fit, 2 classes | 3,274 | 1,457 | 2,121 | 3,026 | 1.5× | 0.5× |
| LogisticRegression fit, 2 classes, newton-cholesky | 2,016 | 923 | 984 | 605 | 2.0× | 1.5× |
| LogisticRegression fit, 3 classes | 8,167 | 3,205 | 4,218 | 9,267 | 1.9× | 0.3× |
| LogisticRegression predictProba, 3 classes | 50.8 | 65.9 | 29.7 | 33.2 | 1.7× | 2.0× |

Iterations: Lasso 4 (scikit-learn 4), logistic regression 100 and 100 (scikit-learn 100 and 100).

Run 2026-11-15: smaltml 0.1.0 on Node 22.23.4; scikit-learn 1.9.1, NumPy 2.5.4 on Python 3.13.18 (openblas 0.3.31.dev, openblas 0.3.34.106.0, openmp). CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores. smaltml's runs on several threads use fit() with threads 'auto' (up to that many); predictions and StandardScaler run on one thread in every smaltml column.

<!-- /bench:results -->

## WebGPU

smaltml's GPU path, timed against its CPU path in the same browser page, on a machine with a GPU.

- **Where:** Chromium, on the GPU through Vulkan (`bench/gpu.mjs` passes the flags), the high-performance adapter
  where there are two. The script refuses a software adapter such as SwiftShader: that is the CPU, and timing it says
  nothing.
- **Timing:** the median of at least 5 runs and a second in all (3 when a run takes over a second), after a warm-up.
  A GPU call is timed end to end: converting to 32-bit floats, uploading, the shader and reading the result back.
- **Cases:** `matMul` over a grid of shapes, the product behind every prediction on the GPU, beside the device
  `'auto'` picks for it. The predictions and `LogisticRegression` fits of the CPU benchmark, on its data and
  parameters. `LogisticRegression` fits by rows, up to 30 iterations. The CPU column is one thread; fits also run on 4
  (`fit(X, y, { threads: 4 })`).
- **Rerun:** `npm run bench:gpu` (Playwright's Chromium: `npx playwright install chromium` first), or
  `npm run bench:gpu -- --browser /usr/bin/chromium` for another one; then `npm run bench:report`.

<!-- bench:gpu -->

### matMul

`'auto'` runs on the GPU from `GPU_MIN_WORK` (4,194,304) multiply-adds, n · p · k. **slower** marks the cases where that picks the slower device.

| A · B | Multiply-adds | CPU (ms) | GPU (ms) | GPU ÷ CPU | `auto` picks |
| --- | ---: | ---: | ---: | ---: | --- |
| 1,000 × 16 · 16 × 1 | 16,000 | 0.05 | 4.41 | 88.1× | cpu |
| 1,000 × 16 · 16 × 3 | 48,000 | 0.28 | 4.16 | 15.1× | cpu |
| 1,000 × 16 · 16 × 8 | 128,000 | 0.57 | 4.66 | 8.2× | cpu |
| 1,000 × 64 · 64 × 1 | 64,000 | 0.25 | 5.10 | 20.4× | cpu |
| 1,000 × 64 · 64 × 3 | 192,000 | 0.45 | 4.77 | 10.6× | cpu |
| 1,000 × 64 · 64 × 8 | 512,000 | 0.67 | 5.14 | 7.7× | cpu |
| 4,000 × 16 · 16 × 1 | 64,000 | 0.10 | 4.27 | 42.6× | cpu |
| 4,000 × 16 · 16 × 3 | 192,000 | 0.74 | 4.50 | 6.1× | cpu |
| 4,000 × 16 · 16 × 8 | 512,000 | 1.08 | 4.04 | 3.7× | cpu |
| 4,000 × 64 · 64 × 1 | 256,000 | 0.33 | 5.15 | 15.6× | cpu |
| 4,000 × 64 · 64 × 3 | 768,000 | 1.08 | 6.55 | 6.1× | cpu |
| 4,000 × 64 · 64 × 8 | 2,048,000 | 2.71 | 4.76 | 1.8× | cpu |
| 16,000 × 16 · 16 × 1 | 256,000 | 0.36 | 6.63 | 18.2× | cpu |
| 16,000 × 16 · 16 × 3 | 768,000 | 1.13 | 5.04 | 4.5× | cpu |
| 16,000 × 16 · 16 × 8 | 2,048,000 | 3.03 | 6.92 | 2.3× | cpu |
| 16,000 × 64 · 64 × 1 | 1,024,000 | 1.32 | 7.21 | 5.5× | cpu |
| 16,000 × 64 · 64 × 3 | 3,072,000 | 3.97 | 7.51 | 1.9× | cpu |
| 16,000 × 64 · 64 × 8 | 8,192,000 | 10.7 | 8.32 | 0.8× | gpu |
| 64,000 × 16 · 16 × 1 | 1,024,000 | 1.45 | 7.19 | 5.0× | cpu |
| 64,000 × 16 · 16 × 3 | 3,072,000 | 4.47 | 8.42 | 1.9× | cpu |
| 64,000 × 16 · 16 × 8 | 8,192,000 | 11.8 | 11.5 | 1.0× | gpu |
| 64,000 × 64 · 64 × 1 | 4,096,000 | 5.76 | 23.7 | 4.1× | cpu |
| 64,000 × 64 · 64 × 3 | 12,288,000 | 17.6 | 25.8 | 1.5× | gpu, **slower** |
| 64,000 × 64 · 64 × 8 | 32,768,000 | 46.7 | 39.7 | 0.9× | gpu |
| 256,000 × 16 · 16 × 1 | 4,096,000 | 6.25 | 26.2 | 4.2× | cpu |
| 256,000 × 16 · 16 × 3 | 12,288,000 | 18.2 | 31.5 | 1.7× | gpu, **slower** |
| 256,000 × 16 · 16 × 8 | 32,768,000 | 46.9 | 38.5 | 0.8× | gpu |
| 256,000 × 64 · 64 × 1 | 16,384,000 | 23.8 | 86.5 | 3.6× | gpu, **slower** |
| 256,000 × 64 · 64 × 3 | 49,152,000 | 69.4 | 83.6 | 1.2× | gpu, **slower** |
| 256,000 × 64 · 64 × 8 | 131,072,000 | 199 | 89.1 | 0.4× | gpu |

### 10k × 20, the CPU benchmark's data and parameters

| Operation | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 1 thread | Iterations, CPU / GPU | Run to the end: iterations, CPU / GPU | Run to the end: largest probability difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LinearRegression predict | 0.20 |  | 5.22 | 26.1× |  |  |  |
| LogisticRegression predictProba, 3 classes | 1.01 |  | 5.55 | 5.5× |  |  |  |
| LogisticRegression fit, 2 classes | 78.4 | 88.7 | 281 | 3.6× | 100 / 100 | 144 / 157 | 8.0e-4 |
| LogisticRegression fit, 3 classes | 158 | 185 | 291 | 1.8× | 100 / 100 | 511 / 363 | 3.4e-2 |

The timed fits stop at the default 100 iterations, before they converge on this data; the last two columns fit again with `maxIter: 20000`, and compare the answers once both have finished.

### 200k × 50, the CPU benchmark's data and parameters

| Operation | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 1 thread | Iterations, CPU / GPU | Run to the end: iterations, CPU / GPU | Run to the end: largest probability difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LinearRegression predict | 7.26 |  | 76.4 | 10.5× |  |  |  |
| LogisticRegression predictProba, 3 classes | 29.8 |  | 61.3 | 2.1× |  |  |  |
| LogisticRegression fit, 2 classes | 2,409 | 974 | 492 | 0.2× | 100 / 100 | 287 / 335 | 1.4e-3 |
| LogisticRegression fit, 3 classes | 6,076 | 2,301 | 776 | 0.1× | 100 / 100 | 432 / 629 | 8.8e-3 |

The timed fits stop at the default 100 iterations, before they converge on this data; the last two columns fit again with `maxIter: 20000`, and compare the answers once both have finished.

### LogisticRegression fits by rows, 20 columns, up to 30 iterations

| Rows | Classes | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 4 threads | Iterations, CPU / GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 2 | 15.8 | 18.0 | 368 | 20.4× | 18 / 18 |
| 10,000 | 3 | 42.7 | 47.8 | 395 | 8.3× | 26 / 23 |
| 50,000 | 2 | 70.9 | 36.3 | 299 | 8.3× | 18 / 18 |
| 50,000 | 3 | 205 | 90.9 | 416 | 4.6× | 26 / 22 |
| 200,000 | 2 | 302 | 134 | 459 | 3.4× | 18 / 20 |
| 200,000 | 3 | 912 | 360 | 364 | 1.0× | 29 / 23 |
| 1,000,000 | 2 | 1,573 | 556 | 813 | 1.5× | 18 / 19 |
| 1,000,000 | 3 | 4,376 | 1,452 | 1,114 | 0.8× | 28 / 25 |

Run 2026-11-15: smaltml 0.1.0 in Chromium 157.0.8103.38. GPU: WebGPU adapter nvidia ampere. CPU: 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz, 16 cores. Not the machine of the tables above.

<!-- /bench:gpu -->

## Notes

- One machine and one run: timings move by about 10% between runs on a shared virtual machine. Compare the ratios
  more than the milliseconds.
- The scikit-learn tables are CPU timings, on a machine with no GPU. The WebGPU tables come from another machine,
  named under them, and compare smaltml's GPU path only with its own CPU path. smaltml's GPU path covers predictions
  and `LogisticRegression` fits ([The CPU, WebGPU and workers](../guide/hardware)).
