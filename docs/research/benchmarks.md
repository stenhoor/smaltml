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
  parameters; there the CPU column is one thread, and fits also run on 4 (`fit(X, y, { threads: 4 })`).
  `LogisticRegression` fits by size, with the default parameters on columns of one scale, against the CPU on every
  core (`threads: 'auto'`).
- **Rerun:** `npm run bench:gpu` (Playwright's Chromium: `npx playwright install chromium` first), or
  `npm run bench:gpu -- --browser /usr/bin/chromium` for another one; then `npm run bench:report`.

<!-- bench:gpu -->

### matMul

`'auto'` runs on the GPU from `GPU_MIN_WORK` (4,194,304) multiply-adds, n · p · k. **slower** marks the cases where that picks the slower device.

| A · B | Multiply-adds | CPU (ms) | GPU (ms) | GPU ÷ CPU | `auto` picks |
| --- | ---: | ---: | ---: | ---: | --- |
| 1,000 × 16 · 16 × 1 | 16,000 | 0.05 | 3.53 | 78.3× | cpu |
| 1,000 × 16 · 16 × 3 | 48,000 | 0.16 | 3.58 | 23.1× | cpu |
| 1,000 × 16 · 16 × 8 | 128,000 | 0.20 | 3.52 | 17.6× | cpu |
| 1,000 × 64 · 64 × 1 | 64,000 | 0.09 | 3.59 | 39.9× | cpu |
| 1,000 × 64 · 64 × 3 | 192,000 | 0.27 | 3.60 | 13.3× | cpu |
| 1,000 × 64 · 64 × 8 | 512,000 | 0.73 | 3.80 | 5.2× | cpu |
| 4,000 × 16 · 16 × 1 | 64,000 | 0.11 | 2.99 | 27.2× | cpu |
| 4,000 × 16 · 16 × 3 | 192,000 | 0.30 | 3.19 | 10.8× | cpu |
| 4,000 × 16 · 16 × 8 | 512,000 | 0.75 | 4.19 | 5.6× | cpu |
| 4,000 × 64 · 64 × 1 | 256,000 | 0.36 | 4.49 | 12.5× | cpu |
| 4,000 × 64 · 64 × 3 | 768,000 | 1.03 | 4.08 | 3.9× | cpu |
| 4,000 × 64 · 64 × 8 | 2,048,000 | 2.75 | 4.92 | 1.8× | cpu |
| 16,000 × 16 · 16 × 1 | 256,000 | 0.37 | 5.23 | 14.1× | cpu |
| 16,000 × 16 · 16 × 3 | 768,000 | 1.16 | 5.28 | 4.6× | cpu |
| 16,000 × 16 · 16 × 8 | 2,048,000 | 3.13 | 5.28 | 1.7× | cpu |
| 16,000 × 64 · 64 × 1 | 1,024,000 | 1.36 | 7.17 | 5.3× | cpu |
| 16,000 × 64 · 64 × 3 | 3,072,000 | 4.59 | 6.97 | 1.5× | cpu |
| 16,000 × 64 · 64 × 8 | 8,192,000 | 11.1 | 8.36 | 0.8× | gpu |
| 64,000 × 16 · 16 × 1 | 1,024,000 | 1.49 | 7.86 | 5.3× | cpu |
| 64,000 × 16 · 16 × 3 | 3,072,000 | 4.55 | 8.90 | 2.0× | cpu |
| 64,000 × 16 · 16 × 8 | 8,192,000 | 12.4 | 11.4 | 0.9× | gpu |
| 64,000 × 64 · 64 × 1 | 4,096,000 | 5.62 | 22.8 | 4.1× | cpu |
| 64,000 × 64 · 64 × 3 | 12,288,000 | 18.0 | 24.5 | 1.4× | gpu, **slower** |
| 64,000 × 64 · 64 × 8 | 32,768,000 | 45.8 | 36.0 | 0.8× | gpu |
| 256,000 × 16 · 16 × 1 | 4,096,000 | 6.31 | 26.7 | 4.2× | cpu |
| 256,000 × 16 · 16 × 3 | 12,288,000 | 19.5 | 30.1 | 1.5× | gpu, **slower** |
| 256,000 × 16 · 16 × 8 | 32,768,000 | 48.5 | 39.1 | 0.8× | gpu |
| 256,000 × 64 · 64 × 1 | 16,384,000 | 23.5 | 118 | 5.0× | gpu, **slower** |
| 256,000 × 64 · 64 × 3 | 49,152,000 | 71.5 | 88.3 | 1.2× | gpu, **slower** |
| 256,000 × 64 · 64 × 8 | 131,072,000 | 211 | 86.1 | 0.4× | gpu |

### 10k × 20, the CPU benchmark's data and parameters

| Operation | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 1 thread | Iterations, CPU / GPU | Run to the end: iterations, CPU / GPU | Run to the end: largest probability difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LinearRegression predict | 0.27 |  | 4.77 | 18.0× |  |  |  |
| LogisticRegression predictProba, 3 classes | 1.09 |  | 5.48 | 5.0× |  |  |  |
| LogisticRegression fit, 2 classes | 77.0 | 96.6 | 287 | 3.7× | 100 / 100 | 144 / 157 | 8.0e-4 |
| LogisticRegression fit, 3 classes | 157 | 179 | 294 | 1.9× | 100 / 100 | 511 / 363 | 3.4e-2 |

The timed fits stop at the default 100 iterations, before they converge on this data; the last two columns fit again with `maxIter: 20000`, and compare the answers once both have finished.

### 200k × 50, the CPU benchmark's data and parameters

| Operation | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 1 thread | Iterations, CPU / GPU | Run to the end: iterations, CPU / GPU | Run to the end: largest probability difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LinearRegression predict | 7.17 |  | 75.6 | 10.5× |  |  |  |
| LogisticRegression predictProba, 3 classes | 29.4 |  | 60.8 | 2.1× |  |  |  |
| LogisticRegression fit, 2 classes | 2,385 | 961 | 507 | 0.2× | 100 / 100 | 287 / 335 | 1.4e-3 |
| LogisticRegression fit, 3 classes | 6,036 | 2,299 | 775 | 0.1× | 100 / 100 | 432 / 629 | 8.8e-3 |

The timed fits stop at the default 100 iterations, before they converge on this data; the last two columns fit again with `maxIter: 20000`, and compare the answers once both have finished.

### LogisticRegression fits by size, default parameters, columns of one scale

The CPU column runs as `'auto'` would, on up to every core (16 here). Work is rows × columns × classes (1 for two classes): the multiply-adds of one evaluation.

| Rows | Columns | Classes | Work | CPU, every core (ms) | GPU (ms) | GPU speed-up | Iterations, CPU / GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 10 | 2 | 100,000 | 8.90 | 42.6 | 0.2× | 13 / 13 |
| 10,000 | 10 | 3 | 300,000 | 17.9 | 46.3 | 0.4× | 14 / 14 |
| 50,000 | 10 | 2 | 500,000 | 33.7 | 53.0 | 0.6× | 10 / 10 |
| 10,000 | 50 | 2 | 500,000 | 17.9 | 49.0 | 0.4× | 13 / 13 |
| 10,000 | 10 | 8 | 800,000 | 58.9 | 65.5 | 0.9× | 19 / 19 |
| 100,000 | 10 | 2 | 1,000,000 | 67.3 | 55.9 | 1.2× | 10 / 10 |
| 10,000 | 100 | 2 | 1,000,000 | 29.1 | 50.5 | 0.6× | 14 / 14 |
| 50,000 | 10 | 3 | 1,500,000 | 92.2 | 73.4 | 1.3× | 16 / 16 |
| 10,000 | 50 | 3 | 1,500,000 | 39.5 | 51.5 | 0.8× | 13 / 13 |
| 200,000 | 10 | 2 | 2,000,000 | 41.4 | 71.0 | 0.6× | 10 / 10 |
| 50,000 | 50 | 2 | 2,500,000 | 44.0 | 83.6 | 0.5× | 14 / 14 |
| 100,000 | 10 | 3 | 3,000,000 | 195 | 72.5 | 2.7× | 17 / 17 |
| 10,000 | 100 | 3 | 3,000,000 | 83.0 | 57.8 | 1.4× | 15 / 15 |
| 50,000 | 10 | 8 | 4,000,000 | 287 | 70.2 | 4.1× | 19 / 19 |
| 10,000 | 50 | 8 | 4,000,000 | 135 | 62.0 | 2.2× | 18 / 18 |
| 500,000 | 10 | 2 | 5,000,000 | 133 | 118 | 1.1× | 12 / 12 |
| 100,000 | 50 | 2 | 5,000,000 | 59.8 | 95.6 | 0.6× | 15 / 15 |
| 50,000 | 100 | 2 | 5,000,000 | 67.7 | 80.6 | 0.8× | 14 / 14 |
| 200,000 | 10 | 3 | 6,000,000 | 78.6 | 82.6 | 1.0× | 12 / 12 |
| 50,000 | 50 | 3 | 7,500,000 | 101 | 76.6 | 1.3× | 15 / 15 |
| 100,000 | 10 | 8 | 8,000,000 | 558 | 82.3 | 6.8× | 16 / 16 |
| 10,000 | 100 | 8 | 8,000,000 | 263 | 88.6 | 3.0× | 20 / 20 |
| 1,000,000 | 10 | 2 | 10,000,000 | 224 | 242 | 0.9× | 11 / 11 |
| 200,000 | 50 | 2 | 10,000,000 | 113 | 145 | 0.8× | 14 / 14 |
| 100,000 | 100 | 2 | 10,000,000 | 101 | 151 | 0.7× | 15 / 15 |
| 500,000 | 10 | 3 | 15,000,000 | 224 | 177 | 1.3× | 15 / 15 |
| 100,000 | 50 | 3 | 15,000,000 | 147 | 117 | 1.3× | 17 / 17 |
| 50,000 | 100 | 3 | 15,000,000 | 184 | 133 | 1.4× | 15 / 15 |
| 200,000 | 10 | 8 | 16,000,000 | 286 | 103 | 2.8× | 20 / 20 |
| 50,000 | 50 | 8 | 20,000,000 | 290 | 120 | 2.4× | 20 / 20 |
| 200,000 | 100 | 2 | 20,000,000 | 156 | 142 | 1.1× | 13 / 13 |
| 1,000,000 | 10 | 3 | 30,000,000 | 356 | 242 | 1.5× | 10 / 10 |
| 200,000 | 50 | 3 | 30,000,000 | 248 | 186 | 1.3× | 15 / 15 |
| 100,000 | 100 | 3 | 30,000,000 | 220 | 237 | 0.9× | 16 / 16 |
| 500,000 | 10 | 8 | 40,000,000 | 551 | 193 | 2.9× | 16 / 16 |
| 100,000 | 50 | 8 | 40,000,000 | 402 | 211 | 1.9× | 20 / 20 |
| 50,000 | 100 | 8 | 40,000,000 | 414 | 238 | 1.7× | 15 / 15 |
| 200,000 | 100 | 3 | 60,000,000 | 478 | 466 | 1.0× | 17 / 17 |
| 1,000,000 | 10 | 8 | 80,000,000 | 1,067 | 255 | 4.2× | 17 / 17 |
| 200,000 | 50 | 8 | 80,000,000 | 520 | 295 | 1.8× | 15 / 15 |
| 100,000 | 100 | 8 | 80,000,000 | 639 | 233 | 2.7× | 20 / 20 |
| 200,000 | 100 | 8 | 160,000,000 | 1,287 | 482 | 2.7× | 21 / 21 |

Run 2026-11-15: smaltml 0.1.0 in Chromium 157.0.8103.38. GPU: WebGPU adapter nvidia ampere. CPU: 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz, 16 cores. Not the machine of the tables above.

<!-- /bench:gpu -->

## Notes

- One machine and one run: timings move by about 10% between runs on a shared virtual machine. Compare the ratios
  more than the milliseconds.
- The scikit-learn tables are CPU timings, on a machine with no GPU. The WebGPU tables come from another machine,
  named under them, and compare smaltml's GPU path only with its own CPU path. smaltml's GPU path covers predictions
  and `LogisticRegression` fits ([The CPU, WebGPU and workers](../guide/hardware)).
