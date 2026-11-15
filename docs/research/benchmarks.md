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

## Notes

- One machine and one run: timings move by about 10% between runs on a shared virtual machine. Compare the ratios
  more than the milliseconds.
- WebGPU isn't timed here: these are CPU timings, and the machine has no GPU (the browser tests run WebGPU on
  SwiftShader, a CPU emulation, for correctness only). smaltml's GPU path covers predictions and `LogisticRegression`
  fits ([The CPU, WebGPU and workers](../guide/hardware)).
