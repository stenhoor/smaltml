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
- **smaltml:** the built package (`dist/`), in Node, on one thread, calling the synchronous methods.
- **scikit-learn:** on one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and on every core, with its default
  BLAS.
- **Rerun:** `npm run bench` (smaltml), `npm run bench:python` (scikit-learn; needs `compare/requirements.txt`), then
  `npm run bench:report`, which writes the tables below from `bench/results/`.

`smaltml ÷ …` is how many times longer smaltml takes: below 1 is faster.

## Results

<!-- bench:results -->

### 10k × 20

| Operation | smaltml | scikit-learn, 1 thread | scikit-learn, 4 threads | smaltml ÷ 1 thread | smaltml ÷ 4 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| StandardScaler fit + transform | 1.48 | 1.34 | 1.33 | 1.1× | 1.1× |
| LinearRegression fit | 8.13 | 3.05 | 3.75 | 2.7× | 2.2× |
| LinearRegression predict | 0.22 | 0.15 | 0.15 | 1.5× | 1.5× |
| Ridge fit | 3.31 | 1.24 | 1.25 | 2.7× | 2.7× |
| Lasso fit (alpha 0.1) | 4.51 | 1.22 | 1.49 | 3.7× | 3.0× |
| LogisticRegression fit, 2 classes | 85.7 | 41.6 | 41.6 | 2.1× | 2.1× |
| LogisticRegression fit, 3 classes | 252 | 102 | 104 | 2.5× | 2.4× |
| LogisticRegression predictProba, 3 classes | 1.21 | 0.88 | 0.94 | 1.4× | 1.3× |

Iterations: Lasso 5 (scikit-learn 5), logistic regression 100 and 100 (scikit-learn 100 and 100).

### 200k × 50

| Operation | smaltml | scikit-learn, 1 thread | scikit-learn, 4 threads | smaltml ÷ 1 thread | smaltml ÷ 4 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| StandardScaler fit + transform | 119 | 102 | 145 | 1.2× | 0.8× |
| LinearRegression fit | 1,619 | 415 | 339 | 3.9× | 4.8× |
| LinearRegression predict | 31.8 | 10.0 | 8.63 | 3.2× | 3.7× |
| Ridge fit | 627 | 64.2 | 76.9 | 9.8× | 8.2× |
| Lasso fit (alpha 0.1) | 365 | 97.5 | 65.0 | 3.7× | 5.6× |
| LogisticRegression fit, 2 classes | 7,181 | 1,632 | 2,508 | 4.4× | 2.9× |
| LogisticRegression fit, 3 classes | 24,856 | 3,877 | 8,028 | 6.4× | 3.1× |
| LogisticRegression predictProba, 3 classes | 106 | 30.8 | 22.5 | 3.4× | 4.7× |

Iterations: Lasso 4 (scikit-learn 4), logistic regression 100 and 100 (scikit-learn 100 and 100).

Run 2026-11-15: smaltml 0.1.0 on Node 22.23.4, 1 thread; scikit-learn 1.9.1, NumPy 2.5.4 on Python 3.13.18 (openblas 0.3.31.dev, openblas 0.3.34.106.0, openmp). CPU: Intel(R) Xeon(R) Processor @ 2.30GHz, 4 cores.

<!-- /bench:results -->

## Notes

- One machine and one run: timings move by about 10% between runs on a shared virtual machine. Compare the ratios
  more than the milliseconds.
- WebGPU isn't timed here: these are CPU timings, and the machine has no GPU. smaltml's GPU path is for predictions
  today ([The CPU, WebGPU and workers](../guide/hardware)).
