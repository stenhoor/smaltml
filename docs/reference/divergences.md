---
description: 'Where smaltml differs from scikit-learn, and what scikit-learn has that smaltml does not yet.'
---

# Where smaltml differs

smaltml follows scikit-learn's models, parameters, defaults and results (ADR 0024). Where it differs, or doesn't
have something yet, it fails with an error rather than give a different answer. This page lists each case.

## Differences

| Where | scikit-learn | smaltml |
| --- | --- | --- |
| `LinearRegression` on columns that don't determine one answer | The minimum-norm solution | `ML_SINGULAR` |
| Seeded shuffles (`trainTestSplit`) | NumPy's random numbers | smaltml's own: a seed repeats itself, but not scikit-learn's split |
| `logLoss` with rows that don't sum to 1 | Warns | Uses them as given |
| Names | snake_case, fitted attributes end in `_` | camelCase, no underscore; `random_state` is `seed` |
| Labels | Any comparable values | Numbers or strings, not mixed |
| Encoders: `None` and `NaN` in a column of numbers | Two different values | Both missing, one category |
| `OneHotEncoder` names for numbers | Python's (`size_3.0` for floats) | JavaScript's (`size_3`) |
| Encoders and `ColumnTransformer` output | Sparse by default | Dense always |
| `StandardScaler.partialFit`, a column with no value in the first batch | `NaN` for that column from then on | The column's statistics over the values seen, as `fit` on every row gives |
| A fit that stops before it converges | A `ConvergenceWarning` | A console warning, `ML_CONVERGENCE`, and `converged` `false`: an attribute scikit-learn doesn't have |

## Between JavaScript engines

JavaScript fixes the result of `+`, `−`, `×`, `÷` and `Math.sqrt` to the last bit, but not of `Math.exp`, `Math.log`,
`Math.log1p` or `Math.hypot`. smaltml uses the engine's own, so a fit can differ between engines. Measured on
2026-11-15, on 200,000 inputs per function and range, against results computed exactly:

- Node 26.13 got `Math.exp`, `Math.log` and `Math.log1p` one bit off the correctly rounded result in up to 9.6% of
  the inputs; Chromium 157 got all of them right. Both gave the same `Math.hypot`, up to 2 bits off.
- A one-bit difference changes where an iterative fit stops. `LogisticRegression` on the benchmark data (unscaled
  columns), run to the default `tol`: the largest probability difference between Node and Chromium was 1.2e-3 to
  2.0e-2. With `tol: 1e-12`, 4.2e-6 to 1.7e-4.
- The GPU computes in 32-bit floats, with its own functions: its fits agree with the CPU's within a tolerance, not to
  the last bit ([The CPU, WebGPU and workers](../guide/hardware)).

## Not yet supported

Each of these fails with `ML_PARAM` or isn't a parameter yet:

- `LinearRegression`, `Ridge`, `Lasso` and `ElasticNet`: a two-dimensional `y`. Fit several targets with
  [`MultiOutputRegressor`](./multioutput/multi-output-regressor), which gives the same results.
- `Ridge`: solvers other than Cholesky (`solver`; `positive` is solved exactly, where scikit-learn uses `lbfgs`).
- `Lasso` and `ElasticNet`: `selection='random'`, `warm_start`, `precompute`.
- `MultiOutputRegressor`: `n_jobs`; `crossValScore` and `GridSearchCV` on it (scikit-learn averages R² over the
  targets), and a `pipeline` that ends in it.
- `LogisticRegression`: the solvers `liblinear`, `newton-cg` and `sag` (`lbfgs`, `newton-cholesky` and `saga` are
  here), `warm_start`, and weights in a fit on the GPU. `saga` takes the same row order for a seed every time, but not
  scikit-learn's for the same `random_state`: run to convergence, both reach the one optimum.
- `StratifiedKFold`: the same guarantees as scikit-learn's, but its own allocation, so the folds can differ.
- `crossValScore` and `GridSearchCV`: weights or other fit parameters (`params`), `groups`, `n_jobs`, `error_score`,
  `return_train_score`, several scorers at once, and a `refit` that names a scorer.
- `rocAucScore`: `'ovo'` and `max_fpr`. The classification metrics: `sample_weight`, and the warning on a zero
  denominator.
- `StandardScaler`: `sample_weight`.
- `SimpleImputer`: text columns, `missing_values` other than `NaN`, `add_indicator`.
- `OneHotEncoder` and `OrdinalEncoder`: categories given in advance (`categories`), infrequent categories
  (`min_frequency`, `max_categories`), `handle_unknown='infrequent_if_exist'`, sparse output.
- `ColumnTransformer`: columns chosen by a function or a type, `transformer_weights`, sparse output.
- The other scalers, encoders, imputers, models and metrics of ADR 0024's later phases.
