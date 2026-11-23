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
| `OrdinalEncoder`: a missing value in a text column | `None` is a category with a code; `NaN` stays `NaN` | Missing (`null`) stays `NaN`, as scikit-learn's `NaN` does, and is no category for `minFrequency` and `maxCategories` |
| Encoders: `minFrequency` of 1 | An `int` 1 is a count (nothing is infrequent); a `float` 1.0 fails | JavaScript has one kind of number: 1 is the count |
| Encoders and `ColumnTransformer` output | Sparse by default | Dense always |
| `StandardScaler.partialFit`, a column with no value in the first batch | `NaN` for that column from then on | The column's statistics over the values seen, as `fit` on every row gives |
| Several targets (a `y` with columns) on `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | `coef_` a matrix, `intercept_` a vector, `predict` a matrix; `n_iter_` a list | `coefs`, `intercepts` and `predictTargets`; `coef`, `intercept` and `predict` stay for one target and fail with `ML_SHAPE` ([ADR 0037](../adr/0037-several-targets)); `nIter` the most any target took |
| `groups` with a splitter that doesn't use them | Warns, and splits without them | `ML_PARAM`: the split asked for isn't made |
| `crossValidate`'s result | One dictionary: `test_r2`, `train_r2`, `fit_time`, `score_time` | `test.r2`, `train.r2`, `fitTime`, `scoreTime` |
| `GridSearchCV` with several scorers | `mean_test_<name>` and the rest in `cv_results_` | `cvResultsByScorer.scores[name]`; `cvResults` is for one scorer and fails with several |
| Decision trees where two splits tie | `random_state`'s column order picks one | `seed`'s order picks one: the same seed repeats smaltml's tree, not scikit-learn's ([ADR 0039](../adr/0039-trees-and-forests)) |
| Random forests with bootstrap, `maxFeatures` below every column, or both | `random_state`'s samples and columns | smaltml's own, from `seed`: the same seed repeats smaltml's forest, not scikit-learn's. The compare tests hold the mean score over several seeds to scikit-learn's ([ADR 0039](../adr/0039-trees-and-forests)) |
| Extra trees, and trees with `splitter: 'random'` | `random_state`'s thresholds, columns and samples | smaltml's own, from `seed`, as for random forests: the compare tests hold the mean score over several seeds to scikit-learn's ([ADR 0039](../adr/0039-trees-and-forests)) |
| `BoostedTreesRegressor`'s numbers, against XGBoost's `XGBRegressor` | XGBoost keeps some values in 32-bit floats | 64-bit floats: the same trees, and leaves and predictions within 1e-5 on the compare data ([ADR 0044](../adr/0044-phase-3-boosted-trees)) |
| `maxSamples` of 1 | A `float` 1.0 is every row's worth; an `int` 1 is one row | One kind of number: 1 is one row; `null` is every row's worth |
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

- `crossValScore` and `GridSearchCV` with a `y` that has a column per target (scikit-learn averages R² over them).
- `Ridge`: solvers other than Cholesky (`solver`; `positive` is solved exactly, where scikit-learn uses `lbfgs`).
- `Lasso` and `ElasticNet`: `selection='random'`, `warm_start`, `precompute`.
- `DecisionTreeRegressor` and `DecisionTreeClassifier`: missing values in X (`NaN`, which scikit-learn's trees send
  down a learned side), `ccp_alpha` (cost-complexity pruning) and `cost_complexity_pruning_path`, the criteria
  `absolute_error` and `poisson`, `monotonic_cst`, several outputs, `decision_path`, `export_text` and `plot_tree`.
- `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor` and `ExtraTreesClassifier`: what the trees
  don't support (above); `classWeight: 'balanced_subsample'`, `warm_start`, `n_jobs` (`fit`'s `threads` instead),
  `monotonic_cst`, `apply` and `decision_path`, several outputs.
- `ExtraTreeRegressor` and `ExtraTreeClassifier`, the single extra trees: a `DecisionTreeRegressor` or
  `DecisionTreeClassifier` with `splitter: 'random'` is one (with `maxFeatures: 'sqrt'` for the classifier, as
  scikit-learn's default).
- `BoostedTreesRegressor`, against XGBoost's `XGBRegressor` ([ADR 0044](../adr/0044-phase-3-boosted-trees), each in
  its own step): missing values in X (`NaN`, sent down a learned side); columns with more distinct values than
  `maxBin` (XGBoost's quantile sketch); objectives other than `reg:squarederror`; `subsample` and the `colsample_*`
  below 1; early stopping and evaluation sets; XGBoost's own model files; categorical columns; `grow_policy`
  `lossguide`, `max_leaves`, `max_delta_step`, the `dart` booster, monotone and interaction constraints, and
  `pred_contribs`. The classifier, `BoostedTreesClassifier`, comes next.
- `MultiOutputRegressor`: `n_jobs`; `crossValScore` and `GridSearchCV` on it (scikit-learn averages R² over the
  targets), and a `pipeline` that ends in it.
- `LogisticRegression`: the solvers `liblinear`, `newton-cg` and `sag` (`lbfgs`, `newton-cholesky` and `saga` are
  here), `warm_start`, and weights in a fit on the GPU. `saga` takes the same row order for a seed every time, but not
  scikit-learn's for the same `random_state`: run to convergence, both reach the one optimum.
- `StratifiedKFold`: the same guarantees as scikit-learn's, but its own allocation, so the folds can differ.
- `GroupKFold`: `shuffle` and `random_state`. Other group splitters (`LeaveOneGroupOut`, `GroupShuffleSplit`,
  `StratifiedGroupKFold`).
- `crossValScore`, `crossValidate` and `GridSearchCV`: weights or other fit parameters (`params`), `n_jobs`,
  `error_score`, a callable `refit`; `crossValidate`'s `return_estimator` and `return_indices`.
- `rocAucScore`: `'ovo'` and `max_fpr`. The classification metrics: `sample_weight`, and the warning on a zero
  denominator.
- `StandardScaler`: `sample_weight`.
- `SimpleImputer`: text columns, `missing_values` other than `NaN`, `add_indicator`.
- `OneHotEncoder` and `OrdinalEncoder`: categories given in advance (`categories`), sparse output.
- `ColumnTransformer`: columns chosen by a function or a type, `transformer_weights`, sparse output.
- The other scalers, encoders, imputers, models and metrics of ADR 0024's later phases.
