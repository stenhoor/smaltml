# ADR 0037: Several regression targets, with separate accessors

- **Status:** accepted, 2026-11-17. The owner chose the approach (2026-11-16: a wrapper first, then a two-dimensional
  `y` on the models; 2026-11-17: separate accessors for the types).
- **Date:** 2026-11-17
- **Amends:** [ADR 0024](./0024-smaltml-scope-and-architecture): section 2 (scikit-learn's names and shapes), for
  this one case.

## Context

scikit-learn's linear regressors take a two-dimensional `y`, a column per target. Then `coef_` is a matrix (a row
per target), `intercept_` a vector and `predict` a matrix (a column per target). With a one-dimensional `y` they are a
vector, a number and a vector. Python doesn't type them, so one name serves both shapes.

smaltml is TypeScript. `coef` was a `Float64Array`, `intercept` a `number` and `predict` returned a `Float64Array`,
and code across the family reads them so: the docs examples (type-checked by smaltty), pipelines, the registry's
`predict()`, the worker. A name whose type depends on the data a model was fitted on can't be typed honestly in a
way that keeps that code compiling.

For LinearRegression, Ridge, Lasso and ElasticNet the targets don't interact: scikit-learn's two-dimensional fit is
each target fitted on its own (the compare tests check this). So the question is only how the results are read.

## Options considered

1. **Union types:** `coef: Float64Array | Matrix`, `intercept: number | Float64Array`, `predict(): Float64Array |
   Matrix`. scikit-learn's names, honest types, but every caller narrows: a breaking change for all TypeScript code
   that reads them.
2. **An overloaded `fit`** that returns the model typed for several targets. scikit-learn's names and no break, but the
   types are right only on `fit`'s return value: a variable typed before the fit still says `Float64Array` while it
   holds a matrix.
3. **Separate accessors.** The one-target names keep their types; new names hold several targets.
4. **Only a wrapper,** `MultiOutputRegressor` (scikit-learn's own, which fits a model per column).

## Decision

1. **The wrapper first.** `MultiOutputRegressor` (`@smaltworks/smaltml/multioutput`, 2026-11-16) fits a copy of a
   regressor per column of `Y`. It stays: it works with any regressor.
2. **Separate accessors on the models** (option 3). LinearRegression, Ridge, Lasso and ElasticNet take a `y` with a
   column per target: rows, named columns or a Matrix, as X can be.
   - `coefs` (a Matrix, a row per target and a column per feature), `intercepts` (a `Float64Array`), `targetNames`
     and `predictTargets()` / `predictTargetsSync()` (a Matrix, a column per target with `y`'s names) hold several
     targets. They work after a one-target fit too, with one row or column.
   - `coef`, `intercept`, `predict()` and `predictSync()` keep their types and are for one target. On a model fitted
     to a `y` with columns, even one column, they fail with `ML_SHAPE` and name the accessors to read instead. They
     never return a matrix under a vector's type.
   - Ridge, Lasso and ElasticNet's `nIter` is the most iterations any target took, and `converged` whether every
     target converged. scikit-learn gives a list of iterations for Lasso and ElasticNet.
3. **Each target is fitted on its own,** with the solver it would use alone, and XᵀX is computed once for all of them
   (the kernel cache GridSearchCV uses). A target's coefficients equal its one-target fit's to the last bit.
4. **Saved models.** A model fitted on several targets saves `coefs`, `intercepts`, `targetNames` (and `nIters` and
   `convergedEach`) in place of `coef` and `intercept`: format version 3. One-target models save as before; older
   documents still load.
5. **Not yet:** scoring several targets (`crossValScore` and `GridSearchCV` fail with `ML_PARAM`; scikit-learn averages
   R² over the targets), and a pipeline that ends in a `MultiOutputRegressor`.

## Consequences

- No caller changes: one-target code compiles and gives the same results. `Predictor.fit`'s `y` widens to
  `Labels | MatrixInput`, which classifiers still satisfy.
- smaltml's names differ from scikit-learn's for several targets (`coefs`, `intercepts`, `predictTargets`). They are
  listed in [Where smaltml differs](../reference/divergences#differences).
- A later version could add union-typed aliases, if the family wants scikit-learn's names more than the types.

## Numbering

0037 was the next free number when this was written: the highest in the family's repositories this session could
read was 0036 (smaltdb), and a search of the owner's repositories on GitHub found no 0037. That search covers only the
repositories GitHub's code search reaches for this account. That they are all of them is an assumption; renumber if
another ADR took 0037.
