---
description: 'MultiOutputRegressor: several targets at once, a regressor per column of Y. scikit-learn MultiOutputRegressor.'
---

# MultiOutputRegressor

<!-- docs:skip -->
```ts
class MultiOutputRegressor implements Model {
  constructor(estimator: Model);
  fit(X: MatrixInput, Y: MatrixInput, options?: { threads?: number | 'auto'; sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, Y: MatrixInput, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
  predictSync(X: MatrixInput): Matrix;
  readonly estimator: Model;
  readonly estimators: readonly Model[];
  readonly targetNames: readonly string[] | undefined;
  readonly nFeaturesIn: number;
}
```

Predicts several numbers for each row: a fresh copy of a regressor is fitted to each column of `Y`, and `predict`
gives a column for each. scikit-learn's
[`MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html).
From `@smaltworks/smaltml/multioutput`.

## Parameters

- `estimator`: the regressor each target gets a copy of: `LinearRegression`, `Ridge`, `Lasso` or `ElasticNet`, with
  any of their parameters. It isn't fitted itself.

`fit` and `fitSync` take `Y` with a column per target: rows (an array of arrays) or named columns. `sampleWeight` goes
to each target's fit, and `fit`'s `threads` too.

## Returns

- `predict` and `predictSync`: a Matrix with a column per target, in `Y`'s order and with its column names.
  `predict`'s `device` goes to each target's `predict`.
- `estimators`: the fitted regressors, one per target. `targetNames`: `Y`'s column names, when it had them.
- `nFeaturesIn`: the columns of X.

## Errors

- `ML_PARAM`: an `estimator` that isn't a regressor (a classifier or a transformer). And, not yet supported:
  `crossValScore` or `GridSearchCV` on a `MultiOutputRegressor`, or a `pipeline` that ends in one.
- `ML_SHAPE`: a `Y` with one dimension, no columns, or a different number of rows from X.
- `ML_NOT_FITTED`: `predict` before `fit`.
- And each target's own errors, from its fit.

## Notes

LinearRegression, Ridge, Lasso and ElasticNet fit each target on its own, with nothing shared between targets. So a
`MultiOutputRegressor` gives what scikit-learn gives both for `MultiOutputRegressor` and for these models fitted to a
two-dimensional `y`: the compare tests check both, within 1e-9 (1e-8 for Lasso and ElasticNet). Each target reads X on
its own, so a fit costs about as many passes over X as there are targets. The models don't take a two-dimensional
`y` themselves yet ([Where smaltml differs](../divergences#not-yet-supported)).

`cloneModel` and saved models reach the estimator's parameters as `estimator__name`, as in scikit-learn:
`cloneModel(m, { estimator__alpha: 2 })`.

## Examples

```ts
import { MultiOutputRegressor, Ridge } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 1], [4, 2], [5, 3]];
const Y = { price: [3, 5, 7, 9, 11], weight: [1, 2.5, 3, 4.5, 6] };
const both = new MultiOutputRegressor(new Ridge({ alpha: 0.1 })).fitSync(X, Y);
both.predictSync([[6, 3], [7, 4]])
```

| price | weight |
| --- | --- |
| 12.7925 | 6.55497 |
| 14.8675 | 8.0053 |

<!-- docs:error ML_PARAM -->
```ts
import { LogisticRegression } from '@smaltworks/smaltml';

new MultiOutputRegressor(new LogisticRegression())
```

## Working demo

A flat's rent and its heating cost, from its area and its floor:

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const flats = { area: [48, 62, 75, 90, 110, 130], floor: [1, 3, 2, 5, 4, 6] };
const costs = { rent: [1150, 1390, 1580, 1840, 2120, 2450], heating: [62, 70, 85, 96, 120, 135] };
const model = await new MultiOutputRegressor(new LinearRegression()).fit(flats, costs);
await model.predict({ area: [70, 120], floor: [2, 5] })
```

| rent | heating |
| --- | --- |
| 1497.47 | 81.4839 |
| 2287.2 | 127.123 |

## See also

[`LinearRegression`](../linear/linear-regression), [`Ridge`](../linear/ridge), [`Lasso`](../linear/lasso),
[`ElasticNet`](../linear/elastic-net), [`cloneModel`](../selection/cross-val-score#notes)
