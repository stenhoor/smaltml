---
description: 'LinearRegression: ordinary least squares for one target. scikit-learn LinearRegression.'
---

# LinearRegression

<!-- docs:skip -->
```ts
class LinearRegression implements Predictor {
  constructor(params?: { fitIntercept?: boolean });
  fit(X: MatrixInput, y: ArrayLike<number>): Promise<this>;
  predict(X: MatrixInput): Promise<Float64Array>;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nFeaturesIn: number;
  readonly featureNamesIn: readonly string[] | undefined;
}
```

Ordinary least squares: the coefficients `w` and intercept `b` that minimize the sum of `(y − X·w − b)²`. scikit-learn's
[`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
for one target.

## Parameters

- `fitIntercept` (default `true`): fit an intercept. With `false`, the fit passes through the origin.

## Returns

`predict` returns `X·w + b`, one value per row. After `fit`, `coef` holds one coefficient per column and `intercept`
the intercept (0 without `fitIntercept`).

## Errors

- `ML_SINGULAR`: the columns don't determine one answer: a column that is a combination of others, or fewer rows than
  columns.
- `ML_VALUE`: a missing or infinite value in `X` or `y`.
- `ML_SHAPE`: `y`'s length differs from `X`'s rows, or `predict` gets a different number of columns.
- `ML_NOT_FITTED`: `predict` before `fit`.

## Notes

Solved by Householder QR on centred data, in 64-bit floats. Results agree with scikit-learn within 1e-8 (relative) in
the compare tests. Where scikit-learn returns one of many equally good solutions (the minimum-norm one), smaltml fails
with `ML_SINGULAR` instead ([Where smaltml differs](../divergences)).

## Examples

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const m = await new LinearRegression().fit([[1], [2], [3], [4]], [3, 5, 7, 9]);
({ coef: m.coef, intercept: m.intercept })
```

| coef | intercept |
| --- | --- |
| [2] | 1 |

<!-- docs:error ML_SINGULAR -->
```ts
await new LinearRegression().fit([[1, 2], [2, 4], [3, 6]], [1, 2, 3])
```

## Working demo

```ts
const sizes = { area: [48, 62, 75, 90, 110, 130], rooms: [2, 3, 3, 4, 4, 5] };
const rent = [1150, 1390, 1580, 1840, 2120, 2450];
const flats = await new LinearRegression().fit(sizes, rent);
await flats.predict({ area: [70, 120], rooms: [3, 4] })
```

| value |
| --- |
| 1507.32 |
| 2263.24 |

## See also

[Training models](../../guide/training), [`r2Score`](../metrics/r2-score), [`LogisticRegression`](./logistic-regression)
