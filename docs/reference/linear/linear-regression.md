---
description: 'LinearRegression: ordinary least squares for one target. scikit-learn LinearRegression.'
---

# LinearRegression

<!-- docs:skip -->
```ts
class LinearRegression implements Predictor {
  constructor(params?: { fitIntercept?: boolean; positive?: boolean });
  fit(X: MatrixInput, y: ArrayLike<number>, options?: { threads?: number | 'auto'; sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: ArrayLike<number>, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array>;
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
- `positive` (default `false`): hold every coefficient at 0 or more (non-negative least squares), for quantities that
  can only add: amounts, prices, counts. The intercept isn't held.

`fit` and `fitSync` take `sampleWeight`: one weight per row, 0 or more, as scikit-learn's `sample_weight`. A row's
squared error counts that many times, so whole-number weights fit as the rows repeated would.

## Returns

`predict` returns `X·w + b`, one value per row; `device` chooses where the product runs
([The CPU, WebGPU and workers](../../guide/hardware)). After `fit`, `coef` holds one coefficient per column and `intercept`
the intercept (0 without `fitIntercept`).

## Errors

- `ML_SINGULAR`: the columns don't determine one answer: a column that is a combination of others, or fewer rows than
  columns. With `positive`, only where the columns it keeps above 0 don't.
- `ML_PARAM`: `positive` that isn't `true` or `false`.
- `ML_VALUE`: a missing or infinite value in `X` or `y`.
- `ML_SHAPE`: `y`'s length differs from `X`'s rows, or `predict` gets a different number of columns.
- `ML_NOT_FITTED`: `predict` before `fit`.

## Notes

Solved in 64-bit floats by the corrected semi-normal equations: a Cholesky factor of XᵀX (one blocked pass over X,
centred as it is read), then up to three steps of iterative refinement on the residual. Where the columns are nearly
dependent, the refinement doesn't settle and the fit uses Householder QR instead, which is slower and as accurate as
the data allows. Results agree with scikit-learn within 1e-8 (relative) in the compare tests. Where scikit-learn returns one of many equally good solutions (the minimum-norm one), smaltml fails
with `ML_SINGULAR` instead ([Where smaltml differs](../divergences)).

With `positive`, the fit is Lawson and Hanson's active-set method on the same normal equations: it frees one
coefficient at a time, the one whose gradient most wants it above 0, solves on the free ones by Cholesky, and holds at
exactly 0 any that would go below. It stops when no held coefficient's gradient points up. scikit-learn's
`positive=True` (SciPy's `nnls`) is the same method: the compare tests agree within 1e-9, and give two identical
columns' weight to the first, as SciPy does.

## Examples

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const m = await new LinearRegression().fit([[1], [2], [3], [4]], [3, 5, 7, 9]);
({ coef: m.coef, intercept: m.intercept })
```

| coef | intercept |
| --- | --- |
| [2] | 1 |

The last point is off the line. Weighted 0, it doesn't count at all:

```ts
const off = [[1], [2], [3], [4], [5]];
const ys = [3, 5, 7, 9, 20];
({ plain: new LinearRegression().fitSync(off, ys).coef, weighted: new LinearRegression().fitSync(off, ys, { sampleWeight: [1, 1, 1, 1, 0] }).coef })
```

| plain | weighted |
| --- | --- |
| 3.8 | 2 |

With `positive`, a coefficient that would come out negative is held at 0 instead, and the others refit without it:

```ts
const mixes = [[1, 4], [2, 3], [3, 3], [4, 1], [5, 1]];
const yields = [1.5, 3.7, 5.9, 8.5, 10.7];
({ free: new LinearRegression().fitSync(mixes, yields).coef, held: new LinearRegression({ positive: true }).fitSync(mixes, yields).coef })
```

| free | held |
| --- | --- |
| 2.16 | 2.32 |
| -0.2 | 0 |

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
