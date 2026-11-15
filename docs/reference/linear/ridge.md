---
description: 'Ridge: linear regression with an L2 penalty, solved in closed form. scikit-learn Ridge.'
---

# Ridge

<!-- docs:skip -->
```ts
class Ridge implements Predictor {
  constructor(params?: { alpha?: number; fitIntercept?: boolean });
  fit(X: MatrixInput, y: ArrayLike<number>): Promise<this>;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array>;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nIter: number;
}
```

Linear regression with an L2 penalty: minimizes `‖y − X·w − b‖² + alpha · ‖w‖²`. scikit-learn's
[`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), for one target.

## Parameters

- `alpha` (default `1`): the penalty's strength. `0` is ordinary least squares.
- `fitIntercept` (default `true`): fit an intercept, which is never penalized.

## Returns

`predict` returns `X·w + b`; `device` chooses where the product runs. After `fit`: `coef`, `intercept`, and `nIter`
(0: the solution is closed-form).

## Errors

- `ML_PARAM`: a negative `alpha`.
- `ML_SINGULAR`: `alpha` 0 with columns that don't determine one answer. Any `alpha` above 0 has one.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Solved as `(XᵀX + alpha·I) w = Xᵀy` on centred data, by Cholesky, in 64-bit floats; XᵀX takes one blocked pass over
X, centred as it is read. Results agree with scikit-learn's
within 1e-9 in the compare tests.

## Examples

```ts
import { Ridge } from '@smaltworks/smaltml';

const X = [[1, 1], [2, 2.1], [3, 2.9], [4, 4.2]];
const y = [2, 4, 6, 8];
({ weak: new Ridge({ alpha: 0.01 }).fitSync(X, y).coef, strong: new Ridge({ alpha: 10 }).fitSync(X, y).coef })
```

| weak | strong |
| --- | --- |
| 1.65289 | 0.490548 |
| 0.330579 | 0.508036 |

## Working demo

```ts
await new Ridge({ alpha: 1 }).fitSync(X, y).predict([[5, 5]])
```

| value |
| --- |
| 9.41938 |

## See also

[`Lasso`](./lasso), [`ElasticNet`](./elastic-net), [`LinearRegression`](./linear-regression), [Training models](../../guide/training#penalized-linear-regression)
