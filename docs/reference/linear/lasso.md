---
description: 'Lasso: linear regression with an L1 penalty, which sets some coefficients to exactly 0. scikit-learn Lasso.'
---

# Lasso

<!-- docs:skip -->
```ts
class Lasso implements Predictor {
  constructor(params?: { alpha?: number; fitIntercept?: boolean; maxIter?: number; tol?: number });
  fit(X: MatrixInput, y: ArrayLike<number>, options?: { threads?: number | 'auto' }): Promise<this>;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array>;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nIter: number;
  readonly converged: boolean | null;
}
```

Linear regression with an L1 penalty: minimizes `‖y − X·w − b‖² / (2n) + alpha · ‖w‖₁`. The penalty sets the least
useful coefficients to exactly 0. scikit-learn's
[`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), for one target.

## Parameters

- `alpha` (default `1`): the penalty's strength. Larger values zero more coefficients.
- `fitIntercept` (default `true`): fit an intercept, which is never penalized.
- `maxIter` (default `1000`): the most passes over the coefficients.
- `tol` (default `1e-4`): the stopping tolerance (Notes).

## Returns

`predict` returns `X·w + b`. After `fit`: `coef` (some exactly 0), `intercept`, `nIter`, the passes it took, and
`converged`: whether the duality gap reached `tol` within `maxIter`. Where it didn't, `fit` warns on the console
(`ML_CONVERGENCE`, [Errors](../errors#warnings)), as scikit-learn raises a `ConvergenceWarning`. `converged` is `null`
for a model saved by smaltml before it recorded this.

## Errors

- `ML_PARAM`: a negative `alpha` or `tol`, or `maxIter` below 1.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Fitted by cyclic coordinate descent, on the Gram matrix XᵀX when there are at least as many rows as columns (each
pass then costs p², not a read of X), with scikit-learn's documented stopping rule: once the largest change in a pass
is at most `tol` times the largest coefficient, it stops when the duality gap is at most `tol · ‖y‖² / n`. Run to
convergence, results agree with scikit-learn's within 1e-7 and zero the same coefficients; with the defaults, within
1e-3. `Lasso` is `ElasticNet` with `l1Ratio` 1.

## Examples

```ts
import { Lasso } from '@smaltworks/smaltml';

const X = [[1, 0.2, 5], [2, -0.1, 3], [3, 0.4, 4], [4, -0.3, 1], [5, 0.1, 2], [6, -0.2, 6]];
const y = [3, 5, 7, 9, 11, 13];
new Lasso({ alpha: 0.1 }).fitSync(X, y).coef
```

| value |
| --- |
| 1.96571 |
| 0 |
| 0 |

## Working demo

A stronger penalty zeroes everything, and predicts the mean:

```ts
const flat = new Lasso({ alpha: 100 }).fitSync(X, y);
({ coef: flat.coef, intercept: flat.intercept })
```

| coef | intercept |
| --- | --- |
| [0, 0, 0] | 8 |

## See also

[`ElasticNet`](./elastic-net), [`Ridge`](./ridge), [Training models](../../guide/training#penalized-linear-regression)
