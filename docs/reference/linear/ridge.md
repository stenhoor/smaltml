---
description: 'Ridge: linear regression with an L2 penalty, solved in closed form. scikit-learn Ridge.'
---

# Ridge

<!-- docs:skip -->
```ts
class Ridge implements Predictor {
  constructor(params?: { alpha?: number; fitIntercept?: boolean; positive?: boolean });
  fit(X: MatrixInput, y: ArrayLike<number> | MatrixInput, options?: { threads?: number | 'auto'; sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: ArrayLike<number> | MatrixInput, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array>;
  predictTargets(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
  predictTargetsSync(X: MatrixInput): Matrix;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly coefs: Matrix;
  readonly intercepts: Float64Array;
  readonly targetNames: readonly string[] | undefined;
  readonly nIter: number;
  readonly converged: boolean | null;
}
```

Linear regression with an L2 penalty: minimizes `‖y − X·w − b‖² + alpha · ‖w‖²`. scikit-learn's
[`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), for one target.

## Parameters

- `alpha` (default `1`): the penalty's strength. `0` is ordinary least squares.
- `fitIntercept` (default `true`): fit an intercept, which is never penalized.
- `positive` (default `false`): hold every coefficient at 0 or more.

`fit` and `fitSync` take `sampleWeight`: one weight per row, 0 or more, as scikit-learn's `sample_weight`. A row's
squared error counts that many times, so whole-number weights fit as the rows repeated would. The penalty isn't weighted: it is `alpha·‖w‖²` whatever the weights add up to, as in
scikit-learn.

`y` is one value per row, or a column per target (rows, named columns or a Matrix, as scikit-learn's two-dimensional
`y`): each target is fitted on its own, with XᵀX computed once for all of them ([ADR 0037](../../adr/0037-several-targets)).

## Returns

`predict` returns `X·w + b`; `device` chooses where the product runs. After `fit`: `coef`, `intercept`, `nIter`
(0: the solution is closed-form), and `converged`, always `true` for the same reason (`null` for a model saved by
smaltml before it recorded this).

With several targets, read `coefs` (a Matrix: a row per target, a column per feature), `intercepts` (one per target),
`targetNames` (`y`'s column names, when it had them) and `predictTargets` (a Matrix with a column per target and those
names). They work after a one-target fit too, with one row or column. `coef`, `intercept` and `predict` are for one
target.

## Errors

- `ML_SHAPE` for `coef`, `intercept` and `predict` on a model fitted to a `y` with columns: read `coefs`,
  `intercepts` and `predictTargets`. Also a `y` whose rows differ from X's.
- `ML_PARAM`: a negative `alpha`, or `positive` that isn't `true` or `false`.
- `ML_SINGULAR`: `alpha` 0 with columns that don't determine one answer. Any `alpha` above 0 has one.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Solved as `(XᵀX + alpha·I) w = Xᵀy` on centred data, by Cholesky, in 64-bit floats; XᵀX takes one blocked pass over
X, centred as it is read. Results agree with scikit-learn's
within 1e-9 in the compare tests. With `positive`, the same equations are solved with w ≥ 0 by Lawson and Hanson's
active set, exactly ([`LinearRegression`](./linear-regression#notes)). scikit-learn solves that case by L-BFGS-B
(`solver='lbfgs'`, which `positive=True` chooses), to its `tol`: run with `tol` 1e-12 there, the two agree within 1e-6.

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
