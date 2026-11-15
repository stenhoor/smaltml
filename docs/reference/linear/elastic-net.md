---
description: 'ElasticNet: linear regression with a mix of L1 and L2 penalties. scikit-learn ElasticNet.'
---

# ElasticNet

<!-- docs:skip -->
```ts
class ElasticNet implements Predictor {
  constructor(params?: { alpha?: number; l1Ratio?: number; fitIntercept?: boolean; maxIter?: number; tol?: number });
  fit(X: MatrixInput, y: ArrayLike<number>, options?: { threads?: number | 'auto'; sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: ArrayLike<number>, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array>;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nIter: number;
  readonly converged: boolean | null;
}
```

Linear regression with a mix of the lasso's and ridge's penalties: minimizes
`‖y − X·w − b‖² / (2n) + alpha · l1Ratio · ‖w‖₁ + alpha · (1 − l1Ratio) / 2 · ‖w‖²`. scikit-learn's
[`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), for one target.

## Parameters

- `alpha` (default `1`): the penalty's strength.
- `l1Ratio` (default `0.5`): the L1 share, from 0 (all L2) to 1 (the lasso).
- `fitIntercept`, `maxIter`, `tol`: as for [`Lasso`](./lasso).

`fit` and `fitSync` take `sampleWeight`, as [`Lasso`](./lasso) does.

## Returns

As [`Lasso`](./lasso).

## Errors

- `ML_PARAM`: `l1Ratio` outside 0 to 1, and as for `Lasso`.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Fitted as `Lasso` is, by coordinate descent with the same stopping rule; the duality gap is the lasso's on X with
`√(n · alpha · (1 − l1Ratio)) · I` rows added. With `l1Ratio` 0 that gap can't close (its dual point scales to 0), so
the gap is the elastic net's own dual at the residual, which needs no scaling when the L2 term is above 0: such a fit
stops when it converges, as scikit-learn's does, instead of running to `maxIter`. With correlated columns, it tends to keep them together where the
lasso picks one.

## Examples

```ts
import { ElasticNet } from '@smaltworks/smaltml';

const X = [[1, 1.1], [2, 1.9], [3, 3.2], [4, 3.9], [5, 5.1]];
const y = [2, 4, 6, 8, 10];
new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 }).fitSync(X, y).coef
```

| value |
| --- |
| 1.09825 |
| 0.849396 |

## Working demo

```ts
({ lasso: new ElasticNet({ alpha: 0.5, l1Ratio: 1 }).fitSync(X, y).coef, mixed: new ElasticNet({ alpha: 0.5, l1Ratio: 0.2 }).fitSync(X, y).coef })
```

| lasso | mixed |
| --- | --- |
| 1.75 | 0.900732 |
| 0 | 0.869175 |

## See also

[`Lasso`](./lasso), [`Ridge`](./ridge), [Training models](../../guide/training#penalized-linear-regression)
