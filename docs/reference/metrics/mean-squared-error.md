---
description: 'meanSquaredError: the mean of the squared errors. scikit-learn mean_squared_error.'
---

# meanSquaredError

<!-- docs:skip -->
```ts
function meanSquaredError(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number;
```

The mean of the squared errors, `Σ (yᵢ − ŷᵢ)² / n`, in the target's units squared. scikit-learn's
[`mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html).

## Parameters

- `yTrue`: the true values. `yPred`: the predictions, one per value.

## Returns

A number, 0 or more; 0 is a perfect prediction.

## Errors

- `ML_SHAPE`: empty, or lengths that differ.

## Examples

```ts
import { meanSquaredError } from '@smaltworks/smaltml';

meanSquaredError([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
```

| value |
| --- |
| 0.375 |

## Working demo

```ts
const actual = [120, 135, 150, 160];
const forecast = [118, 140, 149, 166];
({ mse: meanSquaredError(actual, forecast), rmse: Math.sqrt(meanSquaredError(actual, forecast)) })
```

| mse | rmse |
| --- | --- |
| 16.5 | 4.06202 |

## See also

[`r2Score`](./r2-score), [Evaluating models](../../guide/evaluating)
