---
description: 'meanAbsoluteError: the mean of the absolute errors. scikit-learn mean_absolute_error.'
---

# meanAbsoluteError

<!-- docs:skip -->
```ts
function meanAbsoluteError(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number;
```

The mean of the absolute errors, `Σ |yᵢ − ŷᵢ| / n`, in the target's units. scikit-learn's [`mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

## Parameters

- `yTrue`: the true values. `yPred`: the predictions, one per value.

## Returns

A number, 0 or more; 0 is a perfect prediction.

## Errors

- `ML_SHAPE`: empty, or lengths that differ.

## Examples

```ts
import { meanAbsoluteError } from '@smaltworks/smaltml';

meanAbsoluteError([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
```

| value |
| --- |
| 0.5 |

## Working demo

Unlike the squared error, one large miss doesn't dominate:

```ts
({ mae: meanAbsoluteError([10, 10, 10, 10], [11, 9, 10, 30]) })
```

| mae |
| --- |
| 5.5 |

## See also

[`meanSquaredError`](./mean-squared-error), [`r2Score`](./r2-score)
