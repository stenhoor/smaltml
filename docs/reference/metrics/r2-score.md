---
description: 'r2Score: the coefficient of determination, the share of the variation predictions explain. scikit-learn r2_score.'
---

# r2Score

<!-- docs:skip -->
```ts
function r2Score(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number;
```

The coefficient of determination: `1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²`. scikit-learn's
[`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

## Parameters

- `yTrue`: the true values. `yPred`: the predictions, one per value.

## Returns

1 for a perfect prediction, 0 for no better than predicting the mean, and negative for worse. When `yTrue` is constant,
1 for a perfect prediction and 0 otherwise.

## Errors

- `ML_SHAPE`: empty, or lengths that differ.

## Examples

```ts
import { r2Score } from '@smaltworks/smaltml';

r2Score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
```

| value |
| --- |
| 0.948608 |

## Working demo


| perfect | mean | worse |
| --- | --- | --- |
| 1 | 0 | -3 |
```ts
const truth = [1, 2, 3, 4, 5];
({ perfect: r2Score(truth, truth), mean: r2Score(truth, [3, 3, 3, 3, 3]), worse: r2Score(truth, [5, 4, 3, 2, 1]) })
```

| perfect | mean | worse |
| --- | --- | --- |
| 1 | 0 | -3 |

## See also

[`meanSquaredError`](./mean-squared-error), [Evaluating models](../../guide/evaluating)
