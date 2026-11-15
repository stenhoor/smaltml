---
description: 'rocAucScore: the area under the ROC curve, from scores or probabilities. scikit-learn roc_auc_score.'
---

# rocAucScore

<!-- docs:skip -->
```ts
function rocAucScore(yTrue: Labels, yScore: ArrayLike<number> | MatrixInput, options?: { multiClass?: 'ovr'; average?: 'macro' | 'weighted'; labels?: Label[] }): number;
```

The area under the ROC curve: the chance that a random row of the positive class scores above a random row of the other, ties counting half. scikit-learn's [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

## Parameters

- `yTrue`: the true labels.
- `yScore`: for two classes, the second (greater) class's score or probability, one per row, or `predictProba`'s two columns; for more, a column per class.
- `multiClass`: `'ovr'`, each class against the rest. Required with more than two classes, as in scikit-learn.
- `average` (default `'macro'`): with more classes, the mean of their areas, or weighted by their rows.
- `labels`: the classes, in the order of yScore's columns. Default: yTrue's labels, sorted.

## Returns

A number from 0 to 1: 1 ranks every positive above every negative, 0.5 is chance.

## Errors

- `ML_LABELS`: one class only, or a class with no rows.
- `ML_PARAM`: more than two classes without `multiClass: 'ovr'`.
- `ML_SHAPE`: scores that don't match the rows or the classes.

## Notes

Computed from the ranks (the Mann–Whitney form), which equals the trapezoid area under the curve. `'ovo'` and `max_fpr` aren't supported.

## Examples

```ts
import { rocAucScore } from '@smaltworks/smaltml';

rocAucScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
```

| value |
| --- |
| 0.75 |

## Working demo

```ts
const y = ['a', 'b', 'c', 'a', 'b', 'c'];
const P = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]];
rocAucScore(y, P, { multiClass: 'ovr' })
```

| value |
| --- |
| 0.9375 |

## See also

[`logLoss`](./log-loss), [`LogisticRegression`](../linear/logistic-regression)
