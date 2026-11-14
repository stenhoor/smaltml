---
description: 'logLoss: the mean negative log-likelihood of the true labels under predicted probabilities. scikit-learn log_loss.'
---

# logLoss

<!-- docs:skip -->
```ts
function logLoss(yTrue: Labels, yProba: MatrixInput | ArrayLike<number>, options?: { labels?: readonly Label[] }): number;
```

The mean negative log-likelihood of the true labels: `−Σ log pᵢ(yᵢ) / n`. scikit-learn's
[`log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

## Parameters

- `yTrue`: the true labels.
- `yProba`: the probabilities, a column per class in the order of the labels (as `predictProba` gives them); or, for
  two classes, the second class's probability alone.
- `labels`: the classes, in the order of `yProba`'s columns. Default: `yTrue`'s labels, sorted. Pass the model's
  `classes` when `yTrue` may not hold every class.

## Returns

A number, 0 or more; lower is better. A confident wrong answer costs the most.

## Errors

- `ML_SHAPE`: rows or columns that don't match.
- `ML_LABELS`: fewer than two classes, or a true label not among the labels.

## Notes

Probabilities are clipped to [ε, 1 − ε], ε the float64 epsilon, so a zero probability costs about 36, not infinity.
Rows aren't renormalized.

## Examples

```ts
import { logLoss } from '@smaltworks/smaltml';

logLoss(['spam', 'ham', 'ham', 'spam'], [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]])
```

| value |
| --- |
| 0.216162 |

## Working demo


| sure | unsure | wrong |
| --- | --- | --- |
| 0.0100503 | 0.510826 | 4.60517 |
```ts
const outcome = [1, 0, 1, 1];
({ sure: logLoss(outcome, [0.99, 0.01, 0.99, 0.99]), unsure: logLoss(outcome, [0.6, 0.4, 0.6, 0.6]), wrong: logLoss(outcome, [0.01, 0.99, 0.01, 0.01]) })
```

| sure | unsure | wrong |
| --- | --- | --- |
| 0.0100503 | 0.510826 | 4.60517 |

## See also

[`accuracyScore`](./accuracy-score), [`LogisticRegression`](../linear/logistic-regression)
