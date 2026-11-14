---
description: 'accuracyScore: the share of labels predicted right. scikit-learn accuracy_score.'
---

# accuracyScore

<!-- docs:skip -->
```ts
function accuracyScore(yTrue: Labels, yPred: Labels): number;
```

The share of rows whose predicted label equals the true one. scikit-learn's
[`accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).

## Parameters

- `yTrue`: the true labels. `yPred`: the predicted labels, one per row. Labels are numbers or strings.

## Returns

A number from 0 to 1.

## Errors

- `ML_SHAPE`: empty, or lengths that differ. `ML_LABELS`: a label that is neither a number nor a string.

## Examples

```ts
import { accuracyScore } from '@smaltworks/smaltml';

accuracyScore(['cat', 'dog', 'dog', 'cat'], ['cat', 'dog', 'cat', 'cat'])
```

| value |
| --- |
| 0.75 |

## Working demo

```ts
const truth = [0, 1, 1, 0, 1, 0, 1, 1];
const guess = [0, 1, 0, 0, 1, 1, 1, 1];
`${accuracyScore(truth, guess) * 100}% right`
```

| value |
| --- |
| 75% right |

## See also

[`logLoss`](./log-loss), [Evaluating models](../../guide/evaluating)
