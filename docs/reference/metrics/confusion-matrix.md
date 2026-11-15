---
description: 'confusionMatrix: counts of true labels against predicted labels. scikit-learn confusion_matrix.'
---

# confusionMatrix

<!-- docs:skip -->
```ts
function confusionMatrix(yTrue: Labels, yPred: Labels, options?: { labels?: Label[] }): Matrix;
```

Counts of true labels (rows) against predicted labels (columns): entry (i, j) is how many rows of class i were predicted as class j. scikit-learn's [`confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

## Parameters

- `yTrue`, `yPred`: the true and predicted labels.
- `labels`: the classes, in the order of the rows and columns. Default: every label in either list, sorted.

## Returns

A k × k Matrix of counts, its columns named after the labels.

## Errors

- `ML_SHAPE`: empty, or lengths that differ.

## Examples

```ts
import { confusionMatrix } from '@smaltworks/smaltml';

confusionMatrix([0, 1, 2, 2, 1], [0, 2, 2, 2, 1])
```

| 0 | 1 | 2 |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 1 |
| 0 | 0 | 2 |

## Working demo

```ts
confusionMatrix(['cat', 'dog', 'dog'], ['dog', 'dog', 'dog'], { labels: ['dog', 'cat'] })
```

| dog | cat |
| --- | --- |
| 2 | 0 |
| 1 | 0 |

## See also

[`precisionScore`](./precision-score), [`accuracyScore`](./accuracy-score)
