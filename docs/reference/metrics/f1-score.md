---
description: 'f1Score: the harmonic mean of precision and recall. scikit-learn f1_score.'
---

# f1Score

<!-- docs:skip -->
```ts
function f1Score(yTrue: Labels, yPred: Labels, options?: { average?: 'binary' | 'micro' | 'macro' | 'weighted' | null; posLabel?: Label; labels?: Label[]; zeroDivision?: number }): number | Float64Array;
```

The harmonic mean of precision and recall: `2·TP / (2·TP + FP + FN)`. scikit-learn's [`f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

## Parameters

- `yTrue`: the true labels. `yPred`: the predicted labels, one per row.
- `average` (default `'binary'`): `'binary'` scores the `posLabel` class alone (two classes only); `'micro'` counts
  every row alike; `'macro'` averages the classes alike; `'weighted'` averages them by their rows; `null` gives one
  score per class.
- `posLabel` (default `1`): the positive class for `'binary'`.
- `labels`: the classes to score, and their order for `average: null`. Default: every label in either list, sorted.
- `zeroDivision` (default `0`): the score where its denominator is 0.

## Returns

A number from 0 to 1, or, with `average: null`, a Float64Array with one per class.

## Errors

- `ML_LABELS`: `'binary'` with more than two classes, or a `posLabel` that isn't one of them.
- `ML_PARAM`: an unknown `average`. `ML_SHAPE`: empty, or lengths that differ.

## Notes

scikit-learn warns where a denominator is 0 and gives `zero_division`; smaltml gives `zeroDivision` without a warning.

## Examples

```ts
import { f1Score } from '@smaltworks/smaltml';

f1Score([1, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 1])
```

| value |
| --- |
| 0.75 |

## Working demo

```ts
const truth = ['a', 'b', 'c', 'a', 'b', 'c', 'a'];
const guess = ['a', 'b', 'b', 'a', 'c', 'c', 'b'];
({ macro: f1Score(truth, guess, { average: 'macro' }), perClass: f1Score(truth, guess, { average: null }) })
```

| macro | perClass |
| --- | --- |
| 0.566667 | [0.8, 0.4, 0.5] |

## See also

[`precisionScore`](./precision-score), [`recallScore`](./recall-score), [`f1Score`](./f1-score), [`confusionMatrix`](./confusion-matrix)
