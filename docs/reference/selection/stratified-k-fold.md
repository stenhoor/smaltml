---
description: 'StratifiedKFold: k folds that keep each class in proportion. scikit-learn StratifiedKFold.'
---

# StratifiedKFold

<!-- docs:skip -->
```ts
class StratifiedKFold implements Splitter {
  constructor(params?: { nSplits?: number; shuffle?: boolean; seed?: number });
  split(n: number, y: Labels): Fold[];
}
```

K folds whose test sets each hold every class in the proportion of the whole. scikit-learn's [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), and the default for a classifier in `crossValScore` and `GridSearchCV`.

## Parameters

As [`KFold`](./k-fold); `split` also takes `y`, the labels.

## Returns

The folds, as `KFold` gives them.

## Errors

- `ML_PARAM`: no labels, and as `KFold`.
- `ML_SHAPE`: labels that don't match the rows.

## Notes

It keeps scikit-learn's documented guarantees: each class in proportion in every test set, test sets within one row of each other in size, folds that don't depend on what the labels are called, and, without shuffling, each class's rows in order. The allocation is smaltml's own, so the folds can differ from scikit-learn's.

## Examples

```ts
import { StratifiedKFold } from '@smaltworks/smaltml';

const y = ['a', 'a', 'a', 'a', 'b', 'b', 'a', 'a', 'b', 'a'];
new StratifiedKFold({ nSplits: 3 }).split(y.length, y).map((f) => Array.from(f.test, (i) => y[i]).join(''))
```

| value |
| --- |
| aaab |
| aba |
| aba |

## Working demo

```ts
new StratifiedKFold({ nSplits: 2 }).split(y.length, y).map((f) => f.test.length)
```

| value |
| --- |
| 5 |
| 5 |

## See also

[`KFold`](./k-fold), [`crossValScore`](./cross-val-score)
