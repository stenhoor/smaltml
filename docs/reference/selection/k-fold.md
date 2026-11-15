---
description: 'KFold: k folds of consecutive rows for cross-validation. scikit-learn KFold.'
---

# KFold

<!-- docs:skip -->
```ts
class KFold implements Splitter {
  constructor(params?: { nSplits?: number; shuffle?: boolean; seed?: number });
  split(n: number): Fold[];
}
interface Fold { train: Uint32Array; test: Uint32Array }
```

Cuts n rows into k folds of consecutive rows; each fold is the test set once, and the rest the training set. scikit-learn's [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).

## Parameters

- `nSplits` (default `5`): how many folds, at least 2.
- `shuffle` (default `false`): shuffle the rows first.
- `seed`: the shuffle's seed (with `shuffle` only).

## Returns

`split` returns the folds, each the training and test rows by position. The first `n % k` test sets have one row more.

## Errors

- `ML_PARAM`: fewer than 2 folds, more folds than rows, or a seed without shuffle.

## Notes

Without shuffling, the folds are scikit-learn's, row for row. A shuffle is smaltml's own: a seed repeats it.

## Examples

```ts
import { KFold } from '@smaltworks/smaltml';

new KFold({ nSplits: 3 }).split(7).map((f) => ({ train: Array.from(f.train), test: Array.from(f.test) }))
```

| train | test |
| --- | --- |
| [3, 4, 5, 6] | [0, 1, 2] |
| [0, 1, 2, 5, 6] | [3, 4] |
| [0, 1, 2, 3, 4] | [5, 6] |

## Working demo

```ts
new KFold({ nSplits: 3, shuffle: true, seed: 3 }).split(7).map((f) => Array.from(f.test))
```

| value |
| --- |
| [[1, 4, 6], [0, 3], [2, 5]] |

## See also

[`StratifiedKFold`](./stratified-k-fold), [`crossValScore`](./cross-val-score)
