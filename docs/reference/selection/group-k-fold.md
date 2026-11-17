---
description: 'GroupKFold: folds that keep each group''s rows together, so a model is tested on groups it never saw. scikit-learn GroupKFold.'
---

# GroupKFold

<!-- docs:skip -->
```ts
class GroupKFold implements Splitter {
  constructor(params?: { nSplits?: number });
  split(n: number, y: Labels | undefined, groups: ArrayLike<number | string>): Fold[];
}
```

Cuts the rows into k folds by group: all of a group's rows are in one test fold, and in the training rows of every
other fold. Use it when rows of one group are alike (several visits of one patient, the orders of one customer): a
fold that trains on some of a group and tests on the rest scores the model better than it will do on a group it
hasn't seen. scikit-learn's
[`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

## Parameters

- `nSplits` (default `5`): how many folds, at least 2 and at most the number of groups.

`split` takes each row's group, numbers or text. [`crossValScore`](./cross-val-score) and
[`GridSearchCV`](./grid-search-cv)'s `fit` take them as `groups`, and pass them here.

## Returns

`split` returns the folds, each the training and test rows by position.

## Errors

- `ML_PARAM`: fewer than 2 folds, more folds than groups, no groups, or a parameter other than `nSplits`
  (scikit-learn's `shuffle` isn't supported yet).
- `ML_SHAPE`: groups that don't have one value per row.
- `ML_LABELS`: groups that mix numbers and text.

## Notes

The folds are scikit-learn's, row for row: the groups, largest first, each go to the test fold with the fewest rows
so far, so the test folds are about the same size. Of two groups as large, the one sorted later goes first; of two
folds as full, the lower one takes it. The compare tests check the folds against scikit-learn's, with ties, text
groups and 40 groups.

## Examples

```ts
import { GroupKFold } from '@smaltworks/smaltml';

const visits = ['ann', 'ann', 'bob', 'bob', 'bob', 'cy', 'dee', 'dee'];
new GroupKFold({ nSplits: 3 }).split(visits.length, undefined, visits).map((f) => Array.from(f.test, (i) => visits[i]))
```

| value |
| --- |
| [[bob, bob, bob], [cy, dee, dee], [ann, ann]] |

<!-- docs:error ML_PARAM -->
```ts
new GroupKFold({ nSplits: 5 }).split(visits.length, undefined, visits)
```

## Working demo

Each patient's visits stay in one fold, so every score is on patients the model didn't see:

```ts
import { crossValScore, LinearRegression } from '@smaltworks/smaltml';

const patients = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5];
const dose = patients.map((p, i) => [i % 4, p]);
const effect = dose.map(([d, p]) => 3 * d! + 0.5 * p! + (p! % 2));
await crossValScore(new LinearRegression(), dose, effect, { cv: new GroupKFold({ nSplits: 3 }), groups: patients })
```

| value |
| --- |
| 0.96424 |
| 0.847655 |
| 0.980843 |

## See also

[`KFold`](./k-fold), [`StratifiedKFold`](./stratified-k-fold), [`crossValScore`](./cross-val-score),
[`GridSearchCV`](./grid-search-cv)
