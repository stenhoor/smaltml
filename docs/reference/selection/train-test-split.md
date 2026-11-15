---
description: 'trainTestSplit: splits rows into a training set and a test set, seeded. scikit-learn train_test_split.'
---

# trainTestSplit

<!-- docs:skip -->
```ts
function trainTestSplit<Y extends Labels>(X: MatrixInput, y: Y, options?: TrainTestSplitOptions): { XTrain: Matrix; XTest: Matrix; yTrain: Y; yTest: Y };
function trainTestSplit(X: MatrixInput, y?: undefined, options?: TrainTestSplitOptions): { XTrain: Matrix; XTest: Matrix };
interface TrainTestSplitOptions { testSize?: number; trainSize?: number; shuffle?: boolean; seed?: number; stratify?: Labels }
```

Splits `X`, and `y` row for row, into a training set and a test set. scikit-learn's
[`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

## Parameters

- `testSize`: the test set's share of the rows (between 0 and 1), or its number of rows. Default `0.25`, or what
  `trainSize` leaves.
- `trainSize`: the training set's share or number of rows. Default: what `testSize` leaves.
- `shuffle` (default `true`): shuffle the rows first. Without it, the first rows train and the last rows test.
- `seed`: the shuffle's seed. The same seed gives the same split.
- `stratify`: labels to keep in proportion, usually `y`. Each class then has the same share of the training set and of
  the test set as of all the rows, as near as whole rows allow: with a rare class, both sets get some of it.

## Returns

`XTrain` and `XTest` as Matrices with `X`'s column names; `yTrain` and `yTest` as arrays (a `Float64Array` when `y`
is one).

## Errors

- `ML_PARAM`: a size outside (0, 1) that isn't a whole number of rows from 1 to n − 1, sizes that add up to more than
  the rows, or an empty training set.
- `ML_SHAPE`: `y`'s length (or `stratify`'s) differs from `X`'s rows.
- `ML_PARAM` with `stratify`: `shuffle: false`, or a set with fewer rows than there are classes.
- `ML_LABELS` with `stratify`: a class with only one row.

## Notes

Sizes are scikit-learn's: a share of the rows rounds up for the test set and down for the training set. The shuffle
is smaltml's own seeded generator, so a seed repeats its split, but doesn't give scikit-learn's split for the same
number. With `stratify`, each class's count in each set follows scikit-learn's rule (its share, rounded down, then
one more for the classes with the largest remainders), checked against scikit-learn in the compare tests; which rows
go where is smaltml's shuffle.

## Examples

```ts
import { trainTestSplit } from '@smaltworks/smaltml';

const rows = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
const { XTest } = trainTestSplit(rows, undefined, { testSize: 3, shuffle: false });
XTest
```

| 0 |
| --- |
| 7 |
| 8 |
| 9 |

With `stratify`, a class that is one row in ten is one row in ten of each set:

```ts
const kinds = ['a', 'a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'b'];
const parts = trainTestSplit(kinds.map((_, i) => [i]), kinds, { testSize: 0.5, stratify: kinds, seed: 1 });
({ train: parts.yTrain.filter((v) => v === 'b').length, test: parts.yTest.filter((v) => v === 'b').length })
```

| train | test |
| --- | --- |
| 2 | 2 |

## Working demo

| test | labels |
| --- | --- |
| 2 | a |
| 1 | b |
| 4 | a |

```ts
const labels = ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'];
const split = trainTestSplit(rows, labels, { testSize: 0.3, seed: 42 });
({ test: split.XTest.column(0), labels: split.yTest })
```

| test | labels |
| --- | --- |
| 2 | a |
| 1 | b |
| 4 | a |

## See also

[Evaluating models](../../guide/evaluating)
