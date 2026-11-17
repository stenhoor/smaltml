---
description: 'crossValidate: scores per fold for one scorer or several, on the test rows and the training rows, with times. scikit-learn cross_validate.'
---

# crossValidate

<!-- docs:skip -->
```ts
function crossValidate(model: Model, X: TableInput, y: Labels, options?: {
  cv?: number | Splitter | Fold[];
  scoring?: ScorerName | Scorer | ScorerName[] | Record<string, ScorerName | Scorer>;
  groups?: ArrayLike<number | string>;
  returnTrainScore?: boolean;
}): Promise<{ test: Record<string, Float64Array>; train?: Record<string, Float64Array>; fitTime: Float64Array; scoreTime: Float64Array }>;
```

Fits a fresh copy of the model on each fold's training rows and scores it on the fold's test rows, with several
scorers at once, and on the training rows too if asked. [`crossValScore`](./cross-val-score) with more. scikit-learn's
[`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).
From `@smaltworks/smaltml/selection`.

## Parameters

- `model`, `cv`, `groups`: as [`crossValScore`](./cross-val-score).
- `scoring`: one scorer, as `crossValScore` takes, or several: a list of scorers' names (`['r2',
  'neg_mean_absolute_error']`), or a record that names them, each a scorer's name or a function
  (`{ fit: 'r2', error: myScorer }`). Default: accuracy for a classifier, R² otherwise.
- `returnTrainScore` (default `false`): score each fold's training rows too. It costs a prediction on them.

## Returns

- `test`: each scorer's score per fold, under its name; one scorer's is named `score`.
- `train`: the same on the training rows, with `returnTrainScore`.
- `fitTime` and `scoreTime`: each fold's fit, and its scoring on the test rows, in seconds.

## Errors

- `ML_PARAM`: as for `crossValScore`; an empty list or record of scorers, or a scorer named twice.
- `ML_SHAPE`: y that doesn't match X. The model's own errors.

## Notes

scikit-learn's result is one dictionary, `test_r2`, `train_r2`, `fit_time`, `score_time` (`test_score` for one
scorer). smaltml's groups it as `test.r2`, `train.r2`, `fitTime` and `scoreTime`. The scores agree with scikit-learn's
in the compare tests, within 1e-9 (1e-5 for logistic regression). The times are this machine's.

## Examples

```ts
import { crossValidate, KFold, Ridge } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 1], [4, 3], [5, 2], [6, 4], [7, 3], [8, 5]];
const y = [3, 6, 7, 12, 12, 17, 17, 22];
const res = await crossValidate(new Ridge(), X, y, { cv: new KFold({ nSplits: 4 }), scoring: ['r2', 'neg_mean_absolute_error'], returnTrainScore: true });
({ test: res.test.r2, train: res.train!.r2 })
```

| test | train |
| --- | --- |
| 0.9045 | 0.99833 |
| 0.957471 | 0.998818 |
| 0.98972 | 0.997809 |
| 0.9 | 0.996139 |

<!-- docs:error ML_PARAM -->
```ts
await crossValidate(new Ridge(), X, y, { scoring: ['r2', 'r2'] })
```

## Working demo

Training scores much higher than test scores show a model that fits its training rows too closely:

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const wide = Array.from({ length: 12 }, (_, i) => [i, (i * 7) % 5, (i * 3) % 4, (i * 5) % 7, (i * 11) % 6]);
const noisy = wide.map((r, i) => r[0]! + ((i * 13) % 5) - 2);
const fits = await crossValidate(new LinearRegression(), wide, noisy, { cv: new KFold({ nSplits: 3 }), returnTrainScore: true });
({ test: fits.test.score, train: fits.train!.score })
```

| test | train |
| --- | --- |
| 0.14 | 0.882583 |
| -0.381374 | 0.952527 |
| 0.285714 | 0.759904 |

## See also

[`crossValScore`](./cross-val-score), [`GridSearchCV`](./grid-search-cv), [`KFold`](./k-fold), [Metrics](../metrics/r2-score)
