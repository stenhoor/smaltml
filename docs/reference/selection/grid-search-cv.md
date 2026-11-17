---
description: 'GridSearchCV: cross-validates every combination of parameters and keeps the best. scikit-learn GridSearchCV.'
---

# GridSearchCV

<!-- docs:skip -->
```ts
class GridSearchCV implements Model {
  constructor(estimator: Model, paramGrid: ParamGrid, options?: {
    cv?: number | Splitter | Fold[];
    scoring?: ScorerName | Scorer | ScorerName[] | Record<string, ScorerName | Scorer>;
    refit?: boolean | string;
    returnTrainScore?: boolean;
  });
  fit(X: TableInput, y: Labels, options?: { groups?: ArrayLike<number | string> }): Promise<this>;
  predict(X: TableInput): Promise<Float64Array | Label[]>;
  predictProba(X: TableInput): Promise<Matrix>;
  readonly bestParams: Record<string, unknown>;
  readonly bestScore: number;
  readonly bestIndex: number;
  readonly bestEstimator: Model;
  readonly cvResults: { params; meanTestScore; stdTestScore; rankTestScore; splitTestScores; meanTrainScore?; stdTrainScore?; splitTrainScores? };
  readonly cvResultsByScorer: { params; scores: Record<string, { meanTestScore; stdTestScore; rankTestScore; splitTestScores; meanTrainScore?; stdTrainScore?; splitTrainScores? }> };
}
type ParamGrid = Record<string, unknown[]> | Record<string, unknown[]>[];
function parameterGrid(grid: ParamGrid): Record<string, unknown>[];
```

Cross-validates every combination of a parameter grid, keeps the best, and fits it on all the rows. scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). From `@smaltworks/smaltml/selection`.

## Parameters

- `estimator`: the model to tune; a pipeline's step parameters are `step__param`.
- `paramGrid`: each parameter's values, or a list of such grids.
- `cv`: as [`crossValScore`](./cross-val-score). `fit` takes `groups` for a `cv` that uses them, as `crossValScore`
  does.
- `scoring`: one scorer, as `crossValScore` takes, or several: a list of scorers' names, or a record that names them
  (`{ fit: 'r2', error: myScorer }`), as [`crossValidate`](./cross-validate) takes.
- `refit` (default `true`): fit the best parameters on all the rows as `bestEstimator`. With several scorers, name the
  one that chooses the best, or `false`; `true` fails, as in scikit-learn.
- `returnTrainScore` (default `false`): score each fold's training rows too, as `meanTrainScore`, `stdTrainScore` and
  `splitTrainScores`. A training score far above the test score shows a model that overfits.

## Returns

After `fit`: `cvResults` with every candidate's parameters, mean and standard deviation of fold scores, and rank (1 is best; ties share a rank), and the train scores with `returnTrainScore`; `bestParams`, `bestScore`, `bestIndex`, and `bestEstimator`. `predict` and `predictProba` use `bestEstimator`.

With several scorers, `cvResultsByScorer` holds each scorer's results under its name (scikit-learn's
`mean_test_<name>` and the rest), and `bestScore` and the other best fields follow the scorer `refit` names.
`cvResultsByScorer` works with one scorer too, under `score`.

## Errors

- `ML_PARAM`: an empty list in the grid, or a parameter the estimator doesn't have. With several scorers: `refit: true`,
  a `refit` that isn't one of them, `cvResults` (read `cvResultsByScorer`), or the best fields with `refit: false`.
  A scorer named twice.
- `ML_NOT_FITTED`: results before `fit`.

## Notes

The candidates come in scikit-learn's order (names sorted, the last changing fastest), so `cvResults` lines up with scikit-learn's, and the best is the first of the highest mean. It saves to JSON with its results and best estimator (scorers by name: one given as a function isn't saved); `loadModel` reads it once `@smaltworks/smaltml/selection` is loaded.

## Examples

```ts
import { GridSearchCV, LogisticRegression, KFold } from '@smaltworks/smaltml';

const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]];
const y = [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1];
const search = await new GridSearchCV(new LogisticRegression(), { C: [0.01, 1, 100] }, { cv: new KFold({ nSplits: 3 }) }).fit(X, y);
({ best: search.bestParams, means: search.cvResults.meanTestScore, ranks: search.cvResults.rankTestScore })
```

| best | means | ranks |
| --- | --- | --- |
| \{"C":1\} | [0.166667, 0.833333, 0.833333] | [3, 1, 1] |

Two scorers, the best by the error, and the training scores beside the test scores:

```ts
const two = await new GridSearchCV(new LogisticRegression(), { C: [0.01, 1, 100] }, {
  cv: new KFold({ nSplits: 3 }),
  scoring: { accuracy: 'accuracy', loss: 'neg_log_loss' },
  refit: 'loss',
  returnTrainScore: true,
}).fit(X, y);
({ best: two.bestParams, test: two.cvResultsByScorer.scores.loss!.meanTestScore, train: two.cvResultsByScorer.scores.loss!.meanTrainScore })
```

| best | test | train |
| --- | --- | --- |
| \{"C":1\} | [-1.08488, -0.578163, -0.779812] | [-0.529589, -0.33784, -0.324258] |

## Working demo

```ts
await search.predict([[2.5], [9.5]])
```

| value |
| --- |
| 0 |
| 1 |

## See also

[`crossValScore`](./cross-val-score), [`pipeline`](../pipeline/pipeline), [Evaluating models: grid search](../../guide/evaluating#tuning-with-a-grid-search)
