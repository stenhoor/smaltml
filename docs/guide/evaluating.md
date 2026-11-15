---
description: 'Measure how well a model does on rows it has not seen: test sets, metrics, cross-validation, and tuning parameters by grid search.'
---

# Evaluating models

A model is only as good as its predictions on rows it hasn't seen. Hold some rows back, fit on the rest, and score
the predictions for the rows held back.

## Holding rows back

`trainTestSplit` shuffles the rows and splits them into a training set and a test set, a quarter of the rows by
default. A `seed` gives the same split every time:

```ts
import { trainTestSplit } from '@smaltworks/smaltml';

// Twenty flats: area in square metres and rooms, and the rent each asks.
const X = [[38, 1], [45, 2], [52, 2], [55, 2], [61, 3], [64, 2], [70, 3], [72, 3], [78, 3], [83, 4],
           [88, 3], [91, 4], [96, 4], [102, 4], [108, 5], [115, 4], [120, 5], [128, 5], [135, 6], [142, 5]];
const rent = [980, 1120, 1260, 1290, 1420, 1450, 1580, 1590, 1700, 1820,
              1870, 1950, 2020, 2110, 2260, 2300, 2420, 2510, 2680, 2700];

const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, rent, { testSize: 0.25, seed: 7 });
({ train: XTrain.rows, test: XTest.rows })
```

| train | test |
| --- | --- |
| 15 | 5 |

## Regression metrics

```ts
import { LinearRegression, meanSquaredError, r2Score } from '@smaltworks/smaltml';

const model = await new LinearRegression().fit(XTrain, yTrain);
const predicted = await model.predict(XTest);
({ mse: meanSquaredError(yTest, predicted), r2: r2Score(yTest, predicted) })
```

| mse | r2 |
| --- | --- |
| 171.251 | 0.999437 |

`meanSquaredError` is the mean of the squared errors, in the target's units squared. `r2Score` is the share of the
target's variation the model explains: 1 is perfect, 0 is no better than predicting the mean.

## Classification metrics

```ts
import { LogisticRegression, accuracyScore, logLoss } from '@smaltworks/smaltml';

const hours = [[1, 4], [2, 8], [3, 5], [4, 7], [5, 6], [6, 8], [2, 5], [7, 4], [8, 7], [3, 8], [5, 4], [6, 5]];
const passed = ['fail', 'fail', 'fail', 'pass', 'pass', 'pass', 'fail', 'pass', 'pass', 'fail', 'fail', 'pass'];
const split = trainTestSplit(hours, passed, { testSize: 4, seed: 3 });

const classifier = await new LogisticRegression().fit(split.XTrain, split.yTrain);
const accuracy = accuracyScore(split.yTest, await classifier.predict(split.XTest));
const loss = logLoss(split.yTest, await classifier.predictProba(split.XTest), { labels: classifier.classes });
({ accuracy, logLoss: loss })
```

| accuracy | logLoss |
| --- | --- |
| 0.75 | 0.476208 |


`accuracyScore` is the share of labels predicted right. `logLoss` scores the probabilities: lower is better, and a
confident wrong answer costs the most. Pass the model's `classes` as `labels` when the test rows may not hold every
class.

## Precision, recall and the confusion matrix

Accuracy hides which mistakes a model makes. `confusionMatrix` counts true labels (rows) against predicted ones
(columns); `precisionScore`, `recallScore` and `f1Score` measure a class at a time:

```ts
import { confusionMatrix, precisionScore, recallScore, f1Score } from '@smaltworks/smaltml';

const truth = ['spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham', 'ham'];
const guess = ['spam', 'ham', 'ham', 'ham', 'spam', 'spam', 'ham', 'ham'];
confusionMatrix(truth, guess)
```

| ham | spam |
| --- | --- |
| 4 | 1 |
| 1 | 2 |

```ts
const spam = { posLabel: 'spam' };
({ precision: precisionScore(truth, guess, spam), recall: recallScore(truth, guess, spam), f1: f1Score(truth, guess, spam) })
```

| precision | recall | f1 |
| --- | --- | --- |
| 0.666667 | 0.666667 | 0.666667 |

With more than two classes, `average` combines them: `'macro'` (each class alike), `'weighted'` (by each class's
rows) or `'micro'` (every row alike). `rocAucScore` scores probabilities rather than labels.

## Cross-validation

One test set is one sample. `crossValScore` fits a fresh copy of the model on each of k folds' training rows, and
scores it on the fold's test rows: k scores, from every row taking a turn in a test set.

```ts
import { crossValScore, KFold } from '@smaltworks/smaltml';

const scores = await crossValScore(new LinearRegression(), X, rent, { cv: new KFold({ nSplits: 5 }) });
({ scores, mean: scores.reduce((a, b) => a + b, 0) / scores.length })
```

| scores | mean |
| --- | --- |
| [0.930761, 0.9389, 0.938561, 0.966219, 0.770818] | 0.909052 |

The default is 5 folds, kept in proportion by class (`StratifiedKFold`) for a classifier, and the model's usual score:
accuracy for a classifier, R² otherwise. `scoring` names another, as scikit-learn names them (`'f1_macro'`,
`'neg_mean_squared_error'` …).

## Tuning with a grid search

`GridSearchCV` cross-validates every combination of the parameters you list, keeps the best, and fits it on all the
rows. Name a pipeline step's parameter as `step__parameter`:

```ts
import { GridSearchCV, Ridge, StandardScaler, pipeline } from '@smaltworks/smaltml';

const search = await new GridSearchCV(
  pipeline(new StandardScaler(), new Ridge()),
  { ridge__alpha: [0.1, 1, 10, 100] },
  { cv: new KFold({ nSplits: 4 }) },
).fit(X, rent);
({ best: search.bestParams, score: search.bestScore })
```

| best | score |
| --- | --- |
| \{"ridge__alpha":0.1\} | 0.933863 |

```ts
await search.predict([[75, 3]])
```

| value |
| --- |
| 1646.8 |

## See also

- [`trainTestSplit`](../reference/selection/train-test-split), [`crossValScore`](../reference/selection/cross-val-score),
  [`GridSearchCV`](../reference/selection/grid-search-cv), [Metrics](../reference/metrics/)
