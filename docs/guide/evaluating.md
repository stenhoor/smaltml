---
description: 'Measure how well a model does on rows it has not seen: trainTestSplit, then the metrics for regression and classification.'
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

## See also

- [`trainTestSplit`](../reference/selection/train-test-split), [Metrics](../reference/metrics/)
