---
description: 'Train regression and classification models: linear, ridge, lasso, elastic-net and logistic regression, and pipelines that scale data first.'
---

# Training models

Every model is fitted the same way: create it with its parameters, `fit` it on `X` and `y`, then `predict`.

## Predicting a number: linear regression

`LinearRegression` fits a straight line (a plane, with more columns) through the data by least squares:

```ts
import { LinearRegression } from '@smaltworks/smaltml';

// Delivery times in minutes, from distance in km and items in the order.
const X = [[1.2, 1], [3.5, 2], [2.0, 5], [6.1, 3], [4.4, 1], [8.0, 6], [5.2, 4], [7.3, 2]];
const minutes = [14, 25, 24, 38, 27, 52, 37, 42];

const linear = await new LinearRegression().fit(X, minutes);
({ coef: linear.coef, intercept: linear.intercept })
```

| coef | intercept |
| --- | --- |
| [4.32549, 1.81477] | 6.5468 |

Each kilometre adds about 4.3 minutes and each item about 1.8, on top of a 6.5-minute start.

```ts
await linear.predict([[3.0, 2], [10.0, 4]])
```

| value |
| --- |
| 23.1528 |
| 57.0608 |

## Penalized linear regression

With many columns, or columns that move together, a penalty keeps the coefficients in check. `Ridge` shrinks them all
towards zero; `Lasso` sets the least useful to exactly zero, so it also chooses columns; `ElasticNet` mixes the two.
`alpha` sets the penalty's strength:

```ts
import { Ridge, Lasso } from '@smaltworks/smaltml';

// The delivery data again, with a column that is pure noise: the lasso drops it.
const withNoise = X.map((row, i) => [...row, [0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4][i]]);
const ridge = await new Ridge({ alpha: 1 }).fit(withNoise, minutes);
const lasso = await new Lasso({ alpha: 0.5 }).fit(withNoise, minutes);
({ ridge: ridge.coef, lasso: lasso.coef })
```

| ridge | lasso |
| --- | --- |
| 4.22961 | 4.26576 |
| 1.77154 | 1.67598 |
| 0.0653734 | 0 |

Scale the columns first (a `StandardScaler` in a pipeline), so the penalty weighs them alike.

## Predicting a label: logistic regression

`LogisticRegression` predicts a class, and how likely each class is. Labels are numbers or strings:

```ts
import { LogisticRegression } from '@smaltworks/smaltml';

// Hours studied and hours slept, and whether the exam was passed.
const hours = [[1, 4], [2, 8], [3, 5], [4, 7], [5, 6], [6, 8], [2, 5], [7, 4], [8, 7], [3, 8]];
const passed = ['fail', 'fail', 'fail', 'pass', 'pass', 'pass', 'fail', 'pass', 'pass', 'fail'];

const classifier = await new LogisticRegression().fit(hours, passed);
await classifier.predict([[2, 6], [6, 6]])
```

| value |
| --- |
| fail |
| pass |

`predictProba` gives each class's probability, a column per class in the order of `classes`:

```ts
await classifier.predictProba([[2, 6], [6, 6]])
```

| fail | pass |
| --- | --- |
| 0.917699 | 0.0823009 |
| 0.0606206 | 0.939379 |

With more than two classes, it fits the multinomial (softmax) model. `C` sets the penalty: smaller values hold the
coefficients closer to zero.

## Pipelines

A pipeline runs steps in order as one model: the steps before the last transform `X`, and the last one predicts.
Fitting the pipeline fits each step on what the steps before it give:

```ts
import { pipeline, StandardScaler } from '@smaltworks/smaltml';

const model = await pipeline(new StandardScaler(), new LogisticRegression()).fit(hours, passed);
await model.predict([[2, 6], [6, 6]])
```

| value |
| --- |
| fail |
| pass |

The pipeline saves, loads and predicts as one model, so the scaling learned on the training rows always comes with
it.

## See also

- [`LinearRegression`](../reference/linear/linear-regression), [`Ridge`](../reference/linear/ridge), [`Lasso`](../reference/linear/lasso),
  [`ElasticNet`](../reference/linear/elastic-net), [`LogisticRegression`](../reference/linear/logistic-regression),
  [`pipeline`](../reference/pipeline/pipeline)
- [Evaluating models](./evaluating)
