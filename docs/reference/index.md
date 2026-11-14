---
description: 'Every class and function in smaltml, by area, with the entry point each comes from.'
---

# Reference

One page per class or function, grouped by area. Every page has its signature, parameters, what it returns, the errors
it raises, and examples with their tested results.

## Data

- [`Matrix`](./matrix): the table of numbers smaltml holds data in. `@smaltworks/smaltml/core`

## Preprocessing

- [`StandardScaler`](./preprocessing/standard-scaler): scales each column to mean 0 and standard deviation 1.
  `@smaltworks/smaltml/preprocessing`

## Linear models

- [`LinearRegression`](./linear/linear-regression): ordinary least squares. `@smaltworks/smaltml/linear`
- [`LogisticRegression`](./linear/logistic-regression): classes and their probabilities from a linear model.
  `@smaltworks/smaltml/linear`

## Model selection

- [`trainTestSplit`](./selection/train-test-split): splits rows into a training set and a test set.
  `@smaltworks/smaltml/selection`

## Pipelines

- [`pipeline`](./pipeline/pipeline): steps that run in order as one model. `@smaltworks/smaltml/pipeline`

## Metrics

[Metrics](./metrics/), from `@smaltworks/smaltml/metrics`:

- [`meanSquaredError`](./metrics/mean-squared-error), [`r2Score`](./metrics/r2-score) for regression
- [`accuracyScore`](./metrics/accuracy-score), [`logLoss`](./metrics/log-loss) for classification

## Saving and loading

- [`saveModel`](./io/save-model) and [`loadModel`](./io/load-model): models as JSON. `@smaltworks/smaltml/io`

## Rules

- [Errors](./errors): every error code.
- [Where smaltml differs](./divergences): from scikit-learn.
