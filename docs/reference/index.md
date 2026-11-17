---
description: 'Every class and function in smaltml, by area, with the entry point each comes from.'
---

# Reference

One page per class or function, grouped by area. Every page has its signature, parameters, what it returns, the errors
it raises, and examples with their tested results.

## Data

- [`Matrix`](./matrix): the table of numbers smaltml holds data in. `@smaltworks/smaltml/core`

## Preprocessing

From `@smaltworks/smaltml/preprocessing`:

- [`StandardScaler`](./preprocessing/standard-scaler): scales each column to mean 0 and standard deviation 1.
- [`MinMaxScaler`](./preprocessing/min-max-scaler): scales each column to a range.
- [`RobustScaler`](./preprocessing/robust-scaler): centres on the median, scales by the interquartile range.
- [`SimpleImputer`](./preprocessing/simple-imputer): fills missing values.
- [`OneHotEncoder`](./preprocessing/one-hot-encoder): one 0/1 column per category.
- [`OrdinalEncoder`](./preprocessing/ordinal-encoder): categories as numbers 0, 1, 2 …

## Linear models

- [`LinearRegression`](./linear/linear-regression): ordinary least squares. `@smaltworks/smaltml/linear`
- [`Ridge`](./linear/ridge), [`Lasso`](./linear/lasso), [`ElasticNet`](./linear/elastic-net): least squares with an L2,
  L1 or mixed penalty. `@smaltworks/smaltml/linear`
- [`LogisticRegression`](./linear/logistic-regression): classes and their probabilities from a linear model.
  `@smaltworks/smaltml/linear`

## Model selection

From `@smaltworks/smaltml/selection`:

- [`trainTestSplit`](./selection/train-test-split): splits rows into a training set and a test set.
- [`KFold`](./selection/k-fold), [`StratifiedKFold`](./selection/stratified-k-fold), [`GroupKFold`](./selection/group-k-fold):
  folds for cross-validation.
- [`crossValScore`](./selection/cross-val-score): a score per fold.
- [`crossValidate`](./selection/cross-validate): scores per fold for several scorers, on test and training rows.
- [`GridSearchCV`](./selection/grid-search-cv): the best of a grid of parameters, by cross-validation.

## Pipelines

- [`pipeline`](./pipeline/pipeline): steps that run in order as one model. `@smaltworks/smaltml/pipeline`
- [`ColumnTransformer`](./pipeline/column-transformer): different steps on different columns, side by side.
  `@smaltworks/smaltml/pipeline`

## Several targets

- [`MultiOutputRegressor`](./multioutput/multi-output-regressor): a regressor per column of Y.
  `@smaltworks/smaltml/multioutput`

## Metrics

[Metrics](./metrics/), from `@smaltworks/smaltml/metrics`:

- [`meanSquaredError`](./metrics/mean-squared-error), [`meanAbsoluteError`](./metrics/mean-absolute-error),
  [`r2Score`](./metrics/r2-score) for regression
- [`accuracyScore`](./metrics/accuracy-score), [`precisionScore`](./metrics/precision-score),
  [`recallScore`](./metrics/recall-score), [`f1Score`](./metrics/f1-score), [`rocAucScore`](./metrics/roc-auc-score),
  [`confusionMatrix`](./metrics/confusion-matrix), [`logLoss`](./metrics/log-loss) for classification

## Saving and loading

- [`saveModel`](./io/save-model) and [`loadModel`](./io/load-model): models as JSON. `@smaltworks/smaltml/io`

## Model registry

- [`predict`](./registry/predict): predictions from a model, or from the model saved under a name.
  `@smaltworks/smaltml/registry`
- [`ModelRegistry`](./registry/model-registry): fitted models by name and version, in a
  [`MemoryStore`](./registry/memory-store) or a [`TableStore`](./registry/table-store).

## Hardware

- [`matMul`](./gpu/mat-mul): the matrix product, on the CPU or the GPU. `@smaltworks/smaltml/gpu`
- [`gpuAvailable`](./gpu/gpu-available): whether WebGPU is there. `@smaltworks/smaltml/gpu`
- [`startWorker`](./worker/start-worker) and [`connect`](./worker/connect): smaltml in a Web Worker.
  `@smaltworks/smaltml/worker`

## Rules

- [Errors](./errors): every error code.
- [Where smaltml differs](./divergences): from scikit-learn.
