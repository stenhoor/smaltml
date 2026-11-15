---
description: 'Metrics: how good predictions are, for regression and for classification. scikit-learn metrics.'
---

# Metrics

Functions that score predictions against the true values, by scikit-learn's definitions. Each takes the true values
first and the predictions second.

## Regression

- [`meanSquaredError`](./mean-squared-error): the mean of the squared errors.
- [`meanAbsoluteError`](./mean-absolute-error): the mean of the absolute errors.
- [`r2Score`](./r2-score): the share of the target's variation the predictions explain.

## Classification

- [`accuracyScore`](./accuracy-score): the share of labels predicted right.
- [`logLoss`](./log-loss): the mean negative log-likelihood of the true labels under the predicted probabilities.
- [`precisionScore`](./precision-score), [`recallScore`](./recall-score), [`f1Score`](./f1-score): per class, or
  averaged over the classes.
- [`rocAucScore`](./roc-auc-score): the area under the ROC curve.
- [`confusionMatrix`](./confusion-matrix): true labels against predicted ones.

## Rules they share

- `yTrue` and `yPred` must have the same length, at least one value: else `ML_SHAPE`.
- Regression metrics take numbers; classification metrics take labels, numbers or strings.
