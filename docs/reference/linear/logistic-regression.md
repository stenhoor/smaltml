---
description: 'LogisticRegression: classes and their probabilities from a linear model, fitted by L-BFGS. scikit-learn LogisticRegression.'
---

# LogisticRegression

<!-- docs:skip -->
```ts
class LogisticRegression implements Classifier {
  constructor(params?: { C?: number; l1Ratio?: number; tol?: number; fitIntercept?: boolean; maxIter?: number });
  fit(X: MatrixInput, y: Labels): Promise<this>;
  predict(X: MatrixInput): Promise<Label[]>;
  predictProba(X: MatrixInput): Promise<Matrix>;
  decisionFunctionSync(X: MatrixInput): Matrix;
  readonly classes: readonly Label[];
  readonly coef: Matrix;
  readonly intercept: Float64Array;
  readonly nIter: number;
}
type Label = number | string;
```

A linear model of each class's log-odds, fitted by L-BFGS to minimize the log loss plus an L2 penalty. scikit-learn's
[`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
with its default solver, `lbfgs`.

## Parameters

- `C` (default `1`): the inverse of the penalty's strength. The objective is the mean log loss plus
  `‖w‖² / (2·C·n)`; smaller `C` holds the coefficients closer to zero. `Infinity` turns the penalty off.
- `l1Ratio` (default `0`): the mix of L1 in the penalty. Only `0` is supported so far.
- `tol` (default `1e-4`): stop when the largest component of the objective's gradient is at most this.
- `fitIntercept` (default `true`): fit an intercept, which is never penalized.
- `maxIter` (default `100`): the most L-BFGS iterations.

## Returns

- `predict`: the most probable class for each row.
- `predictProba`: a Matrix with a column per class, in the order of `classes`; each row sums to 1.
- `decisionFunctionSync`: each row's score, `X·wᵀ + b`.
- `classes`: the labels seen in `fit`, sorted.
- `coef`: one row for two classes (the log-odds of the second class), else one row per class; one column per feature.
- `intercept`: one value per row of `coef`. `nIter`: the iterations the fit took.

## Errors

- `ML_LABELS`: one class only, or labels that mix numbers and strings.
- `ML_PARAM`: `C` ≤ 0, `l1Ratio` other than 0, `tol` ≤ 0, or `maxIter` below 1.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Two classes fit one row of coefficients; more classes fit the multinomial (softmax) model, as scikit-learn does. Run
to convergence, the coefficients agree with scikit-learn's within 1e-5 (relative) in the compare tests; with the
default `tol` both stop early, and agree within about 2e-3.

## Examples

```ts
import { LogisticRegression } from '@smaltworks/smaltml';

const clf = await new LogisticRegression().fit([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 1, 1, 1]);
await clf.predictProba([[1], [2.5], [4]])
```

| 0 | 1 |
| --- | --- |
| 0.842976 | 0.157024 |
| 0.499938 | 0.500062 |
| 0.156959 | 0.843041 |

<!-- docs:error ML_LABELS -->
```ts
await new LogisticRegression().fit([[0], [1]], ['yes', 'yes'])
```

## Working demo

Three species from two measurements:


| value |
| --- |
| setosa |
| versicolor |
| virginica |
```ts
const petals = [[1.4, 0.2], [1.3, 0.2], [1.5, 0.3], [4.5, 1.5], [4.1, 1.3], [4.7, 1.4], [6.0, 2.5], [5.6, 2.2], [5.8, 1.8]];
const species = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica'];
const iris = await new LogisticRegression({ C: 10 }).fit(petals, species);
await iris.predict([[1.6, 0.4], [4.4, 1.4], [5.9, 2.1]])
```

| value |
| --- |
| setosa |
| versicolor |
| virginica |

## See also

[Training models](../../guide/training), [`accuracyScore`](../metrics/accuracy-score), [`logLoss`](../metrics/log-loss)
