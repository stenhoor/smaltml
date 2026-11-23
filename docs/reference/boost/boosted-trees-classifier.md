---
description: 'BoostedTreesClassifier: gradient-boosted classification trees, as XGBoost grows them. XGBoost XGBClassifier.'
---

# BoostedTreesClassifier

<!-- docs:skip -->
```ts
class BoostedTreesClassifier implements Classifier {
  constructor(params?: {
    nEstimators?: number;
    learningRate?: number;
    maxDepth?: number;
    minChildWeight?: number;
    gamma?: number;
    regLambda?: number;
    regAlpha?: number;
    maxBin?: number;
    baseScore?: number | null;
    objective?: 'binary:logistic' | 'multi:softprob';
    subsample?: number;
    colsampleBytree?: number;
    colsampleBylevel?: number;
    colsampleBynode?: number;
    seed?: number;
  });
  fit(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Label[]>;
  predictSync(X: MatrixInput): Label[];
  predictProba(X: MatrixInput): Promise<Matrix>;
  predictProbaSync(X: MatrixInput): Matrix;
  readonly classes: readonly Label[];
  readonly objective: 'binary:logistic' | 'multi:softprob';
  readonly trees: readonly BoostTree[];
  readonly baseScore: Float64Array;
  readonly featureImportances: Float64Array;
}
```

Gradient-boosted classification trees: each round grows trees that correct the log odds so far. With two classes
(`binary:logistic`), one tree a round on the second class's log odds; with more (`multi:softprob`), a tree per class a
round on that class's margin, the probabilities their softmax. The trees are grown as XGBoost's histogram method grows
them. XGBoost's
[`XGBClassifier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), with its
parameters and defaults ([ADR 0044](../../adr/0044-phase-3-boosted-trees)). From `@smaltworks/smaltml/boost`.

## Parameters

As [`BoostedTreesRegressor`](./boosted-trees-regressor)'s, but for these two:

- `baseScore` (default: fitted): with two classes, the second class's starting probability, between 0 and 1; fitted,
  its weighted share. With more classes each class's starting margin is fitted (giving one isn't supported yet).
- `objective` (default `'binary:logistic'`): the loss. More than two classes turn `'binary:logistic'` into
  `'multi:softprob'`, as XGBoost's does; `'multi:softprob'` needs more than two classes.

y is any labels, numbers or strings, as for every smaltml classifier: `classes` sorts them, and XGBoost's class 0, 1,
… are `classes[0]`, `classes[1]`, …. `fit` and `fitSync` take `sampleWeight`, one weight per row, 0 or more.

## Returns

- `predict`: each row's most likely class (of equal probabilities, the first).
- `predictProba`: each class's probability, a column per class in the order of `classes`.
- `objective`: the loss the fit used.
- `trees`: the trees, round by round; with more than two classes, a tree per class in each round (tree r · K + k is
  round r's for class k), as XGBoost's model holds them.
- `baseScore`: XGBoost's `base_score`. Two classes: the second class's starting probability (one value). More: each
  class's starting margin, the log of its weighted share less the mean of those logs (they sum to 0).
- `featureImportances`: each column's mean gain over the splits on it in every tree, normalized to sum to 1: XGBoost's
  `feature_importances_`.

## Errors

- `ML_PARAM`: a parameter outside its range; `'multi:softprob'` with two classes; a `baseScore` with more than two
  classes, or sampling below 1, which aren't supported yet.
- `ML_LABELS`: y has one class only.
- `ML_VALUE`: a missing value (`NaN`) in X, which isn't supported yet; a column with more distinct values than
  `maxBin`; a negative `sampleWeight`; a class whose rows all weigh 0, which isn't supported yet.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

The gradients are log loss's: p − y, with Hessian p(1 − p) for two classes and 2p(1 − p) for more, each times the
row's weight; with more than two classes, every class's probabilities come from the round's starting margins, before
any of its trees grows. The compare tests hold smaltml to XGBoost 3.4.1 on data whose columns have at most `maxBin`
distinct values: the same trees, split for split and in the same order, the same predicted classes, and probabilities,
importances and base scores within 1e-5.

## Examples

```ts
import { BoostedTreesClassifier } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]];
const y = ['low', 'low', 'low', 'high', 'low', 'high', 'high', 'high'];
const boosted = new BoostedTreesClassifier({ nEstimators: 20, maxDepth: 2, minChildWeight: 0 }).fitSync(X, y);
boosted.predictProbaSync([[2, 1], [7, 0]])
```

| high | low |
| --- | --- |
| 0.188333 | 0.811667 |
| 0.799347 | 0.200653 |

## Working demo

Three kinds of fruit from their weight and colour:

```ts
const weight = [120, 150, 130, 300, 320, 280, 80, 70, 90, 140, 310, 75];
const red = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1];
const fruit = ['apple', 'apple', 'apple', 'melon', 'melon', 'melon', 'plum', 'plum', 'plum', 'apple', 'melon', 'plum'];
const sorter = await new BoostedTreesClassifier({ nEstimators: 10, maxDepth: 2, minChildWeight: 0 }).fit({ weight, red }, fruit);
({ objective: sorter.objective, classes: sorter.classes, predict: await sorter.predict({ weight: [100, 290], red: [1, 0] }) })
```

| objective | classes | predict |
| --- | --- | --- |
| multi:softprob | [apple, melon, plum] | [plum, melon] |

## See also

[`BoostedTreesRegressor`](./boosted-trees-regressor), [`RandomForestClassifier`](../ensemble/random-forest-classifier),
[`LogisticRegression`](../linear/logistic-regression), [`logLoss`](../metrics/log-loss)
