---
description: 'RandomForestClassifier: many classification trees on samples of the rows and columns, their class probabilities averaged. scikit-learn RandomForestClassifier.'
---

# RandomForestClassifier

<!-- docs:skip -->
```ts
class RandomForestClassifier implements Classifier {
  constructor(params?: {
    nEstimators?: number;
    criterion?: 'gini' | 'entropy' | 'log_loss';
    classWeight?: 'balanced' | Record<string, number> | null;
    bootstrap?: boolean;
    maxSamples?: number | null;
    oobScore?: boolean;
    maxDepth?: number | null;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    minWeightFractionLeaf?: number;
    maxFeatures?: number | 'sqrt' | 'log2' | null;
    maxLeafNodes?: number | null;
    minImpurityDecrease?: number;
    seed?: number;
  });
  fit(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number>; threads?: number | 'auto' }): Promise<this>;
  fitSync(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Label[]>;
  predictSync(X: MatrixInput): Label[];
  predictProba(X: MatrixInput): Promise<Matrix>;
  predictProbaSync(X: MatrixInput): Matrix;
  readonly classes: readonly Label[];
  readonly estimators: readonly DecisionTreeClassifier[];
  readonly featureImportances: Float64Array;
  readonly oobScore: number;
  readonly oobDecisionFunction: Matrix;
  readonly seed: number;
}
```

Many classification trees, each grown on a bootstrap sample of the rows and, at each split, from a random choice of
the columns (√columns by default); a row's class probabilities are the trees' mean, and the prediction is the most
probable class. scikit-learn's
[`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/ensemble`.

## Parameters

- `criterion` (default `'gini'`): as for [`DecisionTreeClassifier`](../tree/decision-tree-classifier#parameters).
- `classWeight`: weights for the classes. `'balanced'` gives each class the total weight over (classes × the class's
  weight), so every class weighs the same in all (with no `sampleWeight`, a class's weight is its count of rows); an
  object gives a class's weight by its label. They multiply the rows' weights, so with `bootstrap` they change each
  row's chance of being drawn, as scikit-learn 1.9 does. `'balanced_subsample'` isn't supported yet.
- `maxFeatures` (default `'sqrt'`): how many columns each split looks at, as for the trees.
- `nEstimators`, `bootstrap`, `maxSamples`, `oobScore`, `seed`, and each tree's `maxDepth`, `minSamplesSplit`,
  `minSamplesLeaf`, `minWeightFractionLeaf`, `maxLeafNodes`, `minImpurityDecrease`: as for
  [`RandomForestRegressor`](./random-forest-regressor#parameters). The out-of-bag score is the accuracy.

`fit` and `fitSync` take `sampleWeight`, as for [`RandomForestRegressor`](./random-forest-regressor#parameters),
multiplied by the row's class weight. `fit` also takes `threads`, as for
[`RandomForestRegressor`](./random-forest-regressor#parameters): the same forest as `fitSync` on any number of threads.

## Returns

- `predict`: each row's most probable class (of equal probabilities, the first in `classes`).
- `predictProba`: each row's class probabilities, the trees' mean, a column per class in the order of `classes`.
- `classes`: the labels seen in fit, sorted.
- `estimators`, `featureImportances`: as for [`RandomForestRegressor`](./random-forest-regressor#returns).
- `oobScore`: the accuracy of the out-of-bag probabilities' most probable class, with `oobScore: true`.
- `oobDecisionFunction`: each row's class probabilities by the trees that didn't draw it. Not saved with the model.

## Errors

- `ML_PARAM`: as for [`RandomForestRegressor`](./random-forest-regressor#errors); a `classWeight` that names a class y
  doesn't have; `classWeight: 'balanced_subsample'`.
- `ML_LABELS`: labels that mix numbers and strings.
- `ML_VALUE`: as for [`RandomForestRegressor`](./random-forest-regressor#errors); `'balanced'` with a class whose rows
  all weigh 0.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

As for [`RandomForestRegressor`](./random-forest-regressor#notes): smaltml's own seeded samples, held to scikit-learn's
forests exactly without bootstrap, and by the mean score over several seeds otherwise, with class weights and sample
weights.

## Examples

```ts
import { RandomForestClassifier } from '@smaltworks/smaltml';

const X = [[1, 1], [2, 1], [1, 2], [6, 5], [7, 6], [6, 7], [1, 6], [2, 7]];
const y = ['low', 'low', 'low', 'high', 'high', 'high', 'mixed', 'mixed'];
const forest = new RandomForestClassifier({ nEstimators: 25, seed: 0 }).fitSync(X, y);
({ classes: forest.classes, predict: forest.predictSync([[1.5, 1.5], [6.5, 6], [1.5, 6.5]]) })
```

| classes | predict |
| --- | --- |
| high | low |
| low | high |
| mixed | mixed |

## Working demo

How sure the forest is about three new flowers:

```ts
const petals = { length: [1.4, 1.3, 1.5, 4.5, 4.1, 4.7, 6.0, 5.6, 5.8], width: [0.2, 0.2, 0.3, 1.5, 1.3, 1.4, 2.5, 2.2, 1.8] };
const species = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica'];
const flowers = await new RandomForestClassifier({ nEstimators: 50, seed: 0 }).fit(petals, species);
await flowers.predictProba({ length: [1.5, 4.4, 5.9], width: [0.3, 1.4, 2.1] })
```

| setosa | versicolor | virginica |
| --- | --- | --- |
| 0.98 | 0.02 | 0 |
| 0 | 0.98 | 0.02 |
| 0 | 0 | 1 |

## See also

[`RandomForestRegressor`](./random-forest-regressor), [`DecisionTreeClassifier`](../tree/decision-tree-classifier),
[`accuracyScore`](../metrics/accuracy-score)
