---
description: 'DecisionTreeClassifier: a tree of yes-or-no questions about the columns, each leaf giving class shares. scikit-learn DecisionTreeClassifier.'
---

# DecisionTreeClassifier

<!-- docs:skip -->
```ts
class DecisionTreeClassifier implements Classifier {
  constructor(params?: {
    criterion?: 'gini' | 'entropy' | 'log_loss';
    classWeight?: 'balanced' | Record<string, number> | null;
    maxDepth?: number | null;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    minWeightFractionLeaf?: number;
    maxFeatures?: number | 'sqrt' | 'log2' | null;
    maxLeafNodes?: number | null;
    minImpurityDecrease?: number;
    seed?: number;
  });
  fit(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Label[]>;
  predictSync(X: MatrixInput): Label[];
  predictProba(X: MatrixInput): Promise<Matrix>;
  predictProbaSync(X: MatrixInput): Matrix;
  apply(X: MatrixInput): Int32Array;
  getDepth(): number;
  getNLeaves(): number;
  readonly classes: readonly Label[];
  readonly featureImportances: Float64Array;
  readonly tree: TreeArrays;
}
```

A tree of questions, each "is this column at most this value?", grown so each leaf's rows are as much of one class as
possible; a leaf gives its rows' class shares and predicts the largest. scikit-learn's
[`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html),
grown by CART ([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/tree`.

## Parameters

- `criterion` (default `'gini'`): what a split reduces: the Gini impurity, or the entropy in bits (`'entropy'`, and
  `'log_loss'`, the same).
- `classWeight`: weights for the classes. `'balanced'` gives each class n / (classes × its rows), counting rows (not
  their sample weights), as scikit-learn's trees do; an object gives a class's weight by its label.
- `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `minWeightFractionLeaf`, `maxFeatures`, `maxLeafNodes`,
  `minImpurityDecrease`, `seed`: as for [`DecisionTreeRegressor`](./decision-tree-regressor#parameters).

`fit` and `fitSync` take `sampleWeight`, one weight per row, multiplied by its class's `classWeight`.

## Returns

- `predict`: each row's most likely class (of equal shares, the first in `classes`).
- `predictProba`: each row's class shares in its leaf, a column per class in the order of `classes`.
- `apply`, `featureImportances`, `tree`, `getDepth()`, `getNLeaves()`: as for
  [`DecisionTreeRegressor`](./decision-tree-regressor#returns). `tree.value` holds each node's class shares, as
  scikit-learn's does.

## Errors

- `ML_PARAM`: a parameter outside its range; a `classWeight` that names a class y doesn't have.
- `ML_LABELS`: labels that mix numbers and strings.
- `ML_VALUE`: a missing value (`NaN`) in X, which trees don't take yet.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

As for [`DecisionTreeRegressor`](./decision-tree-regressor#notes): the compare tests hold the tree to scikit-learn's
node for node, with the Gini impurity and the entropy, class weights and sample weights, and best-first growth. Small
nodes tie more often, two columns splitting the same rows the same way; there the trees can differ.

## Examples

```ts
import { DecisionTreeClassifier } from '@smaltworks/smaltml';

const X = [[1, 1], [2, 1], [1, 2], [6, 5], [7, 6], [6, 7], [1, 6], [2, 7]];
const y = ['low', 'low', 'low', 'high', 'high', 'high', 'mixed', 'mixed'];
const tree = new DecisionTreeClassifier({ seed: 0 }).fitSync(X, y);
({ classes: tree.classes, predict: tree.predictSync([[1.5, 1.5], [6.5, 6], [1.5, 6.5]]) })
```

| classes | predict |
| --- | --- |
| high | low |
| low | high |
| mixed | mixed |

## Working demo

Which flowers are which, from two measurements, and how sure the tree is:

```ts
const petals = { length: [1.4, 1.3, 1.5, 4.5, 4.1, 4.7, 6.0, 5.6, 5.8], width: [0.2, 0.2, 0.3, 1.5, 1.3, 1.4, 2.5, 2.2, 1.8] };
const species = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica'];
const flowers = await new DecisionTreeClassifier({ maxDepth: 2, seed: 0 }).fit(petals, species);
await flowers.predictProba({ length: [1.5, 4.4, 5.9], width: [0.3, 1.4, 2.1] })
```

| setosa | versicolor | virginica |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

## See also

[`DecisionTreeRegressor`](./decision-tree-regressor), [`LogisticRegression`](../linear/logistic-regression),
[`accuracyScore`](../metrics/accuracy-score)
