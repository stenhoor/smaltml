---
description: 'DecisionTreeRegressor: a tree of yes-or-no questions about the columns, each leaf predicting a mean. scikit-learn DecisionTreeRegressor.'
---

# DecisionTreeRegressor

<!-- docs:skip -->
```ts
class DecisionTreeRegressor implements Predictor {
  constructor(params?: {
    criterion?: 'squared_error' | 'friedman_mse';
    maxDepth?: number | null;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    minWeightFractionLeaf?: number;
    maxFeatures?: number | 'sqrt' | 'log2' | null;
    maxLeafNodes?: number | null;
    minImpurityDecrease?: number;
    seed?: number;
  });
  fit(X: MatrixInput, y: ArrayLike<number>, options?: { sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: ArrayLike<number>, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Float64Array>;
  predictSync(X: MatrixInput): Float64Array;
  apply(X: MatrixInput): Int32Array;
  getDepth(): number;
  getNLeaves(): number;
  readonly featureImportances: Float64Array;
  readonly tree: TreeArrays;
}
```

A tree of questions, each "is this column at most this value?", grown so the rows in each leaf have values as close as
possible; a leaf predicts their mean. scikit-learn's
[`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html),
grown by CART ([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/tree`.

## Parameters

- `criterion` (default `'squared_error'`): what a split reduces, the squared error. `'friedman_mse'` is the same
  (scikit-learn 1.9 maps it to `'squared_error'`). `'absolute_error'` and `'poisson'` aren't supported yet.
- `maxDepth` (default no limit): the deepest a leaf may be; the root is depth 0.
- `minSamplesSplit` (default `2`): the fewest rows a node needs to be split: a whole number, or a share of the rows
  (above 0, at most 1).
- `minSamplesLeaf` (default `1`): the fewest rows each leaf keeps: a whole number, or a share of the rows (below 1).
- `minWeightFractionLeaf` (default `0`): the smallest share of the total weight each leaf keeps, up to 0.5.
- `maxFeatures` (default every column): how many columns each split looks at: a whole number, a share of the columns
  (below 1), `'sqrt'` or `'log2'`. Which ones is drawn from `seed`.
- `maxLeafNodes` (default no limit): the most leaves. The tree then grows best first: the split that improves most,
  next.
- `minImpurityDecrease` (default `0`): a split must decrease the weighted impurity by at least this much.
- `seed`: the order the columns are looked at, which decides between splits that score exactly the same, and which
  columns `maxFeatures` takes. The same seed grows the same tree.

`fit` and `fitSync` take `sampleWeight`, one weight per row.

## Returns

- `predict`: each row's leaf mean.
- `apply`: each row's leaf, by node index.
- `featureImportances`: each column's share of the total impurity decrease, summing to 1. scikit-learn's
  `feature_importances_`.
- `tree`: the nodes as arrays, as scikit-learn's `tree_` holds them: `left` and `right` (−1 at a leaf), `feature` and
  `threshold` (−2 at a leaf), `impurity`, `nSamples`, `weightedNSamples`, `value`.
- `getDepth()` and `getNLeaves()`.

## Errors

- `ML_PARAM`: a parameter outside its range; a criterion or `ccpAlpha` that isn't supported yet.
- `ML_VALUE`: a missing value (`NaN`) in X (trees don't take missing values yet: fill them first, with
  [`SimpleImputer`](../preprocessing/simple-imputer)), or an infinite one.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

A node is split on the column and threshold that reduce the weighted squared error most, trying the midpoint between
each two neighbouring values. It stays a leaf where it reaches `maxDepth`, has too few rows, is pure, or no split
reduces the impurity by `minImpurityDecrease`. Like scikit-learn, the tree reads X as 32-bit floats, so its thresholds
are scikit-learn's. On data where no two splits tie, the tree is scikit-learn's node for node: the compare tests check
every node's column, threshold, impurity, rows and value within 1e-10. Where splits tie, the column order (`seed`,
scikit-learn's `random_state`) decides, and smaltml's draws aren't scikit-learn's.

## Examples

```ts
import { DecisionTreeRegressor } from '@smaltworks/smaltml';

const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
const y = [1.1, 0.9, 1.0, 5.2, 4.8, 5.0, 9.1, 8.9];
const tree = new DecisionTreeRegressor({ maxDepth: 2, seed: 0 }).fitSync(X, y);
({ predict: tree.predictSync([[2.5], [5.5], [7.5]]), leaves: tree.getNLeaves() })
```

| predict | leaves |
| --- | --- |
| [0.95, 5, 9] | 4 |

<!-- docs:error ML_PARAM -->
```ts
new DecisionTreeRegressor({ criterion: 'poisson' })
```

## Working demo

A house's price from its size and age: the importances say which column the tree used most.

```ts
const houses = { size: [50, 62, 70, 85, 90, 110, 120, 140], age: [30, 5, 22, 40, 3, 15, 35, 8] };
const price = [150, 210, 190, 210, 300, 320, 300, 420];
const pricing = await new DecisionTreeRegressor({ maxDepth: 2, seed: 0 }).fit(houses, price);
({ importances: pricing.featureImportances, guess: await pricing.predict({ size: [100], age: [10] }) })
```

| importances | guess |
| --- | --- |
| [1, 0] | [306.667] |

## See also

[`DecisionTreeClassifier`](./decision-tree-classifier), [`LinearRegression`](../linear/linear-regression),
[`GridSearchCV`](../selection/grid-search-cv)
