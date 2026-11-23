---
description: 'BoostedTreesRegressor: gradient-boosted regression trees, as XGBoost grows them. XGBoost XGBRegressor.'
---

# BoostedTreesRegressor

<!-- docs:skip -->
```ts
class BoostedTreesRegressor implements Regressor {
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
    objective?: 'reg:squarederror';
    subsample?: number;
    colsampleBytree?: number;
    colsampleBylevel?: number;
    colsampleBynode?: number;
    seed?: number;
  });
  fit(X: MatrixInput, y: VectorInput, options?: { sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: VectorInput, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Float64Array>;
  predictSync(X: MatrixInput): Float64Array;
  readonly trees: readonly BoostTree[];
  readonly baseScore: number;
  readonly featureImportances: Float64Array;
}
```

Gradient-boosted regression trees: each round grows a tree that corrects the predictions so far, and adds its leaves'
values, times the learning rate. The trees are grown as XGBoost's histogram method grows them: each column cut into
bins, each node split where the gradient sums say the loss drops most. XGBoost's
[`XGBRegressor`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor), with its
parameters and defaults ([ADR 0044](../../adr/0044-phase-3-boosted-trees)). From `@smaltworks/smaltml/boost`.

## Parameters

- `nEstimators` (default `100`): how many rounds, one tree each.
- `learningRate` (default `0.3`): each tree's leaf values are multiplied by it (XGBoost's `eta`).
- `maxDepth` (default `6`): the deepest a leaf may be; trees grow a level at a time.
- `minChildWeight` (default `1`): the smallest Hessian sum each child keeps; for squared error, the sum of its rows'
  weights.
- `gamma` (default `0`): the smallest gain a split must reach (XGBoost's `min_split_loss`), checked as the tree grows.
- `regLambda` (default `1`), `regAlpha` (default `0`): L2 and L1 regularization of the leaf values.
- `maxBin` (default `256`): the most bins a column is cut into. Each column's distinct values are its cuts, as
  XGBoost's are while a column has at most `maxBin` of them; a column with more fails, for now.
- `baseScore` (default: fitted): every row's starting prediction; fitted, it is the weighted mean of y.
- `objective` (default `'reg:squarederror'`): the loss; squared error only, so far.
- `subsample`, `colsampleBytree`, `colsampleBylevel`, `colsampleBynode` (default `1`): sampling of rows and columns;
  only 1 so far.
- `seed`: for the sampling, when it comes.

`fit` and `fitSync` take `sampleWeight`, one weight per row, 0 or more: a row's gradient and Hessian are multiplied by
it, and the base score is the weighted mean.

## Returns

- `predict`: each row's prediction, the base score plus each tree's leaf for the row.
- `trees`: the trees, one per round, as arrays: children (−1 at a leaf), column, cut (`x < cut` goes left), the side
  for missing values (`defaultLeft`, 1 for left), leaf value, gain and cover (the Hessian sum), as XGBoost's model
  holds them.
- `baseScore`: the starting prediction.
- `featureImportances`: each column's mean gain over the splits on it, normalized to sum to 1: XGBoost's
  `feature_importances_`.

## Errors

- `ML_PARAM`: a parameter outside its range; an `objective` other than squared error, or sampling below 1, which
  aren't supported yet.
- `ML_VALUE`: an infinite value in X (as in XGBoost); a column with more distinct values than `maxBin`; a negative `sampleWeight`, or every row of weight 0.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

A missing value (`NaN`, or `null` in a named column) needs no filling: each split learns which side its missing values
take, the one that gains more, as XGBoost's does (and right where the node has none). A column can also split its
missing values from its present ones. Prediction sends a missing value down the side each split learned.

X is read as 32-bit floats, as XGBoost reads it, so the cuts and comparisons are XGBoost's. smaltml then computes in
64-bit floats, where XGBoost keeps some values in 32-bit ones. The compare tests hold smaltml to XGBoost 3.4.1 on data
whose columns have at most `maxBin` distinct values: the same trees, split for split, and leaves, predictions,
importances and base score within 1e-5. Two splits of a node that send the same rows each way have the same gain; the
lower column wins, then the lower cut, as in XGBoost.

## Examples

```ts
import { BoostedTreesRegressor } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]];
const y = [1.1, 2.3, 2.9, 4.2, 5.1, 5.8, 7.2, 7.9];
const boosted = new BoostedTreesRegressor({ nEstimators: 30, maxDepth: 2, minChildWeight: 0 }).fitSync(X, y);
boosted.predictSync([[2, 1], [7, 0]])
```

| value |
| --- |
| 2.30664 |
| 7.19521 |

With missing values: each split learns where they go.

```ts
const gaps = [[1, 0], [2, NaN], [3, 0], [NaN, 1], [5, 0], [6, NaN], [7, 0], [NaN, 1]];
const fromGaps = new BoostedTreesRegressor({ nEstimators: 30, maxDepth: 2, minChildWeight: 0 }).fitSync(gaps, y);
fromGaps.predictSync([[NaN, 1], [7, NaN]])
```

| value |
| --- |
| 6.04558 |
| 7.17848 |

## Working demo

Which columns the trees lean on:

```ts
const rooms = [3, 4, 2, 5, 3, 4, 2, 6, 3, 5, 4, 2];
const area = [70, 95, 48, 130, 72, 101, 51, 160, 69, 122, 90, 45];
const floor = [1, 3, 2, 1, 4, 2, 1, 3, 2, 5, 1, 3];
const price = area.map((a, i) => 2.1 * a + 8 * rooms[i]! + (i % 3));
const homes = await new BoostedTreesRegressor({ nEstimators: 50, maxDepth: 2, minChildWeight: 0 }).fit({ rooms, area, floor }, price);
({ baseScore: homes.baseScore, importances: homes.featureImportances })
```

| baseScore | importances |
| --- | --- |
| 213.942 | [0.684418, 0.31546, 0.000121208] |

## See also

[`RandomForestRegressor`](../ensemble/random-forest-regressor), [`DecisionTreeRegressor`](../tree/decision-tree-regressor),
[`r2Score`](../metrics/r2-score)
