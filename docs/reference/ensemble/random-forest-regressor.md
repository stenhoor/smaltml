---
description: 'RandomForestRegressor: many regression trees on samples of the rows and columns, their predictions averaged. scikit-learn RandomForestRegressor.'
---

# RandomForestRegressor

<!-- docs:skip -->
```ts
class RandomForestRegressor implements Regressor {
  constructor(params?: {
    nEstimators?: number;
    criterion?: 'squared_error' | 'friedman_mse';
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
  fit(X: MatrixInput, y: VectorInput, options?: { sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: VectorInput, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput): Promise<Float64Array>;
  predictSync(X: MatrixInput): Float64Array;
  readonly estimators: readonly DecisionTreeRegressor[];
  readonly featureImportances: Float64Array;
  readonly oobScore: number;
  readonly oobPrediction: Float64Array;
  readonly seed: number;
}
```

Many regression trees, each grown on a bootstrap sample of the rows (drawn with replacement) and, at each split, from
a random choice of the columns; a prediction is the trees' mean. Averaging trees that err in different ways gives a
steadier prediction than one deep tree. scikit-learn's
[`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/ensemble`.

## Parameters

- `nEstimators` (default 100): how many trees.
- `criterion` (default `'squared_error'`): as for [`DecisionTreeRegressor`](../tree/decision-tree-regressor#parameters).
- `bootstrap` (default `true`): each tree's rows are a sample drawn with replacement; `false` gives each tree every row.
- `maxSamples` (default `null`, as many draws as rows): how many rows each sample draws, a whole number (more than the
  rows is allowed) or a share above 0 and below 1 of the rows (with `sampleWeight`, of the weights' sum), rounded down
  and at least 1. Only with `bootstrap`.
- `oobScore` (default `false`): score the forest on each row with the trees whose sample left it out (out of bag), as
  R²: `oobScore` and `oobPrediction` after fit. Only with `bootstrap`.
- `maxFeatures` (default `null`, every column): how many columns each split looks at, as for the trees. A forest of
  every column differs between its trees only by their samples.
- `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `minWeightFractionLeaf`, `maxLeafNodes`, `minImpurityDecrease`:
  each tree's, as for [`DecisionTreeRegressor`](../tree/decision-tree-regressor#parameters).
- `seed`: the seed for the samples and each tree's columns. The same seed grows the same forest; without one, a new
  seed is drawn, and `seed` after fit says which.

`fit` and `fitSync` take `sampleWeight`, one weight per row, 0 or more. With `bootstrap`, a row's chance of being drawn
is in proportion to its weight, and each tree weighs a row by the times it was drawn; without, each tree weighs the
rows by `sampleWeight`. Both as scikit-learn 1.9 does.

## Returns

- `predict`: each row's prediction, the trees' mean.
- `estimators`: the fitted trees, in the order they were grown.
- `featureImportances`: the trees' importances, averaged; they sum to 1.
- `oobScore`: R² of `oobPrediction` against y, with `oobScore: true`.
- `oobPrediction`: each row's mean prediction by the trees that didn't draw it. Not saved with the model.

## Errors

- `ML_PARAM`: a parameter outside its range; `maxSamples` or `oobScore` without `bootstrap`; `oobScore` or
  `oobPrediction` of a forest fitted without `oobScore: true`.
- `ML_VALUE`: a missing value (`NaN`) in X; a negative `sampleWeight`, or every row of weight 0.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

A row that every tree drew has no out-of-bag prediction: it gets 0, counts in `oobScore` as such and the fit warns
(`ML_OOB`), as scikit-learn does. More trees make it rarer.

## Notes

smaltml draws its own samples from its own seeded generator, so a seeded forest repeats itself to the last bit but
isn't scikit-learn's forest for the same `random_state`. The compare tests hold it to scikit-learn's in two ways:
without bootstrap and with every column, where every tree is the one tree, to the last digits; and otherwise by the
mean score over several seeds, on held-out rows and out of bag, within 0.02, and within 0.005 on 2000 rows.

## Examples

```ts
import { RandomForestRegressor } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]];
const y = [1.1, 2.3, 2.9, 4.2, 5.1, 5.8, 7.2, 7.9];
const forest = new RandomForestRegressor({ nEstimators: 20, seed: 0 }).fitSync(X, y);
forest.predictSync([[2.5, 0], [6.5, 1]])
```

| value |
| --- |
| 2.42 |
| 5.905 |

## Working demo

Which columns the forest leans on, and how well it does on rows it didn't train on:

```ts
const rooms = [3, 4, 2, 5, 3, 4, 2, 6, 3, 5, 4, 2];
const area = [70, 95, 48, 130, 72, 101, 51, 160, 69, 122, 90, 45];
const floor = [1, 3, 2, 1, 4, 2, 1, 3, 2, 5, 1, 3];
const price = area.map((a, i) => 2.1 * a + 8 * rooms[i]! + (i % 3));
const homes = await new RandomForestRegressor({ nEstimators: 50, oobScore: true, seed: 1 }).fit({ rooms, area, floor }, price);
({ importances: homes.featureImportances, oobScore: homes.oobScore })
```

| importances | oobScore |
| --- | --- |
| [0.542105, 0.449431, 0.00846328] | 0.904232 |

## See also

[`RandomForestClassifier`](./random-forest-classifier), [`DecisionTreeRegressor`](../tree/decision-tree-regressor),
[`r2Score`](../metrics/r2-score)
