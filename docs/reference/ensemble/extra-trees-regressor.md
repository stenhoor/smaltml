---
description: 'ExtraTreesRegressor: many regression trees with random thresholds, their predictions averaged. scikit-learn ExtraTreesRegressor.'
---

# ExtraTreesRegressor

<!-- docs:skip -->
```ts
class ExtraTreesRegressor implements Regressor {
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
  fit(X: MatrixInput, y: VectorInput, options?: { sampleWeight?: ArrayLike<number>; threads?: number | 'auto' }): Promise<this>;
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

Extremely randomized trees: many regression trees, each grown on every row and, at each split, from a random choice
of the columns, each column with one threshold drawn at random between the node's smallest and largest value; the
best of those splits is kept. A prediction is the trees' mean. Random thresholds make the trees differ more than a
random forest's, so their mean is often steadier, and a split is cheaper to find. scikit-learn's
[`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/ensemble`.

## Parameters

- `bootstrap` (default `false`): every tree sees every row; `true` gives each tree a sample drawn with replacement, as
  a random forest's.
- `nEstimators`, `criterion`, `maxSamples`, `oobScore`, `maxFeatures` (default `null`, every column), `seed`, and each
  tree's `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `minWeightFractionLeaf`, `maxLeafNodes`,
  `minImpurityDecrease`: as for [`RandomForestRegressor`](./random-forest-regressor#parameters). `maxSamples` and
  `oobScore` need `bootstrap: true`. The `seed` also draws the thresholds.

The trees are [`DecisionTreeRegressor`](../tree/decision-tree-regressor)s with `splitter: 'random'`. `fit` and
`fitSync` take `sampleWeight`, and `fit` takes `threads`, as for
[`RandomForestRegressor`](./random-forest-regressor#parameters): the same forest as `fitSync` on any number of threads.

## Returns

- `predict`: each row's prediction, the trees' mean.
- `estimators`, `featureImportances`, `oobScore`, `oobPrediction`: as for
  [`RandomForestRegressor`](./random-forest-regressor#returns).

## Errors

- `ML_PARAM`: a parameter outside its range; `maxSamples` or `oobScore` without `bootstrap: true`; `oobScore` or
  `oobPrediction` of a forest fitted without `oobScore: true`.
- `ML_VALUE`: a missing value (`NaN`) in X; a negative `sampleWeight`, or every row of weight 0.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

smaltml draws its own thresholds and columns from its own seeded generator, so a seeded forest repeats itself to the
last bit but isn't scikit-learn's forest for the same `random_state`. The compare tests hold it to scikit-learn's by
the mean score over several seeds, on held-out rows and out of bag, within 0.02, and within 0.005 on 2000 rows.

## Examples

```ts
import { ExtraTreesRegressor } from '@smaltworks/smaltml';

const X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]];
const y = [1.1, 2.3, 2.9, 4.2, 5.1, 5.8, 7.2, 7.9];
const forest = new ExtraTreesRegressor({ nEstimators: 20, seed: 0 }).fitSync(X, y);
forest.predictSync([[2.5, 0], [6.5, 1]])
```

| value |
| --- |
| 2.63 |
| 6.395 |

## Working demo

Which columns the trees lean on, and how well they do on rows they didn't train on:

```ts
const rooms = [3, 4, 2, 5, 3, 4, 2, 6, 3, 5, 4, 2];
const area = [70, 95, 48, 130, 72, 101, 51, 160, 69, 122, 90, 45];
const floor = [1, 3, 2, 1, 4, 2, 1, 3, 2, 5, 1, 3];
const price = area.map((a, i) => 2.1 * a + 8 * rooms[i]! + (i % 3));
const homes = await new ExtraTreesRegressor({ nEstimators: 50, bootstrap: true, oobScore: true, seed: 1 }).fit({ rooms, area, floor }, price);
({ importances: homes.featureImportances, oobScore: homes.oobScore })
```

| importances | oobScore |
| --- | --- |
| [0.524026, 0.450057, 0.0259168] | 0.875429 |

## See also

[`ExtraTreesClassifier`](./extra-trees-classifier), [`RandomForestRegressor`](./random-forest-regressor),
[`DecisionTreeRegressor`](../tree/decision-tree-regressor)
