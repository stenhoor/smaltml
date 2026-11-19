---
description: 'ExtraTreesClassifier: many classification trees with random thresholds, their class probabilities averaged. scikit-learn ExtraTreesClassifier.'
---

# ExtraTreesClassifier

<!-- docs:skip -->
```ts
class ExtraTreesClassifier implements Classifier {
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

Extremely randomized trees for classes: many classification trees, each grown on every row and, at each split, from
a random choice of the columns, each column with one threshold drawn at random between the node's smallest and largest
value; the best of those splits is kept. Class probabilities are the trees' mean, and the prediction the most probable
class. scikit-learn's
[`ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
([ADR 0039](../../adr/0039-trees-and-forests)). From `@smaltworks/smaltml/ensemble`.

## Parameters

- `bootstrap` (default `false`): every tree sees every row; `true` gives each tree a sample drawn with replacement.
- `maxFeatures` (default `'sqrt'`): how many columns each split looks at, as for the trees.
- `criterion`, `classWeight`, `nEstimators`, `maxSamples`, `oobScore`, `seed`, and each tree's `maxDepth`,
  `minSamplesSplit`, `minSamplesLeaf`, `minWeightFractionLeaf`, `maxLeafNodes`, `minImpurityDecrease`: as for
  [`RandomForestClassifier`](./random-forest-classifier#parameters). `maxSamples` and `oobScore` need
  `bootstrap: true`. The `seed` also draws the thresholds.

The trees are [`DecisionTreeClassifier`](../tree/decision-tree-classifier)s with `splitter: 'random'`. `fit` and
`fitSync` take `sampleWeight`, multiplied by the row's class weight, and `fit` takes `threads`, as for
[`RandomForestClassifier`](./random-forest-classifier#parameters): the same forest as `fitSync` on any number of
threads.

## Returns

- `predict`, `predictProba`, `classes`, `estimators`, `featureImportances`, `oobScore`, `oobDecisionFunction`: as for
  [`RandomForestClassifier`](./random-forest-classifier#returns).

## Errors

- `ML_PARAM`: a parameter outside its range; `classWeight: 'balanced_subsample'`, or naming a label that isn't in y;
  `maxSamples` or `oobScore` without `bootstrap: true`; `oobScore` or `oobDecisionFunction` of a forest fitted without
  `oobScore: true`.
- `ML_VALUE`: a missing value (`NaN`) in X; a negative `sampleWeight`, or every row of weight 0.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

As for [`ExtraTreesRegressor`](./extra-trees-regressor#notes): smaltml's own seeded draws, held to scikit-learn's by
the mean accuracy over several seeds, on held-out rows and out of bag, within 0.02.

## Examples

```ts
import { ExtraTreesClassifier } from '@smaltworks/smaltml';

const X = [[1, 1], [2, 1], [1, 2], [6, 5], [7, 6], [6, 7], [1, 6], [2, 7]];
const y = ['low', 'low', 'low', 'high', 'high', 'high', 'mixed', 'mixed'];
const forest = new ExtraTreesClassifier({ nEstimators: 25, seed: 0 }).fitSync(X, y);
({ classes: forest.classes, predict: forest.predictSync([[1.5, 1.5], [6.5, 6], [1.5, 6.5]]) })
```

| classes | predict |
| --- | --- |
| high | low |
| low | high |
| mixed | mixed |

## Working demo

How sure the trees are about three new flowers:

```ts
const petals = { length: [1.4, 1.3, 1.5, 4.5, 4.1, 4.7, 6.0, 5.6, 5.8], width: [0.2, 0.2, 0.3, 1.5, 1.3, 1.4, 2.5, 2.2, 1.8] };
const species = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica'];
const flowers = await new ExtraTreesClassifier({ nEstimators: 50, seed: 0 }).fit(petals, species);
await flowers.predictProba({ length: [1.5, 4.4, 5.9], width: [0.3, 1.4, 2.1] })
```

| setosa | versicolor | virginica |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

## See also

[`ExtraTreesRegressor`](./extra-trees-regressor), [`RandomForestClassifier`](./random-forest-classifier),
[`DecisionTreeClassifier`](../tree/decision-tree-classifier)
