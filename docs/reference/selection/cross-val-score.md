---
description: 'crossValScore: a score per fold, from fresh copies of a model fitted on the other folds. scikit-learn cross_val_score.'
---

# crossValScore

<!-- docs:skip -->
```ts
function crossValScore(model: Model, X: TableInput, y: Labels, options?: { cv?: number | Splitter | Fold[]; scoring?: ScorerName | Scorer }): Promise<Float64Array>;
function cloneModel<M extends Model>(model: M, params?: Record<string, unknown>): M;
```

Fits a fresh copy of the model on each fold's training rows and scores it on the fold's test rows. scikit-learn's [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html). From `@smaltworks/smaltml/selection`.

## Parameters

- `model`: a model, fitted or not; it isn't changed. Each fold fits a copy (`cloneModel`).
- `cv` (default `5`): a number of folds (stratified for a classifier), a splitter such as `KFold`, or folds.
- `scoring`: a scorer's name, as scikit-learn names them (`accuracy`, `r2`, `neg_mean_squared_error`, `neg_root_mean_squared_error`, `neg_mean_absolute_error`, `neg_log_loss`, `roc_auc`, `f1`, `f1_macro`, `f1_micro`, `f1_weighted`, `precision`, `precision_macro`, `recall`, `recall_macro`), or a function `(model, X, y) => Promise<number>`. Default: accuracy for a classifier, R² otherwise. Error metrics are negated, so higher is always better.

## Returns

A Float64Array with one score per fold.

## Errors

- `ML_PARAM`: an unknown scorer, or a scorer the model can't serve (probabilities from a regressor).
- `ML_SHAPE`: y that doesn't match X. The model's own errors.

## Notes

`cloneModel(model, params)` is the copy it fits: unfitted, with parameters changed, as `{ C: 10 }` or `{ step__param: v }` for a step of a pipeline or a ColumnTransformer.

## Examples

```ts
import { crossValScore, KFold, LinearRegression } from '@smaltworks/smaltml';

const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];
await crossValScore(new LinearRegression(), X, y, { cv: new KFold({ nSplits: 4 }), scoring: 'neg_mean_absolute_error' })
```

| value |
| --- |
| -0.0942857 |
| -0.200429 |
| -0.183262 |
| -0.18 |

## Working demo

```ts
import { cloneModel, Ridge } from '@smaltworks/smaltml';

(cloneModel(new Ridge({ alpha: 1 }), { alpha: 5 }) as Ridge).alpha
```

| value |
| --- |
| 5 |

## See also

[`GridSearchCV`](./grid-search-cv), [`KFold`](./k-fold), [Evaluating models: cross-validation](../../guide/evaluating#cross-validation)
