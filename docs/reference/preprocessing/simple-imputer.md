---
description: 'SimpleImputer: fills missing values with each column''s mean, median, most frequent value or a constant. scikit-learn SimpleImputer.'
---

# SimpleImputer

<!-- docs:skip -->
```ts
class SimpleImputer implements Transformer {
  constructor(params?: { strategy?: 'mean' | 'median' | 'most_frequent' | 'constant'; fillValue?: number; keepEmptyFeatures?: boolean });
  fit(X: MatrixInput): Promise<this>;
  transform(X: MatrixInput): Promise<Matrix>;
  fitTransform(X: MatrixInput): Promise<Matrix>;
  readonly statistics: Float64Array;
}
```

Fills missing values (`NaN`, or `null` in named columns) in each column. scikit-learn's
[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), for numbers.

## Parameters

- `strategy` (default `'mean'`): fill with each column's `'mean'`, `'median'` or `'most_frequent'` value (the
  smallest of a tie), or a `'constant'`.
- `fillValue` (default `0`): the value for `'constant'`.
- `keepEmptyFeatures` (default `false`): keep a column with no values at all, filled with 0 (or `fillValue` for
  `'constant'`). Without it, such a column is dropped.

## Returns

`transform` returns a Matrix without the dropped columns, keeping the others' names. `statistics` holds each column's
fill value, `NaN` for a dropped column.

## Errors

- `ML_PARAM`: an unknown strategy, or a `fillValue` that isn't a finite number.
- `ML_SHAPE`, `ML_VALUE` (an infinite value), `ML_NOT_FITTED`: as for every model.

## Notes

Text columns aren't imputed here: `OneHotEncoder` makes a missing value a category of its own. Results equal
scikit-learn's to 1e-12 in the compare tests.

## Examples

```ts
import { SimpleImputer } from '@smaltworks/smaltml';

new SimpleImputer().fitSync([[1, NaN], [3, 10], [NaN, 20]]).transformSync([[NaN, NaN]])
```

| 0 | 1 |
| --- | --- |
| 2 | 15 |

A column with nothing to fill from is dropped:

```ts
await new SimpleImputer({ strategy: 'constant', fillValue: -1 }).fitTransform({ a: [1, NaN], empty: [NaN, NaN] })
```

| a |
| --- |
| 1 |
| -1 |

## Working demo

```ts
const readings = { temp: [21.5, NaN, 22.1, 21.9, NaN], humidity: [40, 42, NaN, 41, 39] };
const filled = new SimpleImputer({ strategy: 'median' }).fitSync(readings);
({ statistics: filled.statistics, temp: filled.transformSync(readings).column(0) })
```

| statistics | temp |
| --- | --- |
| [21.9, 40.5] | [21.5, 21.9, 22.1, 21.9, 21.9] |

## See also

[Preparing data: missing values](../../guide/preparing-data#missing-values), [`ColumnTransformer`](../pipeline/column-transformer)
