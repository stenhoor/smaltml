---
description: 'ColumnTransformer: different steps on different columns, side by side, such as scaling numbers and encoding text. scikit-learn ColumnTransformer.'
---

# ColumnTransformer

<!-- docs:skip -->
```ts
class ColumnTransformer implements Transformer {
  constructor(parts: ColumnPart[], params?: { remainder?: 'drop' | 'passthrough' });
  fit(X: TableInput, y?: Labels): Promise<this>;
  transform(X: TableInput): Promise<Matrix>;
  fitTransform(X: TableInput, y?: Labels): Promise<Matrix>;
  getFeatureNamesOut(): string[];
  part(name: string): Transformer | 'passthrough' | 'drop';
}
type ColumnPart = [name: string, transformer: Transformer | 'passthrough' | 'drop', columns: string[] | number[]];
```

Runs a different transformer on different columns and puts their outputs side by side. scikit-learn's
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), with
dense output. From `@smaltworks/smaltml/pipeline`.

## Parameters

- `parts`: each a name, a transformer (or `'passthrough'` to keep the columns as they are, or `'drop'`), and its
  columns, by name or by position. A transformer can be a pipeline.
- `remainder` (default `'drop'`): what happens to the columns no part names: dropped, or kept (`'passthrough'`).

## Returns

`transform` returns a Matrix of the parts' outputs in order, then the remainder. Its columns are named `part__column`
(`getFeatureNamesOut`); the remainder's are `remainder__column`. Columns selected by position from data without names
are called `x0`, `x1` …, as scikit-learn calls them.

## Errors

- `ML_PARAM`: no parts, two parts with one name, a column that doesn't exist, or a name used on data without column
  names.
- `ML_VALUE`: text in a column that passes through: encode it.
- `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

It fits each part on its own columns. In a pipeline, the preparation is fitted on the training rows and saved with
the model. Results equal scikit-learn's to 1e-12 in the compare tests.

## Examples

```ts
import { ColumnTransformer, StandardScaler, OneHotEncoder } from '@smaltworks/smaltml';

const people = { age: [31, 45, 27], city: ['Leeds', 'York', 'Leeds'], id: [101, 102, 103] };
const ct = new ColumnTransformer([['age', new StandardScaler(), ['age']], ['city', new OneHotEncoder(), ['city']]]);
await ct.fitTransform(people)
```

| age__age | city__city_Leeds | city__city_York |
| --- | --- | --- |
| -0.431934 | 1 | 0 |
| 1.38219 | 0 | 1 |
| -0.950255 | 1 | 0 |

## Working demo

With `remainder: 'passthrough'`, the `id` column comes along:

```ts
const keep = new ColumnTransformer([['city', new OneHotEncoder(), ['city']]], { remainder: 'passthrough' });
await keep.fitTransform(people)
```

| city__city_Leeds | city__city_York | remainder__age | remainder__id |
| --- | --- | --- | --- |
| 1 | 0 | 31 | 101 |
| 0 | 1 | 45 | 102 |
| 1 | 0 | 27 | 103 |

## See also

[`pipeline`](./pipeline), [Preparing data: different columns](../../guide/preparing-data#different-columns-different-steps)
