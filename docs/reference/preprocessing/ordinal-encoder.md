---
description: 'OrdinalEncoder: each column''s categories as numbers 0, 1, 2 … in sorted order. scikit-learn OrdinalEncoder.'
---

# OrdinalEncoder

<!-- docs:skip -->
```ts
class OrdinalEncoder implements Transformer {
  constructor(params?: {
    handleUnknown?: 'error' | 'use_encoded_value';
    unknownValue?: number;
    minFrequency?: number | null;
    maxCategories?: number | null;
  });
  fit(X: TableInput): Promise<this>;
  transform(X: TableInput): Promise<Matrix>;
  fitTransform(X: TableInput): Promise<Matrix>;
  readonly categories: Category[][];
  readonly infrequentCategories: (Category[] | null)[];
}
```

Turns each column's categories into numbers 0, 1, 2 … in sorted order. scikit-learn's
[`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

## Parameters

- `handleUnknown` (default `'error'`): what to do with a category `transform` didn't see in `fit`: fail, or give it
  `unknownValue` (`'use_encoded_value'`).
- `unknownValue`: the number for unknown categories, which `'use_encoded_value'` needs. `NaN` is allowed.
- `minFrequency` (default none): a category seen fewer times than this is infrequent. A whole number of rows, 1 or
  more, or a share of the rows, above 0 and below 1 (`0.05`: fewer than 5% of them).
- `maxCategories` (default no limit): the most categories each column keeps, counting the one the infrequent
  categories become. Where there are more, the least frequent become infrequent too; of two seen as often, the one
  sorted first.

## Returns

`transform` returns a Matrix with one column per input column, keeping their names. A missing value stays missing
(`NaN`). `categories` holds each column's categories, sorted. Without infrequent ones, a category's code is its place
there. With them, the frequent categories are coded 0, 1, 2 … in sorted order, and the infrequent ones (in
`infrequentCategories`, or `null` where a column has none) share the next code. The missing value is no category for
`minFrequency` and `maxCategories`: it stays `NaN`, as scikit-learn's `NaN` does.

## Errors

- `ML_LABELS`: an unseen category, with `handleUnknown: 'error'`.
- `ML_PARAM`: `'use_encoded_value'` without an `unknownValue`; `minFrequency` or `maxCategories` as for
  `OneHotEncoder`.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for `OneHotEncoder`.

## Examples

```ts
import { OrdinalEncoder } from '@smaltworks/smaltml';

new OrdinalEncoder().fitSync({ grade: ['B', 'A', 'C', 'A'] }).transformSync({ grade: ['A', 'C', null] })
```

| grade |
| --- |
| 0 |
| 2 |
| NaN |

## Working demo

```ts
const grades = new OrdinalEncoder({ handleUnknown: 'use_encoded_value', unknownValue: -1 }).fitSync({ grade: ['B', 'A', 'C'] });
({ categories: grades.categories[0], codes: grades.transformSync({ grade: ['C', 'F'] }).column(0) })
```

| categories | codes |
| --- | --- |
| [A, B, C] | [2, -1] |

## See also

[`OneHotEncoder`](./one-hot-encoder), [Preparing data: text columns](../../guide/preparing-data#text-columns)
