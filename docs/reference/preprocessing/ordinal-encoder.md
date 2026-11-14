---
description: 'OrdinalEncoder: each column''s categories as numbers 0, 1, 2 … in sorted order. scikit-learn OrdinalEncoder.'
---

# OrdinalEncoder

<!-- docs:skip -->
```ts
class OrdinalEncoder implements Transformer {
  constructor(params?: { handleUnknown?: 'error' | 'use_encoded_value'; unknownValue?: number });
  fit(X: TableInput): Promise<this>;
  transform(X: TableInput): Promise<Matrix>;
  fitTransform(X: TableInput): Promise<Matrix>;
  readonly categories: Category[][];
}
```

Turns each column's categories into numbers 0, 1, 2 … in sorted order. scikit-learn's
[`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

## Parameters

- `handleUnknown` (default `'error'`): what to do with a category `transform` didn't see in `fit`: fail, or give it
  `unknownValue` (`'use_encoded_value'`).
- `unknownValue`: the number for unknown categories, which `'use_encoded_value'` needs. `NaN` is allowed.

## Returns

`transform` returns a Matrix with one column per input column, keeping their names. A missing value stays missing
(`NaN`). `categories` holds each column's categories in the order of their codes.

## Errors

- `ML_LABELS`: an unseen category, with `handleUnknown: 'error'`.
- `ML_PARAM`: `'use_encoded_value'` without an `unknownValue`.
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
