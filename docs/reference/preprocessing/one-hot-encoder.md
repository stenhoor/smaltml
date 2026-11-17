---
description: 'OneHotEncoder: one 0/1 column per category of each text or number column. scikit-learn OneHotEncoder.'
---

# OneHotEncoder

<!-- docs:skip -->
```ts
class OneHotEncoder implements Transformer {
  constructor(params?: {
    drop?: null | 'first' | 'if_binary';
    handleUnknown?: 'error' | 'ignore' | 'infrequent_if_exist';
    minFrequency?: number | null;
    maxCategories?: number | null;
  });
  fit(X: TableInput): Promise<this>;
  transform(X: TableInput): Promise<Matrix>;
  fitTransform(X: TableInput): Promise<Matrix>;
  getFeatureNamesOut(): string[];
  readonly categories: Category[][];
  readonly infrequentCategories: (Category[] | null)[];
  readonly dropIdx: (number | null)[];
}
type Category = string | number | null;
```

Turns each column into one 0/1 column per category, named `column_category`. scikit-learn's
[`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), with
dense output.

## Parameters

- `drop` (default `null`): leave out each column's first category (`'first'`), or the first only where a column has
  two categories (`'if_binary'`), so the indicators aren't redundant for a linear model. With infrequent categories,
  the first output column goes: the first frequent category's.
- `handleUnknown` (default `'error'`): what to do with a category `transform` didn't see in `fit`: fail, give it all
  zeros (`'ignore'`), or count it as infrequent where the column has infrequent categories, else all zeros
  (`'infrequent_if_exist'`).
- `minFrequency` (default none): a category seen fewer times than this is infrequent. A whole number of rows, 1 or
  more, or a share of the rows, above 0 and below 1 (`0.05`: fewer than 5% of them).
- `maxCategories` (default no limit): the most categories each column keeps, counting the one the infrequent
  categories become. Where there are more, the least frequent become infrequent too; of two seen as often, the one
  sorted first.

## Returns

`transform` returns a Matrix of 0s and 1s, its columns named as `getFeatureNamesOut` gives them. `categories` holds
each column's categories, sorted (numbers in order, text by code unit), with a missing value (`null`) last; infrequent ones too. `infrequentCategories` holds each column's infrequent
categories, which share one indicator column, last, named `column_infrequent_sklearn` as scikit-learn names it (or
`null` where a column has none). `dropIdx` is the index in `categories` of each column's dropped category.

## Errors

- `ML_LABELS`: an unseen category, with `handleUnknown: 'error'`.
- `ML_VALUE`: a column that mixes numbers and text, or holds text now and numbers when fitted.
- `ML_PARAM`: a `minFrequency` that isn't a whole number of 1 or more or a share between 0 and 1, or a
  `maxCategories` that isn't a whole number of 1 or more.
- `ML_PARAM`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

A missing value (`null`, or `NaN` in a column of numbers) is a category of its own, named `nan` (numbers) or `None`
(text), as scikit-learn names them. Column names write numbers as JavaScript does: `size_3`, where scikit-learn
writes `size_3.0` for a column of floats. The missing value counts as a category for `minFrequency` and
`maxCategories`, so it can be infrequent. The compare tests check which categories are infrequent, the output and
its names, and `dropIdx` against scikit-learn's, exactly.

## Examples

```ts
import { OneHotEncoder } from '@smaltworks/smaltml';

const enc = new OneHotEncoder().fitSync({ colour: ['red', 'blue', 'red', 'green'] });
enc.transformSync({ colour: ['green', 'red'] })
```

| colour_blue | colour_green | colour_red |
| --- | --- | --- |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

<!-- docs:error ML_LABELS -->
```ts
enc.transformSync({ colour: ['purple'] })
```

Rare categories, seen fewer than twice, share one column:

```ts
const animals = { pet: ['cat', 'cat', 'dog', 'dog', 'dog', 'eel', 'yak'] };
const rare = new OneHotEncoder({ minFrequency: 2 }).fitSync(animals);
({ names: rare.getFeatureNamesOut(), infrequent: rare.infrequentCategories[0] })
```

| names | infrequent |
| --- | --- |
| [pet_cat, pet_dog, pet_infrequent_sklearn] | [eel, yak] |

## Working demo

```ts
const sizes = new OneHotEncoder({ drop: 'if_binary', handleUnknown: 'ignore' })
  .fitSync({ size: ['S', 'M', 'L', 'M'], member: ['yes', 'no', 'no', 'yes'] });
sizes.transformSync({ size: ['M', 'XL'], member: ['yes', 'no'] })
```

| size_L | size_M | size_S | member_yes |
| --- | --- | --- | --- |
| 0 | 1 | 0 | 1 |
| 0 | 0 | 0 | 0 |

## See also

[`OrdinalEncoder`](./ordinal-encoder), [`ColumnTransformer`](../pipeline/column-transformer), [Preparing data: text columns](../../guide/preparing-data#text-columns)
