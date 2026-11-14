---
description: 'StandardScaler: scales each column to mean 0 and standard deviation 1. scikit-learn StandardScaler.'
---

# StandardScaler

<!-- docs:skip -->
```ts
class StandardScaler implements Transformer {
  constructor(params?: { withMean?: boolean; withStd?: boolean });
  fit(X: MatrixInput): Promise<this>;
  transform(X: MatrixInput): Promise<Matrix>;
  fitTransform(X: MatrixInput): Promise<Matrix>;
  inverseTransform(X: MatrixInput): Promise<Matrix>;
  readonly mean: Float64Array | null;
  readonly var: Float64Array | null;
  readonly scale: Float64Array | null;
  readonly nSamplesSeen: number | Float64Array;
  readonly nFeaturesIn: number;
  readonly featureNamesIn: readonly string[] | undefined;
}
```

Scales each column to mean 0 and standard deviation 1: `z = (x − mean) / scale`. scikit-learn's
[`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

## Parameters

- `withMean` (default `true`): subtract each column's mean.
- `withStd` (default `true`): divide by each column's standard deviation.

## Returns

`transform` returns a new Matrix with the same column names. After `fit`:

- `mean`: each column's mean (`null` when `withMean` and `withStd` are both false).
- `var`: each column's population variance (`null` without `withStd`).
- `scale`: what each column is divided by, its standard deviation, or 1 where the column is constant (`null` without
  `withStd`).
- `nSamplesSeen`: the rows fitted on, or, when some values were missing, the count for each column.

## Errors

- `ML_NOT_FITTED`: `transform` before `fit`.
- `ML_SHAPE`: no rows, or a different number of columns from the fit.
- `ML_VALUE`: an infinite value.

## Notes

Missing values (`NaN`) are left out of the mean and variance, and stay missing in `transform`, as in scikit-learn.
The `*Sync` twins (`fitSync`, `transformSync`, `inverseTransformSync`) run on the calling thread.

## Examples

```ts
import { StandardScaler } from '@smaltworks/smaltml';

const scaler = new StandardScaler().fitSync([[0, 10], [2, 20], [4, 60]]);
scaler.transformSync([[2, 30]])
```

| 0 | 1 |
| --- | --- |
| 0 | 0 |

Without centring, only the scale changes:

| 0 | 1 |
| --- | --- |
| 1.22474 | 1.38873 |

```ts
new StandardScaler({ withMean: false }).fitSync([[0, 10], [2, 20], [4, 60]]).transformSync([[2, 30]])
```


| 0 | 1 |
| --- | --- |
| 1.22474 | 1.38873 |
## Working demo

```ts
const heights = { cm: [158, 171, 165, 180, 176], kg: [52, 70, 61, 85, 74] };
const s = new StandardScaler().fitSync(heights);
const z = s.transformSync(heights);
({ zMean: z.column(0).reduce((a, b) => a + b, 0) / z.rows, back: s.inverseTransformSync(z).column(0) })
```

| zMean | back |
| --- | --- |
| 0 | [158, 171, 165, 180, 176] |

## See also

[Preparing data](../../guide/preparing-data), [`pipeline`](../pipeline/pipeline)
