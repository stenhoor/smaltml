---
description: 'StandardScaler: scales each column to mean 0 and standard deviation 1. scikit-learn StandardScaler.'
---

# StandardScaler

<!-- docs:skip -->
```ts
class StandardScaler implements Transformer {
  constructor(params?: { withMean?: boolean; withStd?: boolean });
  fit(X: MatrixInput): Promise<this>;
  partialFit(X: MatrixInput): Promise<this>;
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

`transform` returns a new Matrix with the same column names. `partialFit` adds a batch of rows to the fit: the
statistics become those of every row seen so far, as if `fit` had had them all at once (within rounding), so data
that comes in batches, or doesn't fit in memory, can be scaled. Before any fit, it fits; `fit` always starts again.
After `fit` or `partialFit`:

- `mean`: each column's mean (`null` when `withMean` and `withStd` are both false).
- `var`: each column's population variance (`null` without `withStd`).
- `scale`: what each column is divided by, its standard deviation, or 1 where the column is constant (`null` without
  `withStd`).
- `nSamplesSeen`: the rows fitted on (over every batch), or, when some values were missing, the count for each column.

## Errors

- `ML_NOT_FITTED`: `transform` before `fit`.
- `ML_SHAPE`: no rows, or a different number of columns from the fit; for `partialFit`, columns named differently.
- `ML_VALUE`: an infinite value.

## Notes

Missing values (`NaN`) are left out of the mean and variance, and stay missing in `transform`, as in scikit-learn.
`partialFit` combines each batch's means and variances with the earlier ones by Chan, Golub and LeVeque's pairwise
update, column by column, so a column's missing values count in no batch. Where a column has no value in the first
batch, scikit-learn's `partial_fit` keeps it `NaN` from then on; smaltml gives the statistics of the values it has
seen ([Where smaltml differs](../divergences#differences)). The `*Sync` twins (`fitSync`, `partialFitSync`,
`transformSync`, `inverseTransformSync`) run on the calling thread.

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

```ts
new StandardScaler({ withMean: false }).fitSync([[0, 10], [2, 20], [4, 60]]).transformSync([[2, 30]])
```

| 0 | 1 |
| --- | --- |
| 1.22474 | 1.38873 |

Rows that come in batches: the scaler fitted batch by batch has the statistics of one fit on every row:

```ts
const batches = [[[1, 10], [2, 30]], [[4, 15], [7, 20], [11, 60]], [[3, 5]]];
const streamed = new StandardScaler();
for (const batch of batches) streamed.partialFitSync(batch);
({ mean: streamed.mean, nSamplesSeen: streamed.nSamplesSeen })
```

| mean | nSamplesSeen |
| --- | --- |
| [4.66667, 23.3333] | 6 |

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
