---
description: 'MinMaxScaler: scales each column to a range, [0, 1] by default. scikit-learn MinMaxScaler.'
---

# MinMaxScaler

<!-- docs:skip -->
```ts
class MinMaxScaler implements Transformer {
  constructor(params?: { featureRange?: [number, number]; clip?: boolean });
  fit(X: MatrixInput): Promise<this>;
  transform(X: MatrixInput): Promise<Matrix>;
  fitTransform(X: MatrixInput): Promise<Matrix>;
  inverseTransform(X: MatrixInput): Promise<Matrix>;
  readonly dataMin: Float64Array;
  readonly dataMax: Float64Array;
  readonly dataRange: Float64Array;
  readonly scale: Float64Array;
  readonly min: Float64Array;
}
```

Scales each column to a range: the fitted minimum goes to the range's low end and the maximum to its high end,
`z = x · scale + min`. scikit-learn's
[`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

## Parameters

- `featureRange` (default `[0, 1]`): the range, low end first.
- `clip` (default `false`): clip transformed values to the range, for rows beyond the fitted minimum and maximum.

## Returns

`transform` returns a Matrix with the same column names. After `fit`: `dataMin`, `dataMax` and `dataRange` per column;
`scale` and `min`, the multiplier and the offset. A constant column has a range of 0, and is scaled by 1 instead.

## Errors

- `ML_PARAM`: a range whose low end isn't below its high end.
- `ML_SHAPE`, `ML_VALUE` (an infinite value), `ML_NOT_FITTED`: as for every model.

## Notes

Missing values (`NaN`) are left out of the fit and stay missing. Results equal scikit-learn's to 1e-12 in the compare
tests.

## Examples

```ts
import { MinMaxScaler } from '@smaltworks/smaltml';

new MinMaxScaler().fitSync([[10], [15], [30]]).transformSync([[10], [20], [40]])
```

| 0 |
| --- |
| 0 |
| 0.5 |
| 1.5 |

With `clip`, the row beyond the fitted maximum stays at 1:

```ts
new MinMaxScaler({ clip: true }).fitSync([[10], [15], [30]]).transformSync([[10], [20], [40]])
```

| 0 |
| --- |
| 0 |
| 0.5 |
| 1 |

## Working demo

```ts
const temps = new MinMaxScaler({ featureRange: [-1, 1] }).fitSync({ celsius: [-5, 10, 25, 35] });
({ scaled: temps.transformSync({ celsius: [0, 30] }).column(0), back: temps.inverseTransformSync([[1]]).column(0) })
```

| scaled | back |
| --- | --- |
| [-0.75, 0.75] | [35] |

## See also

[`StandardScaler`](./standard-scaler), [`RobustScaler`](./robust-scaler), [Preparing data](../../guide/preparing-data)
