---
description: 'RobustScaler: centres on the median and scales by the interquartile range, which outliers barely move. scikit-learn RobustScaler.'
---

# RobustScaler

<!-- docs:skip -->
```ts
class RobustScaler implements Transformer {
  constructor(params?: { withCentering?: boolean; withScaling?: boolean; quantileRange?: [number, number]; unitVariance?: boolean });
  fit(X: MatrixInput): Promise<this>;
  transform(X: MatrixInput): Promise<Matrix>;
  fitTransform(X: MatrixInput): Promise<Matrix>;
  inverseTransform(X: MatrixInput): Promise<Matrix>;
  readonly center: Float64Array | null;
  readonly scale: Float64Array | null;
}
```

Scales each column by statistics that a few extreme values barely move: `z = (x − median) / (q75 − q25)`.
scikit-learn's [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

## Parameters

- `withCentering` (default `true`): subtract each column's median.
- `withScaling` (default `true`): divide by each column's interquartile range.
- `quantileRange` (default `[25, 75]`): the two percentiles whose difference is the scale.
- `unitVariance` (default `false`): also divide by the normal distribution's spread over that range, so normally
  distributed data comes out with variance 1.

## Returns

`transform` returns a Matrix with the same column names. After `fit`: `center`, each column's median (`null` without
centring), and `scale` (`null` without scaling). A column whose range is 0 is scaled by 1.

## Errors

- `ML_PARAM`: percentiles outside 0 to 100, or the low one above the high one.
- `ML_SHAPE`, `ML_VALUE` (an infinite value), `ML_NOT_FITTED`: as for every model.

## Notes

Percentiles interpolate linearly between the two nearest values, as NumPy's do. Missing values are left out of the
fit and stay missing. Results equal scikit-learn's to 1e-12 in the compare tests.

## Examples

```ts
import { RobustScaler } from '@smaltworks/smaltml';

const r = new RobustScaler().fitSync([[1], [2], [3], [4], [100]]);
({ center: r.center, scale: r.scale })
```

| center | scale |
| --- | --- |
| 3 | 2 |

## Working demo

```ts
r.transformSync([[3], [100]])
```

| 0 |
| --- |
| 0 |
| 48.5 |

## See also

[`StandardScaler`](./standard-scaler), [`MinMaxScaler`](./min-max-scaler), [Preparing data](../../guide/preparing-data)
