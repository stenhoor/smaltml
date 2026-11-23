---
description: 'Get data into shape for a model: named columns, missing values, scaling, encoding text, and different steps for different columns.'
---

# Preparing data

Models learn from numbers. This page covers getting a table into the shape a model takes, and scaling its columns.

## From a query result to X and y

A columnar query result, from SQL, smaltdf or anything else, is an object of named columns. Take the columns the model
should learn from as `X`, and the one it should predict as `y`:

```ts
import { Matrix } from '@smaltworks/smaltml';

// Customers: age, visits a month, monthly spend, and whether they renewed.
const customers = {
  age:     [23, 35, 47, 52, 29, 41, 60, 33, 38, 26, 55, 44],
  visits:  [ 2,  6,  9,  3,  1,  8, 10,  4,  7,  2,  5,  9],
  spend:   [20, 64, 88, 35, 12, 75, 93, 41, 70, 18, 52, 80],
  renewed: ['no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes'],
};
const { renewed, ...features } = customers;
const X = Matrix.from(features);
X
```

| age | visits | spend |
| --- | --- | --- |
| 23 | 2 | 20 |
| 35 | 6 | 64 |
| 47 | 9 | 88 |
| 52 | 3 | 35 |
| 29 | 1 | 12 |
| 41 | 8 | 75 |
| 60 | 10 | 93 |
| 33 | 4 | 41 |
| 38 | 7 | 70 |
| 26 | 2 | 18 |
| 55 | 5 | 52 |
| 44 | 9 | 80 |

## Missing values

`NaN` marks a missing value; in named columns, `null` does too. Models that can't take a missing value refuse it with
`ML_VALUE` and name the column:

<!-- docs:error ML_VALUE -->
```ts
import { LinearRegression } from '@smaltworks/smaltml';

await new LinearRegression().fit({ age: [23, NaN, 47], visits: [2, 6, 9] }, [20, 64, 88])
```

Boosted trees ([`BoostedTreesRegressor`](../reference/boost/boosted-trees-regressor), 
[`BoostedTreesClassifier`](../reference/boost/boosted-trees-classifier)) take them as they are: each split learns which
side its missing values go, as XGBoost's do. For the other models, `SimpleImputer` fills them with each column's mean, median or most frequent value, or a constant:

```ts
import { SimpleImputer } from '@smaltworks/smaltml';

const imputer = new SimpleImputer({ strategy: 'median' });
await imputer.fitTransform({ age: [23, NaN, 47, 31], visits: [2, 6, NaN, 4] })
```

| age | visits |
| --- | --- |
| 23 | 2 |
| 31 | 6 |
| 47 | 4 |
| 31 | 4 |

## Scaling columns

Columns measured on different scales (age in years, spend in pounds) pull unevenly on models with a penalty, such as
logistic regression. `StandardScaler` moves each column to mean 0 and standard deviation 1:

```ts
import { StandardScaler } from '@smaltworks/smaltml';

const scaler = await new StandardScaler().fit(X);
const scaled = await scaler.transform(X);
({ mean: scaler.mean, scale: scaler.scale })
```

| mean | scale |
| --- | --- |
| 40.25 | 11.2481 |
| 5.5 | 2.98608 |
| 54 | 27.2336 |

`MinMaxScaler` scales each column to a range, [0, 1] by default; `RobustScaler` uses the median and the interquartile
range, which a few extreme values barely move:

```ts
import { MinMaxScaler, RobustScaler } from '@smaltworks/smaltml';

const spend = { spend: [12, 18, 20, 35, 41, 52, 64, 70, 75, 2400] };
const minmax = await new MinMaxScaler().fitTransform(spend);
const robust = await new RobustScaler().fitTransform(spend);
({ minmax: minmax.column(0), robust: robust.column(0) })
```

| minmax | robust |
| --- | --- |
| 0 | -0.77095 |
| 0.00251256 | -0.636872 |
| 0.00335008 | -0.592179 |
| 0.00963149 | -0.256983 |
| 0.0121441 | -0.122905 |
| 0.0167504 | 0.122905 |
| 0.0217755 | 0.391061 |
| 0.0242881 | 0.52514 |
| 0.0263819 | 0.636872 |
| 1 | 52.5922 |

One very large spend squeezes every other row towards 0 under `MinMaxScaler`; under `RobustScaler` the others keep
their spread.

Fit the scaler on the training rows only, then transform both the training and the test rows with it, so the test
rows tell you nothing the model wouldn't know. A [pipeline](./training#pipelines) does this for you.

## Text columns

Models take numbers, so a column of text needs encoding. `OneHotEncoder` makes one 0/1 column per category;
`OrdinalEncoder` numbers the categories 0, 1, 2 … in sorted order. A missing value is a category of its own for
`OneHotEncoder`, and stays missing for `OrdinalEncoder`:

```ts
import { OneHotEncoder, OrdinalEncoder } from '@smaltworks/smaltml';

const plans = { plan: ['basic', 'pro', 'basic', null, 'team'] };
await new OneHotEncoder().fitTransform(plans)
```

| plan_basic | plan_pro | plan_team | plan_None |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 0 | 0 | 0 | 1 |
| 0 | 0 | 1 | 0 |

```ts
await new OrdinalEncoder().fitTransform(plans)
```

| plan |
| --- |
| 0 |
| 1 |
| 0 |
| NaN |
| 2 |

A category `transform` didn't see in `fit` fails with `ML_LABELS`, unless `handleUnknown` says what to do with it
(all zeros for `OneHotEncoder`, a number of your choice for `OrdinalEncoder`).

## Different columns, different steps

`ColumnTransformer` runs a step on some columns and another step on others, and puts the results side by side. In a
pipeline, the whole preparation fits on the training rows and travels with the model:

```ts
import { ColumnTransformer, LogisticRegression, pipeline } from '@smaltworks/smaltml';

const accounts = {
  plan:   ['basic', 'pro', 'basic', 'team', 'pro', 'basic', 'team', 'pro'],
  seats:  [1, 5, 2, 20, 8, 1, 35, 3],
  spend:  [10, 90, 25, 400, 150, NaN, 610, 60],
};
const churned = ['yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'];

const prepare = new ColumnTransformer([
  ['numbers', pipeline(new SimpleImputer(), new StandardScaler()), ['seats', 'spend']],
  ['plan', new OneHotEncoder({ handleUnknown: 'ignore' }), ['plan']],
]);
const model = await pipeline(prepare, new LogisticRegression()).fit(accounts, churned);
prepare.getFeatureNamesOut()
```

| value |
| --- |
| numbers__seats |
| numbers__spend |
| plan__plan_basic |
| plan__plan_pro |
| plan__plan_team |

```ts
await model.predict({ plan: ['team', 'basic'], seats: [12, 1], spend: [300, 15] })
```

| value |
| --- |
| no |
| yes |

## See also

- [`Matrix`](../reference/matrix), [`StandardScaler`](../reference/preprocessing/standard-scaler),
  [`MinMaxScaler`](../reference/preprocessing/min-max-scaler), [`RobustScaler`](../reference/preprocessing/robust-scaler),
  [`SimpleImputer`](../reference/preprocessing/simple-imputer), [`OneHotEncoder`](../reference/preprocessing/one-hot-encoder),
  [`OrdinalEncoder`](../reference/preprocessing/ordinal-encoder), [`ColumnTransformer`](../reference/pipeline/column-transformer)
