---
description: 'Get data into shape for a model: rows or named columns, missing values, and scaling columns with StandardScaler.'
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

Fill or drop missing values before fitting. Imputers that fill them come in a later release.

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

Fit the scaler on the training rows only, then transform both the training and the test rows with it, so the test
rows tell you nothing the model wouldn't know. A [pipeline](./training#pipelines) does this for you.

## See also

- [`Matrix`](../reference/matrix), [`StandardScaler`](../reference/preprocessing/standard-scaler)
