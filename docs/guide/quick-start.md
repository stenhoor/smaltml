---
description: 'Five minutes, three steps. Install smaltml, fit a model from your code, then change it and run it on this page.'
---

# Quick start

Five minutes, three steps: install smaltml, fit a model from your code, then change it and run it on this page.

## 1. Install

```sh
npm install --save-exact @smaltworks/smaltml@0.1.0
```

smaltml 0.1.0 is not on npm yet. [Installation](./getting-started) has the entry points and what else to know.

## 2. Fit a model from your code

<!-- docs:skip -->
```ts
import { LinearRegression } from '@smaltworks/smaltml';

// Flats: floor area in square metres and number of rooms, and the rent each one asks.
const X = [[48, 2], [62, 3], [75, 3], [90, 4], [110, 4], [130, 5]];
const rent = [1150, 1390, 1580, 1840, 2120, 2450];

const model = await new LinearRegression().fit(X, rent);
console.log(await model.predict([[80, 3]])); // Float64Array [ 1649.44… ]
```

`fit` learns from rows of numbers and the value each row should predict; `predict` gives a value for each new row.
Every model works this way.

## 3. Try it here

This example runs in your browser. Change it and press **Run** (or Ctrl+Enter). **Reset** brings back the example,
and **Share** copies a link to your version.

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const X = [[48, 2], [62, 3], [75, 3], [90, 4], [110, 4], [130, 5]];
const rent = [1150, 1390, 1580, 1840, 2120, 2450];

const model = await new LinearRegression().fit(X, rent);
await model.predict([[80, 3], [100, 4]])
```

| value |
| --- |
| 1649.44 |
| 1979 |

The table under the code is the tested result: the docs' tests run every example and check what it prints. The last
line's value is what a Run shows.


| coef | intercept |
| --- | --- |
| [14.212, 45.318] | 376.525 |
The examples on a page build on each other, as a notebook's cells do. This one uses the model fitted above:

```ts
({ coef: model.coef, intercept: model.intercept })
```

| coef | intercept |
| --- | --- |
| [14.212, 45.318] | 376.525 |

Each square metre adds the first coefficient to the rent, each room the second, from the intercept.

## Next

- [Installation](./getting-started): the package, its entry points and the data smaltml takes.
- [Training models](./training): linear and logistic regression, and pipelines.
- [Evaluating models](./evaluating): test sets and metrics.
