---
description: 'Save a fitted model as JSON, keep it in a file or a table, and load it back to predict.'
---

# Saving and loading models

A fitted model saves to a JSON document: its class, its parameters and what it learned. Keep it as a file, or as a
row in a table, and load it back to predict, in another page, a worker or a notebook.

```ts
import { LinearRegression, saveModel, loadModel } from '@smaltworks/smaltml';

const model = await new LinearRegression().fit({ area: [48, 62, 75, 90], rooms: [2, 3, 3, 4] }, [1150, 1390, 1580, 1840]);
const json = saveModel(model);
const doc = JSON.parse(json);
({ format: doc.format, version: doc.version, model: doc.model, params: doc.params, names: doc.state.featureNamesIn })
```

| format | version | model | params | names |
| --- | --- | --- | --- | --- |
| smaltml | 2 | LinearRegression | \{"fitIntercept":true\} | [area, rooms] |

`loadModel` turns the JSON (or the parsed document) back into the model, ready to predict:

```ts
const loaded = loadModel(json) as LinearRegression;
await loaded.predict([[80, 3]])
```

| value |
| --- |
| 1654.6 |

## What a saved model holds

- `format` and `version`: `smaltml` and the document's version. A later smaltml still loads an older document; a
  document newer than the smaltml reading it fails with `ML_FORMAT`. Version 2 (2026-11-15) adds `converged` to the
  state of `LogisticRegression`, `Ridge`, `Lasso` and `ElasticNet`; a version 1 document loads with `converged`
  null, not recorded.
- `model`: the class, such as `LinearRegression` or `Pipeline`. A pipeline holds each step's document.
- `smaltml`: the version that saved it.
- `params`: the parameters, as the constructor takes them.
- `state`: what fitting learned (`null` before it is fitted), such as `coef` and `intercept`, and the column names
  when the data had them.

## See also

- [Predicting by name](./predicting): a registry that keeps saved models by name and version, in memory or a table.
- [`saveModel`](../reference/io/save-model), [`loadModel`](../reference/io/load-model)
