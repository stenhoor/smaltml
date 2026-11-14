---
description: 'loadModel: a model from the JSON that saveModel wrote, ready to predict.'
---

# loadModel

<!-- docs:skip -->
```ts
function loadModel(json: string | ModelDocument): Model;
```

A model from the JSON text `saveModel` wrote, or from its parsed document. A fitted model comes back fitted, ready to
predict.

## Parameters

- `json`: the JSON text, or the document it parses to.

## Returns

The model, as a `Model`. Cast it to its class (`as LinearRegression`) to reach its own methods and attributes.

## Errors

- `ML_FORMAT`: not JSON, not a smaltml document, a document version newer than this smaltml reads, or a model class it
  doesn't know.

## Notes

A later smaltml still loads a document an earlier one saved.

## Examples

```ts
import { loadModel, LinearRegression } from '@smaltworks/smaltml';

const doc = {
  format: 'smaltml', version: 1, model: 'LinearRegression', smaltml: '0.1.0',
  params: { fitIntercept: true },
  state: { coef: [2], intercept: 1, nFeaturesIn: 1 },
};
await (loadModel(doc) as LinearRegression).predict([[10]])
```

| value |
| --- |
| 21 |

<!-- docs:error ML_FORMAT -->
```ts
loadModel({ ...doc, version: 2 })
```

## Working demo

```ts
const original = await new LinearRegression().fit([[1], [2], [3]], [1, 4, 9]);
const copy = loadModel(JSON.stringify(original)) as LinearRegression;
({ original: await original.predict([[4]]), copy: await copy.predict([[4]]) })
```

| original | copy |
| --- | --- |
| 12.6667 | 12.6667 |

## See also

[`saveModel`](./save-model), [Saving and loading models](../../guide/saving)
