---
description: 'saveModel: a model as JSON text, its parameters and what it learned.'
---

# saveModel

<!-- docs:skip -->
```ts
function saveModel(model: Model): string;
```

The model as JSON text: a document with its class, parameters and, once fitted, what it learned.
`JSON.stringify(model)` gives the same, since every model has `toJSON()`.

## Parameters

- `model`: any smaltml model, fitted or not, pipelines included.

## Returns

JSON text: `{ format: 'smaltml', version: 4, model, smaltml, params, state }`. `state` is `null` before the model is
fitted. [Saving and loading models](../../guide/saving) describes each field.

## Errors

- `ML_VALUE`: a learned value that JSON can't hold (not finite).

## Examples

```ts
import { saveModel, StandardScaler } from '@smaltworks/smaltml';

saveModel(new StandardScaler({ withStd: false }))
```

| value |
| --- |
| \{"format":"smaltml","version":4,"model":"StandardScaler","smaltml":"0.1.0","params":\{"withMean":true,"withStd":false\},"state":null\} |

## Working demo


| mean | var | scale | nSamplesSeen | nFeaturesIn | featureNamesIn |
| --- | --- | --- | --- | --- | --- |
| [15] | [6] | [2.44949] | 3 | 1 | [temp] |
```ts
const fitted = new StandardScaler().fitSync({ temp: [12, 18, 15] });
JSON.parse(saveModel(fitted)).state
```

| mean | var | scale | nSamplesSeen | nFeaturesIn | featureNamesIn |
| --- | --- | --- | --- | --- | --- |
| [15] | [6] | [2.44949] | 3 | 1 | [temp] |

## See also

[`loadModel`](./load-model), [Saving and loading models](../../guide/saving)
