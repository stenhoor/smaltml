---
description: 'predict and predictProba: predictions from a model, or from the model saved under a name in a registry.'
---

# predict

<!-- docs:skip -->
```ts
function predict(model: Model | string, X: MatrixInput, options?: PredictOptions): Promise<Float64Array | Label[]>;
function predictProba(model: Model | string, X: MatrixInput, options?: PredictOptions): Promise<Matrix>;
interface PredictOptions { device?: 'auto' | 'cpu' | 'gpu'; registry?: ModelRegistry; version?: number }
```

The predictions of a model, or of the model saved under a name. `predictProba` gives a classifier's probabilities the
same way. From `@smaltworks/smaltml/registry`.

## Parameters

- `model`: a fitted model, or the name of one saved in a registry.
- `X`: the rows to predict, as rows, named columns or a `Matrix`.
- `registry`: where a name is looked up. Default: the registry `useRegistry` set, else one in memory.
- `version`: the version of a named model. Default: its latest.
- `device`: where the work runs ([The CPU, WebGPU and workers](../../guide/hardware)).

## Returns

What the model's own `predict` or `predictProba` returns: values for a regression, labels for a classifier, or a
Matrix of probabilities.

## Errors

- `ML_FORMAT`: no model is saved under the name (or at that version).
- `ML_PARAM`: the model doesn't predict, or doesn't give probabilities.
- The model's own errors, such as `ML_SHAPE` for the wrong number of columns.

## Examples

```ts
import { predict, ModelRegistry, LinearRegression } from '@smaltworks/smaltml';

const registry = new ModelRegistry();
await registry.save('rent', await new LinearRegression().fit([[48], [62], [75], [90]], [1150, 1390, 1580, 1840]));
await predict('rent', [[80]], { registry })
```

| value |
| --- |
| 1673.05 |

<!-- docs:error ML_FORMAT -->
```ts
await predict('missing', [[80]], { registry })
```

## Working demo

```ts
const fitted = await new LinearRegression().fit([[1], [2], [3]], [2, 4, 6]);
({ byModel: await predict(fitted, [[4]]), byName: await predict('rent', [[100]], { registry }) })
```

| byModel | byName |
| --- | --- |
| 8 | 1998.47 |

## See also

[`ModelRegistry`](./model-registry), [Predicting by name](../../guide/predicting)
