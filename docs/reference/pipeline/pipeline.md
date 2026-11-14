---
description: 'pipeline: steps that run in order as one model, such as a scaler and then a classifier. scikit-learn make_pipeline.'
---

# pipeline

<!-- docs:skip -->
```ts
function pipeline(...steps: Step[]): Pipeline;
type Step = Model | readonly [name: string, model: Model];
class Pipeline implements Model {
  constructor(steps: readonly Step[]);
  fit(X: MatrixInput, y?: Labels): Promise<this>;
  predict(X: MatrixInput): Promise<Float64Array | Label[]>;
  predictProba(X: MatrixInput): Promise<Matrix>;
  transform(X: MatrixInput): Promise<Matrix>;
  step(name: string): Model;
  readonly steps: readonly (readonly [string, Model])[];
  readonly classes: readonly Label[];
}
```

Steps that run in order as one model. Every step but the last transforms `X` for the next; the last one predicts or
transforms. scikit-learn's
[`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) and
`Pipeline`.

## Parameters

- `steps`: the models, in order. A step is named after its class in lower case (`standardscaler`), or by the name
  you give it as `[name, model]`.

## Returns

A `Pipeline`. `fit` fits each step on what the steps before it give. `predict`, `predictProba` and `transform` run `X`
through the steps and call the last step's method. `step(name)` is the step with that name; `classes` is the last
step's, when it is a classifier.

## Errors

- `ML_PARAM`: no steps, two steps with one name, a step before the last that doesn't transform, a step name that
  doesn't exist, or a method the last step doesn't have.

## Notes

A pipeline saves and loads as one model, with every step's document in it.

## Examples

```ts
import { pipeline, StandardScaler, LinearRegression } from '@smaltworks/smaltml';

const p = pipeline(new StandardScaler(), ['model', new LinearRegression()]);
p.steps.map(([name, model]) => ({ name, kind: model.kind }))
```

| name | kind |
| --- | --- |
| standardscaler | StandardScaler |
| model | LinearRegression |

## Working demo

| prediction | coef |
| --- | --- |
| [44.8889] | [7.47426, 4.05361] |

```ts
await p.fit([[1, 100], [2, 300], [3, 200], [4, 500]], [10, 22, 26, 41]);
({ prediction: await p.predict([[5, 400]]), coef: (p.step('model') as LinearRegression).coef })
```

| prediction | coef |
| --- | --- |
| [44.8889] | [7.47426, 4.05361] |

## See also

[Training models: pipelines](../../guide/training#pipelines), [`StandardScaler`](../preprocessing/standard-scaler)
