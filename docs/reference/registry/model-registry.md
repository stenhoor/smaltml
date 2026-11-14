---
description: 'ModelRegistry: fitted models kept by name and version, loaded once and predicted with by name.'
---

# ModelRegistry

<!-- docs:skip -->
```ts
class ModelRegistry {
  constructor(store?: ModelStore);
  save(name: string, model: Model): Promise<number>;
  load(name: string, version?: number): Promise<Model>;
  list(): Promise<ModelEntry[]>;
  delete(name: string, version?: number): Promise<number>;
  predict(name: string, X: MatrixInput, options?: PredictOptions): Promise<Float64Array | Label[]>;
  predictProba(name: string, X: MatrixInput, options?: PredictOptions): Promise<Matrix>;
  readonly store: ModelStore;
}
function useRegistry(registry: ModelRegistry): void;
function defaultRegistry(): ModelRegistry;
interface ModelEntry { name: string; version: number; model: string; savedAt: string }
```

Fitted models by name and version. From `@smaltworks/smaltml/registry`.

## Parameters

- `store`: where the models' documents are kept. Default: a [`MemoryStore`](./memory-store). A
  [`TableStore`](./table-store) keeps them in a database table.

## Returns

- `save` stores a fitted model as the next version of `name` and returns the version, from 1.
- `load` returns the model at `version`, or the latest. It decodes each document once and keeps the model.
- `list` returns every saved version, by name, then version, with the model's class and when it was saved.
- `delete` removes one version, or every version of a name, and returns how many it removed.
- `predict` and `predictProba` load a model by name and call it ([`predict`](./predict)).
- `useRegistry` sets the registry `predict` looks names up in; `defaultRegistry` returns it (one in memory until
  set).

## Errors

- `ML_NOT_FITTED`: `save` with a model that isn't fitted.
- `ML_FORMAT`: `load` with a name or version that isn't saved.
- `ML_PARAM`: an empty name.

## Examples

```ts
import { ModelRegistry, StandardScaler } from '@smaltworks/smaltml';

const registry = new ModelRegistry();
await registry.save('scale', await new StandardScaler().fit([[1], [2], [3]]));
await registry.save('scale', await new StandardScaler().fit([[10], [20], [30]]));
(await registry.list()).map((e) => ({ name: e.name, version: e.version, model: e.model }))
```

| name | version | model |
| --- | --- | --- |
| scale | 1 | StandardScaler |
| scale | 2 | StandardScaler |

## Working demo

```ts
const first = (await registry.load('scale', 1)) as StandardScaler;
const latest = (await registry.load('scale')) as StandardScaler;
({ first: first.mean, latest: latest.mean })
```

| first | latest |
| --- | --- |
| 2 | 20 |

## See also

[`predict`](./predict), [`TableStore`](./table-store), [Predicting by name](../../guide/predicting)
