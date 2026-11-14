---
description: 'MemoryStore: a model store that keeps saved models in memory, for the life of the page.'
---

# MemoryStore

<!-- docs:skip -->
```ts
class MemoryStore implements ModelStore {
  put(name: string, doc: ModelDocument): Promise<number>;
  get(name: string, version?: number): Promise<ModelDocument | undefined>;
  list(): Promise<ModelEntry[]>;
  delete(name: string, version?: number): Promise<number>;
}
```

A model store in memory: for tests, docs, and apps that keep models somewhere of their own. `new ModelRegistry()`
uses one. From `@smaltworks/smaltml/registry`.

## Returns

The `ModelStore` methods: `put` stores a copy of a document as the next version and returns it; `get` returns a copy
of a version, or the latest, or `undefined`.

## Errors

- `ML_PARAM`: an empty name.

## Notes

Models last as long as the page or process. Implement `ModelStore` (`put`, `get`, `list`, `delete`) to keep them
anywhere else.

## Examples

```ts
import { MemoryStore, LinearRegression } from '@smaltworks/smaltml';

const store = new MemoryStore();
await store.put('line', (await new LinearRegression().fit([[1], [2]], [1, 2])).toJSON())
```

| value |
| --- |
| 1 |

## Working demo

```ts
(await store.get('line'))?.state
```

| coef | intercept | nFeaturesIn |
| --- | --- | --- |
| [1] | 0 | 1 |

## See also

[`ModelRegistry`](./model-registry), [`TableStore`](./table-store)
