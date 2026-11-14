---
description: 'TableStore: a model store that keeps saved models as rows of a table in smaltdb, or any SQL database.'
---

# TableStore

<!-- docs:skip -->
```ts
class TableStore implements ModelStore {
  constructor(db: SqlDatabase, options?: { table?: string });
  readonly table: string;
}
interface SqlDatabase {
  query(sql: string, params?: unknown[]): Promise<readonly Record<string, unknown>[]>;
  exec(sql: string, params?: unknown[]): Promise<unknown>;
}
```

A model store that keeps models as rows of a table, so they last with the data, and every page or notebook that opens
the database finds them. From `@smaltworks/smaltml/registry`.

## Parameters

- `db`: a SQL database: smaltdb's `Database` with smaltql, or anything with `query` and `exec` that take `?`
  parameters and return rows as objects.
- `table` (default `smaltml_models`): the table, a plain name of letters, digits and `_`.

## Returns

The `ModelStore` methods, as [`MemoryStore`](./memory-store) has them. The table is created on first use when it isn't
there:

<!-- docs:skip -->
```sql
CREATE TABLE IF NOT EXISTS smaltml_models (
  name TEXT NOT NULL, version INTEGER NOT NULL, model TEXT NOT NULL,
  smaltml TEXT NOT NULL, saved_at TEXT NOT NULL, document TEXT NOT NULL,
  PRIMARY KEY (name, version)
)
```

## Errors

- `ML_PARAM`: a table name that isn't plain, or an empty model name.
- `ML_FORMAT`: a row whose document isn't JSON.
- The database's own errors.

## Notes

`document` holds the model as `saveModel` writes it; `model` and `smaltml` repeat its class and version so queries
can filter on them. Versions count up per name.

## Examples

<!-- docs:skip -->
```ts
import { open } from '@smaltworks/smaltql';
import { ModelRegistry, TableStore } from '@smaltworks/smaltml';

const registry = new ModelRegistry(new TableStore(await open('analytics')));
await registry.save('churn', model);
```

## See also

[`ModelRegistry`](./model-registry), [Predicting by name](../../guide/predicting)
