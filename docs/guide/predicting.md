---
description: 'Keep fitted models by name and version in a model registry, in memory or in a smaltdb table, and predict with them by name.'
---

# Predicting by name

A notebook or a pipeline fits a model; an app predicts with it later. A **model registry** keeps fitted models by
name, with a new version each time one is saved, and `predict` finds a model by its name.

## Saving a model by name

```ts
import { ModelRegistry, LogisticRegression, StandardScaler, pipeline } from '@smaltworks/smaltml';

// Customers: visits a month and monthly spend, and whether they renewed.
const X = [[2, 20], [6, 64], [9, 88], [3, 35], [1, 12], [8, 75], [10, 93], [4, 41]];
const renewed = ['no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'];

const registry = new ModelRegistry();
const churn = await pipeline(new StandardScaler(), new LogisticRegression()).fit(X, renewed);
await registry.save('churn', churn)
```

| value |
| --- |
| 1 |

`save` returns the version: 1 for the first model saved under a name, then 2, 3 and so on.

## Predicting by name

```ts
import { predict } from '@smaltworks/smaltml';

await predict('churn', [[7, 70], [2, 15]], { registry })
```

| value |
| --- |
| yes |
| no |

`predict` takes a model or a name. A name is looked up in the registry you pass, or in the default one, which
`useRegistry(registry)` sets. The registry loads a model once and keeps it, so repeated predictions don't decode it
again. `predictProba` does the same for probabilities, and `registry.predict(name, X)` is the same call.

## Versions

Saving under the same name adds a version. `predict` uses the latest unless you ask for one:

```ts
await registry.save('churn', await new LogisticRegression({ C: 0.05 }).fit(X, renewed));
(await registry.list()).map((e) => ({ name: e.name, version: e.version, model: e.model }))
```

| name | version | model |
| --- | --- | --- |
| churn | 1 | Pipeline |
| churn | 2 | LogisticRegression |

```ts
import { predictProba } from '@smaltworks/smaltml';

const latest = await predictProba('churn', [[7, 70]], { registry });
const first = await predictProba('churn', [[7, 70]], { registry, version: 1 });
({ latest: latest.row(0), first: first.row(0) })
```

| latest | first |
| --- | --- |
| 0.0418112 | 0.258542 |
| 0.958189 | 0.741458 |

## Keeping models in smaltdb

`new ModelRegistry()` keeps models in memory, for the page's life. A `TableStore` keeps them as rows of a table in a
SQL database, such as smaltdb with smaltql, so they last, and every page or notebook that opens the database finds
them:

<!-- docs:skip -->
```ts
import { open } from '@smaltworks/smaltql';
import { ModelRegistry, TableStore, useRegistry, predict } from '@smaltworks/smaltml';

const db = await open('analytics');
useRegistry(new ModelRegistry(new TableStore(db))); // the table smaltml_models, created when it isn't there

await predict('churn', customers);                   // the latest churn model, from the table
```

Each row holds a model's name, version, class, the smaltml version that saved it, when, and the saved document as
JSON. A SQL query can list them: `SELECT name, version, model, saved_at FROM smaltml_models`.

## See also

- [`predict`](../reference/registry/predict), [`ModelRegistry`](../reference/registry/model-registry),
  [`TableStore`](../reference/registry/table-store), [Saving and loading models](./saving)
