---
description: 'For AI agents that write code with smaltml. How to use it, its rules, and where the complete agent guide is.'
---

# For AI agents

This page is written for AI agents that write code with smaltml, and it opens the agent guide. The site publishes
Markdown made for agents beside its pages:

- [`/agents.md`](https://stenhoor.github.io/smaltml/agents.md): this page, then the reference, in one file. Every
  example in it is tested, with its result.
- [`/llms.txt`](https://stenhoor.github.io/smaltml/llms.txt): an index of every page, each as Markdown.
- Any page as Markdown: its address with `.md`, such as
  [`guide/training.md`](https://stenhoor.github.io/smaltml/guide/training.md).

## What smaltml is

smaltml is machine learning on tables, in TypeScript: scikit-learn's models, parameters and results, for the browser
and Node. It has no runtime dependencies. The package is `@smaltworks/smaltml`, which is not on npm yet.

## How to use it

<!-- docs:skip -->
```ts
import { LogisticRegression, StandardScaler, pipeline, trainTestSplit, accuracyScore, saveModel, loadModel, SmaltmlError } from '@smaltworks/smaltml';

const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize: 0.25, seed: 1 });
const model = pipeline(new StandardScaler(), new LogisticRegression({ C: 1 }));
await model.fit(XTrain, yTrain);
const accuracy = accuracyScore(yTest, await model.predict(XTest));
const json = saveModel(model);   // a string; loadModel(json) gives the model back
try {
  model.predictSync([[1, 2, 3, 4, 5]]);
} catch (e) {
  if (!(e instanceof SmaltmlError && e.code === 'ML_SHAPE')) throw e;
}
```

## Rules

- **Names are scikit-learn's in camelCase:** `LinearRegression`, `fitIntercept`, `maxIter`, `predictProba`,
  `meanSquaredError`, `trainTestSplit`. Fitted attributes drop the trailing underscore: `coef_` is `coef`.
  `random_state` is `seed`.
- **Defaults are scikit-learn's,** and results agree with it within each model's stated tolerance.
- **`X` is rows of numbers, an object of named columns, or a `Matrix`.** Text columns are for `OneHotEncoder`,
  `OrdinalEncoder` and `ColumnTransformer` (`[['num', new StandardScaler(), ['age']], ['cat', new OneHotEncoder(), ['city']]]`),
  which give numbers. `y` is numbers (regression) or labels,
  numbers or strings (classification). `NaN` is a missing value; most models reject it with `ML_VALUE`.
- **Calls are asynchronous** (`await model.fit(X, y)`); `fitSync`, `predictSync` and the others are the synchronous
  twins. `predict` and `predictProba` take `{ device: 'auto' | 'cpu' | 'gpu' }`: the GPU computes in 32-bit floats, and
  `'cpu'` gives exactly the synchronous result. `startWorker()` (from `@smaltworks/smaltml/worker`) fits and predicts in
  a Web Worker and returns ordinary fitted models.
- **Models by name:** `registry.save('churn', model)` keeps a version; `predict('churn', X, { registry })` predicts with
  the latest (`version` picks one). `new ModelRegistry(new TableStore(db))` keeps them in a smaltdb table.
- **Tuning:** `new GridSearchCV(pipeline(new StandardScaler(), new Ridge()), { ridge__alpha: [0.1, 1, 10] }).fit(X, y)`,
  then `bestParams`, `bestScore`, `predict`. `crossValScore(model, X, y, { cv: 5, scoring: 'f1_macro' })` scores folds.
- **Errors are `SmaltmlError`s with a `code`:** `ML_SHAPE`, `ML_VALUE`, `ML_PARAM`, `ML_NOT_FITTED`, `ML_SINGULAR`,
  `ML_FORMAT`, `ML_LABELS` ([Errors](../reference/errors)).
- **What isn't supported fails with an error, never a wrong answer.** [Where smaltml
  differs](../reference/divergences) lists what scikit-learn has that smaltml doesn't yet.
- **Seeded randomness is smaltml's own:** `trainTestSplit` with a `seed` repeats itself, but doesn't reproduce
  scikit-learn's split for the same number.
