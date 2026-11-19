---
description: 'Install smaltml, choose the entry points you need, and pass it data: rows, named columns or a Matrix.'
---

# Installation

```sh
npm install --save-exact @smaltworks/smaltml@0.1.0
```

smaltml 0.1.0 is not on npm yet: it is a work in progress, and versions before 1.0 can change. Pin the exact version.
smaltml has no runtime dependencies. It is plain ES2022 modules and runs in every browser the smalt family supports,
in Web Workers and in Node 20 or later.

## Entry points

Import everything from `@smaltworks/smaltml`, or each area from its own entry point, so an app loads only what it
uses:

| Entry point | What it holds |
| --- | --- |
| `@smaltworks/smaltml` | Everything below |
| `@smaltworks/smaltml/core` | `Matrix`, `SmaltmlError`, `Random`, the model and table types, `useKernels`, `kernelsInUse`, `KERNELS_VERSION`, `VERSION` |
| `@smaltworks/smaltml/preprocessing` | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder` |
| `@smaltworks/smaltml/linear` | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression` |
| `@smaltworks/smaltml/tree` | `DecisionTreeRegressor`, `DecisionTreeClassifier` |
| `@smaltworks/smaltml/ensemble` | `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier` |
| `@smaltworks/smaltml/simd` | `useSimd`, `simdSupport`: the WebAssembly SIMD kernels, which smaltml turns on by itself where it can |
| `@smaltworks/smaltml/metrics` | `meanSquaredError`, `meanAbsoluteError`, `r2Score`, `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `rocAucScore`, `confusionMatrix`, `logLoss` |
| `@smaltworks/smaltml/selection` | `trainTestSplit`, `KFold`, `StratifiedKFold`, `GroupKFold`, `crossValScore`, `crossValidate`, `cloneModel`, `GridSearchCV` |
| `@smaltworks/smaltml/pipeline` | `pipeline`, `Pipeline`, `ColumnTransformer` |
| `@smaltworks/smaltml/multioutput` | `MultiOutputRegressor` |
| `@smaltworks/smaltml/io` | `saveModel`, `loadModel` |
| `@smaltworks/smaltml/registry` | `predict`, `predictProba`, `ModelRegistry`, `MemoryStore`, `TableStore`, `useRegistry` |
| `@smaltworks/smaltml/gpu` | `matMul`, `matMulSync`, `gpuAvailable`, `gpuDevice`, `GPU_MIN_WORK` |
| `@smaltworks/smaltml/parallel` | `configureThreads`, `stopThreads`, `sharedMemoryAvailable`, `PARALLEL_MIN_VALUES` |
| `@smaltworks/smaltml/threads-script` | The script the thread pool's workers run |
| `@smaltworks/smaltml/worker` | `startWorker`, `connect`, `serve` (not in the main entry point) |
| `@smaltworks/smaltml/worker-script` | The worker script `startWorker` runs |

## Data in

A model takes `X`, the rows to learn from, in any of three shapes, and `y`, one value or label per row:

```ts
import { Matrix } from '@smaltworks/smaltml';

const rows = [[31, 1200], [45, 800], [27, 1500]];             // rows of numbers
const columns = { age: [31, 45, 27], spend: [1200, 800, 1500] }; // named columns, as a query result has them
const X = Matrix.from(columns);
({ rows: X.rows, cols: X.cols, names: X.names })
```

| rows | cols | names |
| --- | --- | --- |
| 3 | 2 | [age, spend] |

Named columns keep their names: errors name the column, and a saved model records them. Columns of text are for the
encoders and `ColumnTransformer` ([Preparing data](./preparing-data#text-columns)); every other model takes numbers. `NaN` (or `null` in a column)
means a missing value. Labels for a classifier are numbers or strings.

smaltml holds data as a [`Matrix`](../reference/matrix): numbers in 64-bit floats, stored column by column.

## Asynchronous and synchronous calls

`fit`, `predict`, `predictProba` and `transform` return promises, so that work can move to WebGPU or a worker without
changing your code ([The CPU, WebGPU and workers](./hardware)). Each has a synchronous twin (`fitSync`, `predictSync` …) that runs on the calling thread, for
code that is already off the page's main thread, such as a notebook kernel or a test:

```ts
import { LinearRegression } from '@smaltworks/smaltml';

const m = new LinearRegression().fitSync([[1], [2], [3]], [2, 4, 6]);
m.predictSync([[4]])
```

| value |
| --- |
| 8 |

## Next

- [Preparing data](./preparing-data)
- [Training models](./training)
- [The reference](../reference/)
