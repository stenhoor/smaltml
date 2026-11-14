<picture><source media="(prefers-color-scheme: dark)" srcset="docs/public/lockup-on-dark.svg"><img alt="smaltml" src="docs/public/lockup.svg" height="56"></picture>

<a href="https://smaltworks.dev/stability"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/public/work-in-progress-on-dark.svg"><img alt="Work in progress. Come build with us." src="docs/public/work-in-progress.svg" height="56"></picture></a>

**Small. Alternative. Predict.** Machine learning for smaltdb, on the hardware you have.

[Documentation](https://stenhoor.github.io/smaltml/) · [Playground](https://stenhoor.github.io/smaltml/playground) · [smaltworks.dev](https://smaltworks.dev)

**smaltml** trains models on tables and predicts with them, in TypeScript, in the browser and in Node. It follows
scikit-learn: the same models, parameters, defaults and results, checked against scikit-learn on the same data. Every
model runs on the CPU in 64-bit floats; WebGPU will speed up the heavy steps where the browser has it. Fitted models
save to JSON and load back the same.

> **Status: 0.1.0, work in progress.** [ADR 0024](docs/adr/0024-smaltml-scope-and-architecture.md)'s phase 1a and
> part of 1b: scalers (standard, min–max, robust), `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`,
> `ColumnTransformer`, `LinearRegression`, `LogisticRegression`, `trainTestSplit`, `pipeline`, four metrics, saving and
> loading models, predictions on WebGPU, a Web Worker for fits, and `predict()` by model name from a model registry,
> in memory or in a smaltdb table.

## Install

```sh
npm install --save-exact @smaltworks/smaltml@0.1.0 # not on npm yet
```

```ts
import { LogisticRegression, StandardScaler, pipeline, trainTestSplit, accuracyScore, saveModel } from '@smaltworks/smaltml';

const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { seed: 1 });
const model = await pipeline(new StandardScaler(), new LogisticRegression()).fit(XTrain, yTrain);
accuracyScore(yTest, await model.predict(XTest));
const json = saveModel(model); // loadModel(json) gives it back
```

## Develop

| Command | What it does |
| --- | --- |
| `npm ci` | Installs dependencies |
| `npm run check` | Typecheck, then the tests: the compare cases, every docs example, and the entry points |
| `npm run test:browser` | In headless Chromium: every playground of the site, then WebGPU and the worker against the CPU |
| `npm run compare` | Reruns the compare cases through scikit-learn (`pip install -r compare/requirements.txt`) |
| `npm run docs:dev` / `docs:build` | The documentation site |
| `npm run build` | `dist/` and type declarations |

<a href="https://smaltworks.dev"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/public/a-smaltworks-project-on-dark.svg"><img alt="A smaltworks project" src="docs/public/a-smaltworks-project.svg" height="28"></picture></a>
