---
layout: home
hero:
  name: 'sm<span class="alt">alt</span><span class="suffix kit">ml</span>'
  text: 'Small. Alternative.<br>Predict.'
  tagline: Machine learning for smaltdb, on the hardware you have.
  image:
    light: /hardware-mark.svg
    dark: /hardware-mark-on-dark.svg
    alt: smaltml
  actions:
    - theme: brand
      text: Getting started
      link: /guide/getting-started
    - theme: alt
      text: Quick start
      link: /guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/stenhoor/smaltml
features:
  - title: Classic machine learning on tables
    details: Scaling, linear and logistic regression, pipelines and metrics first, with scikit-learn's parameters, defaults and results. Trees, boosting and clustering follow.
  - title: The hardware you have
    details: Every model runs on the CPU, in 64-bit floats, in the browser and in Node. WebGPU will speed up the heavy steps where the browser has it.
  - title: Models as JSON
    details: A fitted model, pipelines included, saves to a versioned JSON document and loads back the same, from a file or a row in a table.
  - title: Checked against Python
    details: Every model is compared with scikit-learn on the same data, and the results are kept as tests that run without Python.
---

<!-- docs:skip -->
```ts
import { LogisticRegression, StandardScaler, pipeline, accuracyScore } from '@smaltworks/smaltml';

const model = pipeline(new StandardScaler(), new LogisticRegression());
await model.fit(XTrain, yTrain);
accuracyScore(yTest, await model.predict(XTest));
```
