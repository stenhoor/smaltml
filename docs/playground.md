---
description: 'Try smaltml in your browser: fit a model, predict and score it, with nothing to install.'
---

# Playground

Fit a model, predict and score it, in your browser. Change the code and press **Run** (or Ctrl+Enter). The last
line's value is the result.

```ts
import { LogisticRegression, StandardScaler, pipeline, trainTestSplit, accuracyScore } from '@smaltworks/smaltml';

// Petal length and width of three iris species.
const X = [[1.4, 0.2], [1.3, 0.2], [1.5, 0.2], [1.7, 0.4], [1.4, 0.3], [1.6, 0.2],
           [4.7, 1.4], [4.5, 1.5], [4.9, 1.5], [4.0, 1.3], [4.6, 1.5], [3.9, 1.1],
           [6.0, 2.5], [5.1, 1.9], [5.9, 2.1], [5.6, 1.8], [5.8, 2.2], [6.6, 2.1]];
const y = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
           'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
           'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica'];

const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { testSize: 6, seed: 1 });
const model = await pipeline(new StandardScaler(), new LogisticRegression()).fit(XTrain, yTrain);
const predicted = await model.predict(XTest);
({ accuracy: accuracyScore(yTest, predicted), predicted, actual: yTest })
```

| accuracy | predicted | actual |
| --- | --- | --- |
| 0.833333 | [versicolor, virginica, virginica, versicolor, virginica, versicolor] | [versicolor, virginica, virginica, virginica, virginica, versicolor] |
