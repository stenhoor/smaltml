---
description: 'LogisticRegression: classes and their probabilities from a linear model, fitted by L-BFGS or Newton–Cholesky. scikit-learn LogisticRegression.'
---

# LogisticRegression

<!-- docs:skip -->
```ts
class LogisticRegression implements Classifier {
  constructor(params?: { C?: number; l1Ratio?: number; tol?: number; fitIntercept?: boolean; maxIter?: number; solver?: 'lbfgs' | 'newton-cholesky' | 'saga'; seed?: number; classWeight?: 'balanced' | Record<string, number> });
  fit(X: MatrixInput, y: Labels, options?: { threads?: number | 'auto'; device?: 'auto' | 'cpu' | 'gpu'; sampleWeight?: ArrayLike<number> }): Promise<this>;
  fitSync(X: MatrixInput, y: Labels, options?: { sampleWeight?: ArrayLike<number> }): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Label[]>;
  predictProba(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
  decisionFunction(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
  decisionFunctionSync(X: MatrixInput): Matrix;
  readonly classes: readonly Label[];
  readonly coef: Matrix;
  readonly intercept: Float64Array;
  readonly nIter: number;
  readonly converged: boolean | null;
}
type Label = number | string;
```

A linear model of each class's log-odds, fitted to minimize the log loss plus an L2 penalty. scikit-learn's
[`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
with the solvers `lbfgs` (the default) and `newton-cholesky`.

## Parameters

- `C` (default `1`): the inverse of the penalty's strength. The objective is the mean log loss plus
  `‖w‖² / (2·C·n)`; smaller `C` holds the coefficients closer to zero. `Infinity` turns the penalty off.
- `l1Ratio` (default `0`): the L1 share of the penalty, from `0` (all L2) to `1` (all L1, the lasso's penalty, which
  sets some coefficients to exactly 0); between, the elastic net. The penalty is `(1 − l1Ratio)·‖w‖²/2 + l1Ratio·‖w‖₁`,
  over `C·Σw`. Above 0 it needs `solver: 'saga'`, as in scikit-learn (which deprecated its `penalty` for `l1_ratio`).
- `tol` (default `1e-4`): stop when the largest component of the objective's gradient is at most this.
- `fitIntercept` (default `true`): fit an intercept, which is never penalized.
- `maxIter` (default `100`): the most iterations of the solver.
- `solver` (default `'lbfgs'`): how the fit is solved. `'lbfgs'` works from the gradient alone: many cheap
  iterations. `'newton-cholesky'` takes Newton steps with the exact Hessian, solved by Cholesky: a few costly ones,
  usually the faster fit with many rows, up to a few hundred columns and few classes: its Hessian has a block for
  each pair of classes, so each iteration costs about as many passes over X as there are pairs. It runs on the CPU
  (on several threads, as `lbfgs` does). `'saga'` (SAGA, Defazio, Bach and Lacoste-Julien, 2014) takes one row at a
  time in a random order, with a proximal step for the L1 part: the solver for an L1 or elastic-net penalty. Each of
  its iterations is one pass over the rows; it converges fastest on scaled columns (`StandardScaler`), and runs on this
  thread, on the CPU.
- `seed` (default: a new one each fit): the seed for `'saga'`'s order of rows, scikit-learn's `random_state`. The same
  seed gives the same fit.
- `classWeight` (default: every class 1): weights for the classes, scikit-learn's `class_weight`. `'balanced'` gives
  each class n / (classes × its rows), counting rows by their sample weights, so every class counts as much in all: the
  usual remedy when one class is rare. An object gives a class's weight by its label, `{ yes: 5 }`; classes it leaves
  out weigh 1.

`fit` takes `threads` ([On several cores](../../guide/hardware#on-several-cores)) and `device`: `'gpu'` runs the loss
and its gradient on the GPU in 32-bit floats, for a fit close to the CPU's ([Fits on the
GPU](../../guide/hardware#fits-on-the-gpu)); `'auto'` (the default) and `'cpu'` fit on the CPU. `fit` and `fitSync`
take `sampleWeight`, one weight per row: each row's loss counts that many times, multiplied by its class's
`classWeight`. The loss is then the weighted mean, and the penalty `‖w‖² / (2·C·Σw)`, as in scikit-learn.

## Returns

- `predict`: the most probable class for each row.
- `predictProba`: a Matrix with a column per class, in the order of `classes`; each row sums to 1.
- `decisionFunction` and `decisionFunctionSync`: each row's score, `X·wᵀ + b`. The asynchronous calls take `device`,
  which chooses where the product runs ([The CPU, WebGPU and workers](../../guide/hardware)).
- `classes`: the labels seen in `fit`, sorted.
- `coef`: one row for two classes (the log-odds of the second class), else one row per class; one column per feature.
- `intercept`: one value per row of `coef`. `nIter`: the iterations the fit took.
- `converged`: whether the solver converged within `maxIter`. Where it didn't, `fit` warns on the console
  (`ML_CONVERGENCE`, [Errors](../errors#warnings)), as scikit-learn raises a `ConvergenceWarning`. `null` for a model
  saved by smaltml before it recorded this.

## Errors

- `ML_LABELS`: one class only, or labels that mix numbers and strings.
- `ML_PARAM`: `C` ≤ 0, `l1Ratio` outside 0 to 1, or above 0 with a solver other than `'saga'`, `tol` ≤ 0, `maxIter` below 1, a `solver` it doesn't know,
  `device: 'gpu'` where there is no WebGPU, or with weights (`sampleWeight`, `classWeight`), `'newton-cholesky'` or
  `'saga'` with `device: 'gpu'`, a `seed` that isn't a whole number, a `classWeight` that isn't `'balanced'` or an object of weights 0 or
  more, or one that names a class y doesn't have.
- `ML_SHAPE`, `ML_VALUE`: `sampleWeight` of the wrong length, or with a negative or non-finite weight; every row
  weighing 0.
- `ML_VALUE`, `ML_SHAPE`, `ML_NOT_FITTED`: as for every model.

## Notes

Two classes fit one row of coefficients; more classes fit the multinomial (softmax) model, as scikit-learn does. Run
to convergence, the coefficients agree with scikit-learn's within 1e-5 (relative) in the compare tests; with the
default `tol` both stop early, and agree within about 2e-3 there. `newton-cholesky` stops as scikit-learn's does, when
the largest gradient component and half the Newton decrement are both at most `tol`: it takes the same number of
iterations and agrees within 1e-6, at the default `tol` as well, with two classes or more. With more than two, the
classes' coefficients and intercepts each sum to 0, as scikit-learn's do. With `lbfgs` on data with unscaled columns, the
default `tol` stops further from the minimum, and fits differ more: by up to 1.5e-2 in probability from
scikit-learn, and up to 2.0e-2 between JavaScript engines, on the benchmark data
([Where smaltml differs](../divergences#between-javascript-engines)). Scale the columns first (`StandardScaler`), or
lower `tol`.

`saga` stops as scikit-learn's does: after a pass over the rows in which no coefficient or intercept changes by more
than `tol` times the largest of them. Its row order is smaltml's own, so a seed doesn't give scikit-learn's path;
run to a tight `tol`, the two land on the one optimum: within 1e-5 in the compare tests, with the same coefficients at
exactly 0.

## Examples

```ts
import { LogisticRegression } from '@smaltworks/smaltml';

const clf = await new LogisticRegression().fit([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 1, 1, 1]);
await clf.predictProba([[1], [2.5], [4]])
```

| 0 | 1 |
| --- | --- |
| 0.842976 | 0.157024 |
| 0.499938 | 0.500062 |
| 0.156959 | 0.843041 |

The same fit by Newton–Cholesky, in a few iterations:

```ts
const newton = await new LogisticRegression({ solver: 'newton-cholesky' }).fit([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 1, 1, 1]);
newton.nIter
```

| value |
| --- |
| 4 |

Stopped before it converges, a fit warns `ML_CONVERGENCE` on the console and says so in `converged`:

```ts
const early = await new LogisticRegression({ maxIter: 2 }).fit([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 1, 1, 1]);
({ nIter: early.nIter, converged: early.converged })
```

| nIter | converged |
| --- | --- |
| 2 | false |

An L1 penalty, with `solver: 'saga'`, sets the coefficients of columns that don't help to exactly 0. Here only the
first column decides the class:

```ts
const Xl = [[0, 1, 0.2], [1, 0, 0.9], [2, 1, 0.4], [3, 0, 0.1], [4, 1, 0.8], [5, 0, 0.3], [6, 1, 0.6], [7, 0, 0.5]];
const yl = [0, 0, 0, 0, 1, 1, 1, 1];
new LogisticRegression({ solver: 'saga', l1Ratio: 1, C: 0.5, seed: 0, maxIter: 1000 }).fitSync(Xl, yl).coef
```

| 0 | 1 | 2 |
| --- | --- | --- |
| 0.815177 | 0 | 0 |

One class rare: nine rows of `no` to three of `yes`. `classWeight: 'balanced'` makes each class count as much in all,
so the model leans less towards the common one:

```ts
const Xr = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]];
const yr = ['no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'];
const plain = new LogisticRegression().fitSync(Xr, yr);
const balanced = new LogisticRegression({ classWeight: 'balanced' }).fitSync(Xr, yr);
({ plain: plain.predictSync([[7.5]]), balanced: balanced.predictSync([[7.5]]) })
```

| plain | balanced |
| --- | --- |
| no | yes |

<!-- docs:error ML_LABELS -->
```ts
await new LogisticRegression().fit([[0], [1]], ['yes', 'yes'])
```

## Working demo

Three species from two measurements:


| value |
| --- |
| setosa |
| versicolor |
| virginica |
```ts
const petals = [[1.4, 0.2], [1.3, 0.2], [1.5, 0.3], [4.5, 1.5], [4.1, 1.3], [4.7, 1.4], [6.0, 2.5], [5.6, 2.2], [5.8, 1.8]];
const species = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica'];
const iris = await new LogisticRegression({ C: 10 }).fit(petals, species);
await iris.predict([[1.6, 0.4], [4.4, 1.4], [5.9, 2.1]])
```

| value |
| --- |
| setosa |
| versicolor |
| virginica |

## See also

[Training models](../../guide/training), [`accuracyScore`](../metrics/accuracy-score), [`logLoss`](../metrics/log-loss)
