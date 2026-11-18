---
description: 'Every error code smaltml raises: what it means and what to do.'
---

# Errors

smaltml's errors are `SmaltmlError`s: an `Error` with a `code` that says what went wrong, and a message that says
where (the column, the row, the parameter).

<!-- docs:skip -->
```ts
class SmaltmlError extends Error {
  readonly code: ErrorCode;
}
```

| Code | Meaning | What to do |
| --- | --- | --- |
| `ML_SHAPE` | The data's shape doesn't fit: rows of different lengths, `X` and `y` of different lengths, no rows, or a different number of columns from the fit. | Check the rows and columns you pass. |
| `ML_VALUE` | A missing (`NaN`) or infinite value where the model can't take one, or a value that isn't a number. | Fill or drop missing values; encode text as numbers. |
| `ML_PARAM` | A parameter out of range or of the wrong kind, or a method the model doesn't have. | The model's reference page lists its parameters. |
| `ML_NOT_FITTED` | `predict`, `transform` or another method that needs a fitted model, before `fit`. | Call `fit` first. |
| `ML_SINGULAR` | The data doesn't determine one answer, such as linear regression on a column that repeats others. | Drop the repeated column, or add rows. |
| `ML_FORMAT` | A saved model this smaltml can't read. | Save it again, or update smaltml. |
| `ML_LABELS` | Labels a model can't use: one class only, or numbers mixed with strings. | Check `y`. |

## Warnings

Where scikit-learn warns and goes on, smaltml warns on the console (`console.warn`), with a message that starts
`smaltml CODE:`, and the result stands.

| Code | Meaning | What to do |
| --- | --- | --- |
| `ML_OOB` | A forest fitted with `oobScore: true` had rows that every tree's bootstrap sample drew, so no tree could score them out of bag; their `oobPrediction` or `oobDecisionFunction` row is 0s and they count in `oobScore` as such, as scikit-learn does. | Use more trees (`nEstimators`). |
| `ML_CONVERGENCE` | An iterative fit (`LogisticRegression`, `Lasso`, `ElasticNet`) stopped at `maxIter` before it converged; its `converged` is `false`. scikit-learn's `ConvergenceWarning`. | Scale the columns (`StandardScaler`), raise `maxIter`, or for `Lasso` and `ElasticNet` raise `alpha`. |

## Examples

<!-- docs:error ML_NOT_FITTED -->
```ts
import { LinearRegression } from '@smaltworks/smaltml';

await new LinearRegression().predict([[1]])
```

```ts
import { LinearRegression, SmaltmlError } from '@smaltworks/smaltml';

let caught: unknown;
try {
  await new LinearRegression().predict([[1]]);
} catch (e) {
  caught = e;
}
const error = caught as SmaltmlError;
({ isSmaltmlError: error instanceof SmaltmlError, code: error.code, message: error.message })
```

| isSmaltmlError | code | message |
| --- | --- | --- |
| true | ML_NOT_FITTED | this LinearRegression isn't fitted yet: call fit() first |
