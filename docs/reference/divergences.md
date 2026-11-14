---
description: 'Where smaltml differs from scikit-learn, and what scikit-learn has that smaltml does not yet.'
---

# Where smaltml differs

smaltml follows scikit-learn's models, parameters, defaults and results (ADR 0024). Where it differs, or doesn't
have something yet, it fails with an error rather than give a different answer. This page lists each case.

## Differences

| Where | scikit-learn | smaltml |
| --- | --- | --- |
| `LinearRegression` on columns that don't determine one answer | The minimum-norm solution | `ML_SINGULAR` |
| Seeded shuffles (`trainTestSplit`) | NumPy's random numbers | smaltml's own: a seed repeats itself, but not scikit-learn's split |
| `logLoss` with rows that don't sum to 1 | Warns | Uses them as given |
| Names | snake_case, fitted attributes end in `_` | camelCase, no underscore; `random_state` is `seed` |
| Labels | Any comparable values | Numbers or strings, not mixed |

## Not yet supported

Each of these fails with `ML_PARAM` or isn't a parameter yet:

- `LinearRegression`: several targets at once, `positive`, `sample_weight`.
- `LogisticRegression`: L1 and elastic-net penalties (`l1Ratio` other than 0), solvers other than `lbfgs`,
  `class_weight`, `sample_weight`, `warm_start`.
- `trainTestSplit`: `stratify`.
- `StandardScaler`: `partial_fit`, `sample_weight`.
- The other scalers, encoders, imputers, models and metrics of ADR 0024's later phases.
