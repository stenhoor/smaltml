# ADR 0044: Phase 3, gradient-boosted trees with XGBoost's semantics

- **Status:** proposed, 2026-11-23. The owner chose phase 3 next on 2026-11-23, and approved `xgboost-cpu` 3.4.1 as
  the compare tests' reference the same day.
- **Date:** 2026-11-23
- **Number:** 0044, the next in the family's one sequence (smaltworks `adr/README.md`: "The next number is 0044",
  checked 2026-11-22; no open pull request in the family takes it, checked 2026-11-23).
- **Builds on:** [ADR 0024](./0024-smaltml-scope-and-architecture), sections 1 (phase 3: histogram-based boosting
  with XGBoost's semantics: its objectives, missing values sent down a learned default branch, base score, early
  stopping, feature importance), 6 (saved as JSON, boosted trees also in XGBoost's format), 7 (the compare tests) and
  8 (speed against Python's native libraries); [ADR 0039](./0039-trees-and-forests) (phase 2's trees, which phase 3
  doesn't reuse).

## Context

XGBoost's scikit-learn interface (`XGBRegressor`, `XGBClassifier`) is the reference, as scikit-learn is for phases 1
and 2: the same models, parameters, defaults and results, written from XGBoost's documentation and the paper (Chen and
Guestrin, "XGBoost: A Scalable Tree Boosting System", 2016), never from its code (ADR 0024, section 2).

What XGBoost 3.4.1 does by default, read from the installed package on 2026-11-23 (`get_params()`, the booster's
`save_config()` and `get_quantile_cut()`):

- **Training:** 100 rounds; `eta` (learning rate) 0.3, stored as the 32-bit float 0.300000012; `max_depth` 6, grown
  depth first (`grow_policy` `depthwise`); `min_child_weight` 1; `gamma` (`min_split_loss`) 0; `lambda` 1; `alpha` 0;
  `subsample` and the three `colsample_by*` 1; `max_bin` 256; `tree_method` `auto`, which runs the histogram method
  (`grow_quantile_histmaker`).
- **Objectives:** `reg:squarederror` for the regressor; `binary:logistic` for the classifier with two classes,
  `multi:softprob` with more.
- **Base score:** fitted from the data (`boost_from_average`): the mean of y for squared error; the weighted share of
  the positive class for `binary:logistic`; a value per class for `multi:softprob` (they summed to about 0 on the data
  tried).
- **Histogram cuts:** a column with at most `max_bin` distinct values is cut at each distinct value but the smallest,
  after a first cut at −∞ and before a last at about twice the largest; a split sends `x < cut` left. With more
  distinct values, 257 cuts from a weighted quantile sketch.
- **Missing values** (`missing` NaN by default) go down a branch each split learns. Categorical columns are on by
  default in the scikit-learn interface (`enable_categorical`), for pandas categories.

## Decision (proposed)

1. **An area of its own:** `@smaltworks/smaltml/boost`, an entry point beside `tree` and `ensemble`; it imports
   `core` and `parallel`, not phase 2's trees, whose exact splits over sorted values aren't histogram splits.

2. **The models, `BoostedTreesRegressor` and `BoostedTreesClassifier`** (the owner, 2026-11-23: smaltml's own names,
   documented as XGBoost's `XGBRegressor` and `XGBClassifier`, rather than another project's), with XGBoost's
   scikit-learn parameters and defaults in camelCase (`nEstimators`, `learningRate`,
   `maxDepth`, `minChildWeight`, `gamma`, `regLambda`, `regAlpha`, `maxBin`, `subsample`, `colsampleBytree`,
   `colsampleBylevel`, `colsampleBynode`, `baseScore`, `objective`, `seed`), `sampleWeight`, `fit`/`fitSync`,
   `predict`, `predictProba` for the classifier, `featureImportances` (XGBoost's `importance_type`: gain by default,
   weight, cover, total gain, total cover), early stopping (`earlyStoppingRounds` with an evaluation set and
   `evalMetric`), saved as JSON.

3. **The algorithm, XGBoost's histogram method:** each column bucketed once by its cuts; each round, the objective's
   gradients and Hessians; trees grown depth first from per-node histograms of gradient and Hessian sums (a child's
   histogram as its parent's less its sibling's); a split's gain and a leaf's weight from those sums with `lambda`,
   `alpha`, `gamma` and `min_child_weight`; leaf weights times `learningRate`; missing values sent down the side that
   gains more, learned per split. Cuts: a column's distinct values while there are at most `maxBin`, as XGBoost's;
   beyond that our own weighted quantile sketch, from the paper's description.

4. **Numbers:** smaltml computes in 64-bit floats on the CPU, as every model does (ADR 0024, section 3). XGBoost keeps
   some values in 32-bit floats (its parameters, at least), so results agree within a tolerance each compare case
   states, not bit for bit; which of XGBoost's sums are 32-bit is measured in the first pull request, and decides that
   tolerance.

5. **Compare tests** (ADR 0024, section 7), against `xgboost-cpu` 3.4.1, pinned in `compare/requirements.txt` beside
   scikit-learn and NumPy:
   - **Exact:** on data whose columns have at most `maxBin` distinct values (the cuts are the values) and no sampling,
     the same trees (each node's column, cut and default direction) and the same predictions, within the stated
     tolerance.
   - **Statistical:** with sampling (`subsample`, `colsample_*`), whose draws are smaltml's own, and with more distinct
     values than `maxBin`, where the sketches' cuts can differ: held-out scores within a margin of XGBoost's, set from
     its spread as phase 2's forests' were.

6. **Saved models:** smaltml's JSON, as every model; and XGBoost's own JSON model format, written and read
   (`save_model`/`load_model` with a `.json` file), so a model moves between smaltml and XGBoost both ways. The
   compare tests load XGBoost's saved models in smaltml and check their predictions.

7. **Hardware:** the CPU first. Threads build histograms over shares of the rows and add them in a fixed order, so a
   fit on any number of threads equals `fitSync` to the last bit, as the forests do. WebGPU histograms come after, kept
   only where measured faster. The benchmarks compare with XGBoost on one thread (`nthread` 1), as phases 1 and 2 do
   with scikit-learn.

8. **The order of work,** each its own pull request with its compare cases, reference pages and examples:
   1. The regressor with squared error: cuts from distinct values, depth-first growth, `lambda`, `alpha`, `gamma`,
      `minChildWeight`, the base score, `sampleWeight`, gain importances; no missing values or sampling yet (they fail
      with `ML_PARAM` or `ML_VALUE`, and are listed in the divergences).
   2. The classifier: `binary:logistic` and `multi:softprob`.
   3. Missing values, sent down a learned default branch.
   4. Our weighted quantile sketch, for columns with more distinct values than `maxBin`.
   5. Sampling: `subsample` and the `colsample_*`.
   6. Early stopping, with an evaluation set and its metrics.
   7. XGBoost's JSON model format, both ways.
   8. Threads; then WebGPU, measured.

## Still to decide

1. ~~**The names:** XGBoost's own (`XGBRegressor`, `XGBClassifier`), or smaltml's own.~~ Decided 2026-11-23 (the
   owner): `BoostedTreesRegressor` and `BoostedTreesClassifier` (item 2).
2. **More objectives:** `reg:logistic`, `count:poisson`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:gamma`,
   `reg:tweedie`, `rank:*`, `survival:*`: which, and when.
3. **Categorical columns,** which XGBoost's scikit-learn interface takes by default from pandas categories: whether
   smaltml takes them natively, or asks for encoded columns (`OneHotEncoder`, `OrdinalEncoder`).
4. **Loss-guided growth** (`grow_policy` `lossguide`, `max_leaves`), the `dart` booster, monotone and interaction
   constraints, contributions (`pred_contribs`, SHAP values) and XGBoost's binary model format (UBJSON).

## Step 1, the regressor (2026-11-23)

`BoostedTreesRegressor`, squared error, in `src/boost/` (the `boost` entry point, importing only `core`). What XGBoost
3.4.1 does, read from its saved trees (`get_dump` with stats) and checked against the paper's formulas:

- The base score is the weighted mean of y; a row's gradient is its weight times (prediction − y), its Hessian its
  weight.
- A split's gain, as XGBoost reports it, is T(G_L)²/(H_L + λ) + T(G_R)²/(H_R + λ) − T(G)²/(H + λ), with no factor of ½,
  where T shrinks a sum towards 0 by alpha; a leaf is −T(G)/(H + λ) times the learning rate.
- A split is made where its gain reaches `gamma` (checked as the tree grows, not pruned after: a weak split above a
  strong one stays a leaf) and each child's Hessian sum reaches `min_child_weight`.
- Of splits with the same gain, the lower column wins. Splits that send the same rows each way have the same gain, but
  64-bit sums in another order differ in the last bits, so gains within 1e-12 of each other, relatively, count as one.
- `feature_importances_` is each column's mean gain per split, normalized to sum to 1.
- A cut at 0 is +0 (−0 in the data is the same value).

Four compare cases, on columns of 3 to 41 distinct values: XGBoost's trees split for split (3,147 splits in the
defaults' 100 trees), and leaves, predictions, importances and base score within 1e-5; the differences measured were
below 1e-6, XGBoost's 32-bit arithmetic.

## Consequences

- `compare/requirements.txt` gains `xgboost-cpu==3.4.1` (the owner's yes, 2026-11-23: Apache-2.0, a 5.8 MB wheel for
  Linux x86-64, used only by `npm run compare`, never in smaltml's package).
- Phase 3 starts with the regressor, as phase 2 started with the trees: the parts every later step needs.
- A model saved by XGBoost can be loaded and used in a page with no Python, once step 7 lands.
