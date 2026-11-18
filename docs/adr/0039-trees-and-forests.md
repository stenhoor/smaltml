# ADR 0039: Phase 2, trees and forests

- **Status:** proposed, 2026-11-17. The owner started phase 2 on 2026-11-17 ("let's get rolling on phase 2"),
  which decides [ADR 0024](./0024-smaltml-scope-and-architecture)'s "Still to decide", item 4: alongside phase 1's
  release, not after it.
- **Date:** 2026-11-17
- **Number:** 0039, the next free number in the family repositories this session can read (smaltdb has 0038). GitHub's
  code search missed smaltdb's 0038, so it can't be relied on for this; renumber if another repository took 0039.
- **Builds on:** ADR 0024, section 1 (phase 2: decision trees, random forests, extra trees, feature importance) and
  section 7 (the compare tests).

## Context

Phase 2 is scikit-learn's `tree` and the forests of its `ensemble`: CART decision trees for classification and
regression, random forests, extra trees, and feature importance. Phase 3's gradient-boosted trees (XGBoost's
semantics) come after, and use their own histogram-based trees, so phase 2's trees don't have to serve them.

Two things make phase 2 unlike phase 1:

- **Randomness decides the result.** A random forest's trees each see a bootstrap sample of the rows and a random
  subset of the columns at each split; extra trees also draw their thresholds at random. scikit-learn draws them from
  NumPy's generator with `random_state`. smaltml has its own generator (`Random`, `seed`), as phase 1's shuffles do,
  and writes its own code from the documentation (ADR 0024, section 2), so it can't draw the same numbers: a seeded
  smaltml forest repeats itself, never scikit-learn's forest.
- **A single tree is deterministic where no two splits tie.** scikit-learn's `DecisionTreeClassifier` and
  `DecisionTreeRegressor` look at the columns in a random order, which matters only to choose between splits that
  improve the impurity by exactly as much. On data without such ties, smaltml's tree can be the same tree, split for
  split.

## Decision (proposed)

1. **Areas.** `@smaltworks/smaltml/tree` (DecisionTreeClassifier, DecisionTreeRegressor) and
   `@smaltworks/smaltml/ensemble` (the forests), each an entry point, as scikit-learn's modules are. `ensemble` may
   import `tree`; `tree` imports `core` only.

2. **The trees, in this order of work:**
   1. `DecisionTreeRegressor` and `DecisionTreeClassifier`: CART, best split by exhaustive search over midpoints of
      sorted values, depth first; `criterion` (`squared_error`, `friedman_mse`, `absolute_error`, `poisson`; `gini`,
      `entropy`, `log_loss`), `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `minWeightFractionLeaf`,
      `maxFeatures`, `maxLeafNodes` (best first), `minImpurityDecrease`, `ccpAlpha` (cost-complexity pruning),
      `classWeight`, `sampleWeight`, `seed`; `predict`, `predictProba`, `apply`, `featureImportances`, `getDepth`,
      `getNLeaves`. Missing values (`NaN`) as scikit-learn 1.3+ sends them down a learned side. Saved as JSON.
   2. `RandomForestRegressor` and `RandomForestClassifier`: `nEstimators`, `bootstrap`, `maxSamples`, `oobScore`, the
      trees' parameters, `seed`; fitted trees in the thread pool (item 5). Done, 2026-11-18 (below).
   3. `ExtraTreesRegressor` and `ExtraTreesClassifier`: random thresholds, no bootstrap by default.
   Each lands as its own pull request, with its compare cases, reference pages and examples.

3. **The compare tests, two kinds** (ADR 0024, section 7, extended):
   - **Exact, for single trees:** on data whose splits don't tie (continuous columns, no repeated values), smaltml's
     tree has scikit-learn's structure (each node's column, threshold, impurity, sample counts and values) and its
     predictions and probabilities, within rounding (1e-12). Where splits tie, the trees may differ, and the tests use
     data without ties.
   - **Statistical, for forests:** a seeded forest repeats itself to the last bit, and the tests hold smaltml's forest
     to scikit-learn's on what doesn't depend on the draws: on held-out data from several seeds, accuracy and R²
     within a stated margin of scikit-learn's, and feature importances that rank the informative columns first. A
     forest with `bootstrap: false` and `maxFeatures` of every column is checked exactly, as it grows the same trees.
   - The difference is a `DIVERGES`-style entry: a seeded forest doesn't give scikit-learn's forest for the same
     `random_state`, as `trainTestSplit`'s shuffles don't today.

4. **The CPU is the product,** as for every model. WebGPU isn't planned for trees: building a tree is branchy and
   sorts, which GPUs do poorly. Prediction over many rows may use it later, if it measures faster.

5. **Threads for forests.** The thread pool runs kernels today. Forests fit their trees in parallel: each worker
   fits whole trees from its own seeded stream, so a forest on any number of threads equals its `fitSync`, to the last
   bit. This extends the pool's protocol with a "fit trees" task.

6. **Saved models:** a tree saves its nodes as arrays (children, column, threshold, missing-value side, impurity,
   samples, values), as scikit-learn's `tree_` holds them; a forest saves its trees. The format version rises with
   the first tree.

## Still to decide

1. **The statistical margin for forests:** proposed, held-out accuracy within 0.02 and R² within 0.02 of
   scikit-learn's mean over 5 seeds, on each compare dataset; then 0.02 on small data and 0.005 on one dataset of 2000
   rows or more. Set with the first forest, below.
2. **`export_text` and `plot_tree`:** text export is cheap and useful in the notebooks; plotting belongs to smaltkit's
   charts, not smaltml.
3. **Speed:** scikit-learn's trees are Cython. The benchmarks (docs/research) will compare from the first tree, as
   phase 1's did.

## The first forests (2026-11-18)

`RandomForestRegressor` and `RandomForestClassifier` landed without threads (item 5 follows in its own pull request).
What they settled:

- **The margins, measured.** A seed-to-seed spread decides how far two means of a few seeds can fall apart by chance,
  so each margin is held to at least about three standard errors of that difference, from scikit-learn's spread over
  30 other seeds on the same data: 0.02 on 120 rows (20 seeds each, as out-of-bag accuracy there varies by about
  0.02 from seed to seed) and on 2000 rows (5 seeds), and 0.005 for the 2000-row regressor with its defaults (5
  seeds), whose scores vary least (about 0.0013 R² on the test rows, 0.0025 out of bag). Both the held-out score and
  the out-of-bag score are checked. A classifier's out-of-bag accuracy on 2000 rows varies by about 0.006, so 0.005
  would need some 25 seeds; it is held to 0.02.
- **Weights follow scikit-learn 1.9.** With bootstrap, a row's chance of being drawn is in proportion to its weight
  (`sampleWeight` times its class weight), and each tree weighs a row by its draws; without, the trees take the
  weights. `'balanced'` comes from weighted class counts; a `maxSamples` share is of the weights' sum. The first
  version multiplied the weights by the draws, as earlier scikit-learn did, and the compare tests caught it: with
  `classWeight: 'balanced'`, held-out accuracy was 0.6923 against scikit-learn's 0.6782 (40 seeds each).
- **Rows no tree left out** get 0 out of bag and count in the score, with a warning (`ML_OOB`), as scikit-learn does.

## Forests on threads (2026-11-18)

Item 5, as decided: `fit` grows the trees on the thread pool (`threads`, default `'auto'`), and the forest equals
`fitSync`'s to the last bit (test/parallel.test.ts).

- **A stream per tree.** The forest's seed gives each tree one number, its stream; from it the tree draws its bootstrap
  sample and then its own seed for the column draws (`src/ensemble/grow.ts`). A tree depends only on its stream, so
  any thread can grow it. This changed what a seeded forest grows from the first release, where one stream gave every
  tree's draws in turn; that release had been out for minutes, with no saved forests to keep.
- **Whole trees per worker.** The pool takes a new task, `trees`: X in shared memory, the targets, the weights and a
  run of streams; each worker grows its trees with the same code as `fitSync` and sends them back as their saved
  documents, with the out-of-bag rows and predictions. The main thread adds the out-of-bag predictions up in tree
  order, as `fitSync` does.
- **When.** `'auto'` uses the pool where X's values times the trees reach `PARALLEL_MIN_VALUES`, as the kernels'
  threshold counts X's values. X is checked on the main thread first, so a missing value fails with `ML_VALUE`, not
  a worker's message.

## Consequences

- Forests don't match scikit-learn tree for tree; users comparing with Python see the same quality, not the same
  trees. The reference pages say so.
- The thread pool grows a task type beyond kernels (item 5), which smaltml-simd's workers' script must keep up with.
- Phase 3's boosted trees get their own tree code (histograms, XGBoost's rules); phase 2's exact CART trees stay as
  they are.
