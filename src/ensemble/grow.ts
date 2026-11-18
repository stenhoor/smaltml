/**
 * One tree of a forest, from its own seeded stream: its bootstrap sample, its column draws, and its out-of-bag
 * predictions. fitSync() and the thread pool's workers both grow trees with this, so a forest grown on any number of
 * threads equals fitSync()'s to the last bit (ADR 0039, item 5).
 */
import { Matrix } from '../core/matrix.js';
import type { Label, Labels } from '../core/model.js';
import { Random } from '../core/random.js';
import { asPrepared } from '../tree/cart.js';
import { DecisionTreeClassifier, DecisionTreeRegressor, type DecisionTreeClassifierParams, type DecisionTreeRegressorParams } from '../tree/index.js';

/** What every tree of a forest is grown from. */
export interface GrowJob {
  classifier: boolean;
  /** X prepared for trees (32-bit floats, checked, presorted: tree/cart.ts's asPrepared). */
  X: Matrix;
  /** The regressor's targets, or the classifier's labels. */
  y: Float64Array | readonly Label[];
  /** Each row's weight (sampleWeight times its class's weight), or `null` where every row weighs 1. */
  weights: Float64Array | null;
  /** The trees' parameters, with their criterion; never a seed or class weights. */
  params: DecisionTreeRegressorParams | DecisionTreeClassifierParams;
  bootstrap: boolean;
  /** How many rows each bootstrap sample draws. */
  draws: number;
  /** Whether to predict the rows each tree left out. */
  oob: boolean;
}

/** A grown tree, with the rows its bootstrap left out and its predictions for them (class probabilities, column-major). */
export interface Grown<T> {
  tree: T;
  outOfBag: Int32Array | null;
  oob: Float64Array | null;
}

/** The weights summed row by row, for drawing rows in proportion to their weight; `null` without weights. */
export function cumulativeWeights(weights: Float64Array | null): Float64Array | null {
  if (!weights) return null;
  const out = new Float64Array(weights.length);
  let sum = 0;
  for (let i = 0; i < weights.length; i++) out[i] = sum += weights[i]!;
  return out;
}

/** The rows `rows` of X, as a matrix. */
export function pickRows(m: Matrix, rows: Int32Array): Matrix {
  const n = m.rows;
  const k = rows.length;
  const data = new Float64Array(k * m.cols);
  for (let j = 0; j < m.cols; j++) for (let r = 0; r < k; r++) data[j * k + r] = m.data[j * n + rows[r]!]!;
  return new Matrix(k, m.cols, data, m.names);
}

/**
 * Grows the tree whose stream starts at `stream`. With bootstrap, its rows are a sample of `draws` drawn with
 * replacement, each row with a chance in proportion to its weight, and the tree weighs each row by the times it was
 * drawn; without, the tree weighs the rows by their weight. Both as scikit-learn 1.9 does. Then the stream gives the
 * tree's own seed, for its column draws.
 */
export function growTree(job: GrowJob, cumulative: Float64Array | null, stream: number): Grown<DecisionTreeRegressor | DecisionTreeClassifier> {
  const n = job.X.rows;
  const r = new Random(stream);
  let w = job.weights ?? undefined;
  let outOfBag: Int32Array | null = null;
  if (job.bootstrap) {
    const counts = new Float64Array(n);
    if (cumulative) {
      // A draw is the first row whose running sum passes a uniform number below the total.
      const end = cumulative[n - 1]!;
      for (let d = 0; d < job.draws; d++) {
        const u = r.next() * end;
        let lo = 0;
        let hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (cumulative[mid]! > u) hi = mid;
          else lo = mid + 1;
        }
        counts[lo]! += 1;
      }
    } else for (let d = 0; d < job.draws; d++) counts[r.int(n)]! += 1;
    w = counts;
    if (job.oob) {
      const out: number[] = [];
      for (let i = 0; i < n; i++) if (counts[i] === 0) out.push(i);
      outOfBag = Int32Array.from(out);
    }
  }
  const seed = r.uint32() >>> 1;
  const fit = w ? { sampleWeight: w } : {};
  const tree = job.classifier
    ? new DecisionTreeClassifier({ ...(job.params as DecisionTreeClassifierParams), seed }).fitSync(job.X, job.y as Labels, fit)
    : new DecisionTreeRegressor({ ...(job.params as DecisionTreeRegressorParams), seed }).fitSync(job.X, job.y as Float64Array, fit);
  let oob: Float64Array | null = null;
  if (outOfBag && outOfBag.length > 0) {
    // Rows of X, prepared already: the tree takes them as they are.
    const rows = asPrepared(pickRows(job.X, outOfBag), false);
    oob = tree instanceof DecisionTreeClassifier ? tree.predictProbaSync(rows).data : tree.predictSync(rows);
  }
  return { tree, outOfBag, oob };
}
