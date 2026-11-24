/**
 * Gradient-boosted trees by XGBoost's histogram method (ADR 0044), written from XGBoost's documentation and the paper
 * (Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", 2016), and checked against XGBoost 3.4.1's results:
 * our own code. XGBoost is the reference for results, not the source.
 *
 * A column is bucketed once by its cuts. While a column has at most maxBin distinct values (as 32-bit floats) its cuts
 * are those values, as XGBoost's are, and a split before bin b sends `x < edges[b]` left. A tree grows depth first, a
 * level at a time: each node's histogram of gradient and Hessian sums per bin, the best split from it, and the node's
 * children numbered as they are made. A missing value (NaN) is in no bin: each split learns which side its missing
 * values take, as the paper's sparsity-aware split finding does.
 */
import type { Matrix } from '../core/matrix.js';

/** A matrix's columns bucketed: each row's bin per column, row by row, and each column's distinct values. */
export interface Binned {
  rows: number;
  cols: number;
  /** codes[i · cols + f]: row i's bin in column f, the rank of its value among the column's distinct values. */
  codes: Uint16Array;
  /**
   * Each column's distinct values, ascending, then one cut past the largest: a split before bin b sends `x < edges[f][b]`
   * left. The last, before no bin, sends every present value left and the missing ones right.
   */
  edges: Float64Array[];
  /** missing[i · cols + f] is 1 where row i's value in column f is missing (NaN); null where none is. */
  missing: Uint8Array | null;
  /** Where each column's bins start in a histogram of every column's bins. */
  offsets: Int32Array;
  /** Every column's bins, together. */
  bins: number;
}

/** The most bins a column can have: its codes are 16-bit. */
export const MAX_BIN_LIMIT = 65536;

/**
 * The cut past a column's largest value: twice it if positive, 0 if not, plus 1e-5, in 32-bit floats. XGBoost's last
 * cut, read from its `get_quantile_cut()` (2.00001 past 1, 0.00001 past −1 or 0, 200 past 100).
 */
function cutPast(max: number): number {
  return Math.fround(Math.fround(max + Math.abs(max)) + Math.fround(1e-5));
}

/**
 * A column's cuts where it has more distinct values than maxBin: its smallest value, then, for each j from 1 to
 * maxBin − 1, the first value with at least j/maxBin of the column's weight below it (once each), then the cut past its
 * largest value. These are the exact weighted quantiles that XGBoost's sketch approximates, weighted as XGBoost weighs
 * them, by the rows' sample weights (ADR 0044, step 4: the owner chose exact quantiles, 2026-11-24). `order` is the
 * present rows, by value.
 */
function quantileEdges(col: Float64Array, order: Uint32Array, weights: Float64Array | undefined, maxBin: number): Float64Array {
  let W = 0;
  if (weights) for (const i of order) W += weights[i]!;
  // Rows that all weigh 0 have no quantiles by weight: they are counted instead.
  const wt = weights && W > 0 ? weights : null;
  if (!wt) W = order.length;
  const out: number[] = [];
  let below = 0;
  let j = 1;
  for (let k = 0; k < order.length; ) {
    const v = col[order[k]!]!;
    if (out.length === 0) out.push(v + 0);
    else if (j < maxBin && below >= (j * W) / maxBin) {
      out.push(v + 0);
      while (j < maxBin && below >= (j * W) / maxBin) j++;
    }
    // Every row of this value is below the next one. (−0 and +0 are one value.)
    for (; k < order.length && col[order[k]!] === v; k++) below += wt ? wt[order[k]!]! : 1;
  }
  out.push(cutPast(col[order[order.length - 1]!]!));
  return Float64Array.from(out);
}

/**
 * X's columns bucketed. X holds 32-bit floats (the trees read X so, as XGBoost does). A column with at most maxBin
 * distinct values is cut at each of them, as XGBoost's is; one with more, at its weighted quantiles (quantileEdges).
 */
export function binColumns(X: Matrix, maxBin: number, weights?: Float64Array): Binned {
  const n = X.rows;
  const p = X.cols;
  const codes = new Uint16Array(n * p);
  const edges: Float64Array[] = [];
  const offsets = new Int32Array(p + 1);
  let missing: Uint8Array | null = null;
  for (let f = 0; f < p; f++) {
    const col = X.column(f);
    // Sorted, NaN last: the present values are the first `present`.
    const sorted = Float64Array.from(col).sort();
    let present = n;
    while (present > 0 && Number.isNaN(sorted[present - 1]!)) present--;
    let k = 0;
    // −0 and +0 are one value (−0 < 0 is false); kept as +0, as XGBoost's cuts are.
    for (let i = 0; i < present; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[k++] = sorted[i]! + 0;
    let e: Float64Array;
    if (k <= maxBin) {
      e = new Float64Array(k + 1);
      e.set(sorted.subarray(0, k));
      e[k] = k > 0 ? cutPast(e[k - 1]!) : Number.NaN;
    } else {
      const order = new Uint32Array(present);
      for (let i = 0, at = 0; i < n; i++) if (!Number.isNaN(col[i]!)) order[at++] = i;
      order.sort((a, b) => col[a]! - col[b]!);
      e = quantileEdges(col, order, weights, maxBin);
    }
    edges.push(e);
    // The column's bins: one before each cut but the last.
    const bins = e.length - 1;
    for (let i = 0; i < n; i++) {
      const x = col[i]!;
      if (Number.isNaN(x)) {
        missing ??= new Uint8Array(n * p);
        missing[i * p + f] = 1;
        continue;
      }
      // The value's bin: the last cut at or below it, by binary search.
      let a = 0;
      let b = bins - 1;
      while (a < b) {
        const mid = (a + b + 1) >>> 1;
        if (e[mid]! <= x) a = mid;
        else b = mid - 1;
      }
      codes[i * p + f] = a;
    }
    offsets[f + 1] = offsets[f]! + Math.max(bins, 0);
  }
  return { rows: n, cols: p, codes, edges, missing, offsets, bins: offsets[p]! };
}

/** What a tree's growth takes, as XGBoost's parameters. */
export interface GrowParams {
  learningRate: number;
  maxDepth: number;
  minChildWeight: number;
  gamma: number;
  regLambda: number;
  regAlpha: number;
}

/** A boosted tree's nodes as parallel arrays. A leaf has left −1 and its value; a split its column and cut. */
export interface BoostTree {
  left: Int32Array;
  right: Int32Array;
  feature: Int32Array;
  /** The cut: `x < threshold` goes left. NaN at a leaf. */
  threshold: Float64Array;
  /** 1 where a split sends missing values (NaN) left, 0 where right (and at a leaf). */
  defaultLeft: Uint8Array;
  /** A leaf's value (already times the learning rate); 0 at a split. */
  value: Float64Array;
  /** A split's gain (XGBoost's `gain`); 0 at a leaf. */
  gain: Float64Array;
  /** The node's Hessian sum (XGBoost's `cover`). */
  cover: Float64Array;
}

/** XGBoost stops a split whose gain is below this, as numbers below it are rounding. */
const RT_EPS = 1e-6;

/** Gains this close, relatively, are equal: the same split found twice, summed in another order. */
const TIE = 1e-12;

/**
 * Grows one tree on the gradients g and Hessians h, and adds each leaf's value to its rows' predictions. A node's split
 * is the one with the largest gain, T(G_L)²/(H_L + λ) + T(G_R)²/(H_R + λ) − T(G)²/(H + λ), where T shrinks a sum
 * towards 0 by alpha; ties (within TIE) go to the lower column, then the lower cut. A split is made where its gain reaches gamma and
 * each child's Hessian sum reaches minChildWeight; a leaf's value is −T(G)/(H + λ) times the learning rate. Each column's
 * cuts are tried with its missing values on the right, then on the left (the paper's Algorithm 3): the left wins only
 * where it gains more, so missing values go right where it makes no difference, as in XGBoost.
 */
export function growTree(b: Binned, g: Float64Array, h: Float64Array, params: GrowParams, pred: Float64Array): BoostTree {
  const n = b.rows;
  const p = b.cols;
  const { codes, edges, offsets, missing } = b;
  const { regLambda: lambda, regAlpha: alpha, minChildWeight, gamma, maxDepth, learningRate } = params;
  const T = (G: number) => (G > alpha ? G - alpha : G < -alpha ? G + alpha : 0);
  const score = (G: number, H: number) => {
    const t = T(G);
    return (t * t) / (H + lambda);
  };

  const left: number[] = [];
  const right: number[] = [];
  const feature: number[] = [];
  const threshold: number[] = [];
  const defaultLeft: number[] = [];
  const value: number[] = [];
  const gain: number[] = [];
  const cover: number[] = [];
  const newNode = (H: number) => {
    left.push(-1);
    right.push(-1);
    feature.push(-1);
    threshold.push(Number.NaN);
    defaultLeft.push(0);
    value.push(0);
    gain.push(0);
    cover.push(H);
    return left.length - 1;
  };

  const rows = new Uint32Array(n);
  for (let i = 0; i < n; i++) rows[i] = i;
  const scratch = new Uint32Array(n);
  // Gradient and Hessian sums per bin, every column's bins side by side: [G, H] pairs.
  const hist = new Float64Array(b.bins * 2);
  // Each column's missing values in the node: [G, H, count].
  const miss = new Float64Array(p * 3);

  let G0 = 0;
  let H0 = 0;
  for (let i = 0; i < n; i++) {
    G0 += g[i]!;
    H0 += h[i]!;
  }
  // A level at a time, in the order the nodes were made: XGBoost's depth-wise growth.
  let level = [{ id: newNode(H0), start: 0, end: n, G: G0, H: H0 }];
  for (let depth = 0; level.length > 0; depth++) {
    const next: typeof level = [];
    for (const node of level) {
      const { id, start, end, G, H } = node;
      const leaf = () => {
        const w = (-T(G) / (H + lambda)) * learningRate;
        value[id] = w;
        for (let k = start; k < end; k++) pred[rows[k]!]! += w;
      };
      if (depth >= maxDepth) {
        leaf();
        continue;
      }
      hist.fill(0);
      miss.fill(0);
      if (missing === null) {
        for (let k = start; k < end; k++) {
          const i = rows[k]!;
          const gi = g[i]!;
          const hi = h[i]!;
          const c = i * p;
          for (let f = 0; f < p; f++) {
            const at = (offsets[f]! + codes[c + f]!) * 2;
            hist[at]! += gi;
            hist[at + 1]! += hi;
          }
        }
      } else {
        for (let k = start; k < end; k++) {
          const i = rows[k]!;
          const gi = g[i]!;
          const hi = h[i]!;
          const c = i * p;
          for (let f = 0; f < p; f++) {
            if (missing[c + f] === 1) {
              miss[f * 3]! += gi;
              miss[f * 3 + 1]! += hi;
              miss[f * 3 + 2]! += 1;
              continue;
            }
            const at = (offsets[f]! + codes[c + f]!) * 2;
            hist[at]! += gi;
            hist[at + 1]! += hi;
          }
        }
      }
      const parent = score(G, H);
      let bestGain = 0;
      let bestF = -1;
      let bestBin = 0;
      let bestGL = 0;
      let bestHL = 0;
      let bestLeft = 0;
      for (let f = 0; f < p; f++) {
        const o = offsets[f]!;
        const k = offsets[f + 1]! - o;
        const nm = miss[f * 3 + 2]!;
        // Missing values right, with the cuts in ascending order; then, where the node has some, left, with the cuts in
        // descending order (the paper's Algorithm 3). Of tied cuts (no row of the node between them), the first met wins:
        // the lowest with missing values right, the highest with them left, as in XGBoost.
        let GL = 0;
        let HL = 0;
        // A split before bin c: bins 0 … c − 1 go left. With missing values right, c = k too: every present value left,
        // the missing ones right.
        const last = nm > 0 ? k : k - 1;
        for (let c = 1; c <= last; c++) {
          GL += hist[(o + c - 1) * 2]!;
          HL += hist[(o + c - 1) * 2 + 1]!;
          const HR = H - HL;
          if (HL < minChildWeight || HR < minChildWeight) continue;
          const gn = score(GL, HL) + score(G - GL, HR) - parent;
          // Gains within 1e-12 of each other, relatively, are a tie: two columns or cuts that split the rows the same
          // way have the same gain, but sums added in another order differ in the last bits. A tie keeps the split met
          // first: the lower column, then missing values right, as XGBoost's does.
          if (bestF < 0 || gn - bestGain > TIE * Math.abs(bestGain)) {
            bestGain = gn;
            bestF = f;
            bestBin = c;
            bestGL = GL;
            bestHL = HL;
            bestLeft = 0;
          }
        }
        if (nm > 0) {
          let GR = 0;
          let HR = 0;
          for (let c = k - 1; c >= 1; c--) {
            GR += hist[(o + c) * 2]!;
            HR += hist[(o + c) * 2 + 1]!;
            const HLm = H - HR;
            if (HLm < minChildWeight || HR < minChildWeight) continue;
            const gn = score(G - GR, HLm) + score(GR, HR) - parent;
            if (bestF < 0 || gn - bestGain > TIE * Math.abs(bestGain)) {
              bestGain = gn;
              bestF = f;
              bestBin = c;
              bestGL = G - GR;
              bestHL = HLm;
              bestLeft = 1;
            }
          }
        }
      }
      if (bestF < 0 || !(bestGain > RT_EPS) || bestGain < gamma) {
        leaf();
        continue;
      }
      // Partition the node's rows: those in bins below bestBin of bestF first, each side in the order it was.
      let a = start;
      let s = 0;
      if (missing === null) {
        for (let k = start; k < end; k++) {
          const i = rows[k]!;
          const goLeft = codes[i * p + bestF]! < bestBin ? 1 : 0;
          rows[a] = i;
          scratch[s] = i;
          a += goLeft;
          s += 1 - goLeft;
        }
      } else {
        // A missing value goes to the side the split learned.
        for (let k = start; k < end; k++) {
          const i = rows[k]!;
          const goLeft = missing[i * p + bestF] === 1 ? bestLeft : codes[i * p + bestF]! < bestBin ? 1 : 0;
          rows[a] = i;
          scratch[s] = i;
          a += goLeft;
          s += 1 - goLeft;
        }
      }
      rows.set(scratch.subarray(0, s), a);
      feature[id] = bestF;
      threshold[id] = edges[bestF]![bestBin]!;
      defaultLeft[id] = bestLeft;
      gain[id] = bestGain;
      const l = newNode(bestHL);
      const r = newNode(H - bestHL);
      left[id] = l;
      right[id] = r;
      next.push({ id: l, start, end: a, G: bestGL, H: bestHL }, { id: r, start: a, end, G: G - bestGL, H: H - bestHL });
    }
    level = next;
  }
  return {
    left: Int32Array.from(left),
    right: Int32Array.from(right),
    feature: Int32Array.from(feature),
    threshold: Float64Array.from(threshold),
    defaultLeft: Uint8Array.from(defaultLeft),
    value: Float64Array.from(value),
    gain: Float64Array.from(gain),
    cover: Float64Array.from(cover),
  };
}

/**
 * A tree's prediction for row i of X (X as 32-bit floats, column by column): the value of the leaf it falls in. A missing
 * value (NaN) takes the side its split learned.
 */
export function treeValue(t: BoostTree, data: Float64Array, n: number, i: number): number {
  let node = 0;
  while (t.left[node] !== -1) {
    const x = data[t.feature[node]! * n + i]!;
    const left = x === x ? x < t.threshold[node]! : t.defaultLeft[node] === 1;
    node = left ? t.left[node]! : t.right[node]!;
  }
  return t.value[node]!;
}
