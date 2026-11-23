/**
 * Gradient-boosted trees by XGBoost's histogram method (ADR 0044), written from XGBoost's documentation and the paper
 * (Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", 2016), and checked against XGBoost 3.4.1's results:
 * our own code. XGBoost is the reference for results, not the source.
 *
 * A column is bucketed once by its cuts. While a column has at most maxBin distinct values (as 32-bit floats) its cuts
 * are those values, as XGBoost's are, and a split before bin b sends `x < edges[b]` left. A tree grows depth first, a
 * level at a time: each node's histogram of gradient and Hessian sums per bin, the best split from it, and the node's
 * children numbered as they are made.
 */
import { fail } from '../core/errors.js';
import type { Matrix } from '../core/matrix.js';

/** A matrix's columns bucketed: each row's bin per column, row by row, and each column's distinct values. */
export interface Binned {
  rows: number;
  cols: number;
  /** codes[i · cols + f]: row i's bin in column f, the rank of its value among the column's distinct values. */
  codes: Uint16Array;
  /** Each column's distinct values, ascending: a split before bin b sends `x < edges[f][b]` left. */
  edges: Float64Array[];
  /** Where each column's bins start in a histogram of every column's bins. */
  offsets: Int32Array;
  /** Every column's bins, together. */
  bins: number;
}

/** The most bins a column can have: its codes are 16-bit. */
export const MAX_BIN_LIMIT = 65536;

/**
 * X's columns bucketed by their distinct values. X holds 32-bit floats (the trees read X so, as XGBoost does). A column
 * with more than maxBin distinct values isn't supported yet: XGBoost cuts it by a quantile sketch (ADR 0044, step 4).
 */
export function binColumns(X: Matrix, maxBin: number): Binned {
  const n = X.rows;
  const p = X.cols;
  const codes = new Uint16Array(n * p);
  const edges: Float64Array[] = [];
  const offsets = new Int32Array(p + 1);
  for (let f = 0; f < p; f++) {
    const col = X.column(f);
    const sorted = Float64Array.from(col).sort();
    let k = 0;
    // −0 and +0 are one value (−0 < 0 is false); kept as +0, as XGBoost's cuts are.
    for (let i = 0; i < n; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[k++] = sorted[i]! + 0;
    if (k > maxBin) {
      fail('ML_VALUE', `column ${X.columnName(f)} has ${k} distinct values, more than maxBin (${maxBin}): columns with more aren't supported yet; raise maxBin to at least ${k}`);
    }
    const e = sorted.slice(0, k);
    edges.push(e);
    for (let i = 0; i < n; i++) {
      // The value's rank: a binary search in the distinct values.
      const x = col[i]!;
      let a = 0;
      let b = k - 1;
      while (a < b) {
        const mid = (a + b) >>> 1;
        if (e[mid]! < x) a = mid + 1;
        else b = mid;
      }
      codes[i * p + f] = a;
    }
    offsets[f + 1] = offsets[f]! + k;
  }
  return { rows: n, cols: p, codes, edges, offsets, bins: offsets[p]! };
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
 * each child's Hessian sum reaches minChildWeight; a leaf's value is −T(G)/(H + λ) times the learning rate.
 */
export function growTree(b: Binned, g: Float64Array, h: Float64Array, params: GrowParams, pred: Float64Array): BoostTree {
  const n = b.rows;
  const p = b.cols;
  const { codes, edges, offsets } = b;
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
  const value: number[] = [];
  const gain: number[] = [];
  const cover: number[] = [];
  const newNode = (H: number) => {
    left.push(-1);
    right.push(-1);
    feature.push(-1);
    threshold.push(Number.NaN);
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
      const parent = score(G, H);
      let bestGain = 0;
      let bestF = -1;
      let bestBin = 0;
      let bestGL = 0;
      let bestHL = 0;
      for (let f = 0; f < p; f++) {
        const o = offsets[f]!;
        const k = offsets[f + 1]! - o;
        let GL = 0;
        let HL = 0;
        // A split before bin c: bins 0 … c − 1 go left.
        for (let c = 1; c < k; c++) {
          GL += hist[(o + c - 1) * 2]!;
          HL += hist[(o + c - 1) * 2 + 1]!;
          const HR = H - HL;
          if (HL < minChildWeight || HR < minChildWeight) continue;
          const gn = score(GL, HL) + score(G - GL, HR) - parent;
          // Gains within 1e-12 of each other, relatively, are a tie: two columns or cuts that split the rows the same
          // way have the same gain, but sums added in another order differ in the last bits. A tie keeps the lower
          // column, then the lower cut, as XGBoost's does.
          if (bestF < 0 || gn - bestGain > TIE * Math.abs(bestGain)) {
            bestGain = gn;
            bestF = f;
            bestBin = c;
            bestGL = GL;
            bestHL = HL;
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
      for (let k = start; k < end; k++) {
        const i = rows[k]!;
        const goLeft = codes[i * p + bestF]! < bestBin ? 1 : 0;
        rows[a] = i;
        scratch[s] = i;
        a += goLeft;
        s += 1 - goLeft;
      }
      rows.set(scratch.subarray(0, s), a);
      feature[id] = bestF;
      threshold[id] = edges[bestF]![bestBin]!;
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
    value: Float64Array.from(value),
    gain: Float64Array.from(gain),
    cover: Float64Array.from(cover),
  };
}

/** A tree's prediction for row i of X (X as 32-bit floats, column by column): the value of the leaf it falls in. */
export function treeValue(t: BoostTree, data: Float64Array, n: number, i: number): number {
  let node = 0;
  while (t.left[node] !== -1) node = data[t.feature[node]! * n + i]! < t.threshold[node]! ? t.left[node]! : t.right[node]!;
  return t.value[node]!;
}
