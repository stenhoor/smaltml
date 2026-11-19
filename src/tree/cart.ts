import { fail } from '../core/errors.js';
import { assertFinite, Matrix, type MatrixInput } from '../core/matrix.js';
import { Random } from '../core/random.js';

/**
 * CART: a binary tree of splits `x[feature] <= threshold`, grown to reduce an impurity. Written from Breiman et al.,
 * Classification and Regression Trees (1984), and scikit-learn's documentation of its trees (the criteria, the
 * stopping rules, best-first growth with max_leaf_nodes, and feature importances as the total weighted impurity
 * decrease). Our own code: scikit-learn is the reference for results, not the source.
 */

/** A tree's nodes as parallel arrays, as scikit-learn's `tree_` holds them. Leaves have feature and threshold −2. */
export interface TreeArrays {
  left: Int32Array;
  right: Int32Array;
  feature: Int32Array;
  threshold: Float64Array;
  impurity: Float64Array;
  nSamples: Int32Array;
  weightedNSamples: Float64Array;
  /** Per node: the mean (regression), or the class fractions (classification), `width` values each. */
  value: Float64Array;
  /** Values per node: 1 for regression, the number of classes for classification. */
  width: number;
}

export const LEAF = -1;
export const UNDEFINED = -2;

/** What a split is judged by. */
export type Criterion = 'squared_error' | 'friedman_mse' | 'gini' | 'entropy';

export interface GrowParams {
  criterion: Criterion;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  minWeightLeaf: number;
  /** How many columns to look at for each split (all: X.cols). */
  maxFeatures: number;
  /**
   * How a column's split is chosen: `'best'`, the best of every threshold between two neighbouring values; `'random'`,
   * one threshold drawn uniformly between the node's smallest and largest value (extra trees).
   */
  splitter: 'best' | 'random';
  maxLeafNodes: number | null;
  minImpurityDecrease: number;
  seed: number;
}

/** The targets: a value per row (regression), or a class index per row and the number of classes. */
export type Targets = { kind: 'regression'; y: Float64Array } | { kind: 'classes'; y: Int32Array; K: number };

const EPSILON = Number.EPSILON;

/**
 * The statistics a criterion needs for a set of rows, in one array: regression [Σw, Σwy, Σwy²], classification
 * [Σw, Σw for class 0, …, class K−1].
 */
function statsSize(t: Targets): number {
  return t.kind === 'regression' ? 3 : 1 + t.K;
}

function addRow(s: Float64Array, t: Targets, i: number, w: number, sign = 1): void {
  s[0]! += sign * w;
  if (t.kind === 'regression') {
    const v = t.y[i]!;
    s[1]! += sign * w * v;
    s[2]! += sign * w * v * v;
  } else s[1 + t.y[i]!]! += sign * w;
}

function impurityOf(s: Float64Array, criterion: Criterion): number {
  const w = s[0]!;
  if (!(w > 0)) return 0;
  if (criterion === 'squared_error' || criterion === 'friedman_mse') {
    const mean = s[1]! / w;
    return s[2]! / w - mean * mean;
  }
  let out = criterion === 'gini' ? 1 : 0;
  for (let k = 1; k < s.length; k++) {
    const p = s[k]! / w;
    if (criterion === 'gini') out -= p * p;
    else if (p > 0) out -= p * Math.log2(p);
  }
  return out;
}

/** impurityOf for Gini, without the criterion's test: the same operations in the same order. */
function giniOf(s: Float64Array): number {
  const w = s[0]!;
  if (!(w > 0)) return 0;
  let out = 1;
  for (let k = 1; k < s.length; k++) {
    const p = s[k]! / w;
    out -= p * p;
  }
  return out;
}

/**
 * A score that orders a node's candidate splits as the impurity decrease does (higher is better), cheaper to compute:
 * for squared error Σ_left² / w_left + Σ_right² / w_right; for Friedman's criterion his improvement,
 * w_left · w_right · (mean_left − mean_right)² / (w_left + w_right), up to the node's weight; for the class criteria
 * −(w_left · impurity_left + w_right · impurity_right).
 */
function proxy(l: Float64Array, r: Float64Array, criterion: Criterion): number {
  const wl = l[0]!;
  const wr = r[0]!;
  if (criterion === 'squared_error') return (l[1]! * l[1]!) / wl + (r[1]! * r[1]!) / wr;
  if (criterion === 'friedman_mse') {
    const diff = wr * l[1]! - wl * r[1]!;
    return (diff * diff) / (wl * wr);
  }
  return -(wl * impurityOf(l, criterion) + wr * impurityOf(r, criterion));
}

/**
 * The improvement a split brings, as min_impurity_decrease and best-first growth measure it: the node's share of the
 * total weight times its impurity less its children's, each by its share of the node's weight. Friedman's criterion
 * gives the same (scikit-learn 1.9 maps it to squared error: they were always equivalent).
 */
function improvement(node: Float64Array, l: Float64Array, r: Float64Array, criterion: Criterion, total: number): number {
  const w = node[0]!;
  return (w / total) * (impurityOf(node, criterion) - (r[0]! / w) * impurityOf(r, criterion) - (l[0]! / w) * impurityOf(l, criterion));
}

interface Split {
  feature: number;
  threshold: number;
  /** Where the node's rows start in each column's order, and how many go left. */
  start: number;
  pos: number;
  improvement: number;
  /** The statistics of the rows that go left and of those that go right. */
  ls: Float64Array;
  rs: Float64Array;
}

/**
 * Each column's rows sorted by value, then by row, and their values in that order: the classic presort of CART. A
 * node's rows are a range of every column's order, split by a stable partition, so a node never sorts and the order is
 * the one a sort at each node would give. The values are 32-bit floats, as the trees read X.
 */
export interface Presorted {
  order: Uint32Array[];
  values: Float32Array[];
}

/**
 * X's rows sorted by each column, by value then by row. X holds 32-bit floats (prepareX), so a value sorts by its
 * bits, turned so they order as the values do; a stable radix sort of those, 11 bits a pass, keeps equal values in row
 * order.
 */
export function presort(X: Matrix): Presorted {
  const n = X.rows;
  const order: Uint32Array[] = [];
  const values: Float32Array[] = [];
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const keys = new Uint32Array(n);
  const keys2 = new Uint32Array(n);
  const rows1 = new Uint32Array(n);
  const rows2 = new Uint32Array(n);
  const counts = new Uint32Array(2048);
  for (let f = 0; f < X.cols; f++) {
    const col = X.column(f);
    let o = rows1;
    let k1 = keys;
    let o2 = rows2;
    let k2 = keys2;
    for (let i = 0; i < n; i++) {
      const x = col[i]!;
      if (Math.fround(x) !== x) throw new Error('trees presort 32-bit floats only (prepareX)');
      f32[0] = x + 0; // −0 sorts with +0, as the values compare
      const b = u32[0]!;
      k1[i] = b & 0x80000000 ? ~b : b | 0x80000000;
      o[i] = i;
    }
    for (const shift of [0, 11, 22]) {
      counts.fill(0);
      for (let i = 0; i < n; i++) counts[(k1[i]! >>> shift) & 2047]!++;
      let sum = 0;
      for (let d = 0; d < 2048; d++) {
        const c = counts[d]!;
        counts[d] = sum;
        sum += c;
      }
      for (let i = 0; i < n; i++) {
        const k = k1[i]!;
        const at = counts[(k >>> shift) & 2047]!++;
        k2[at] = k;
        o2[at] = o[i]!;
      }
      [k1, k2] = [k2, k1];
      [o, o2] = [o2, o];
    }
    const sortedRows = o.slice();
    const v = new Float32Array(n);
    for (let k = 0; k < n; k++) v[k] = col[sortedRows[k]!]!;
    order.push(sortedRows);
    values.push(v);
  }
  return { order, values };
}

/** Matrices already prepared for trees, with their presort where one was made. */
const PREPARED = new WeakMap<Matrix, Presorted | null>();

/**
 * X as the trees read it, checked: scikit-learn's trees read X as 32-bit floats, and split between those values, so
 * the same here, and the thresholds and every comparison are the same. A matrix prepared already is returned as it is.
 */
export function prepareX(X: MatrixInput): Matrix {
  if (X instanceof Matrix && PREPARED.has(X)) return X;
  const m = Matrix.from(X);
  if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
  // scikit-learn's trees learn where missing values go; smaltml's don't yet.
  for (let k = 0; k < m.data.length; k++) {
    if (Number.isNaN(m.data[k]!)) fail('ML_VALUE', `X column ${m.columnName(Math.floor(k / m.rows))} has a missing value (NaN): trees don't take missing values yet; fill them first (SimpleImputer)`);
  }
  assertFinite(m);
  const data = new Float64Array(m.data.length);
  for (let k = 0; k < data.length; k++) data[k] = Math.fround(m.data[k]!);
  return new Matrix(m.rows, m.cols, data, m.names);
}

/**
 * Marks a matrix whose values are 32-bit floats, already checked, as prepared for trees (a forest's X, or rows of it),
 * and presorts it if `sort`: every tree grown on it then shares the one presort.
 */
export function asPrepared(m: Matrix, sort: boolean): Matrix {
  PREPARED.set(m, sort ? presort(m) : null);
  return m;
}

/** Grows a tree on X's rows of weight above 0, with these targets and weights; `sorted` is X's presort, if made. */
export function grow(X: Matrix, t: Targets, weights: Float64Array, params: GrowParams, sorted?: Presorted): TreeArrays {
  const n = X.rows;
  const p = X.cols;
  const S = statsSize(t);
  const width = t.kind === 'regression' ? 1 : t.K;
  const rng = new Random(params.seed);
  let total = 0;
  let kept = 0;
  for (let i = 0; i < n; i++) {
    total += weights[i]!;
    if (weights[i]! > 0) kept++;
  }

  const left: number[] = [];
  const right: number[] = [];
  const feature: number[] = [];
  const threshold: number[] = [];
  const impurity: number[] = [];
  const nSamples: number[] = [];
  const weighted: number[] = [];
  const value: number[] = [];

  // Rows of weight 0 take no part, as in scikit-learn: not in a node's sample count, and not a threshold's neighbour.
  // Each column's order and values, of the rows kept: written whether kept or not, and the cursor moves only for a
  // kept row (one slot to spare for the last write).
  const presorted = sorted ?? PREPARED.get(X) ?? presort(X);
  const keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) keep[i] = weights[i]! > 0 ? 1 : 0;
  const cols: Uint32Array[] = [];
  const vals: Float32Array[] = [];
  for (let f = 0; f < p; f++) {
    const order = presorted.order[f]!;
    const v = presorted.values[f]!;
    if (kept === n) {
      cols.push(order.slice());
      vals.push(v.slice());
      continue;
    }
    const o = new Uint32Array(kept + 1);
    const w = new Float32Array(kept + 1);
    let a = 0;
    for (let k = 0; k < n; k++) {
      const i = order[k]!;
      o[a] = i;
      w[a] = v[k]!;
      a += keep[i]!;
    }
    cols.push(o);
    vals.push(w);
  }
  const all = new Uint32Array(kept);
  for (let i = 0, a = 0; i < n; i++) if (weights[i]! > 0) all[a++] = i;

  const statsOf = (rows: Uint32Array, a: number, b: number) => {
    const s = new Float64Array(S);
    for (let k = a; k < b; k++) {
      const i = rows[k]!;
      addRow(s, t, i, weights[i]!);
    }
    return s;
  };

  const addNode = (m: number, s: Float64Array): number => {
    const id = left.length;
    left.push(LEAF);
    right.push(LEAF);
    feature.push(UNDEFINED);
    threshold.push(UNDEFINED);
    impurity.push(impurityOf(s, params.criterion));
    nSamples.push(m);
    weighted.push(s[0]!);
    if (t.kind === 'regression') value.push(s[0]! > 0 ? s[1]! / s[0]! : 0);
    else for (let k = 0; k < t.K; k++) value.push(s[0]! > 0 ? s[1 + k]! / s[0]! : 0);
    return id;
  };

  const l = new Float64Array(S);
  const r = new Float64Array(S);
  const labels = t.kind === 'classes' ? t.y : null;
  const gini = params.criterion === 'gini';
  /** The best split of the rows `start` to `end` of each column's order, or null where the node is a leaf. */
  const findSplit = (start: number, end: number, s: Float64Array, depth: number, nodeImpurity: number): Split | null => {
    const m = end - start;
    const w = s[0]!;
    if (depth >= params.maxDepth || m < params.minSamplesSplit || m < 2 * params.minSamplesLeaf || w < 2 * params.minWeightLeaf || nodeImpurity <= EPSILON) {
      return null;
    }
    // Columns in a random order: it decides only between splits that score exactly the same, and which columns
    // maxFeatures looks at. Constant columns don't count towards maxFeatures.
    const order = rng.permutation(p);
    let best: Split | null = null;
    let bestScore = -Infinity;
    let visited = 0;
    for (const f of order) {
      if (visited >= params.maxFeatures) break;
      const sortedRows = cols[f]!;
      const values = vals[f]!.subarray(start, end);
      if (values[0] === values[m - 1]) continue;
      visited++;
      if (params.splitter === 'random') {
        // One threshold, drawn uniformly between the node's smallest and largest value (one at the largest is the
        // smallest, so both sides keep a row); the rows at or below it go left, a binary search in the sorted values.
        const lo = values[0]!;
        const hi = values[m - 1]!;
        let th = lo + (hi - lo) * rng.next();
        if (th === hi) th = lo;
        let a = 0;
        let b = m;
        while (a < b) {
          const mid = (a + b) >>> 1;
          if (values[mid]! <= th) a = mid + 1;
          else b = mid;
        }
        const nl = a;
        if (nl < params.minSamplesLeaf || m - nl < params.minSamplesLeaf) continue;
        // The left rows' sums; the right's are the node's less them.
        l.fill(0);
        for (let k = 0; k < nl; k++) {
          const i = sortedRows[start + k]!;
          addRow(l, t, i, weights[i]!);
        }
        for (let q = 0; q < S; q++) r[q] = s[q]! - l[q]!;
        if (l[0]! < params.minWeightLeaf || r[0]! < params.minWeightLeaf) continue;
        const score = gini ? -(l[0]! * giniOf(l) + r[0]! * giniOf(r)) : proxy(l, r, params.criterion);
        if (score > bestScore) {
          bestScore = score;
          best = { feature: f, threshold: th, start, pos: nl, improvement: 0, ls: l, rs: r };
        }
        continue;
      }
      l.fill(0);
      r.set(s);
      for (let k = 0; k < m - 1; k++) {
        const i = sortedRows[start + k]!;
        const wi = weights[i]!;
        if (labels) {
          // addRow for a class, written out: the same sums in the same order.
          const c = 1 + labels[i]!;
          l[0]! += wi;
          l[c]! += wi;
          r[0]! += -wi;
          r[c]! += -wi;
        } else {
          addRow(l, t, i, wi);
          addRow(r, t, i, wi, -1);
        }
        if (values[k + 1] === values[k]) continue;
        const nl = k + 1;
        const nr = m - nl;
        if (nl < params.minSamplesLeaf || nr < params.minSamplesLeaf) continue;
        if (l[0]! < params.minWeightLeaf || r[0]! < params.minWeightLeaf) continue;
        const score = gini ? -(l[0]! * giniOf(l) + r[0]! * giniOf(r)) : proxy(l, r, params.criterion);
        if (score > bestScore) {
          bestScore = score;
          let th = (values[k]! + values[k + 1]!) / 2;
          if (th === values[k + 1] || !Number.isFinite(th)) th = values[k]!;
          best = { feature: f, threshold: th, start, pos: nl, improvement: 0, ls: l, rs: r };
        }
      }
    }
    if (!best) return null;
    // The children's statistics summed afresh, each in its rows' order by the split column, and the improvement from
    // them.
    const byBest = cols[best.feature]!;
    best.ls = statsOf(byBest, start, start + best.pos);
    best.rs = statsOf(byBest, start + best.pos, end);
    best.improvement = improvement(s, best.ls, best.rs, params.criterion, total);
    if (best.improvement + EPSILON < params.minImpurityDecrease) return null;
    return best;
  };

  const inLeft = new Uint8Array(n);
  const scratch = new Uint32Array(kept);
  const scratchValues = new Float32Array(kept);
  /**
   * Splits the node's range of every column's order in place, by a stable partition: the rows that go left first,
   * then those that go right, each in the order they were. The split column's order is already so.
   */
  const partition = (split: Split, end: number): void => {
    const { start, pos } = split;
    const goes = cols[split.feature]!;
    for (let k = start; k < end; k++) inLeft[goes[k]!] = k < start + pos ? 1 : 0;
    for (let f = 0; f < p; f++) {
      if (f === split.feature) continue;
      const c = cols[f]!;
      const v = vals[f]!;
      let a = start;
      let b = 0;
      // Each row is written to both places and only one cursor moves: no branch to mispredict.
      for (let k = start; k < end; k++) {
        const i = c[k]!;
        const x = v[k]!;
        const g = inLeft[i]!;
        c[a] = i;
        v[a] = x;
        scratch[b] = i;
        scratchValues[b] = x;
        a += g;
        b += 1 - g;
      }
      c.set(scratch.subarray(0, b), a);
      v.set(scratchValues.subarray(0, b), a);
    }
  };
  const rootStats = () => statsOf(all, 0, kept);
  if (params.maxLeafNodes === null) {
    // Depth first, left before right: nodes are numbered in that order. A stack of its own, not recursion, so a
    // deep tree can't overflow the engine's.
    const stack: { start: number; end: number; s: Float64Array; depth: number; parent: number; isLeft: boolean }[] = [
      { start: 0, end: kept, s: rootStats(), depth: 0, parent: -1, isLeft: true },
    ];
    while (stack.length) {
      const { start, end, s, depth, parent, isLeft } = stack.pop()!;
      const id = addNode(end - start, s);
      if (parent >= 0) (isLeft ? left : right)[parent] = id;
      const split = findSplit(start, end, s, depth, impurity[id]!);
      if (!split) continue;
      feature[id] = split.feature;
      threshold[id] = split.threshold;
      const mid = start + split.pos;
      partition(split, end);
      stack.push({ start: mid, end, s: split.rs, depth: depth + 1, parent: id, isLeft: false });
      stack.push({ start, end: mid, s: split.ls, depth: depth + 1, parent: id, isLeft: true });
    }
  } else {
    // Best first: the leaf whose split improves most is split next, until maxLeafNodes leaves. A node's children are
    // numbered when it is split, left then right.
    type Open = { id: number; split: Split; end: number; depth: number; order: number };
    const open: Open[] = [];
    let order = 0;
    const consider = (start: number, end: number, s: Float64Array, depth: number) => {
      const id = addNode(end - start, s);
      const split = findSplit(start, end, s, depth, impurity[id]!);
      if (split) open.push({ id, split, end, depth, order: order++ });
    };
    consider(0, kept, rootStats(), 0);
    let leaves = 1;
    while (open.length > 0 && leaves < params.maxLeafNodes) {
      // The largest improvement; of equals, the one found first.
      let b = 0;
      for (let k = 1; k < open.length; k++) {
        const a = open[k]!;
        const c = open[b]!;
        if (a.split.improvement > c.split.improvement || (a.split.improvement === c.split.improvement && a.order < c.order)) b = k;
      }
      const { id, split, end, depth } = open.splice(b, 1)[0]!;
      feature[id] = split.feature;
      threshold[id] = split.threshold;
      const mid = split.start + split.pos;
      partition(split, end);
      left[id] = left.length;
      consider(split.start, mid, split.ls, depth + 1);
      right[id] = left.length;
      consider(mid, end, split.rs, depth + 1);
      leaves++;
    }
  }

  return {
    left: Int32Array.from(left),
    right: Int32Array.from(right),
    feature: Int32Array.from(feature),
    threshold: Float64Array.from(threshold),
    impurity: Float64Array.from(impurity),
    nSamples: Int32Array.from(nSamples),
    weightedNSamples: Float64Array.from(weighted),
    value: Float64Array.from(value),
    width,
  };
}

/** The leaf each row of X falls in. */
export function applyTree(tree: TreeArrays, X: Matrix): Int32Array {
  const n = X.rows;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let node = 0;
    while (tree.left[node] !== LEAF) {
      const v = X.data[tree.feature[node]! * n + i]!;
      node = v <= tree.threshold[node]! ? tree.left[node]! : tree.right[node]!;
    }
    out[i] = node;
  }
  return out;
}

/** Each column's total weighted impurity decrease over the tree's splits, normalized to sum to 1 (all 0 for a stump). */
export function featureImportances(tree: TreeArrays, p: number): Float64Array {
  const out = new Float64Array(p);
  for (let node = 0; node < tree.left.length; node++) {
    const l = tree.left[node]!;
    if (l === LEAF) continue;
    const r = tree.right[node]!;
    out[tree.feature[node]!]! +=
      tree.weightedNSamples[node]! * tree.impurity[node]! - tree.weightedNSamples[l]! * tree.impurity[l]! - tree.weightedNSamples[r]! * tree.impurity[r]!;
  }
  let sum = 0;
  for (let j = 0; j < p; j++) sum += out[j]!;
  if (sum > 0) for (let j = 0; j < p; j++) out[j]! /= sum;
  return out;
}

/** The tree's depth: 0 for a single leaf. */
export function depthOf(tree: TreeArrays): number {
  let deepest = 0;
  const stack: [number, number][] = [[0, 0]];
  while (stack.length) {
    const [node, d] = stack.pop()!;
    if (d > deepest) deepest = d;
    if (tree.left[node] !== LEAF) stack.push([tree.left[node]!, d + 1], [tree.right[node]!, d + 1]);
  }
  return deepest;
}
