import { Random } from '../core/random.js';
import type { Matrix } from '../core/matrix.js';

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
  /** Rows of the node, reordered so the first `pos` go left. */
  rows: Uint32Array;
  pos: number;
  improvement: number;
}

/** Grows a tree on X's rows with these targets and weights. */
export function grow(X: Matrix, t: Targets, weights: Float64Array, params: GrowParams): TreeArrays {
  const n = X.rows;
  const p = X.cols;
  const S = statsSize(t);
  const width = t.kind === 'regression' ? 1 : t.K;
  const rng = new Random(params.seed);
  let total = 0;
  for (let i = 0; i < n; i++) total += weights[i]!;

  const left: number[] = [];
  const right: number[] = [];
  const feature: number[] = [];
  const threshold: number[] = [];
  const impurity: number[] = [];
  const nSamples: number[] = [];
  const weighted: number[] = [];
  const value: number[] = [];

  const statsOf = (rows: Uint32Array) => {
    const s = new Float64Array(S);
    for (const i of rows) addRow(s, t, i, weights[i]!);
    return s;
  };

  const addNode = (rows: Uint32Array, s: Float64Array): number => {
    const id = left.length;
    left.push(LEAF);
    right.push(LEAF);
    feature.push(UNDEFINED);
    threshold.push(UNDEFINED);
    impurity.push(impurityOf(s, params.criterion));
    nSamples.push(rows.length);
    weighted.push(s[0]!);
    if (t.kind === 'regression') value.push(s[0]! > 0 ? s[1]! / s[0]! : 0);
    else for (let k = 0; k < t.K; k++) value.push(s[0]! > 0 ? s[1 + k]! / s[0]! : 0);
    return id;
  };

  /** The best split of these rows, or null where the node is a leaf. */
  const findSplit = (rows: Uint32Array, s: Float64Array, depth: number, nodeImpurity: number): Split | null => {
    const m = rows.length;
    const w = s[0]!;
    if (depth >= params.maxDepth || m < params.minSamplesSplit || m < 2 * params.minSamplesLeaf || w < 2 * params.minWeightLeaf || nodeImpurity <= EPSILON) {
      return null;
    }
    // Columns in a random order: it decides only between splits that score exactly the same, and which columns
    // maxFeatures looks at. Constant columns don't count towards maxFeatures.
    const order = rng.permutation(p);
    const sorted = new Uint32Array(m);
    const values = new Float64Array(m);
    const l = new Float64Array(S);
    const r = new Float64Array(S);
    let best: Split | null = null;
    let bestScore = -Infinity;
    let visited = 0;
    for (const f of order) {
      if (visited >= params.maxFeatures) break;
      const col = X.column(f);
      sorted.set(rows);
      sorted.sort((a, b) => col[a]! - col[b]! || a - b);
      for (let k = 0; k < m; k++) values[k] = col[sorted[k]!]!;
      if (values[0] === values[m - 1]) continue;
      visited++;
      l.fill(0);
      r.set(s);
      for (let k = 0; k < m - 1; k++) {
        const i = sorted[k]!;
        addRow(l, t, i, weights[i]!);
        addRow(r, t, i, weights[i]!, -1);
        if (values[k + 1] === values[k]) continue;
        const nl = k + 1;
        const nr = m - nl;
        if (nl < params.minSamplesLeaf || nr < params.minSamplesLeaf) continue;
        if (l[0]! < params.minWeightLeaf || r[0]! < params.minWeightLeaf) continue;
        const score = proxy(l, r, params.criterion);
        if (score > bestScore) {
          bestScore = score;
          let th = (values[k]! + values[k + 1]!) / 2;
          if (th === values[k + 1] || !Number.isFinite(th)) th = values[k]!;
          best = { feature: f, threshold: th, rows: sorted.slice(), pos: nl, improvement: 0 };
        }
      }
    }
    if (!best) return null;
    // The improvement, from the chosen split's own statistics.
    const ls = statsOf(best.rows.subarray(0, best.pos));
    const rs = statsOf(best.rows.subarray(best.pos));
    best.improvement = improvement(s, ls, rs, params.criterion, total);
    if (best.improvement + EPSILON < params.minImpurityDecrease) return null;
    return best;
  };

  const all = Uint32Array.from({ length: n }, (_, i) => i);
  if (params.maxLeafNodes === null) {
    // Depth first, left before right: nodes are numbered in that order. A stack of its own, not recursion, so a
    // deep tree can't overflow the engine's.
    const stack: { rows: Uint32Array; depth: number; parent: number; isLeft: boolean }[] = [{ rows: all, depth: 0, parent: -1, isLeft: true }];
    while (stack.length) {
      const { rows, depth, parent, isLeft } = stack.pop()!;
      const s = statsOf(rows);
      const id = addNode(rows, s);
      if (parent >= 0) (isLeft ? left : right)[parent] = id;
      const split = findSplit(rows, s, depth, impurity[id]!);
      if (!split) continue;
      feature[id] = split.feature;
      threshold[id] = split.threshold;
      stack.push({ rows: split.rows.slice(split.pos), depth: depth + 1, parent: id, isLeft: false });
      stack.push({ rows: split.rows.slice(0, split.pos), depth: depth + 1, parent: id, isLeft: true });
    }
  } else {
    // Best first: the leaf whose split improves most is split next, until maxLeafNodes leaves. A node's children are
    // numbered when it is split, left then right.
    type Open = { id: number; split: Split; depth: number; order: number };
    const open: Open[] = [];
    let order = 0;
    const consider = (rows: Uint32Array, depth: number) => {
      const s = statsOf(rows);
      const id = addNode(rows, s);
      const split = findSplit(rows, s, depth, impurity[id]!);
      if (split) open.push({ id, split, depth, order: order++ });
    };
    consider(all, 0);
    let leaves = 1;
    while (open.length > 0 && leaves < params.maxLeafNodes) {
      // The largest improvement; of equals, the one found first.
      let b = 0;
      for (let k = 1; k < open.length; k++) {
        const a = open[k]!;
        const c = open[b]!;
        if (a.split.improvement > c.split.improvement || (a.split.improvement === c.split.improvement && a.order < c.order)) b = k;
      }
      const { id, split, depth } = open.splice(b, 1)[0]!;
      feature[id] = split.feature;
      threshold[id] = split.threshold;
      left[id] = left.length;
      consider(split.rows.slice(0, split.pos), depth + 1);
      right[id] = left.length;
      consider(split.rows.slice(split.pos), depth + 1);
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
