import { fail } from '../core/errors.js';
import { Matrix, toVector, type MatrixInput, type VectorInput } from '../core/matrix.js';
import { toLabels, uniqueLabels, type Label, type Labels } from '../core/model.js';

/** The mean of the absolute errors: Σ |yᵢ − ŷᵢ| / n. scikit-learn's `mean_absolute_error`. */
export function meanAbsoluteError(yTrue: VectorInput, yPred: VectorInput): number {
  const t = toVector(yTrue, yTrue?.length ?? 0, 'yTrue');
  if (t.length === 0) fail('ML_SHAPE', 'yTrue is empty');
  const p = toVector(yPred, t.length, 'yPred');
  let s = 0;
  for (let i = 0; i < t.length; i++) s += Math.abs(t[i]! - p[i]!);
  return s / t.length;
}

export interface ConfusionMatrixOptions {
  /** The labels, in the order of the rows and columns. Default: every label in yTrue or yPred, sorted. */
  labels?: readonly Label[];
}

/**
 * Counts of true labels (rows) against predicted labels (columns): entry (i, j) is how many rows of class i were
 * predicted as class j. scikit-learn's `confusion_matrix`. The Matrix's columns are named after the labels.
 */
export function confusionMatrix(yTrue: Labels, yPred: Labels, options: ConfusionMatrixOptions = {}): Matrix {
  const [t, p] = labelPair(yTrue, yPred);
  const labels = options.labels ? [...options.labels] : uniqueLabels([...t, ...p]);
  const index = new Map(labels.map((l, k) => [l, k]));
  const k = labels.length;
  const out = new Float64Array(k * k);
  for (let i = 0; i < t.length; i++) {
    const a = index.get(t[i]!);
    const b = index.get(p[i]!);
    if (a !== undefined && b !== undefined) out[b * k + a]! += 1;
  }
  return new Matrix(k, k, out, labels.map(String));
}

export type Average = 'binary' | 'micro' | 'macro' | 'weighted' | null;

export interface PrecisionRecallOptions {
  /** How per-class scores combine: `'binary'` (the posLabel class only), `'micro'`, `'macro'`, `'weighted'`, or null for one score per class. Default `'binary'`. */
  average?: Average;
  /** The positive class for `'binary'`. Default 1. */
  posLabel?: Label;
  /** The classes to score, and their order for `average: null`. Default: every label in yTrue or yPred, sorted. */
  labels?: readonly Label[];
  /** The score where its denominator is 0 (no predictions, or no true rows, of a class). Default 0. */
  zeroDivision?: number;
}

/** Per-class counts: true positives, predicted and true totals. */
function counts(t: Label[], p: Label[], labels: readonly Label[]) {
  const index = new Map(labels.map((l, k) => [l, k]));
  const tp = new Float64Array(labels.length);
  const pred = new Float64Array(labels.length);
  const support = new Float64Array(labels.length);
  for (let i = 0; i < t.length; i++) {
    const a = index.get(t[i]!);
    const b = index.get(p[i]!);
    if (a !== undefined) support[a]! += 1;
    if (b !== undefined) pred[b]! += 1;
    if (a !== undefined && a === b) tp[a]! += 1;
  }
  return { tp, pred, support };
}

type Which = 'precision' | 'recall' | 'f1';

function score(which: Which, yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions): number | Float64Array {
  const [t, p] = labelPair(yTrue, yPred);
  const average = options.average === undefined ? 'binary' : options.average;
  const zero = options.zeroDivision ?? 0;
  const present = uniqueLabels([...t, ...p]);
  let labels = options.labels ? [...options.labels] : present;
  if (average === 'binary') {
    const pos = options.posLabel ?? 1;
    if (present.length > 2) fail('ML_LABELS', `average 'binary' needs two classes; y has ${present.length}: choose average 'micro', 'macro' or 'weighted'`);
    if (present.length > 0 && !present.includes(pos)) {
      if (present.length === 2 || typeof pos !== typeof present[0]) fail('ML_LABELS', `posLabel ${String(pos)} isn't one of the labels (${present.map(String).join(', ')})`);
    }
    labels = [pos];
  }
  const { tp, pred, support } = counts(t, p, labels);
  const ratio = (a: number, b: number) => (b === 0 ? zero : a / b);
  const one = (k: number): number => {
    if (which === 'precision') return ratio(tp[k]!, pred[k]!);
    if (which === 'recall') return ratio(tp[k]!, support[k]!);
    return ratio(2 * tp[k]!, pred[k]! + support[k]!);
  };
  if (average === 'micro') {
    const TP = tp.reduce((a, b) => a + b, 0);
    const P = pred.reduce((a, b) => a + b, 0);
    const S = support.reduce((a, b) => a + b, 0);
    return which === 'precision' ? ratio(TP, P) : which === 'recall' ? ratio(TP, S) : ratio(2 * TP, P + S);
  }
  const per = Float64Array.from(labels, (_, k) => one(k));
  if (average === null) return per;
  if (average === 'binary') return per[0]!;
  if (average === 'macro') return per.reduce((a, b) => a + b, 0) / per.length;
  if (average === 'weighted') {
    const total = support.reduce((a, b) => a + b, 0);
    if (total === 0) return zero;
    return per.reduce((a, v, k) => a + v * support[k]!, 0) / total;
  }
  return fail('ML_PARAM', `average is 'binary', 'micro', 'macro', 'weighted' or null, not ${String(average)}`);
}

/** Of the rows predicted as a class, the share that are: TP / (TP + FP). scikit-learn's `precision_score`. */
export function precisionScore(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions & { average: null }): Float64Array;
export function precisionScore(yTrue: Labels, yPred: Labels, options?: PrecisionRecallOptions): number;
export function precisionScore(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions = {}): number | Float64Array {
  return score('precision', yTrue, yPred, options);
}

/** Of the rows of a class, the share predicted as it: TP / (TP + FN). scikit-learn's `recall_score`. */
export function recallScore(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions & { average: null }): Float64Array;
export function recallScore(yTrue: Labels, yPred: Labels, options?: PrecisionRecallOptions): number;
export function recallScore(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions = {}): number | Float64Array {
  return score('recall', yTrue, yPred, options);
}

/** The harmonic mean of precision and recall: 2·TP / (2·TP + FP + FN). scikit-learn's `f1_score`. */
export function f1Score(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions & { average: null }): Float64Array;
export function f1Score(yTrue: Labels, yPred: Labels, options?: PrecisionRecallOptions): number;
export function f1Score(yTrue: Labels, yPred: Labels, options: PrecisionRecallOptions = {}): number | Float64Array {
  return score('f1', yTrue, yPred, options);
}

export interface RocAucOptions {
  /** With more than two classes: `'ovr'`, each class against the rest. Required there, as in scikit-learn. */
  multiClass?: 'ovr';
  /** With more than two classes: the mean of the classes' areas (`'macro'`), or weighted by their rows (`'weighted'`). Default `'macro'`. */
  average?: 'macro' | 'weighted';
  /** The classes, in the order of yScore's columns. Default: yTrue's labels, sorted. */
  labels?: readonly Label[];
}

/**
 * The area under the ROC curve: the chance that a random positive row scores above a random negative one (ties count
 * half). For two classes, yScore is the second (greater) class's score; for more, a column per class with
 * multiClass `'ovr'`. scikit-learn's `roc_auc_score`.
 */
export function rocAucScore(yTrue: Labels, yScore: ArrayLike<number> | MatrixInput, options: RocAucOptions = {}): number {
  const t = toLabels(yTrue, yTrue?.length ?? 0);
  const n = t.length;
  if (n === 0) fail('ML_SHAPE', 'yTrue is empty');
  const classes = options.labels ? [...options.labels] : uniqueLabels(t);
  if (classes.length < 2) fail('ML_LABELS', 'yTrue has one class only: the area under the ROC curve needs two');
  const flat = ArrayBuffer.isView(yScore) || (Array.isArray(yScore) && (yScore.length === 0 || typeof yScore[0] === 'number'));
  if (classes.length === 2 && flat) {
    const s = toVector(yScore as ArrayLike<number>, n, 'yScore');
    return auc(t.map((v) => v === classes[1]), s);
  }
  if (classes.length === 2) {
    const S = Matrix.from(yScore as MatrixInput);
    if (S.rows !== n || S.cols !== 2) fail('ML_SHAPE', `yScore has ${S.rows} × ${S.cols} values; it needs ${n} × 2, or ${n} values`);
    return auc(t.map((v) => v === classes[1]), S.column(1));
  }
  if (options.multiClass !== 'ovr') fail('ML_PARAM', "with more than two classes, pass multiClass: 'ovr' (each class against the rest)");
  if (flat) fail('ML_SHAPE', 'with more than two classes, yScore needs a column per class');
  const S = Matrix.from(yScore as MatrixInput);
  if (S.rows !== n || S.cols !== classes.length) fail('ML_SHAPE', `yScore has ${S.rows} × ${S.cols} values for ${n} rows and ${classes.length} classes`);
  const areas = classes.map((c, k) => auc(t.map((v) => v === c), S.column(k)));
  if ((options.average ?? 'macro') === 'macro') return areas.reduce((a, b) => a + b, 0) / areas.length;
  const support = classes.map((c) => t.filter((v) => v === c).length);
  return areas.reduce((a, v, k) => a + v * support[k]!, 0) / n;
}

/** The Mann–Whitney form of the area: (sum of the positives' ranks − n₊(n₊ + 1)/2) / (n₊ n₋), ties at their mean rank. */
function auc(positive: boolean[], score: ArrayLike<number>): number {
  const n = positive.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[a]! - score[b]!);
  let rankSum = 0;
  let pos = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j < n && score[order[j]!] === score[order[i]!]) j++;
    const rank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) if (positive[order[k]!]) rankSum += rank;
    i = j;
  }
  for (const p of positive) if (p) pos++;
  const neg = n - pos;
  if (pos === 0 || neg === 0) fail('ML_LABELS', 'only one class is present in yTrue for a class: its ROC area is undefined');
  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

function labelPair(yTrue: Labels, yPred: Labels): [Label[], Label[]] {
  const t = toLabels(yTrue, yTrue?.length ?? 0);
  if (t.length === 0) fail('ML_SHAPE', 'yTrue is empty');
  return [t, toLabels(yPred, t.length)];
}
