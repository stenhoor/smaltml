/**
 * Model selection: splitting data to test a model on rows it didn't see, cross-validation, and grid search.
 * scikit-learn's `model_selection`.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { toLabels, uniqueLabels, type Labels } from '../core/model.js';
import { Random } from '../core/random.js';

export interface TrainTestSplitOptions {
  /** The test set's share of the rows (0 to 1), or its number of rows. Default 0.25, or what trainSize leaves. */
  testSize?: number;
  /** The training set's share of the rows (0 to 1), or its number of rows. Default: what testSize leaves. */
  trainSize?: number;
  /** Shuffle the rows before splitting. Default true. */
  shuffle?: boolean;
  /** The seed for the shuffle: the same seed gives the same split. */
  seed?: number;
  /**
   * Labels to keep in proportion: each class has the same share of the training and the test set as of all the rows,
   * as near as whole rows allow. Usually y. scikit-learn's `stratify`. Needs `shuffle`.
   */
  stratify?: Labels;
}

export interface TrainTestSplit<Y> {
  XTrain: Matrix;
  XTest: Matrix;
  yTrain: Y;
  yTest: Y;
}

/**
 * Splits X (and y, row for row) into a training set and a test set. scikit-learn's `train_test_split`: the same
 * sizes, rounded the same way (the test set rounds up). The shuffle is smaltml's own, so a seed gives the same split
 * every time, but not scikit-learn's split.
 */
export function trainTestSplit<Y extends Labels>(X: MatrixInput, y: Y, options?: TrainTestSplitOptions): TrainTestSplit<Y extends readonly string[] ? string[] : Y extends Float64Array ? Float64Array : number[]>;
export function trainTestSplit(X: MatrixInput, y?: undefined, options?: TrainTestSplitOptions): Omit<TrainTestSplit<never>, 'yTrain' | 'yTest'>;
export function trainTestSplit(X: MatrixInput, y?: Labels, options: TrainTestSplitOptions = {}): Partial<TrainTestSplit<unknown>> {
  const m = Matrix.from(X);
  const n = m.rows;
  if (y !== undefined && y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
  const [nTrain, nTest] = sizes(n, options.testSize, options.trainSize);
  const shuffle = options.shuffle ?? true;
  let train: Uint32Array;
  let test: Uint32Array;
  if (options.stratify !== undefined) {
    if (!shuffle) fail('ML_PARAM', 'stratify needs shuffle: a split in order can\'t keep the classes in proportion');
    [train, test] = stratifiedSplit(toLabels(options.stratify, n), nTrain, nTest, new Random(options.seed ?? Date.now()));
  } else {
    const order = shuffle ? new Random(options.seed ?? Date.now()).permutation(n) : Uint32Array.from({ length: n }, (_, i) => i);
    train = order.subarray(0, nTrain);
    test = order.subarray(nTrain, nTrain + nTest);
  }
  const out: Partial<TrainTestSplit<unknown>> = { XTrain: m.take(train), XTest: m.take(test) };
  if (y !== undefined) {
    out.yTrain = pick(y, train);
    out.yTest = pick(y, test);
  }
  return out;
}

/**
 * Rows for each set, class by class: the training set's count of each class first, then the test set's from the rows
 * left, both by scikit-learn's rule (StratifiedShuffleSplit, `_approximate_mode`). Which rows, and the ties in the
 * counts, come from smaltml's own shuffle.
 */
function stratifiedSplit(labels: readonly (string | number)[], nTrain: number, nTest: number, rng: Random): [Uint32Array, Uint32Array] {
  const classes = uniqueLabels(labels);
  const index = new Map(classes.map((c, k) => [c, k]));
  const rows: number[][] = classes.map(() => []);
  labels.forEach((l, i) => rows[index.get(l)!]!.push(i));
  const counts = rows.map((r) => r.length);
  const few = classes.filter((_, k) => counts[k]! < 2);
  if (few.length) fail('ML_LABELS', `stratify: each class needs at least 2 rows; ${few.map(String).join(', ')} ha${few.length === 1 ? 's' : 've'} 1`);
  if (nTrain < classes.length) fail('ML_PARAM', `stratify: a training set of ${nTrain} rows can't hold ${classes.length} classes`);
  if (nTest < classes.length) fail('ML_PARAM', `stratify: a test set of ${nTest} rows can't hold ${classes.length} classes`);
  const inTrain = approximateMode(counts, nTrain, rng);
  const inTest = approximateMode(counts.map((c, k) => c - inTrain[k]!), nTest, rng);
  const train: number[] = [];
  const test: number[] = [];
  rows.forEach((r, k) => {
    const order = rng.permutation(r.length);
    for (let i = 0; i < inTrain[k]!; i++) train.push(r[order[i]!]!);
    for (let i = inTrain[k]!; i < inTrain[k]! + inTest[k]!; i++) test.push(r[order[i]!]!);
  });
  const shuffled = (a: number[]) => Uint32Array.from(rng.permutation(a.length), (i) => a[i]!);
  return [shuffled(train), shuffled(test)];
}

/**
 * How many of `draws` rows to take from each class: each class's share, floored, then one more for the classes with
 * the largest remainders until the total is reached, ties broken at random. scikit-learn's rule.
 */
function approximateMode(counts: readonly number[], draws: number, rng: Random): number[] {
  const total = counts.reduce((a, b) => a + b, 0);
  const continuous = counts.map((c) => (c / total) * draws);
  const out = continuous.map(Math.floor);
  let need = draws - out.reduce((a, b) => a + b, 0);
  const remainder = continuous.map((v, k) => v - out[k]!);
  for (const value of [...new Set(remainder)].sort((a, b) => b - a)) {
    if (need <= 0) break;
    const tied = remainder.flatMap((v, k) => (v === value ? [k] : []));
    const take = Math.min(tied.length, need);
    const order = rng.permutation(tied.length);
    for (let i = 0; i < take; i++) out[tied[order[i]!]!]!++;
    need -= take;
  }
  return out;
}

function pick(y: Labels, index: Uint32Array): unknown {
  if (y instanceof Float64Array) return Float64Array.from(index, (i) => y[i]!);
  return Array.from(index, (i) => y[i]);
}

/** The training and test sizes, as scikit-learn's `_validate_shuffle_split` works them out. */
function sizes(n: number, testSize: number | undefined, trainSize: number | undefined): [number, number] {
  const check = (v: number | undefined, name: string) => {
    if (v === undefined) return;
    const ok = (v > 0 && v < 1) || (Number.isInteger(v) && v > 0 && v < n);
    if (!ok) fail('ML_PARAM', `${name} must be a share between 0 and 1, or a whole number of rows from 1 to ${n - 1}, not ${v}`);
  };
  check(testSize, 'testSize');
  check(trainSize, 'trainSize');
  const test = testSize ?? (trainSize === undefined ? 0.25 : undefined);
  let nTest = test === undefined ? undefined : test < 1 ? Math.ceil(test * n) : test;
  let nTrain = trainSize === undefined ? undefined : trainSize < 1 ? Math.floor(trainSize * n) : trainSize;
  if (nTrain === undefined) nTrain = n - nTest!;
  if (nTest === undefined) nTest = n - nTrain;
  if (nTrain + nTest > n) fail('ML_PARAM', `a training set of ${nTrain} and a test set of ${nTest} need ${nTrain + nTest} rows; X has ${n}`);
  if (nTrain === 0) fail('ML_PARAM', `with ${n} rows and these sizes, the training set would be empty`);
  return [nTrain, nTest];
}

export { GroupKFold, KFold, StratifiedKFold, type Fold, type GroupKFoldParams, type Groups, type KFoldParams, type Splitter } from './folds.js';
export {
  GridSearchCV,
  SCORER_NAMES,
  cloneModel,
  crossValScore,
  crossValidate,
  parameterGrid,
  type CrossValOptions,
  type CrossValidateOptions,
  type CrossValidateResult,
  type CvResults,
  type ScorerResults,
  type Scorers,
  type GridSearchOptions,
  type ParamGrid,
  type Scorer,
  type ScorerName,
} from './search.js';
