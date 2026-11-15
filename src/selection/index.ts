/**
 * Model selection: splitting data to test a model on rows it didn't see, cross-validation, and grid search.
 * scikit-learn's `model_selection`.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import type { Labels } from '../core/model.js';
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
  const order = shuffle ? new Random(options.seed ?? Date.now()).permutation(n) : Uint32Array.from({ length: n }, (_, i) => i);
  const train = order.subarray(0, nTrain);
  const test = order.subarray(nTrain, nTrain + nTest);
  const out: Partial<TrainTestSplit<unknown>> = { XTrain: m.take(train), XTest: m.take(test) };
  if (y !== undefined) {
    out.yTrain = pick(y, train);
    out.yTest = pick(y, test);
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

export { KFold, StratifiedKFold, type Fold, type KFoldParams, type Splitter } from './folds.js';
export {
  GridSearchCV,
  SCORER_NAMES,
  cloneModel,
  crossValScore,
  parameterGrid,
  type CrossValOptions,
  type CvResults,
  type GridSearchOptions,
  type ParamGrid,
  type Scorer,
  type ScorerName,
} from './search.js';
