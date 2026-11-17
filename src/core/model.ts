import { fail } from './errors.js';
import { Matrix, type MatrixInput } from './matrix.js';
import { VERSION } from '../version.js';

/** The format name and version of a saved model (ADR 0024, section 6). */
export const MODEL_FORMAT = 'smaltml';
export const MODEL_FORMAT_VERSION = 4;

/**
 * A saved model: plain JSON, kept as a file or as a row in a table, that loadModel() turns back into the model.
 * `params` are its parameters; `state` what fitting learned (null before it is fitted).
 */
export interface ModelDocument {
  format: typeof MODEL_FORMAT;
  version: number;
  /** The model's class, such as `LinearRegression`. */
  model: string;
  /** The smaltml version that saved it. */
  smaltml: string;
  params: Record<string, unknown>;
  state: Record<string, unknown> | null;
}

/** Class labels: numbers or strings, one per row. */
export type Labels = ArrayLike<number> | readonly string[];

/** One label. */
export type Label = number | string;

/** What every model has: a name, its parameters, and a saved form. */
export interface Model {
  /** The model's class, such as `LinearRegression`. */
  readonly kind: string;
  /** Whether fit() has run. */
  readonly fitted: boolean;
  /** The parameters, with their defaults filled in. */
  getParams(): object;
  /** The model as a ModelDocument, for JSON.stringify(). */
  toJSON(): ModelDocument;
}

/** A model that learns a transformation of X, such as a scaler. */
export interface Transformer extends Model {
  fit(X: MatrixInput, y?: Labels): Promise<this>;
  fitSync(X: MatrixInput, y?: Labels): this;
  transform(X: MatrixInput): Promise<Matrix>;
  transformSync(X: MatrixInput): Matrix;
  fitTransform(X: MatrixInput, y?: Labels): Promise<Matrix>;
}

/** A model that predicts a value or a label for each row. */
export interface Predictor extends Model {
  /** y: labels or values, one per row; a regressor also takes a column per target (a 2-D y). */
  fit(X: MatrixInput, y: Labels | MatrixInput): Promise<this>;
  fitSync(X: MatrixInput, y: Labels | MatrixInput): this;
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Float64Array | Label[]>;
  predictSync(X: MatrixInput): Float64Array | Label[];
}

/** A predictor of labels that also gives each class's probability. */
export interface Classifier extends Predictor {
  /** The labels seen in fit(), sorted: the order of predictProba()'s columns. */
  readonly classes: readonly Label[];
  predict(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Label[]>;
  predictSync(X: MatrixInput): Label[];
  predictProba(X: MatrixInput, options?: { device?: 'auto' | 'cpu' | 'gpu' }): Promise<Matrix>;
  predictProbaSync(X: MatrixInput): Matrix;
}

export function isTransformer(m: Model): m is Transformer {
  return typeof (m as Partial<Transformer>).transformSync === 'function';
}

export function isPredictor(m: Model): m is Predictor {
  return typeof (m as Partial<Predictor>).predictSync === 'function';
}

export function isClassifier(m: Model): m is Classifier {
  return typeof (m as Partial<Classifier>).predictProbaSync === 'function';
}

export function document(kind: string, params: Record<string, unknown>, state: Record<string, unknown> | null): ModelDocument {
  return { format: MODEL_FORMAT, version: MODEL_FORMAT_VERSION, model: kind, smaltml: VERSION, params, state };
}

/** Checks a document's format and kind before a model reads it. */
export function checkDocument(doc: unknown, kind?: string): ModelDocument {
  const d = doc as Partial<ModelDocument> | null;
  if (!d || typeof d !== 'object' || d.format !== MODEL_FORMAT) fail('ML_FORMAT', 'not a smaltml model: its format must be "smaltml"');
  if (typeof d.version !== 'number' || d.version > MODEL_FORMAT_VERSION) {
    fail('ML_FORMAT', `model format version ${d.version} is newer than this smaltml reads (${MODEL_FORMAT_VERSION}): update smaltml`);
  }
  if (kind && d.model !== kind) fail('ML_FORMAT', `a ${d.model} model, not a ${kind}`);
  if (!d.params || typeof d.params !== 'object') fail('ML_FORMAT', 'the model has no params');
  return d as ModelDocument;
}

/**
 * The document of the same model unfitted: its parameters without what it learned. The steps of a pipeline and a
 * ColumnTransformer, and a MultiOutputRegressor's estimator, are unfitted too.
 */
export function unfitDocument(doc: ModelDocument): ModelDocument {
  if (doc.model === 'Pipeline') return { ...doc, state: { steps: (doc.state!.steps as ModelDocument[]).map(unfitDocument) } };
  if (doc.model === 'ColumnTransformer') {
    return { ...doc, state: { parts: (doc.state!.parts as (ModelDocument | string)[]).map((p) => (typeof p === 'string' ? p : unfitDocument(p))) } };
  }
  if (doc.model === 'MultiOutputRegressor') {
    return { ...doc, params: { ...doc.params, estimator: unfitDocument(doc.params.estimator as ModelDocument) }, state: null };
  }
  return { ...doc, state: null };
}

export function notFitted(kind: string): never {
  return fail('ML_NOT_FITTED', `this ${kind} isn't fitted yet: call fit() first`);
}

/** Checks that X has the columns the model was fitted on. */
export function checkColumns(X: Matrix, nFeaturesIn: number, kind: string): void {
  if (X.cols !== nFeaturesIn) fail('ML_SHAPE', `X has ${X.cols} columns; this ${kind} was fitted on ${nFeaturesIn}`);
}

/** Numbers for JSON: a typed array as a plain array, refusing what JSON can't hold. */
export function numbers(v: ArrayLike<number>): number[] {
  const out = Array.from(v);
  for (const x of out) if (!Number.isFinite(x)) fail('ML_VALUE', `can't save ${x}: a saved model holds finite numbers only`);
  return out;
}

/** A Float64Array from a saved list of numbers. */
export function floats(v: unknown, what: string): Float64Array {
  if (!Array.isArray(v) || v.some((x) => typeof x !== 'number')) fail('ML_FORMAT', `${what} must be a list of numbers`);
  return Float64Array.from(v as number[]);
}

/** Labels as an array of numbers or strings, checked against the number of rows. */
export function toLabels(y: Labels | undefined, rows: number): Label[] {
  if (y === undefined || y === null) fail('ML_SHAPE', 'y is needed: one label per row');
  if (y.length !== rows) fail('ML_SHAPE', `y has ${y.length} values; X has ${rows} rows`);
  const out: Label[] = new Array(rows);
  let kind: string | undefined;
  for (let i = 0; i < rows; i++) {
    const v = y[i];
    if (typeof v !== 'number' && typeof v !== 'string') fail('ML_LABELS', `y[${i}] is ${typeof v}: labels are numbers or strings`);
    if (typeof v === 'number' && !Number.isFinite(v)) fail('ML_VALUE', `y[${i}] is ${v}`);
    if (kind && typeof v !== kind) fail('ML_LABELS', `y mixes numbers and strings (y[${i}] is a ${typeof v})`);
    kind = typeof v;
    out[i] = v;
  }
  return out;
}

/** The distinct labels, sorted: numbers in order, strings by code unit, as NumPy sorts them. */
export function uniqueLabels(labels: readonly Label[]): Label[] {
  const set = [...new Set(labels)];
  return typeof set[0] === 'number' ? (set as number[]).sort((a, b) => a - b) : (set as string[]).sort();
}
