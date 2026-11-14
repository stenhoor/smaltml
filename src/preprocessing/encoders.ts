import { fail } from '../core/errors.js';
import { Matrix } from '../core/matrix.js';
import { checkDocument, document, notFitted, type Labels, type ModelDocument, type Transformer } from '../core/model.js';
import { isText, toTable, type Category, type Table, type TableInput } from '../core/table.js';

/**
 * A column's categories, sorted as scikit-learn sorts them: numbers in order, text by code unit, and a missing value
 * (null) last.
 */
function categoriesOf(col: Float64Array | (string | null)[]): Category[] {
  const seen = new Set<Category>();
  let missing = false;
  for (const v of col) {
    if (v === null || (typeof v === 'number' && Number.isNaN(v))) missing = true;
    else seen.add(v);
  }
  const values = [...seen];
  const sorted = isText(col) ? (values as string[]).sort() : (values as number[]).sort((a, b) => a - b);
  return missing ? [...sorted, null] : sorted;
}

/** The value at row i, with NaN as null. */
function at(col: Float64Array | (string | null)[], i: number): Category {
  const v = col[i]!;
  return typeof v === 'number' && Number.isNaN(v) ? null : v;
}

/** A category as text, for column names: missing is `nan` for numbers and `None` for text, as scikit-learn names them. */
function label(c: Category, text: boolean): string {
  return c === null ? (text ? 'None' : 'nan') : String(c);
}

interface EncoderState {
  categories: Category[][];
  /** Whether each column held text. */
  text: boolean[];
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

function fitCategories(t: Table): EncoderState {
  if (t.rows === 0) fail('ML_SHAPE', 'X has no rows');
  return { categories: t.columns.map(categoriesOf), text: t.columns.map(isText), nFeaturesIn: t.columns.length, featureNamesIn: t.named ? t.names : undefined };
}

function checkTable(t: Table, s: EncoderState, kind: string): void {
  if (t.columns.length !== s.nFeaturesIn) fail('ML_SHAPE', `X has ${t.columns.length} columns; this ${kind} was fitted on ${s.nFeaturesIn}`);
  t.columns.forEach((c, j) => {
    if (isText(c) !== s.text[j] && c.some((v) => v !== null && !(typeof v === 'number' && Number.isNaN(v)))) {
      fail('ML_VALUE', `column ${t.names[j]} held ${s.text[j] ? 'text' : 'numbers'} when fitted, and ${s.text[j] ? 'numbers' : 'text'} now`);
    }
  });
}

function saveState(s: EncoderState): Record<string, unknown> {
  return {
    categories: s.categories,
    text: s.text,
    nFeaturesIn: s.nFeaturesIn,
    ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
  };
}

function loadState(s: Record<string, unknown>): EncoderState {
  if (!Array.isArray(s.categories) || !Array.isArray(s.text)) fail('ML_FORMAT', 'an encoder state needs categories and text');
  return {
    categories: s.categories as Category[][],
    text: s.text as boolean[],
    nFeaturesIn: Number(s.nFeaturesIn),
    featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
  };
}

export interface OrdinalEncoderParams {
  /** What to do with a category transform() didn't see in fit(): fail (`'error'`) or give it `unknownValue`. Default `'error'`. */
  handleUnknown?: 'error' | 'use_encoded_value';
  /** The number unknown categories become, with handleUnknown `'use_encoded_value'`. NaN is allowed. */
  unknownValue?: number;
}

/**
 * Turns each column's categories into numbers 0, 1, 2 … in sorted order. scikit-learn's `OrdinalEncoder`. A missing
 * value stays missing (NaN).
 */
export class OrdinalEncoder implements Transformer {
  readonly kind = 'OrdinalEncoder';
  readonly handleUnknown: 'error' | 'use_encoded_value';
  readonly unknownValue: number | undefined;
  #state: EncoderState | null = null;

  constructor(params: OrdinalEncoderParams = {}) {
    this.handleUnknown = params.handleUnknown ?? 'error';
    this.unknownValue = params.unknownValue;
    if (this.handleUnknown !== 'error' && this.handleUnknown !== 'use_encoded_value') fail('ML_PARAM', `handleUnknown is 'error' or 'use_encoded_value', not ${String(this.handleUnknown)}`);
    if (this.handleUnknown === 'use_encoded_value' && typeof this.unknownValue !== 'number') fail('ML_PARAM', "handleUnknown 'use_encoded_value' needs an unknownValue");
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's categories, in the order of their codes; null (missing) last. */
  get categories(): Category[][] {
    return this.#fitted().categories;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): OrdinalEncoderParams {
    return { handleUnknown: this.handleUnknown, ...(this.unknownValue !== undefined ? { unknownValue: this.unknownValue } : {}) };
  }

  async fit(X: TableInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: TableInput, _y?: Labels): this {
    this.#state = fitCategories(toTable(X));
    return this;
  }

  async transform(X: TableInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: TableInput): Matrix {
    const s = this.#fitted();
    const t = toTable(X);
    checkTable(t, s, this.kind);
    const n = t.rows;
    const out = new Float64Array(n * t.columns.length);
    t.columns.forEach((col, j) => {
      const codes = new Map(s.categories[j]!.map((c, k) => [c, k]));
      for (let i = 0; i < n; i++) {
        const v = at(col, i);
        if (v === null) {
          out[j * n + i] = NaN;
          continue;
        }
        const code = codes.get(v);
        if (code === undefined) {
          if (this.handleUnknown === 'error') fail('ML_LABELS', `column ${t.names[j]}, row ${i}, is ${String(v)}, a category fit() didn't see`);
          out[j * n + i] = this.unknownValue!;
        } else out[j * n + i] = code;
      }
    });
    return new Matrix(n, t.columns.length, out, t.named ? t.names : undefined);
  }

  async fitTransform(X: TableInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  toJSON(): ModelDocument {
    const params: Record<string, unknown> = { ...this.getParams() };
    if (Number.isNaN(this.unknownValue)) params.unknownValue = 'NaN';
    return document(this.kind, params, this.#state && saveState(this.#state));
  }

  static fromJSON(doc: ModelDocument): OrdinalEncoder {
    const d = checkDocument(doc, 'OrdinalEncoder');
    const { unknownValue, ...rest } = d.params as Omit<OrdinalEncoderParams, 'unknownValue'> & { unknownValue?: number | string };
    const model = new OrdinalEncoder({ ...rest, unknownValue: unknownValue === 'NaN' ? NaN : (unknownValue as number | undefined) });
    if (d.state) model.#state = loadState(d.state);
    return model;
  }

  #fitted(): EncoderState {
    return this.#state ?? notFitted(this.kind);
  }
}

export interface OneHotEncoderParams {
  /** Leave a category out of each column's indicators: none (`null`), the `'first'`, or the first only where a column has two (`'if_binary'`). Default null. */
  drop?: null | 'first' | 'if_binary';
  /** What to do with a category transform() didn't see in fit(): fail (`'error'`) or give it all zeros (`'ignore'`). Default `'error'`. */
  handleUnknown?: 'error' | 'ignore';
}

/**
 * Turns each column into one 0/1 column per category, named `column_category`. scikit-learn's `OneHotEncoder`, with
 * dense output. A missing value is a category of its own, last.
 */
export class OneHotEncoder implements Transformer {
  readonly kind = 'OneHotEncoder';
  readonly drop: null | 'first' | 'if_binary';
  readonly handleUnknown: 'error' | 'ignore';
  #state: EncoderState | null = null;

  constructor(params: OneHotEncoderParams = {}) {
    this.drop = params.drop ?? null;
    this.handleUnknown = params.handleUnknown ?? 'error';
    if (this.drop !== null && this.drop !== 'first' && this.drop !== 'if_binary') fail('ML_PARAM', `drop is null, 'first' or 'if_binary', not ${String(this.drop)}`);
    if (this.handleUnknown !== 'error' && this.handleUnknown !== 'ignore') fail('ML_PARAM', `handleUnknown is 'error' or 'ignore', not ${String(this.handleUnknown)}`);
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's categories, in the order of their indicator columns; null (missing) last. */
  get categories(): Category[][] {
    return this.#fitted().categories;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  /** The index of each column's dropped category, or null where none is dropped. */
  get dropIdx(): (number | null)[] {
    const s = this.#fitted();
    return s.categories.map((c) => (this.drop === 'first' || (this.drop === 'if_binary' && c.length === 2) ? 0 : null));
  }

  /** The names of the output columns: `column_category`. */
  getFeatureNamesOut(): string[] {
    const s = this.#fitted();
    const names = s.featureNamesIn ?? Array.from({ length: s.nFeaturesIn }, (_, j) => `x${j}`);
    const drop = this.dropIdx;
    return s.categories.flatMap((cats, j) => cats.flatMap((c, k) => (k === drop[j] ? [] : [`${names[j]}_${label(c, s.text[j]!)}`])));
  }

  getParams(): Required<OneHotEncoderParams> {
    return { drop: this.drop, handleUnknown: this.handleUnknown };
  }

  async fit(X: TableInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: TableInput, _y?: Labels): this {
    this.#state = fitCategories(toTable(X));
    return this;
  }

  async transform(X: TableInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: TableInput): Matrix {
    const s = this.#fitted();
    const t = toTable(X);
    checkTable(t, s, this.kind);
    const n = t.rows;
    const names = this.getFeatureNamesOut();
    const out = new Float64Array(n * names.length);
    const drop = this.dropIdx;
    let offset = 0;
    t.columns.forEach((col, j) => {
      const cats = s.categories[j]!;
      const codes = new Map(cats.map((c, k) => [c, k]));
      const d = drop[j];
      for (let i = 0; i < n; i++) {
        const k = codes.get(at(col, i));
        if (k === undefined) {
          if (this.handleUnknown === 'error') fail('ML_LABELS', `column ${t.names[j]}, row ${i}, is ${String(at(col, i))}, a category fit() didn't see`);
          continue;
        }
        if (k === d) continue;
        const outCol = offset + (d !== null && k > d ? k - 1 : k);
        out[outCol * n + i] = 1;
      }
      offset += cats.length - (d === null ? 0 : 1);
    });
    return new Matrix(n, names.length, out, names);
  }

  async fitTransform(X: TableInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  toJSON(): ModelDocument {
    return document(this.kind, { ...this.getParams() }, this.#state && saveState(this.#state));
  }

  static fromJSON(doc: ModelDocument): OneHotEncoder {
    const d = checkDocument(doc, 'OneHotEncoder');
    const model = new OneHotEncoder(d.params as OneHotEncoderParams);
    if (d.state) model.#state = loadState(d.state);
    return model;
  }

  #fitted(): EncoderState {
    return this.#state ?? notFitted(this.kind);
  }
}
