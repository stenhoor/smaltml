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
  /** Each column's infrequent categories, in the order of `categories`, or null where none are. */
  infrequent: (Category[] | null)[];
  /** Whether each column held text. */
  text: boolean[];
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/** What makes a category infrequent: scikit-learn's `min_frequency` and `max_categories`. */
export interface InfrequentParams {
  /**
   * A category seen fewer times than this is infrequent: a whole number of rows (1 or more), or a share of them
   * (above 0, below 1). Infrequent categories become one category. Default: none.
   */
  minFrequency?: number | null;
  /**
   * The most categories each column keeps, counting the one the infrequent categories become: the least frequent go
   * first, and of two as frequent, the one sorted first. A whole number, 1 or more. Default: no limit.
   */
  maxCategories?: number | null;
}

function checkInfrequent(params: InfrequentParams): { minFrequency: number | null; maxCategories: number | null } {
  const minFrequency = params.minFrequency ?? null;
  const maxCategories = params.maxCategories ?? null;
  if (minFrequency !== null && !((Number.isInteger(minFrequency) && minFrequency >= 1) || (minFrequency > 0 && minFrequency < 1))) {
    fail('ML_PARAM', `minFrequency is a whole number of rows, 1 or more, or a share between 0 and 1, not ${minFrequency}`);
  }
  if (maxCategories !== null && !(Number.isInteger(maxCategories) && maxCategories >= 1)) {
    fail('ML_PARAM', `maxCategories is a whole number, 1 or more, not ${maxCategories}`);
  }
  return { minFrequency, maxCategories };
}

/**
 * A column's infrequent categories, in the order of `cats`, or null where none are: those seen fewer than minFrequency
 * times, and, where more than maxCategories − 1 are left, all but the maxCategories − 1 most frequent (of two as
 * frequent, the one sorted later is kept). With `skipMissing`, the missing value is no category here.
 */
function infrequentOf(
  col: Float64Array | (string | null)[],
  cats: Category[],
  rows: number,
  { minFrequency, maxCategories }: { minFrequency: number | null; maxCategories: number | null },
  skipMissing: boolean,
): Category[] | null {
  if (minFrequency === null && maxCategories === null) return null;
  const counts = new Map<Category, number>();
  for (let i = 0; i < col.length; i++) {
    const v = at(col, i);
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const eligible = skipMissing ? cats.filter((c) => c !== null) : cats;
  const rare = new Set<Category>();
  if (minFrequency !== null) {
    const least = minFrequency >= 1 ? minFrequency : minFrequency * rows;
    for (const c of eligible) if ((counts.get(c) ?? 0) < least) rare.add(c);
  }
  if (maxCategories !== null && maxCategories < eligible.length - rare.size + 1) {
    const keep = maxCategories - 1;
    // Most frequent first; of two as frequent, the one sorted later first.
    const order = eligible.map((c, k) => ({ c, k, n: counts.get(c) ?? 0 })).sort((a, b) => b.n - a.n || b.k - a.k);
    order.slice(keep).forEach(({ c }) => rare.add(c));
  }
  const out = eligible.filter((c) => rare.has(c));
  return out.length > 0 ? out : null;
}

function fitCategories(t: Table, infrequent: { minFrequency: number | null; maxCategories: number | null }, skipMissing: boolean): EncoderState {
  if (t.rows === 0) fail('ML_SHAPE', 'X has no rows');
  const categories = t.columns.map(categoriesOf);
  return {
    categories,
    infrequent: t.columns.map((col, j) => infrequentOf(col, categories[j]!, t.rows, infrequent, skipMissing)),
    text: t.columns.map(isText),
    nFeaturesIn: t.columns.length,
    featureNamesIn: t.named ? t.names : undefined,
  };
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
    ...(s.infrequent.some((c) => c !== null) ? { infrequent: s.infrequent } : {}),
    text: s.text,
    nFeaturesIn: s.nFeaturesIn,
    ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
  };
}

function loadState(s: Record<string, unknown>): EncoderState {
  if (!Array.isArray(s.categories) || !Array.isArray(s.text)) fail('ML_FORMAT', 'an encoder state needs categories and text');
  const categories = s.categories as Category[][];
  return {
    categories,
    infrequent: Array.isArray(s.infrequent) ? (s.infrequent as (Category[] | null)[]) : categories.map(() => null),
    text: s.text as boolean[],
    nFeaturesIn: Number(s.nFeaturesIn),
    featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
  };
}

/**
 * Each category's code: the frequent ones 0, 1, 2 … in sorted order, then one code shared by the infrequent ones.
 * The missing value (null) has none: it stays NaN.
 */
function ordinalCodes(cats: Category[], infrequent: Category[] | null): Map<Category, number> {
  const rare = new Set(infrequent ?? []);
  const codes = new Map<Category, number>();
  let k = 0;
  for (const c of cats) if (c !== null && !rare.has(c)) codes.set(c, k++);
  for (const c of rare) codes.set(c, k);
  return codes;
}

export interface OrdinalEncoderParams extends InfrequentParams {
  /** What to do with a category transform() didn't see in fit(): fail (`'error'`) or give it `unknownValue`. Default `'error'`. */
  handleUnknown?: 'error' | 'use_encoded_value';
  /** The number unknown categories become, with handleUnknown `'use_encoded_value'`. NaN is allowed. */
  unknownValue?: number;
}

/**
 * Turns each column's categories into numbers 0, 1, 2 … in sorted order. scikit-learn's `OrdinalEncoder`. A missing
 * value stays missing (NaN). With minFrequency or maxCategories, the infrequent categories share one code, after the
 * others'.
 */
export class OrdinalEncoder implements Transformer {
  readonly kind = 'OrdinalEncoder';
  readonly handleUnknown: 'error' | 'use_encoded_value';
  readonly unknownValue: number | undefined;
  readonly minFrequency: number | null;
  readonly maxCategories: number | null;
  #state: EncoderState | null = null;

  constructor(params: OrdinalEncoderParams = {}) {
    ({ minFrequency: this.minFrequency, maxCategories: this.maxCategories } = checkInfrequent(params));
    this.handleUnknown = params.handleUnknown ?? 'error';
    this.unknownValue = params.unknownValue;
    if (this.handleUnknown !== 'error' && this.handleUnknown !== 'use_encoded_value') fail('ML_PARAM', `handleUnknown is 'error' or 'use_encoded_value', not ${String(this.handleUnknown)}`);
    if (this.handleUnknown === 'use_encoded_value' && typeof this.unknownValue !== 'number') fail('ML_PARAM', "handleUnknown 'use_encoded_value' needs an unknownValue");
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's categories, sorted; null (missing) last. Infrequent ones are here too. */
  get categories(): Category[][] {
    return this.#fitted().categories;
  }
  /** Each column's infrequent categories, which share one code, or null where none are. */
  get infrequentCategories(): (Category[] | null)[] {
    return this.#fitted().infrequent;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): OrdinalEncoderParams {
    return {
      handleUnknown: this.handleUnknown,
      ...(this.unknownValue !== undefined ? { unknownValue: this.unknownValue } : {}),
      ...(this.minFrequency !== null ? { minFrequency: this.minFrequency } : {}),
      ...(this.maxCategories !== null ? { maxCategories: this.maxCategories } : {}),
    };
  }

  async fit(X: TableInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: TableInput, _y?: Labels): this {
    // The missing value stays NaN, so it is no category for minFrequency and maxCategories (as scikit-learn's NaN).
    this.#state = fitCategories(toTable(X), { minFrequency: this.minFrequency, maxCategories: this.maxCategories }, true);
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
      const codes = ordinalCodes(s.categories[j]!, s.infrequent[j]!);
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

export interface OneHotEncoderParams extends InfrequentParams {
  /**
   * Leave a category out of each column's indicators: none (`null`), the `'first'`, or the first only where a column
   * has two (`'if_binary'`). Where the first category is infrequent, the infrequent column goes. Default null.
   */
  drop?: null | 'first' | 'if_binary';
  /**
   * What to do with a category transform() didn't see in fit(): fail (`'error'`), give it all zeros (`'ignore'`), or
   * count it as infrequent where the column has infrequent categories, else all zeros (`'infrequent_if_exist'`).
   * Default `'error'`.
   */
  handleUnknown?: 'error' | 'ignore' | 'infrequent_if_exist';
}

/**
 * A column's output: its frequent categories in sorted order, then one for the infrequent ones (when there are any).
 * `slot` maps each category to its output position.
 */
function oneHotSlots(cats: Category[], infrequent: Category[] | null): { width: number; slot: Map<Category, number>; rareSlot: number | null } {
  const rare = new Set(infrequent ?? []);
  const slot = new Map<Category, number>();
  let k = 0;
  for (const c of cats) if (!rare.has(c)) slot.set(c, k++);
  if (!infrequent) return { width: k, slot, rareSlot: null };
  for (const c of rare) slot.set(c, k);
  return { width: k + 1, slot, rareSlot: k };
}

/**
 * Turns each column into one 0/1 column per category, named `column_category`. scikit-learn's `OneHotEncoder`, with
 * dense output. A missing value is a category of its own, last. With minFrequency or maxCategories, the infrequent
 * categories share one column, last, named `column_infrequent_sklearn` as scikit-learn names it.
 */
export class OneHotEncoder implements Transformer {
  readonly kind = 'OneHotEncoder';
  readonly drop: null | 'first' | 'if_binary';
  readonly handleUnknown: 'error' | 'ignore' | 'infrequent_if_exist';
  readonly minFrequency: number | null;
  readonly maxCategories: number | null;
  #state: EncoderState | null = null;

  constructor(params: OneHotEncoderParams = {}) {
    ({ minFrequency: this.minFrequency, maxCategories: this.maxCategories } = checkInfrequent(params));
    this.drop = params.drop ?? null;
    this.handleUnknown = params.handleUnknown ?? 'error';
    if (this.drop !== null && this.drop !== 'first' && this.drop !== 'if_binary') fail('ML_PARAM', `drop is null, 'first' or 'if_binary', not ${String(this.drop)}`);
    if (!['error', 'ignore', 'infrequent_if_exist'].includes(this.handleUnknown)) {
      fail('ML_PARAM', `handleUnknown is 'error', 'ignore' or 'infrequent_if_exist', not ${String(this.handleUnknown)}`);
    }
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's categories, sorted; null (missing) last. Infrequent ones are here too. */
  get categories(): Category[][] {
    return this.#fitted().categories;
  }
  /** Each column's infrequent categories, which share one indicator column, or null where none are. */
  get infrequentCategories(): (Category[] | null)[] {
    return this.#fitted().infrequent;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  /**
   * The index in `categories` of each column's dropped category, or null where none is dropped. The first output
   * column is dropped: the first frequent category, or the infrequent column where every category is infrequent
   * (then the first category's index), as scikit-learn does.
   */
  get dropIdx(): (number | null)[] {
    const s = this.#fitted();
    return s.categories.map((c, j) => {
      const { width, slot } = oneHotSlots(c, s.infrequent[j]!);
      if (!(this.drop === 'first' || (this.drop === 'if_binary' && width === 2))) return null;
      const k = c.findIndex((cat) => slot.get(cat) === 0);
      return k < 0 ? 0 : k;
    });
  }

  /** The output position each column drops (its first), or null. */
  #droppedSlots(): (number | null)[] {
    return this.dropIdx.map((d) => (d === null ? null : 0));
  }

  /** The names of the output columns: `column_category`, and `column_infrequent_sklearn` for the infrequent ones. */
  getFeatureNamesOut(): string[] {
    const s = this.#fitted();
    const names = s.featureNamesIn ?? Array.from({ length: s.nFeaturesIn }, (_, j) => `x${j}`);
    const drop = this.#droppedSlots();
    return s.categories.flatMap((cats, j) => {
      const { width, slot, rareSlot } = oneHotSlots(cats, s.infrequent[j]!);
      const out: string[] = new Array(width);
      for (const [c, k] of slot) if (k !== rareSlot) out[k] = `${names[j]}_${label(c, s.text[j]!)}`;
      if (rareSlot !== null) out[rareSlot] = `${names[j]}_infrequent_sklearn`;
      return out.filter((_, k) => k !== drop[j]);
    });
  }

  getParams(): OneHotEncoderParams {
    return {
      drop: this.drop,
      handleUnknown: this.handleUnknown,
      ...(this.minFrequency !== null ? { minFrequency: this.minFrequency } : {}),
      ...(this.maxCategories !== null ? { maxCategories: this.maxCategories } : {}),
    };
  }

  async fit(X: TableInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: TableInput, _y?: Labels): this {
    this.#state = fitCategories(toTable(X), { minFrequency: this.minFrequency, maxCategories: this.maxCategories }, false);
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
    const drop = this.#droppedSlots();
    let offset = 0;
    t.columns.forEach((col, j) => {
      const { width, slot, rareSlot } = oneHotSlots(s.categories[j]!, s.infrequent[j]!);
      const d = drop[j];
      for (let i = 0; i < n; i++) {
        let k = slot.get(at(col, i));
        if (k === undefined) {
          if (this.handleUnknown === 'error') fail('ML_LABELS', `column ${t.names[j]}, row ${i}, is ${String(at(col, i))}, a category fit() didn't see`);
          if (this.handleUnknown !== 'infrequent_if_exist' || rareSlot === null) continue;
          k = rareSlot;
        }
        if (k === d) continue;
        const outCol = offset + (d !== null && d !== undefined && k > d ? k - 1 : k);
        out[outCol * n + i] = 1;
      }
      offset += width - (d === null ? 0 : 1);
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
