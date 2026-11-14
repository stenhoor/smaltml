import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, notFitted, type Labels, type ModelDocument, type Transformer } from '../core/model.js';
import { percentile, sortedValues } from './stats.js';

export interface SimpleImputerParams {
  /** How to fill: each column's `'mean'`, `'median'` or `'most_frequent'` value, or a `'constant'`. Default `'mean'`. */
  strategy?: 'mean' | 'median' | 'most_frequent' | 'constant';
  /** The value for `'constant'`. Default 0. */
  fillValue?: number;
  /** Keep columns with no values at all (filled with 0, or fillValue). Default false: they are dropped. */
  keepEmptyFeatures?: boolean;
}

interface State {
  statistics: Float64Array;
  /** The columns transform() keeps. */
  keep: number[];
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Fills missing values (NaN) in each column with a statistic of the column: its mean, median or most frequent value,
 * or a constant. scikit-learn's `SimpleImputer`, for numbers. A column with no values at all is dropped, whatever the
 * strategy, unless keepEmptyFeatures keeps it (filled with 0, or fillValue for `'constant'`).
 */
export class SimpleImputer implements Transformer {
  readonly kind = 'SimpleImputer';
  readonly strategy: NonNullable<SimpleImputerParams['strategy']>;
  readonly fillValue: number;
  readonly keepEmptyFeatures: boolean;
  #state: State | null = null;

  constructor(params: SimpleImputerParams = {}) {
    this.strategy = params.strategy ?? 'mean';
    this.fillValue = params.fillValue ?? 0;
    this.keepEmptyFeatures = params.keepEmptyFeatures ?? false;
    if (!['mean', 'median', 'most_frequent', 'constant'].includes(this.strategy)) {
      fail('ML_PARAM', `strategy is 'mean', 'median', 'most_frequent' or 'constant', not ${String(this.strategy)}`);
    }
    if (typeof this.fillValue !== 'number' || !Number.isFinite(this.fillValue)) {
      fail('ML_PARAM', 'fillValue is a finite number: text columns are filled by the encoders, which treat a missing value as its own category');
    }
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** The value each column's missing values are filled with (NaN for a column with none to fill from). */
  get statistics(): Float64Array {
    return this.#fitted().statistics;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): Required<SimpleImputerParams> {
    return { strategy: this.strategy, fillValue: this.fillValue, keepEmptyFeatures: this.keepEmptyFeatures };
  }

  async fit(X: MatrixInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: MatrixInput, _y?: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const statistics = new Float64Array(m.cols);
    const keep: number[] = [];
    for (let j = 0; j < m.cols; j++) {
      const col = m.column(j);
      for (const v of col) if (!Number.isNaN(v) && !Number.isFinite(v)) fail('ML_VALUE', `X column ${m.columnName(j)} holds ${v}`);
      const sorted = sortedValues(col);
      let stat: number;
      // A column with no values is dropped, whatever the strategy (scikit-learn 1.9), unless kept.
      if (sorted.length === 0) stat = this.keepEmptyFeatures ? (this.strategy === 'constant' ? this.fillValue : 0) : NaN;
      else if (this.strategy === 'constant') stat = this.fillValue;
      else if (this.strategy === 'mean') stat = sorted.reduce((s, v) => s + v, 0) / sorted.length;
      else if (this.strategy === 'median') stat = percentile(sorted, 50);
      else stat = mode(sorted);
      statistics[j] = stat;
      if (!Number.isNaN(stat)) keep.push(j);
    }
    this.#state = { statistics, keep, nFeaturesIn: m.cols, featureNamesIn: m.names };
    return this;
  }

  async transform(X: MatrixInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const out = new Float64Array(n * s.keep.length);
    s.keep.forEach((j, k) => {
      const stat = s.statistics[j]!;
      const col = m.column(j);
      for (let i = 0; i < n; i++) out[k * n + i] = Number.isNaN(col[i]!) ? stat : col[i]!;
    });
    return new Matrix(n, s.keep.length, out, m.names && s.keep.map((j) => m.names![j]!));
  }

  async fitTransform(X: MatrixInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      // A column with nothing to fill from has no statistic: null.
      statistics: Array.from(s.statistics, (v) => (Number.isNaN(v) ? null : v)),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): SimpleImputer {
    const d = checkDocument(doc, 'SimpleImputer');
    const model = new SimpleImputer(d.params as SimpleImputerParams);
    const s = d.state;
    if (s) {
      if (!Array.isArray(s.statistics)) fail('ML_FORMAT', 'statistics must be a list');
      const statistics = Float64Array.from(s.statistics as (number | null)[], (v) => (v === null ? NaN : v));
      model.#state = {
        statistics,
        keep: [...statistics.keys()].filter((j) => !Number.isNaN(statistics[j]!)),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return model;
  }

  #fitted(): State {
    return this.#state ?? notFitted(this.kind);
  }
}

/** The most frequent of sorted values; the smallest among ties, as scikit-learn chooses. */
function mode(sorted: Float64Array): number {
  let best = sorted[0]!;
  let bestCount = 0;
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j < sorted.length && sorted[j] === sorted[i]) j++;
    if (j - i > bestCount) {
      best = sorted[i]!;
      bestCount = j - i;
    }
    i = j;
  }
  return best;
}
