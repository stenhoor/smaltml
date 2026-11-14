import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, type Labels, type ModelDocument, type Transformer } from '../core/model.js';
import { normalQuantile, percentile, sortedValues } from './stats.js';

/** Applies x → (x − sub[j]) · mul[j] to every column, keeping NaN. */
function affine(X: Matrix, sub: ArrayLike<number> | null, mul: ArrayLike<number> | null, clip?: readonly [number, number]): Matrix {
  const n = X.rows;
  const out = new Float64Array(X.data.length);
  for (let j = 0; j < X.cols; j++) {
    const s = sub ? sub[j]! : 0;
    const m = mul ? mul[j]! : 1;
    const off = j * n;
    for (let i = 0; i < n; i++) {
      let v = (X.data[off + i]! - s) * m;
      if (clip) v = Math.min(Math.max(v, clip[0]), clip[1]);
      out[off + i] = v;
    }
  }
  return new Matrix(n, X.cols, out, X.names);
}

/** 1 where a range is zero or within rounding of it, so a constant column isn't divided by zero (scikit-learn's rule). */
function safe(range: number): number {
  return range < 10 * Number.EPSILON || !Number.isFinite(range) ? 1 : range;
}

export interface MinMaxScalerParams {
  /** The range each column is scaled to. Default [0, 1]. */
  featureRange?: readonly [number, number];
  /** Clip transformed values to featureRange (for rows outside the fitted range). Default false. */
  clip?: boolean;
}

interface MinMaxState {
  dataMin: Float64Array;
  dataMax: Float64Array;
  scale: Float64Array;
  min: Float64Array;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Scales each column to a range, [0, 1] by default: x · scale + min, where the fitted minimum maps to the range's low
 * end and the maximum to its high end. scikit-learn's `MinMaxScaler`. Missing values are left out of the fit and stay
 * missing.
 */
export class MinMaxScaler implements Transformer {
  readonly kind = 'MinMaxScaler';
  readonly featureRange: readonly [number, number];
  readonly clip: boolean;
  #state: MinMaxState | null = null;

  constructor(params: MinMaxScalerParams = {}) {
    this.featureRange = params.featureRange ?? [0, 1];
    this.clip = params.clip ?? false;
    const [lo, hi] = this.featureRange;
    if (!(lo < hi)) fail('ML_PARAM', `featureRange's low end must be below its high end, not [${lo}, ${hi}]`);
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's fitted minimum. */
  get dataMin(): Float64Array {
    return this.#fitted().dataMin;
  }
  /** Each column's fitted maximum. */
  get dataMax(): Float64Array {
    return this.#fitted().dataMax;
  }
  /** Each column's fitted range, max − min. */
  get dataRange(): Float64Array {
    const s = this.#fitted();
    return s.dataMax.map((v, j) => v - s.dataMin[j]!);
  }
  /** What each column is multiplied by. */
  get scale(): Float64Array {
    return this.#fitted().scale;
  }
  /** What is added after: x · scale + min. */
  get min(): Float64Array {
    return this.#fitted().min;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): Required<MinMaxScalerParams> {
    return { featureRange: [...this.featureRange] as [number, number], clip: this.clip };
  }

  async fit(X: MatrixInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: MatrixInput, _y?: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const p = m.cols;
    const dataMin = new Float64Array(p);
    const dataMax = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of m.column(j)) {
        if (Number.isNaN(v)) continue;
        if (!Number.isFinite(v)) fail('ML_VALUE', `X column ${m.columnName(j)} holds ${v}`);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      dataMin[j] = lo === Infinity ? NaN : lo;
      dataMax[j] = hi === -Infinity ? NaN : hi;
    }
    const [a, b] = this.featureRange;
    const scale = dataMin.map((lo, j) => (b - a) / safe(dataMax[j]! - lo));
    const min = dataMin.map((lo, j) => a - lo * scale[j]!);
    this.#state = { dataMin, dataMax, scale, min, nFeaturesIn: p, featureNamesIn: m.names };
    return this;
  }

  async transform(X: MatrixInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    // x · scale + min, written as (x − (−min / scale)) · scale.
    const out = new Float64Array(m.data.length);
    for (let j = 0; j < m.cols; j++) {
      const sc = s.scale[j]!;
      const mn = s.min[j]!;
      const off = j * m.rows;
      for (let i = 0; i < m.rows; i++) {
        let v = m.data[off + i]! * sc + mn;
        if (this.clip) v = Math.min(Math.max(v, this.featureRange[0]), this.featureRange[1]);
        out[off + i] = v;
      }
    }
    return new Matrix(m.rows, m.cols, out, m.names);
  }

  async fitTransform(X: MatrixInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  /** Undoes transform(): x = (z − min) / scale. */
  inverseTransformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return affine(m, s.min, s.scale.map((v) => 1 / v));
  }

  async inverseTransform(X: MatrixInput): Promise<Matrix> {
    return this.inverseTransformSync(X);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      dataMin: numbers(s.dataMin),
      dataMax: numbers(s.dataMax),
      scale: numbers(s.scale),
      min: numbers(s.min),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): MinMaxScaler {
    const d = checkDocument(doc, 'MinMaxScaler');
    const model = new MinMaxScaler(d.params as MinMaxScalerParams);
    const s = d.state;
    if (s) {
      model.#state = {
        dataMin: floats(s.dataMin, 'dataMin'),
        dataMax: floats(s.dataMax, 'dataMax'),
        scale: floats(s.scale, 'scale'),
        min: floats(s.min, 'min'),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return model;
  }

  #fitted(): MinMaxState {
    return this.#state ?? notFitted(this.kind);
  }
}

export interface RobustScalerParams {
  /** Subtract each column's median. Default true. */
  withCentering?: boolean;
  /** Divide by each column's interquartile range (or quantileRange). Default true. */
  withScaling?: boolean;
  /** The percentiles whose difference is the scale. Default [25, 75]. */
  quantileRange?: readonly [number, number];
  /** Scale so that normally distributed data has variance 1. Default false. */
  unitVariance?: boolean;
}

interface RobustState {
  center: Float64Array | null;
  scale: Float64Array | null;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Scales each column by statistics that outliers barely move: z = (x − median) / (q75 − q25). scikit-learn's
 * `RobustScaler`, with NumPy's linear percentiles. Missing values are left out of the fit and stay missing.
 */
export class RobustScaler implements Transformer {
  readonly kind = 'RobustScaler';
  readonly withCentering: boolean;
  readonly withScaling: boolean;
  readonly quantileRange: readonly [number, number];
  readonly unitVariance: boolean;
  #state: RobustState | null = null;

  constructor(params: RobustScalerParams = {}) {
    this.withCentering = params.withCentering ?? true;
    this.withScaling = params.withScaling ?? true;
    this.quantileRange = params.quantileRange ?? [25, 75];
    this.unitVariance = params.unitVariance ?? false;
    const [lo, hi] = this.quantileRange;
    if (!(lo >= 0 && hi <= 100 && lo <= hi)) fail('ML_PARAM', `quantileRange must be two percentiles from 0 to 100, low first, not [${lo}, ${hi}]`);
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  /** Each column's median (null without withCentering). */
  get center(): Float64Array | null {
    return this.#fitted().center;
  }
  /** What each column is divided by (null without withScaling). */
  get scale(): Float64Array | null {
    return this.#fitted().scale;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): Required<RobustScalerParams> {
    return { withCentering: this.withCentering, withScaling: this.withScaling, quantileRange: [...this.quantileRange] as [number, number], unitVariance: this.unitVariance };
  }

  async fit(X: MatrixInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: MatrixInput, _y?: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const p = m.cols;
    const center = new Float64Array(p);
    const scale = new Float64Array(p);
    const [lo, hi] = this.quantileRange;
    const adjust = this.unitVariance ? normalQuantile(hi / 100) - normalQuantile(lo / 100) : 1;
    for (let j = 0; j < p; j++) {
      for (const v of m.column(j)) if (!Number.isNaN(v) && !Number.isFinite(v)) fail('ML_VALUE', `X column ${m.columnName(j)} holds ${v}`);
      const sorted = sortedValues(m.column(j));
      center[j] = percentile(sorted, 50);
      const range = percentile(sorted, hi) - percentile(sorted, lo);
      // A zero range becomes 1 first, then the unit-variance adjustment applies, as in scikit-learn.
      scale[j] = safe(range) / adjust;
    }
    this.#state = {
      center: this.withCentering ? center : null,
      scale: this.withScaling ? scale : null,
      nFeaturesIn: p,
      featureNamesIn: m.names,
    };
    return this;
  }

  async transform(X: MatrixInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    return affine(m, s.center, s.scale && s.scale.map((v) => 1 / v));
  }

  async fitTransform(X: MatrixInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  /** Undoes transform(): x = z · scale + center. */
  inverseTransformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const back = affine(m, null, s.scale);
    return s.center ? affine(back, s.center.map((v) => -v), null) : back;
  }

  async inverseTransform(X: MatrixInput): Promise<Matrix> {
    return this.inverseTransformSync(X);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      center: s.center && numbers(s.center),
      scale: s.scale && numbers(s.scale),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): RobustScaler {
    const d = checkDocument(doc, 'RobustScaler');
    const model = new RobustScaler(d.params as RobustScalerParams);
    const s = d.state;
    if (s) {
      model.#state = {
        center: s.center == null ? null : floats(s.center, 'center'),
        scale: s.scale == null ? null : floats(s.scale, 'scale'),
        nFeaturesIn: Number(s.nFeaturesIn),
        featureNamesIn: Array.isArray(s.featureNamesIn) ? (s.featureNamesIn as string[]) : undefined,
      };
    }
    return model;
  }

  #fitted(): RobustState {
    return this.#state ?? notFitted(this.kind);
  }
}
