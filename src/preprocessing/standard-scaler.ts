import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { checkColumns, checkDocument, document, floats, notFitted, numbers, type Labels, type ModelDocument, type Transformer } from '../core/model.js';

export interface StandardScalerParams {
  /** Subtract each column's mean. Default true. */
  withMean?: boolean;
  /** Divide by each column's standard deviation. Default true. */
  withStd?: boolean;
}

interface State {
  mean: Float64Array | null;
  var: Float64Array | null;
  scale: Float64Array | null;
  nSamplesSeen: number | Float64Array;
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
}

/**
 * Scales each column to mean 0 and standard deviation 1: z = (x − mean) / std, with the population standard
 * deviation. scikit-learn's `StandardScaler`. Missing values (`NaN`) are left out of the mean and deviation, and stay
 * missing.
 */
export class StandardScaler implements Transformer {
  readonly kind = 'StandardScaler';
  readonly withMean: boolean;
  readonly withStd: boolean;
  #state: State | null = null;

  constructor(params: StandardScalerParams = {}) {
    this.withMean = params.withMean ?? true;
    this.withStd = params.withStd ?? true;
    if (typeof this.withMean !== 'boolean' || typeof this.withStd !== 'boolean') fail('ML_PARAM', 'withMean and withStd are true or false');
  }

  get fitted(): boolean {
    return this.#state !== null;
  }

  /** Each column's mean (null with withMean and withStd both false). */
  get mean(): Float64Array | null {
    return this.#fitted().mean;
  }

  /** Each column's variance (null with withStd false). */
  get var(): Float64Array | null {
    return this.#fitted().var;
  }

  /** What each column is divided by: its standard deviation, or 1 where that is zero (null with withStd false). */
  get scale(): Float64Array | null {
    return this.#fitted().scale;
  }

  /** The rows fitted on; per column when some values were missing. */
  get nSamplesSeen(): number | Float64Array {
    return this.#fitted().nSamplesSeen;
  }

  /** The number of columns fitted on. */
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }

  /** The columns' names, when X had them. */
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  getParams(): StandardScalerParams {
    return { withMean: this.withMean, withStd: this.withStd };
  }

  async fit(X: MatrixInput, _y?: Labels): Promise<this> {
    return this.fitSync(X);
  }

  fitSync(X: MatrixInput, _y?: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const { mean, variance, counts, missing } = columnStats(m);
    this.#set(mean, variance, counts, missing ? counts : m.rows, m, m.names);
    return this;
  }

  /**
   * Updates the fit with more rows: the means and variances become those of every row seen so far, as if fitSync() had
   * been given them all at once (within rounding). Before any fit it fits. scikit-learn's `partial_fit`, for data that
   * comes in batches or doesn't fit in memory at once.
   */
  partialFitSync(X: MatrixInput, _y?: Labels): this {
    const m = Matrix.from(X);
    if (m.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const s = this.#state;
    if (!s) return this.fitSync(m);
    checkColumns(m, s.nFeaturesIn, this.kind);
    if (s.featureNamesIn && m.names && s.featureNamesIn.some((name, j) => name !== m.names![j])) {
      fail('ML_SHAPE', `X's columns are ${m.names.join(', ')}; ${this.kind} was fitted on ${s.featureNamesIn.join(', ')}`);
    }
    const p = m.cols;
    const b = columnStats(m);
    const seen = s.nSamplesSeen;
    const oldMean = s.mean ?? new Float64Array(p);
    // Without withStd the variance isn't kept: a mean needs none to be combined.
    const oldVar = s.var;
    const mean = new Float64Array(p);
    const variance = new Float64Array(p);
    const counts = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const na = typeof seen === 'number' ? seen : seen[j]!;
      const nb = b.counts[j]!;
      const n = na + nb;
      counts[j] = n;
      if (nb === 0) {
        mean[j] = oldMean[j]!;
        variance[j] = oldVar ? oldVar[j]! : NaN;
        continue;
      }
      if (na === 0) {
        mean[j] = b.mean[j]!;
        variance[j] = b.variance[j]!;
        continue;
      }
      // Chan, Golub and LeVeque: the combined mean, and the sums of squared deviations added with the term for the
      // means' difference.
      const delta = b.mean[j]! - oldMean[j]!;
      mean[j] = oldMean[j]! + (delta * nb) / n;
      if (oldVar) variance[j] = (oldVar[j]! * na + b.variance[j]! * nb + ((delta * delta * na) / n) * nb) / n;
    }
    const missing = typeof seen !== 'number' || b.missing;
    this.#set(mean, variance, counts, missing ? counts : counts[0] ?? 0, m, s.featureNamesIn);
    return this;
  }

  async partialFit(X: MatrixInput, y?: Labels): Promise<this> {
    return this.partialFitSync(X, y);
  }

  #set(
    mean: Float64Array,
    variance: Float64Array,
    counts: Float64Array,
    nSamplesSeen: number | Float64Array,
    m: Matrix,
    names: Matrix['names'],
  ): void {
    const scale = this.withStd ? variance.map((v, j) => scaleOf(v, mean[j]!, counts[j]!)) : null;
    this.#state = {
      mean: this.withMean || this.withStd ? mean : null,
      var: this.withStd ? variance : null,
      scale,
      nSamplesSeen,
      nFeaturesIn: m.cols,
      featureNamesIn: names,
    };
  }

  async transform(X: MatrixInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const out = new Float64Array(m.data.length);
    for (let j = 0; j < m.cols; j++) {
      const mu = this.withMean ? s.mean![j]! : 0;
      const sc = s.scale ? s.scale[j]! : 1;
      const off = j * n;
      for (let i = 0; i < n; i++) out[off + i] = (m.data[off + i]! - mu) / sc;
    }
    return new Matrix(n, m.cols, out, m.names);
  }

  async fitTransform(X: MatrixInput, _y?: Labels): Promise<Matrix> {
    return this.fitSync(X).transformSync(X);
  }

  /** Undoes transform(): x = z × scale + mean. */
  inverseTransformSync(X: MatrixInput): Matrix {
    const s = this.#fitted();
    const m = Matrix.from(X);
    checkColumns(m, s.nFeaturesIn, this.kind);
    const n = m.rows;
    const out = new Float64Array(m.data.length);
    for (let j = 0; j < m.cols; j++) {
      const mu = this.withMean ? s.mean![j]! : 0;
      const sc = s.scale ? s.scale[j]! : 1;
      const off = j * n;
      for (let i = 0; i < n; i++) out[off + i] = m.data[off + i]! * sc + mu;
    }
    return new Matrix(n, m.cols, out, m.names);
  }

  async inverseTransform(X: MatrixInput): Promise<Matrix> {
    return this.inverseTransformSync(X);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(this.kind, { ...this.getParams() }, s && {
      mean: s.mean && numbers(s.mean),
      var: s.var && numbers(s.var),
      scale: s.scale && numbers(s.scale),
      nSamplesSeen: typeof s.nSamplesSeen === 'number' ? s.nSamplesSeen : numbers(s.nSamplesSeen),
      nFeaturesIn: s.nFeaturesIn,
      ...(s.featureNamesIn ? { featureNamesIn: [...s.featureNamesIn] } : {}),
    });
  }

  static fromJSON(doc: ModelDocument): StandardScaler {
    const d = checkDocument(doc, 'StandardScaler');
    const model = new StandardScaler(d.params as StandardScalerParams);
    const s = d.state;
    if (s) {
      model.#state = {
        mean: s.mean == null ? null : floats(s.mean, 'mean'),
        var: s.var == null ? null : floats(s.var, 'var'),
        scale: s.scale == null ? null : floats(s.scale, 'scale'),
        nSamplesSeen: typeof s.nSamplesSeen === 'number' ? s.nSamplesSeen : floats(s.nSamplesSeen, 'nSamplesSeen'),
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

/** Each column's count, mean and population variance, missing values (NaN) left out; an infinity fails. */
function columnStats(m: Matrix): { mean: Float64Array; variance: Float64Array; counts: Float64Array; missing: boolean } {
  const n = m.rows;
  const p = m.cols;
  const mean = new Float64Array(p);
  const variance = new Float64Array(p);
  const counts = new Float64Array(p);
  let missing = false;
  for (let j = 0; j < p; j++) {
    const col = m.column(j);
    // Most columns have no missing value, which their sum shows (a NaN or an infinity makes it non-finite): then
    // two plain passes, without a test per value.
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let k = 0;
    for (; k + 3 < n; k += 4) {
      s0 += col[k]!; s1 += col[k + 1]!; s2 += col[k + 2]!; s3 += col[k + 3]!;
    }
    for (; k < n; k++) s0 += col[k]!;
    const total = s0 + s1 + s2 + s3;
    if (Number.isFinite(total)) {
      const mu = total / n;
      let q0 = 0, q1 = 0, c0 = 0, c1 = 0;
      let i = 0;
      for (; i + 1 < n; i += 2) {
        const d0 = col[i]! - mu;
        const d1 = col[i + 1]! - mu;
        q0 += d0 * d0; q1 += d1 * d1; c0 += d0; c1 += d1;
      }
      for (; i < n; i++) {
        const d0 = col[i]! - mu;
        q0 += d0 * d0;
        c0 += d0;
      }
      const ss = q0 + q1;
      const comp = c0 + c1;
      mean[j] = mu;
      variance[j] = (ss - (comp * comp) / n) / n;
      counts[j] = n;
      continue;
    }
    // Two passes: the mean, then the squared deviations from it, which keeps the variance accurate.
    let count = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = col[i]!;
      if (Number.isNaN(v)) continue;
      if (!Number.isFinite(v)) fail('ML_VALUE', `X column ${m.columnName(j)}, row ${i}, is ${v}`);
      count++;
      sum += v;
    }
    if (count < n) missing = true;
    const mu = count > 0 ? sum / count : NaN;
    let ss = 0;
    let comp = 0;
    for (let i = 0; i < n; i++) {
      const v = col[i]!;
      if (Number.isNaN(v)) continue;
      const dv = v - mu;
      ss += dv * dv;
      comp += dv;
    }
    mean[j] = mu;
    // Corrected two-pass formula (Chan, Golub and LeVeque).
    variance[j] = count > 0 ? (ss - (comp * comp) / count) / count : NaN;
    counts[j] = count;
  }
  return { mean, variance, counts, missing };
}

/**
 * A column's divisor: its standard deviation, or 1 where the column is constant, so it isn't divided by zero.
 * As scikit-learn decides "constant": a variance within rounding error of zero for the column's mean.
 */
function scaleOf(variance: number, mean: number, count: number): number {
  const eps = Number.EPSILON;
  const upper = count * eps * variance + (count * mean * eps) ** 2;
  if (variance <= upper || variance === 0) return 1;
  return Math.sqrt(variance);
}
