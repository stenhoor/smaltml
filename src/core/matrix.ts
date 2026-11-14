import { fail } from './errors.js';

/**
 * A table of numbers, as smaltml holds data: `rows` × `cols`, stored column by column in one `Float64Array`, the
 * shape a columnar query result already has. Columns may have names, which errors and saved models use.
 */
export class Matrix {
  readonly rows: number;
  readonly cols: number;
  /** The values, column-major: column `j` is `data[j * rows .. (j + 1) * rows)`. */
  readonly data: Float64Array;
  /** The columns' names, when the data had them. */
  readonly names: readonly string[] | undefined;

  constructor(rows: number, cols: number, data?: Float64Array, names?: readonly string[]) {
    if (!Number.isInteger(rows) || rows < 0 || !Number.isInteger(cols) || cols < 0) {
      fail('ML_SHAPE', `a matrix needs whole, non-negative rows and columns, not ${rows} × ${cols}`);
    }
    if (data && data.length !== rows * cols) {
      fail('ML_SHAPE', `a ${rows} × ${cols} matrix holds ${rows * cols} values, not ${data.length}`);
    }
    if (names && names.length !== cols) fail('ML_SHAPE', `${names.length} column names for ${cols} columns`);
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
    this.names = names;
  }

  /** A matrix from rows of numbers (`[[1, 2], [3, 4]]`), or from named columns (`{ age: [31, 45], spend: [12, 80] }`). */
  static from(input: MatrixInput): Matrix {
    if (input instanceof Matrix) return input;
    if (Array.isArray(input)) return Matrix.fromRows(input as readonly (ArrayLike<number>)[]);
    if (ArrayBuffer.isView(input)) {
      fail('ML_SHAPE', 'X needs two dimensions: pass rows (an array of arrays) or named columns, not one flat array');
    }
    if (input && typeof input === 'object') return Matrix.fromColumns(input as Readonly<Record<string, ArrayLike<number>>>);
    return fail('ML_SHAPE', 'X must be a Matrix, rows of numbers, or an object of named columns');
  }

  /** A matrix from rows of numbers. Every row must have the same length. */
  static fromRows(rows: readonly ArrayLike<number>[]): Matrix {
    const n = rows.length;
    if (n > 0 && !isArrayLike(rows[0])) {
      fail('ML_SHAPE', 'X needs two dimensions: pass rows (an array of arrays) or named columns, not one flat array');
    }
    const p = n === 0 ? 0 : rows[0]!.length;
    const data = new Float64Array(n * p);
    for (let i = 0; i < n; i++) {
      const row = rows[i]!;
      if (!isArrayLike(row) || row.length !== p) {
        fail('ML_SHAPE', `row ${i} has ${isArrayLike(row) ? row.length : 'no'} values; row 0 has ${p}`);
      }
      for (let j = 0; j < p; j++) data[j * n + i] = toNumber(row[j], i, j);
    }
    return new Matrix(n, p, data);
  }

  /** A matrix from named columns, or from an array of columns. Every column must have the same length. */
  static fromColumns(columns: Readonly<Record<string, ArrayLike<number>>> | readonly ArrayLike<number>[], names?: readonly string[]): Matrix {
    const list = Array.isArray(columns) ? (columns as readonly ArrayLike<number>[]) : Object.values(columns);
    const keys = names ?? (Array.isArray(columns) ? undefined : Object.keys(columns));
    const p = list.length;
    const n = p === 0 ? 0 : list[0]!.length;
    const data = new Float64Array(n * p);
    for (let j = 0; j < p; j++) {
      const col = list[j]!;
      if (!isArrayLike(col) || col.length !== n) {
        const name = keys?.[j] ?? String(j);
        fail('ML_SHAPE', `column ${name} has ${isArrayLike(col) ? col.length : 'no'} values; the first has ${n}`);
      }
      if (col instanceof Float64Array) data.set(col, j * n);
      else for (let i = 0; i < n; i++) data[j * n + i] = toNumber(col[i], i, j, keys?.[j]);
    }
    return new Matrix(n, p, data, keys);
  }

  /** The value in row `i`, column `j`. */
  get(i: number, j: number): number {
    return this.data[j * this.rows + i]!;
  }

  /** Column `j`, as a view into the matrix's data (not a copy). */
  column(j: number): Float64Array {
    return this.data.subarray(j * this.rows, (j + 1) * this.rows);
  }

  /** The name of column `j`, or its number when it has none. */
  columnName(j: number): string {
    return this.names?.[j] ?? `${j}`;
  }

  /** Row `i`, copied. */
  row(i: number): Float64Array {
    const out = new Float64Array(this.cols);
    for (let j = 0; j < this.cols; j++) out[j] = this.data[j * this.rows + i]!;
    return out;
  }

  /** The rows as arrays of numbers. */
  toRows(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) out.push(Array.from(this.row(i)));
    return out;
  }

  /** The rows at `index`, in that order, as a new matrix with the same column names. */
  take(index: ArrayLike<number>): Matrix {
    const m = index.length;
    const data = new Float64Array(m * this.cols);
    for (let j = 0; j < this.cols; j++) {
      const src = j * this.rows;
      const dst = j * m;
      for (let k = 0; k < m; k++) data[dst + k] = this.data[src + index[k]!]!;
    }
    return new Matrix(m, this.cols, data, this.names);
  }
}

/**
 * What smaltml takes as `X`: a Matrix, rows of numbers (`number[][]`), or named columns
 * (`{ age: Float64Array, spend: number[] }`), such as a columnar query result.
 */
export type MatrixInput = Matrix | readonly (readonly number[])[] | readonly ArrayLike<number>[] | Readonly<Record<string, ArrayLike<number>>>;

/** What smaltml takes as `y` for regression: numbers, one per row. */
export type VectorInput = ArrayLike<number>;

/** `y` as a Float64Array, checked against the number of rows. */
export function toVector(y: VectorInput, rows: number, what = 'y'): Float64Array {
  if (!isArrayLike(y)) fail('ML_SHAPE', `${what} must be an array of numbers`);
  if (y.length !== rows) fail('ML_SHAPE', `${what} has ${y.length} values; X has ${rows} rows`);
  if (y instanceof Float64Array) return y;
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const v = y[i];
    if (typeof v !== 'number') fail('ML_VALUE', `${what}[${i}] is ${typeof v}, not a number`);
    out[i] = v;
  }
  return out;
}

/** Fails with ML_VALUE, naming the column, if X holds a missing or infinite value. */
export function assertFinite(X: Matrix, what = 'X'): void {
  const d = X.data;
  for (let k = 0; k < d.length; k++) {
    if (!Number.isFinite(d[k]!)) {
      const j = Math.floor(k / X.rows);
      const i = k - j * X.rows;
      fail('ML_VALUE', `${what} column ${X.columnName(j)}, row ${i}, is ${d[k]}: fill missing values first (NaN means missing)`);
    }
  }
}

/** Fails with ML_VALUE if y holds a missing or infinite value. */
export function assertFiniteVector(y: Float64Array, what = 'y'): void {
  for (let i = 0; i < y.length; i++) {
    if (!Number.isFinite(y[i]!)) fail('ML_VALUE', `${what}[${i}] is ${y[i]}`);
  }
}

function isArrayLike(v: unknown): v is ArrayLike<unknown> {
  return Array.isArray(v) || ArrayBuffer.isView(v);
}

function toNumber(v: unknown, i: number, j: number, name?: string): number {
  if (typeof v === 'number') return v;
  if (v === null || v === undefined) return NaN;
  return fail('ML_VALUE', `column ${name ?? j}, row ${i}, is ${typeof v}, not a number: encode it first`);
}
