import { fail } from './errors.js';
import { Matrix, type MatrixInput } from './matrix.js';

/** A category value: a string or a number; null (or NaN) is missing. */
export type Category = string | number | null;

/**
 * What the encoders and ColumnTransformer take: anything a model takes as X, or named columns that may hold text
 * (`{ colour: ['red', 'blue'], size: [3, 5] }`), or rows mixing text and numbers.
 */
export type TableInput = MatrixInput | Readonly<Record<string, ArrayLike<number> | readonly Category[]>> | readonly (readonly Category[])[];

/** A table's columns: each either numbers (NaN missing) or text (null missing). */
export interface Table {
  rows: number;
  names: string[];
  /** Whether names came with the data (named columns or a Matrix with names). */
  named: boolean;
  columns: (Float64Array | (string | null)[])[];
}

/** Whether a column holds text. */
export function isText(c: Float64Array | (string | null)[]): c is (string | null)[] {
  return !(c instanceof Float64Array);
}

/** The input as columns. A column that mixes numbers and text fails with ML_VALUE. */
export function toTable(input: TableInput): Table {
  if (input instanceof Matrix) {
    return {
      rows: input.rows,
      names: Array.from({ length: input.cols }, (_, j) => input.columnName(j)),
      named: input.names !== undefined,
      columns: Array.from({ length: input.cols }, (_, j) => input.column(j)),
    };
  }
  if (Array.isArray(input)) {
    const rows = input as readonly (ArrayLike<unknown> | unknown)[];
    if (rows.length > 0 && !Array.isArray(rows[0]) && !ArrayBuffer.isView(rows[0])) {
      fail('ML_SHAPE', 'X needs two dimensions: pass rows (an array of arrays) or named columns, not one flat array');
    }
    const p = rows.length === 0 ? 0 : (rows[0] as ArrayLike<unknown>).length;
    const cols: unknown[][] = Array.from({ length: p }, () => new Array(rows.length));
    rows.forEach((r, i) => {
      const row = r as ArrayLike<unknown>;
      if ((!Array.isArray(row) && !ArrayBuffer.isView(row)) || row.length !== p) fail('ML_SHAPE', `row ${i} has ${(row as ArrayLike<unknown>)?.length ?? 'no'} values; row 0 has ${p}`);
      for (let j = 0; j < p; j++) cols[j]![i] = row[j];
    });
    return { rows: rows.length, names: cols.map((_, j) => `${j}`), named: false, columns: cols.map((c, j) => column(c, `${j}`)) };
  }
  if (ArrayBuffer.isView(input)) fail('ML_SHAPE', 'X needs two dimensions: pass rows or named columns, not one flat array');
  if (!input || typeof input !== 'object') fail('ML_SHAPE', 'X must be a Matrix, rows, or an object of named columns');
  const entries = Object.entries(input as Record<string, ArrayLike<unknown>>);
  const n = entries.length ? entries[0]![1].length : 0;
  for (const [name, c] of entries) {
    if ((!Array.isArray(c) && !ArrayBuffer.isView(c)) || c.length !== n) fail('ML_SHAPE', `column ${name} has ${c?.length ?? 'no'} values; the first has ${n}`);
  }
  return { rows: n, names: entries.map(([k]) => k), named: true, columns: entries.map(([k, c]) => column(c, k)) };
}

function column(values: ArrayLike<unknown>, name: string): Float64Array | (string | null)[] {
  if (values instanceof Float64Array) return values;
  let text = false;
  let number = false;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (typeof v === 'string') text = true;
    else if (typeof v === 'number') {
      if (!Number.isNaN(v)) number = true;
    } else if (v !== null && v !== undefined) fail('ML_VALUE', `column ${name}, row ${i}, is ${typeof v}: values are numbers or text`);
  }
  if (text && number) fail('ML_VALUE', `column ${name} mixes numbers and text: make it one or the other`);
  if (text) return Array.from(values as ArrayLike<string | null | undefined>, (v) => (typeof v === 'string' ? v : null));
  return Float64Array.from(values as ArrayLike<number | null | undefined>, (v) => (typeof v === 'number' ? v : NaN));
}

/** A table whose columns are all numbers, as a Matrix; text fails with ML_VALUE, naming the column. */
export function tableToMatrix(t: Table): Matrix {
  const data = new Float64Array(t.rows * t.columns.length);
  t.columns.forEach((c, j) => {
    if (isText(c)) fail('ML_VALUE', `column ${t.names[j]} holds text: encode it first (OneHotEncoder or OrdinalEncoder)`);
    data.set(c, j * t.rows);
  });
  return new Matrix(t.rows, t.columns.length, data, t.named ? t.names : undefined);
}

/** X as a Matrix, whatever form of table it came in, as long as it holds only numbers. */
export function numericMatrix(input: TableInput): Matrix {
  if (input instanceof Matrix) return input;
  return tableToMatrix(toTable(input));
}
