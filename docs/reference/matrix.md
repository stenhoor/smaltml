---
description: 'Matrix: the table of numbers smaltml holds data in, stored column by column in a Float64Array.'
---

# Matrix

<!-- docs:skip -->
```ts
class Matrix {
  constructor(rows: number, cols: number, data?: Float64Array, names?: readonly string[]);
  static from(input: MatrixInput): Matrix;
  static fromRows(rows: readonly ArrayLike<number>[]): Matrix;
  static fromColumns(columns: Record<string, ArrayLike<number>> | readonly ArrayLike<number>[], names?: readonly string[]): Matrix;
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;
  readonly names: readonly string[] | undefined;
  get(i: number, j: number): number;
  column(j: number): Float64Array;
  row(i: number): Float64Array;
  columnName(j: number): string;
  toRows(): number[][];
  take(index: ArrayLike<number>): Matrix;
}
type MatrixInput = Matrix | number[][] | ArrayLike<number>[] | Record<string, ArrayLike<number>>;
```

A table of numbers: `rows` × `cols` 64-bit floats, stored column by column in one `Float64Array`, the shape a columnar
query result has. Every model takes a `MatrixInput` as `X` and turns it into a Matrix.

## Parameters

- `Matrix.from(input)`: a Matrix as it is; rows of numbers (`[[1, 2], [3, 4]]`); or named columns
  (`{ age: [31, 45], spend: new Float64Array([12, 80]) }`), which keep their names.
- `fromColumns(columns, names?)`: columns as an object or an array, with optional names.
- `take(index)`: the rows at `index`, in that order.

## Returns

`column(j)` is a view into `data`, not a copy; `row(i)` and `toRows()` copy.

## Errors

- `ML_SHAPE`: rows of different lengths, columns of different lengths, or one flat array where rows are needed.
- `ML_VALUE`: a value that isn't a number. `null` and `undefined` become `NaN`, a missing value.

## Examples

```ts
import { Matrix } from '@smaltworks/smaltml';

const X = Matrix.from({ age: [31, 45, 27], spend: [1200, 800, 1500] });
({ rows: X.rows, cols: X.cols, column1: X.column(1), row0: X.row(0) })
```

| rows | cols | column1 | row0 |
| --- | --- | --- | --- |
| 3 | 2 | [1200, 800, 1500] | [31, 1200] |

## Working demo


| 0 | 1 |
| --- | --- |
| 5.9 | 3 |
| 5.1 | 3.5 |
```ts
const sample = Matrix.from([[5.1, 3.5], [4.9, 3.0], [6.2, 3.4], [5.9, 3.0]]);
sample.take([3, 0])
```

| 0 | 1 |
| --- | --- |
| 5.9 | 3 |
| 5.1 | 3.5 |

## See also

[Installation: data in](../guide/getting-started#data-in), [Preparing data](../guide/preparing-data)
