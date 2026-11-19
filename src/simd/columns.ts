/**
 * xtvPart and columnSums on WebAssembly SIMD: (X − mean)ᵀ v over a range of rows, and each column's sum, as smaltml's
 * own give them. Both work as smaltml's Gram does: rows in blocks of 512, each block's columns copied as they are into
 * the WebAssembly memory (Float64Array.set), then four columns at a time, each with its own sum over two rows at a
 * time (f64x2), added into the column's total after the block.
 *
 * Columns are padded with zeros to a multiple of 4, and a block's rows to an even number, so the loops need no
 * remainders. The sums run in another order than smaltml's (even rows and odd rows apart, then added), so results
 * agree in all but the last bits. xtvPart subtracts mean · Σv at the end, with Σv summed row by row as smaltml's.
 */
import { Code, I32, ModuleBuilder, Room, V128 } from 'smaltkit/simd';
import { BLOCK, type MatrixLike } from './gram.js';

/**
 * The module: `xtv(buf, stride, mbytes, cols, v, out)` adds each column's products with v into the f64 at `out` + 8j;
 * `sums(buf, stride, mbytes, cols, out)` adds each column's values. `buf` holds `cols` columns (a multiple of 4), each
 * `stride` bytes apart, of which the first `mbytes` (a multiple of 16, at least 16) are rows. With `fused`, each
 * multiply-add is relaxed SIMD's `f64x2.relaxed_madd`.
 */
export function columnsModule(fused: boolean): Uint8Array<ArrayBuffer> {
  return new ModuleBuilder()
    .memory(1)
    .func('xtv', [I32, I32, I32, I32, I32, I32], [], columnLoop(fused, true))
    .func('sums', [I32, I32, I32, I32, I32], [], columnLoop(fused, false))
    .encode();
}

function columnLoop(fused: boolean, withV: boolean): Code {
  const c = new Code(withV ? 6 : 5);
  const [buf, stride, mbytes, cols] = [0, 1, 2, 3];
  const v = withV ? 4 : -1;
  const out = withV ? 5 : 4;
  const j = c.local(I32);
  const i = c.local(I32);
  const col = [c.local(I32), c.local(I32), c.local(I32), c.local(I32)];
  const o = c.local(I32);
  const vv = c.local(V128);
  const acc = [c.local(V128), c.local(V128), c.local(V128), c.local(V128)];

  c.i32(0).set(j);
  c.block().loop(); // for (j = 0; j < cols; j += 4)
  c.get(j).get(cols).i32GeU().brIf(1);
  c.get(buf).get(j).get(stride).i32Mul().i32Add().set(col[0]!);
  for (let q = 1; q < 4; q++) c.get(col[q - 1]!).get(stride).i32Add().set(col[q]!);
  for (const a of acc) c.v128Zero().set(a);
  c.i32(0).set(i);
  c.loop(); //   do { acc[q] += column[j + q][i, i + 1] (· v[i, i + 1]) } while ((i += 16) < mbytes)
  if (withV) c.get(v).get(i).i32Add().v128Load().set(vv);
  for (let q = 0; q < 4; q++) {
    const a = acc[q]!;
    if (!withV) c.get(a).get(col[q]!).get(i).i32Add().v128Load().f64x2Add().set(a);
    else if (fused) c.get(col[q]!).get(i).i32Add().v128Load().get(vv).get(a).f64x2RelaxedMadd().set(a);
    else c.get(a).get(col[q]!).get(i).i32Add().v128Load().get(vv).f64x2Mul().f64x2Add().set(a);
  }
  c.get(i).i32(16).i32Add().tee(i).get(mbytes).i32LtU().brIf(0);
  c.end();
  //   out[j + q] += lane 0 + lane 1
  c.get(out).get(j).i32(8).i32Mul().i32Add().set(o);
  for (let q = 0; q < 4; q++) {
    const a = acc[q]!;
    c.get(o).get(o).f64Load(q * 8).get(a).f64x2ExtractLane(0).get(a).f64x2ExtractLane(1).f64Add().f64Add().f64Store(q * 8);
  }
  c.get(j).i32(4).i32Add().set(j);
  c.br(0).end().end();
  return c;
}

interface ColumnsExports {
  xtv(buf: number, stride: number, mbytes: number, cols: number, v: number, out: number): void;
  sums(buf: number, stride: number, mbytes: number, cols: number, out: number): void;
}

/** One compiled columns module, its memory and the JavaScript that fills it: smaltml's xtvPart and columnSums. */
export class SimdColumns {
  readonly #room: Room<ColumnsExports>;

  constructor(module: WebAssembly.Module) {
    this.#room = new Room(module);
  }

  /**
   * Runs `step` on each block of rows `from` to `to`, its columns copied into the buffer; returns the pp sums at
   * `outAt`. In f64s: the buffer (pp columns of BLOCK rows), then a block of v, then the sums.
   */
  #blocks(X: MatrixLike, from: number, to: number, v: ArrayLike<number> | undefined, step: (ex: ColumnsExports, even: number, pp: number, vAt: number, outAt: number) => void): Float64Array {
    const n = X.rows;
    const p = X.cols;
    const pp = Math.ceil(p / 4) * 4;
    const B = BLOCK;
    const vAt = pp * B;
    const outAt = vAt + B;
    const mem = this.#room.reserve((outAt + pp) * 8);
    // After reserve(), which can make a new instance on a larger memory.
    const ex = this.#room.exports;
    mem.fill(0, 0, outAt + pp); // the padding columns and the sums start at 0
    const d = X.data;
    for (let r0 = from; r0 < to; r0 += B) {
      const m = Math.min(B, to - r0);
      const even = m + (m & 1);
      for (let j = 0; j < p; j++) {
        mem.set(d.subarray(j * n + r0, j * n + r0 + m), j * B);
        if (even > m) mem[j * B + m] = 0;
      }
      if (v) {
        if (v instanceof Float64Array) mem.set(v.subarray(r0, r0 + m), vAt);
        else for (let i = 0; i < m; i++) mem[vAt + i] = v[r0 + i]!;
        if (even > m) mem[vAt + m] = 0;
      }
      step(ex, even, pp, vAt, outAt);
    }
    return mem.slice(outAt, outAt + p);
  }

  /** smaltml's xtvPart(): (X − mean)ᵀ v over rows `from` to `to`, one value per column. */
  xtvPart(X: MatrixLike, v: ArrayLike<number>, mean: ArrayLike<number> | undefined, from: number, to: number): Float64Array {
    const p = X.cols;
    if (p === 0 || to <= from) return new Float64Array(p);
    const out = this.#blocks(X, from, to, v, (ex, even, pp, vAt, outAt) => ex.xtv(0, BLOCK * 8, even * 8, pp, vAt * 8, outAt * 8));
    // Σ (xᵢ − μ) vᵢ = Σ xᵢ vᵢ − μ Σ vᵢ, as smaltml's.
    if (mean) {
      let vsum = 0;
      for (let i = from; i < to; i++) vsum += v[i]!;
      for (let j = 0; j < p; j++) out[j]! -= mean[j]! * vsum;
    }
    return out;
  }

  /** smaltml's columnSums(): each column's sum over every row. */
  columnSums(X: MatrixLike): Float64Array {
    if (X.cols === 0 || X.rows === 0) return new Float64Array(X.cols);
    return this.#blocks(X, 0, X.rows, undefined, (ex, even, pp, _vAt, outAt) => ex.sums(0, BLOCK * 8, even * 8, pp, outAt * 8));
  }
}
