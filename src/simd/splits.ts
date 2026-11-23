/**
 * treeSplits on WebAssembly: the sort-free random split's two passes over a node's rows (extra trees,
 * src/tree/cart.ts's growSortFree), as smaltml's own give them (ownTreeSplits in src/core/kernels.ts). X is copied into
 * the module's memory once per fit; each tree copies in its weights, targets and row list, and partitions the row list
 * in place, through the view begin() returns.
 *
 * The loops are scalar: a node's rows are scattered over X, so each value is read by its row number, and WebAssembly
 * SIMD has no gather. They add the same terms in the same order as smaltml's own, and `select` gives a row's weight or
 * 0 where smaltml's multiplies it by 1 or 0 (a weight is never negative, so the two are the same number): the results
 * are smaltml's own, to the last bit.
 */
import { Code, F64, I32, ModuleBuilder, Room } from 'smaltkit/simd';
import type { TreeSplits } from '../core/kernels.js';
import type { MatrixLike } from './gram.js';

/** The largest memory a module can have: 65,536 pages of 64 KiB. */
const MAX_BYTES = 65536 * 65536;

/** The most class sums the module keeps: more classes than this sum in JavaScript, over the same memory. */
const MAX_SUMS = 1024;

/** `v ← x[rows[k]]`: the f64 at x + 8 · (the i32 at rows + 4k). */
function gather(c: Code, x: number, rows: number, k: number): Code {
  return c.get(x).get(rows).get(k).i32(2).i32Shl().i32Add().i32Load().i32(3).i32Shl().i32Add().f64Load();
}

/** `minmax(x, rows, start, end, out)`: out[0] = min, out[1] = max of x[rows[k]], for k from start to end − 1. */
function minmax(): Code {
  const c = new Code(5);
  const [X, R, ST, EN, OUT] = [0, 1, 2, 3, 4];
  const k = c.local(I32);
  const v = c.local(F64);
  const lo = c.local(F64);
  const hi = c.local(F64);
  c.f64Const(Infinity).set(lo).f64Const(-Infinity).set(hi);
  c.get(ST).set(k);
  c.block().loop(); // for (k = start; k < end; k++)
  c.get(k).get(EN).i32GeU().brIf(1);
  gather(c, X, R, k).set(v);
  c.get(lo).get(v).f64Min().set(lo);
  c.get(hi).get(v).f64Max().set(hi);
  c.get(k).i32(1).i32Add().set(k);
  c.br(0).end().end();
  c.get(OUT).get(lo).f64Store(0);
  c.get(OUT).get(hi).f64Store(8);
  return c;
}

/**
 * `sumsc(x, rows, w, cls, start, end, th, l)`, classes: over the rows rows[k], k from start to end − 1, whose x is at
 * most th, l[0] += their weight and l[1 + their class] += their weight (l zeroed first, by the caller); returns how
 * many. `sumsv(x, rows, w, y, start, end, th, l)`, values: l = [Σw, Σwy, Σwy²] of those rows; returns how many.
 */
function sums(classes: boolean): Code {
  const c = new Code(8);
  const [X, R, W, T, ST, EN, TH, L] = [0, 1, 2, 3, 4, 5, 6, 7];
  const k = c.local(I32);
  const i = c.local(I32);
  const g = c.local(I32);
  const nl = c.local(I32);
  const at = c.local(I32);
  const wi = c.local(F64);
  const v = c.local(F64);
  const t = c.local(F64);
  const s0 = c.local(F64);
  const s1 = c.local(F64);
  const s2 = c.local(F64);
  c.f64Const(0).set(s0).f64Const(0).set(s1).f64Const(0).set(s2);
  c.i32(0).set(nl);
  c.get(ST).set(k);
  c.block().loop(); // for (k = start; k < end; k++)
  c.get(k).get(EN).i32GeU().brIf(1);
  // i = rows[k]; g = x[i] ≤ th; wi = g ? w[i] : 0
  c.get(R).get(k).i32(2).i32Shl().i32Add().i32Load().set(i);
  c.get(X).get(i).i32(3).i32Shl().i32Add().f64Load().get(TH).f64Le().set(g);
  c.get(W).get(i).i32(3).i32Shl().i32Add().f64Load().f64Const(0).get(g).select().set(wi);
  c.get(s0).get(wi).f64Add().set(s0);
  if (classes) {
    // l[1 + cls[i]] += wi
    c.get(L).i32(8).i32Add().get(T).get(i).i32(2).i32Shl().i32Add().i32Load().i32(3).i32Shl().i32Add().set(at);
    c.get(at).get(at).f64Load().get(wi).f64Add().f64Store();
  } else {
    // t = wi · y[i]; s1 += t; s2 += t · y[i]: Σwy and Σwy², as (w · y) · y
    c.get(T).get(i).i32(3).i32Shl().i32Add().f64Load().set(v);
    c.get(wi).get(v).f64Mul().set(t);
    c.get(s1).get(t).f64Add().set(s1);
    c.get(s2).get(t).get(v).f64Mul().f64Add().set(s2);
  }
  c.get(nl).get(g).i32Add().set(nl);
  c.get(k).i32(1).i32Add().set(k);
  c.br(0).end().end();
  c.get(L).get(s0).f64Store(0);
  if (!classes) {
    c.get(L).get(s1).f64Store(8);
    c.get(L).get(s2).f64Store(16);
  }
  c.get(nl);
  return c;
}

/** The module: minmax, sumsc and sumsv. */
export function splitsModule(): Uint8Array<ArrayBuffer> {
  return new ModuleBuilder()
    .memory(1)
    .func('minmax', [I32, I32, I32, I32, I32], [], minmax())
    .func('sumsc', [I32, I32, I32, I32, I32, I32, F64, I32], [I32], sums(true))
    .func('sumsv', [I32, I32, I32, I32, I32, I32, F64, I32], [I32], sums(false))
    .encode();
}

interface SplitsExports {
  minmax(x: number, rows: number, start: number, end: number, out: number): void;
  sumsc(x: number, rows: number, w: number, cls: number, start: number, end: number, th: number, l: number): number;
  sumsv(x: number, rows: number, w: number, y: number, start: number, end: number, th: number, l: number): number;
}

/** The compiled module, for treeSplits(): a TreeSplits per X, or null where X won't fit in a module's memory. */
export class SimdSplits {
  readonly #module: WebAssembly.Module;

  constructor(module: WebAssembly.Module) {
    this.#module = module;
  }

  prepare(X: MatrixLike): TreeSplits | null {
    const n = X.rows;
    const p = X.cols;
    // In bytes: X, the weights, the targets (f64 values, or i32 classes in the first half), the row list (i32, made
    // even), two f64 out, and the class sums.
    const wAt = n * p * 8;
    const tAt = wAt + n * 8;
    const rAt = tAt + n * 8;
    const oAt = rAt + (n + (n & 1)) * 4;
    const lAt = oAt + 16;
    const bytes = lAt + MAX_SUMS * 8;
    if (n === 0 || bytes > MAX_BYTES) return null;
    let room: Room<SplitsExports> | null = new Room<SplitsExports>(this.#module);
    let mem: Float64Array = room.reserve(bytes);
    mem.set(X.data, 0);
    let rows: Uint32Array = new Uint32Array(0);
    let K = 0;
    const live = (): Room<SplitsExports> => {
      if (!room) throw new Error('treeSplits: used after release()');
      return room;
    };
    return {
      begin(weights, targets, list) {
        live();
        mem.set(weights, wAt / 8);
        if ('classes' in targets) {
          new Int32Array(mem.buffer, tAt, n).set(targets.classes);
          K = targets.K;
        } else {
          mem.set(targets.values, tAt / 8);
          K = 0;
        }
        rows = new Uint32Array(mem.buffer, rAt, list.length);
        rows.set(list);
        return rows;
      },
      minMax(f, start, end, out) {
        live().exports.minmax(f * n * 8, rAt, start, end, oAt);
        out[0] = mem[oAt / 8]!;
        out[1] = mem[oAt / 8 + 1]!;
      },
      leftSums(f, start, end, th, l) {
        const ex = live().exports;
        if (K === 0) {
          const nl = ex.sumsv(f * n * 8, rAt, wAt, tAt, start, end, th, lAt);
          l[0] = mem[lAt / 8]!;
          l[1] = mem[lAt / 8 + 1]!;
          l[2] = mem[lAt / 8 + 2]!;
          return nl;
        }
        if (K + 1 <= MAX_SUMS) {
          mem.fill(0, lAt / 8, lAt / 8 + K + 1);
          const nl = ex.sumsc(f * n * 8, rAt, wAt, tAt, start, end, th, lAt);
          for (let q = 0; q <= K; q++) l[q] = mem[lAt / 8 + q]!;
          return nl;
        }
        // More classes than the module keeps sums for: the same loop in JavaScript, over the same memory.
        const col = mem.subarray(f * n, (f + 1) * n);
        const w = mem.subarray(wAt / 8, wAt / 8 + n);
        const cls = new Int32Array(mem.buffer, tAt, n);
        l.fill(0);
        let nl = 0;
        for (let k = start; k < end; k++) {
          const i = rows[k]!;
          const g = col[i]! <= th ? 1 : 0;
          const wi = g * w[i]!;
          l[0]! += wi;
          l[1 + cls[i]!]! += wi;
          nl += g;
        }
        return nl;
      },
      release() {
        room = null;
        mem = new Float64Array(0);
        rows = new Uint32Array(0);
      },
    };
  }
}
