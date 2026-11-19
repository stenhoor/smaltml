import assert from 'node:assert/strict';
import { test } from 'vitest';
import { columnsModule, SimdColumns } from '../src/simd/columns.js';

/** A small, seeded generator, so every run sees the same numbers. */
function random(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

const kernels = [false, true].map((fused) => ({ fused, columns: new SimdColumns(new WebAssembly.Module(columnsModule(fused))) }));

test('both modules are valid WebAssembly', () => {
  for (const fused of [false, true]) assert.ok(WebAssembly.validate(columnsModule(fused)));
});

for (const [n, cols] of [[1, 1], [2, 3], [7, 4], [513, 5], [1300, 11], [2049, 9], [600, 33]] as const) {
  const r = random(n * 37 + cols);
  const d = Float64Array.from({ length: n * cols }, () => r() * 10);
  const v = Float64Array.from({ length: n }, () => r() * 4);
  const mean = Float64Array.from({ length: cols }, () => r() * 3);
  const X = { rows: n, cols, data: d };

  test(`xtvPart on ${n} × ${cols} equals the plain loops within rounding`, () => {
    const ranges = [[0, n], [Math.floor(n / 3), n - Math.floor(n / 5)], [n - 1, n]] as const;
    for (const { fused, columns } of kernels) {
      for (const m of [undefined, mean]) {
        for (const [from, to] of ranges) {
          // v as a plain array too: the kernel copies it either way.
          for (const vv of [v, Array.from(v)]) {
            const got = columns.xtvPart(X, vv, m, from, to);
            assert.equal(got.length, cols);
            for (let j = 0; j < cols; j++) {
              let want = 0;
              let scale = 0;
              for (let i = from; i < to; i++) {
                const t = (d[j * n + i]! - (m ? m[j]! : 0)) * v[i]!;
                want += t;
                scale += Math.abs(t) + (m ? Math.abs(m[j]! * v[i]!) : 0);
              }
              assert.ok(Math.abs(got[j]! - want) <= 1e-13 * (scale || 1), `${fused ? 'fused' : 'plain'}: column ${j}, rows ${from}–${to}: ${got[j]} vs ${want}`);
            }
          }
        }
      }
    }
  });

  test(`columnSums on ${n} × ${cols} equals the plain loops within rounding`, () => {
    for (const { fused, columns } of kernels) {
      const got = columns.columnSums(X);
      assert.equal(got.length, cols);
      for (let j = 0; j < cols; j++) {
        let want = 0;
        let scale = 0;
        for (let i = 0; i < n; i++) {
          want += d[j * n + i]!;
          scale += Math.abs(d[j * n + i]!);
        }
        assert.ok(Math.abs(got[j]! - want) <= 1e-13 * (scale || 1), `${fused ? 'fused' : 'plain'}: column ${j}: ${got[j]} vs ${want}`);
      }
    }
  });
}

test('an empty range gives zeros; X is never written', () => {
  const [{ columns }] = kernels;
  const d = Float64Array.from({ length: 30 }, (_, i) => i - 7.5);
  const X = { rows: 10, cols: 3, data: d };
  const before = d.slice();
  assert.deepEqual([...columns.xtvPart(X, new Float64Array(10).fill(1), undefined, 4, 4)], [0, 0, 0]);
  columns.xtvPart(X, new Float64Array(10).fill(1), [1, 2, 3], 0, 10);
  columns.columnSums(X);
  assert.deepEqual(d, before);
});
