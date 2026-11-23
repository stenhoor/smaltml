import assert from 'node:assert/strict';
import { test } from 'vitest';
import { ownTreeSplits, type TreeSplits } from '../src/core/kernels.js';
import { Matrix } from '../src/core/matrix.js';
import { SimdSplits, splitsModule } from '../src/simd/splits.js';

/** A small, seeded generator, so every run sees the same numbers. */
function random(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

const simd = new SimdSplits(new WebAssembly.Module(splitsModule()));

test('the module is valid WebAssembly', () => {
  assert.ok(WebAssembly.validate(splitsModule()));
});

/** Every pass of the kernel against smaltml's own, on these sizes and targets: equal, to the last bit. */
function same(n: number, cols: number, K: number, seed: number) {
  const r = random(seed);
  // X as trees read it: 32-bit floats, with repeated values, so thresholds meet ties.
  const X = new Matrix(n, cols, Float64Array.from({ length: n * cols }, () => Math.fround(Math.round(r() * 40) / 8)));
  const before = X.data.slice();
  const weights = Float64Array.from({ length: n }, (_, i) => (i % 7 === 3 ? 0 : (r() + 1.2) * 2));
  const targets = K > 0 ? { classes: Int32Array.from({ length: n }, () => Math.floor(((r() + 1) / 2) * K * 0.999)), K } : { values: Float64Array.from({ length: n }, () => r() * 30) };
  // The rows a tree grows on, scrambled, as a node's rows are after splits.
  const rows = Uint32Array.from({ length: n }, (_, i) => i).filter((i) => weights[i]! > 0).sort((a, b) => ((a * 13 + 5) % n) - ((b * 13 + 5) % n));
  const kernel = simd.prepare(X)!;
  const own = ownTreeSplits(X);
  const kr = kernel.begin(weights, targets, rows);
  const or = own.begin(weights, targets, rows);
  assert.deepEqual([...kr], [...or]);
  const S = K > 0 ? K + 1 : 3;
  const m = rows.length;
  for (const [start, end] of [[0, m], [Math.floor(m / 3), m - Math.floor(m / 5)], [m - 1, m]]) {
    for (let f = 0; f < cols; f++) {
      const a = new Float64Array(2);
      const b = new Float64Array(2);
      kernel.minMax(f, start!, end!, a);
      own.minMax(f, start!, end!, b);
      assert.deepEqual([...a], [...b], `minMax, column ${f}, ${start}–${end}`);
      for (const th of [b[0]! - 1, b[0]!, (b[0]! + b[1]!) / 2, b[1]!]) {
        const la = new Float64Array(S);
        const lb = new Float64Array(S);
        assert.equal(kernel.leftSums(f, start!, end!, th, la), own.leftSums(f, start!, end!, th, lb), `count, column ${f}`);
        assert.deepEqual([...la], [...lb], `sums, column ${f}, threshold ${th}`);
      }
    }
  }
  // The row list is the kernel's: a tree partitions it in place, and the passes read it so.
  kr.reverse();
  or.reverse();
  const a = new Float64Array(2);
  const b = new Float64Array(2);
  kernel.minMax(0, 0, Math.ceil(m / 2), a);
  own.minMax(0, 0, Math.ceil(m / 2), b);
  assert.deepEqual([...a], [...b]);
  kernel.release();
  assert.deepEqual([...X.data], [...before], 'a kernel reads X; it never writes it');
  return kernel;
}

for (const [n, cols, K] of [[1, 1, 2], [2, 3, 0], [7, 4, 3], [513, 5, 2], [1300, 11, 0], [2049, 9, 5], [600, 33, 1100]] as const) {
  test(`on ${n} × ${cols}, ${K > 0 ? `${K} classes` : 'values'}: smaltml's own results, to the last bit`, () => {
    same(n, cols, K, n * 31 + cols + K);
  });
}

test('after release(), a pass fails rather than reading freed memory', () => {
  const kernel: TreeSplits = same(9, 2, 0, 3);
  assert.throws(() => kernel.minMax(0, 0, 9, new Float64Array(2)), /release/);
});

test('extra trees fitted with the SIMD kernels save the same documents as with smaltml\'s own', async () => {
  const { useKernels, kernelsInUse } = await import('../src/core/kernel-set.js');
  const { ExtraTreesClassifier, ExtraTreesRegressor, DecisionTreeClassifier } = await import('../src/index.js');
  const set = (await import('../src/simd/kernels.js')).default;
  const r = random(11);
  const X = Array.from({ length: 700 }, () => Array.from({ length: 12 }, () => Math.round(r() * 30) / 4));
  const y = X.map((x) => x[0]! * 2 - x[3]! + (x[7]! > 1 ? 3 : 0) + r());
  const labels = y.map((v) => (v > 3 ? 'high' : v > -3 ? 'mid' : 'low'));
  const weights = X.map((_, i) => (i % 9 === 0 ? 0 : 1 + (i % 4)));
  const fits = () => [
    new ExtraTreesClassifier({ nEstimators: 6, seed: 4 }).fitSync(X, labels).toJSON(),
    new ExtraTreesRegressor({ nEstimators: 6, maxFeatures: 4, seed: 5 }).fitSync(X, y, { sampleWeight: weights }).toJSON(),
    new DecisionTreeClassifier({ splitter: 'random', maxFeatures: 'sqrt', seed: 6 }).fitSync(X, labels).toJSON(),
  ];
  try {
    useKernels(null);
    const own = fits();
    useKernels(await set());
    assert.ok(kernelsInUse().kernels.includes('treeSplits'));
    assert.deepEqual(fits(), own);
  } finally {
    useKernels(null);
  }
});
