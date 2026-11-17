import * as esbuild from 'esbuild';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, afterEach, beforeAll, describe, expect, test } from 'vitest';
import { KERNELS_VERSION, kernelsInUse, useKernels, type KernelSet } from '../src/core/kernel-set.js';
import { gram, ownGramPart, xw } from '../src/core/kernels.js';
import { Matrix } from '../src/core/matrix.js';
import { Random } from '../src/core/random.js';
import {
  ElasticNet,
  Lasso,
  LinearRegression,
  LogisticRegression,
  Ridge,
  configureThreads,
  matMulSync,
  stopThreads,
} from '../src/index.js';
import plugin, { calls } from './kernel-plugin.js';

// The thread pool's workers run the built script, and import the stand-in plugin as a module of its own: the tests
// bundle both from the source once.
let pluginModule = '';
beforeAll(async () => {
  const dir = mkdtempSync(join(tmpdir(), 'smaltml-kernels-'));
  const common = { bundle: true, format: 'esm', platform: 'neutral', logLevel: 'silent' } as const;
  await esbuild.build({ ...common, entryPoints: ['src/parallel/worker.ts'], outfile: join(dir, 'smaltml-threads.js') });
  await esbuild.build({ ...common, entryPoints: ['test/kernel-plugin.ts'], outfile: join(dir, 'plugin.js') });
  configureThreads({ script: pathToFileURL(join(dir, 'smaltml-threads.js')) });
  pluginModule = pathToFileURL(join(dir, 'plugin.js')).href;
});
afterEach(() => {
  useKernels(null);
});
afterAll(() => stopThreads());

/** 40,000 × 12: over the parallel threshold, in three chunks of rows. */
function data() {
  const r = new Random(5);
  const n = 40_000;
  const p = 12;
  const X = new Matrix(n, p, Float64Array.from({ length: n * p }, (_, k) => (r.next() - 0.5) * (1 + (k % 5))));
  const w = Float64Array.from({ length: p }, () => r.next() - 0.5);
  const score = xw(X, w, 0);
  const y = Float64Array.from(score, (s) => s + 0.1 * (r.next() - 0.5));
  const y2 = Float64Array.from(score, (s) => (s + 0.3 * (r.next() - 0.5) > 0 ? 1 : 0));
  return { X, y, y2 };
}
const { X, y, y2 } = data();

type Fittable = { fitSync(X: Matrix, y: Float64Array): { toJSON(): unknown }; fit(X: Matrix, y: Float64Array, o: object): Promise<{ toJSON(): unknown }> };
const models: [string, () => Fittable, () => Float64Array][] = [
  ['LinearRegression', () => new LinearRegression(), () => y],
  ['Ridge', () => new Ridge({ alpha: 3 }), () => y],
  ['Lasso', () => new Lasso({ alpha: 0.01 }), () => y],
  ['ElasticNet', () => new ElasticNet({ alpha: 0.01, l1Ratio: 0.3 }), () => y],
  ['LogisticRegression, newton-cholesky', () => new LogisticRegression({ solver: 'newton-cholesky' }), () => y2],
];

describe('useKernels', () => {
  test('starts with smaltml\'s own kernels, and null puts them back', () => {
    expect(kernelsInUse()).toEqual({ name: 'smaltml', kernels: [], rejected: [] });
    expect(useKernels(plugin())).toEqual({ name: 'nudged', kernels: ['gramPart', 'matMul'], rejected: [] });
    expect(kernelsInUse().name).toBe('nudged');
    const replaced = gram(X);
    expect(useKernels(null)).toEqual({ name: 'smaltml', kernels: [], rejected: [] });
    const before = calls.gramPart;
    const own = gram(X);
    expect(calls.gramPart).toBe(before);
    expect(Array.from(replaced)).not.toEqual(Array.from(own));
  });

  test('a kernel the set leaves out stays smaltml\'s', () => {
    const set = plugin();
    const used = useKernels({ ...set, kernels: { matMul: set.kernels.matMul } });
    expect(used.kernels).toEqual(['matMul']);
    const before = calls.gramPart;
    gram(X);
    expect(calls.gramPart).toBe(before);
  });

  test('matMulSync uses the matMul kernel', () => {
    const A = [[1, 2], [3, 4], [5, 6]];
    const B = [[1, 0, 2], [0, 1, 3]];
    const own = matMulSync(A, B).toRows();
    expect(useKernels(plugin()).kernels).toContain('matMul');
    const before = calls.matMul;
    const replaced = matMulSync(A, B).toRows();
    expect(calls.matMul).toBe(before + 1);
    expect(replaced).not.toEqual(own);
  });

  test('every model fitted on X\'s Gram matrix uses gramPart, and gives other last bits with it', () => {
    for (const [name, make, target] of models) {
      const own = make().fitSync(X, target()).toJSON();
      useKernels(plugin());
      const before = calls.gramPart;
      const replaced = make().fitSync(X, target()).toJSON();
      expect(calls.gramPart, name).toBeGreaterThan(before);
      expect(replaced, name).not.toEqual(own);
      useKernels(null);
    }
  });

  test('a kernel that fails the check at start isn\'t used', () => {
    const wrong: KernelSet = {
      name: 'wrong',
      version: KERNELS_VERSION,
      kernels: {
        // Off by far more than rounding.
        gramPart: (...a) => plugin({ steps: 1 << 20 }).kernels.gramPart!(...a),
        matMul: () => { throw new Error('no'); },
      },
    };
    expect(useKernels(wrong)).toEqual({ name: 'wrong', kernels: [], rejected: ['gramPart', 'matMul'] });
    const short: KernelSet = { name: 'short', version: KERNELS_VERSION, kernels: { gramPart: () => new Float64Array(3) } };
    expect(useKernels(short).rejected).toEqual(['gramPart']);
    const writes: KernelSet = {
      name: 'writes',
      version: KERNELS_VERSION,
      kernels: { gramPart: (X, m, a, b, w, o) => { const out = ownGramPart(X, m, a, b, w, o); X.data[0] = 0; return out; } },
    };
    expect(useKernels(writes).rejected).toEqual(['gramPart']);
  });

  test('a set for another contract version, or with kernels smaltml doesn\'t have, fails with ML_PARAM', () => {
    expect(() => useKernels({ ...plugin(), version: KERNELS_VERSION + 1 })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => useKernels({ name: 'x', version: KERNELS_VERSION, kernels: { exp: () => 0 } as never })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(() => useKernels(3 as never)).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
    expect(kernelsInUse().name).toBe('smaltml');
  });
});

describe('replaced kernels on several threads', () => {
  for (const [name, make, target] of models) {
    test(`${name}: the workers load the set, and fits equal fitSync, bit for bit`, async () => {
      useKernels(plugin({ module: pluginModule }));
      const one = make().fitSync(X, target()).toJSON();
      for (const threads of [2, 4]) {
        const many = await make().fit(X, target(), { threads });
        expect(many.toJSON(), `${threads} threads`).toEqual(one);
      }
    });
  }

  test('a set the workers can\'t load keeps fits on one thread, with fitSync\'s result', async () => {
    useKernels(plugin());
    const one = new Ridge({ alpha: 3 }).fitSync(X, y).toJSON();
    expect((await new Ridge({ alpha: 3 }).fit(X, y, { threads: 4 })).toJSON()).toEqual(one);
    useKernels(plugin({ module: 'file:///nowhere/plugin.js' }));
    expect((await new Ridge({ alpha: 3 }).fit(X, y, { threads: 4 })).toJSON()).toEqual(one);
  });

  test('after useKernels(null), the workers run smaltml\'s own kernels again', async () => {
    useKernels(plugin({ module: pluginModule }));
    await new Ridge({ alpha: 3 }).fit(X, y, { threads: 2 });
    useKernels(null);
    const own = new Ridge({ alpha: 3 }).fitSync(X, y).toJSON();
    expect((await new Ridge({ alpha: 3 }).fit(X, y, { threads: 2 })).toJSON()).toEqual(own);
  });
});
