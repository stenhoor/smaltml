import { afterEach, describe, expect, test, vi } from 'vitest';

/**
 * smaltml turns its SIMD kernels (`@smaltworks/smaltml/simd`) on by itself where the engine has WebAssembly SIMD
 * (ADR 0040). Most of these tests stand in for the entry with a mock, to see the look-up's order; each loads smaltml
 * afresh, as an app would.
 */
afterEach(() => {
  vi.doUnmock('@smaltworks/smaltml/simd');
  vi.resetModules();
});

/** The kernels the SIMD entry gives by default. */
const SIMD_KERNELS = ['gramPart', 'matMul', 'binaryLogisticRows', 'multinomialLogisticRows', 'logisticProducts', 'xtvPart', 'columnSums', 'treeSplits'];

/** A column-major product: the stand-in entry's matMul. */
function product(n: number, p: number, q: number, a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(n * q);
  for (let j = 0; j < q; j++) for (let l = 0; l < p; l++) for (let i = 0; i < n; i++) out[j * n + i]! += a[l * n + i]! * b[j * p + l]!;
  return out;
}

async function smaltml() {
  return import('../src/index.js');
}

describe('kernelsReady: smaltml turns its SIMD kernels on', () => {
  test('where the engine has WebAssembly SIMD (Node has), the real entry turns them on', async () => {
    vi.resetModules();
    const ml = await smaltml();
    expect(await ml.kernelsReady()).toEqual({ name: 'smaltml-simd', kernels: SIMD_KERNELS, rejected: [] });
  });

  test('where the entry can\'t be loaded (a bundled app that leaves it out), smaltml keeps its own kernels', async () => {
    vi.resetModules();
    const ml = await smaltml();
    vi.doMock('@smaltworks/smaltml/simd', () => {
      throw new Error('not in this bundle');
    });
    expect((await ml.kernelsReady()).name).toBe('smaltml');
  });

  test('the entry\'s useSimd() runs once, and async fits wait for it', async () => {
    vi.resetModules();
    const ml = await smaltml();
    const calls: string[] = [];
    vi.doMock('@smaltworks/smaltml/simd', () => ({
      useSimd: async () => {
        calls.push('useSimd');
        // What the entry does: a kernel set through useKernels (here smaltml's own matMul, under the entry's set name).
        ml.useKernels({ name: 'smaltml-simd', version: ml.KERNELS_VERSION, kernels: { matMul: product } });
        return true;
      },
    }));
    // No call yet: a sync call before the look-up uses smaltml's own.
    expect(ml.kernelsInUse().name).toBe('smaltml');
    await new ml.Ridge().fit([[1, 2], [2, 1], [3, 4], [4, 3]], [1, 2, 3, 4]);
    expect(ml.kernelsInUse()).toEqual({ name: 'smaltml-simd', kernels: ['matMul'], rejected: [] });
    await ml.kernelsReady();
    await ml.matMul([[1, 2]], [[3], [4]]);
    expect(calls).toEqual(['useSimd']);
  });

  test("an app's own choice wins: useKernels(null) before the look-up keeps smaltml's", async () => {
    vi.resetModules();
    const ml = await smaltml();
    const useSimd = vi.fn(async () => true);
    vi.doMock('@smaltworks/smaltml/simd', () => ({ useSimd }));
    ml.useKernels(null);
    expect((await ml.kernelsReady()).name).toBe('smaltml');
    expect(useSimd).not.toHaveBeenCalled();
  });

  test('SIMD kernels that fail to start leave smaltml its own kernels', async () => {
    vi.resetModules();
    const ml = await smaltml();
    vi.doMock('@smaltworks/smaltml/simd', () => ({ useSimd: async () => { throw new Error('no SIMD here'); } }));
    expect((await ml.kernelsReady()).name).toBe('smaltml');
  });
});
