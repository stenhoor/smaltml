// The benchmark data: seeded, written as raw little-endian float64 files that both smaltml and Python read, so both
// time the same numbers. X is stored column by column (Fortran order). Not committed: bench/data/ is regenerated.
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';

export const SIZES = [
  { name: '10k × 20', key: '10kx20', n: 10_000, p: 20 },
  { name: '200k × 50', key: '200kx50', n: 200_000, p: 50 },
];

/** A seeded normal generator (mulberry32 and Box–Muller): the same numbers on every machine. */
function normals(seed) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-300)) * Math.cos(2 * Math.PI * u());
}

export function makeData(dir) {
  mkdirSync(dir, { recursive: true });
  for (const { key, n, p } of SIZES) {
    if (existsSync(`${dir}/${key}.X`)) continue;
    const r = normals(n * 31 + p);
    const scale = Array.from({ length: p }, () => 0.5 + 20 * Math.abs(r()));
    const shift = Array.from({ length: p }, () => 10 * r());
    const w = Array.from({ length: p }, () => r());
    const X = new Float64Array(n * p);
    for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) X[j * n + i] = r() * scale[j] + shift[j];
    // y: linear in X plus noise; y2: two classes, y3: three, from the standardized linear score plus noise.
    const y = new Float64Array(n), y2 = new Float64Array(n), y3 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0, z = 0;
      for (let j = 0; j < p; j++) {
        const x = X[j * n + i];
        s += x * w[j];
        z += ((x - shift[j]) / scale[j]) * w[j];
      }
      y[i] = s + r();
      const zn = z + r();
      y2[i] = zn > 0 ? 1 : 0;
      y3[i] = zn < -0.6 ? 0 : zn < 0.6 ? 1 : 2;
    }
    for (const [suffix, v] of [['X', X], ['y', y], ['y2', y2], ['y3', y3]]) writeFileSync(`${dir}/${key}.${suffix}`, Buffer.from(v.buffer));
  }
  writeFileSync(`${dir}/meta.json`, JSON.stringify(SIZES));
}
