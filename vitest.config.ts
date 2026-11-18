import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const src = (p: string) => fileURLToPath(new URL(`./src/${p}`, import.meta.url));

export default defineConfig({
  resolve: {
    alias: [
      { find: /^@smaltworks\/smaltml$/, replacement: src('index.ts') },
      { find: /^@smaltworks\/smaltml\/(\w+)$/, replacement: src('$1/index.ts') },
    ],
  },
  test: {
    include: ['test/**/*.test.ts'],
    // smaltml's own kernels, unless SMALTML_SIMD=1 (npm run test:simd): then its SIMD kernels.
    setupFiles: [process.env.SMALTML_SIMD === '1' ? 'test/simd-setup.ts' : 'test/setup.ts'],
  },
});
