// Builds dist/: one ES module per entry point, and type declarations.
//   node scripts/build.mjs           the package
//   node scripts/build.mjs --minify  minified, for publishing
import * as esbuild from 'esbuild';
import { execFileSync } from 'node:child_process';
import { rmSync } from 'node:fs';

const minify = process.argv.includes('--minify');
rmSync('dist', { recursive: true, force: true });

export const entries = {
  smaltml: 'src/index.ts',
  core: 'src/core/index.ts',
  preprocessing: 'src/preprocessing/index.ts',
  linear: 'src/linear/index.ts',
  metrics: 'src/metrics/index.ts',
  selection: 'src/selection/index.ts',
  pipeline: 'src/pipeline/index.ts',
  io: 'src/io/index.ts',
  gpu: 'src/gpu/index.ts',
  registry: 'src/registry/index.ts',
  worker: 'src/worker/index.ts',
  'smaltml-worker': 'src/worker/worker.ts',
};

await esbuild.build({
  entryPoints: entries,
  outdir: 'dist',
  bundle: true,
  format: 'esm',
  target: 'es2022',
  platform: 'neutral',
  splitting: true,
  chunkNames: 'chunks/[name]-[hash]',
  minify,
  sourcemap: true,
  logLevel: 'warning',
});

execFileSync('node_modules/.bin/tsc', ['-p', 'tsconfig.build.json'], { stdio: 'inherit' });
console.log(`built dist/ (${Object.keys(entries).length} entry points${minify ? ', minified' : ''})`);
