import * as esbuild from 'esbuild';
import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';

/**
 * Every entry point is a package of its own (ADR 0024, section 9): it bundles alone, pulls in only the areas this
 * table allows, and uses neither the DOM nor Node, so the core runs in a page, a worker or Node. A new dependency
 * between areas changes this table on purpose, never by accident.
 */
const ALLOWED: Record<string, string[]> = {
  core: ['core'],
  preprocessing: ['core', 'preprocessing'],
  linear: ['core', 'linear', 'gpu', 'parallel'],
  metrics: ['core', 'metrics'],
  selection: ['core', 'selection', 'io', 'metrics', 'preprocessing', 'linear', 'pipeline', 'gpu', 'parallel'],
  pipeline: ['core', 'pipeline'],
  multioutput: ['core', 'multioutput', 'io', 'preprocessing', 'linear', 'pipeline', 'gpu', 'parallel'],
  io: ['core', 'io', 'preprocessing', 'linear', 'pipeline', 'gpu', 'parallel'],
  gpu: ['core', 'gpu'],
  parallel: ['core', 'parallel'],
  registry: ['core', 'registry', 'io', 'preprocessing', 'linear', 'pipeline', 'gpu', 'parallel'],
  worker: ['core', 'worker', 'io', 'preprocessing', 'linear', 'pipeline', 'gpu', 'parallel'],
};

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as { exports: Record<string, unknown> };

describe('entry points', () => {
  test('package.json exports every area, and only those', () => {
    // ./worker-script and ./threads-script are workers' scripts, not areas.
    const areas = Object.keys(pkg.exports).filter((k) => !['.', './package.json', './worker-script', './threads-script'].includes(k)).map((k) => k.slice(2));
    expect(areas.sort()).toEqual(Object.keys(ALLOWED).sort());
  });

  test.each(Object.keys(ALLOWED))('%s bundles alone, from the areas it may use', async (area) => {
    const result = await esbuild.build({
      entryPoints: [`src/${area}/index.ts`],
      bundle: true,
      write: false,
      format: 'esm',
      platform: 'neutral',
      metafile: true,
      logLevel: 'silent',
    });
    const used = new Set(
      Object.keys(result.metafile.inputs)
        .map((f) => /^src\/(\w+)\//.exec(f)?.[1] ?? (f === 'src/version.ts' ? 'core' : f))
        .filter(Boolean),
    );
    expect([...used].sort()).toEqual(expect.arrayContaining([area]));
    for (const u of used) expect(ALLOWED[area], `${area} pulls in ${u}`).toContain(u);
    const code = result.outputFiles[0]!.text;
    expect(code).not.toMatch(/\bdocument\.|\bwindow\.|from ['"]node:/);
  });
});
