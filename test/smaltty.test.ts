import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { check, VERSION } from 'smaltty';
import { describe, expect, test } from 'vitest';

/**
 * smaltml's own source checked by smaltty, the smalt family's TypeScript checker, with its further checks on, beside
 * tsc (`npm run typecheck`, the gate). tsc reports no errors here, so smaltty must report none: a false error is a
 * smaltty bug. smaltty's downstream check runs these tests before each release (stenhoor/smaltty#38). smaltty catches
 * fewer of tsc's errors (it checks for smalt code, not parity with tsc), so tsc stays the gate for smaltml's source.
 */
const SRC = new URL('../src/', import.meta.url).pathname;
function sources(dir: string): [string, string][] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? sources(p) : p.endsWith('.ts') ? [[`/src/${p.slice(SRC.length)}`, readFileSync(p, 'utf8')] as [string, string]] : [];
  });
}
const files = new Map(sources(SRC));
const CHECKS = ['this', 'overloads', 'variables'] as const;
const show = (d: { file: string; code: number; message: string; start: number }) => {
  const line = files.get(d.file)?.slice(0, d.start).split('\n').length;
  return `${d.file}:${line} TS${d.code} ${d.message}`;
};

describe(`smaltml's source, checked by smaltty ${VERSION}`, () => {
  test('no errors, as tsc reports none', () => {
    expect(check(files, { checks: CHECKS }).map(show)).toEqual([]);
  });

  test('a mistake planted in the source is caught, so the check reads it', () => {
    const planted = new Map(files);
    planted.set('/src/core/errors.ts', `${files.get('/src/core/errors.ts')}\nexport const planted: number = new SmaltmlError('ML_PARAM', 'x');\n`);
    const found = check(planted, { checks: CHECKS });
    expect(found.map((d) => [d.file, d.code])).toEqual([['/src/core/errors.ts', 2322]]);
  });
});
