import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, test } from 'vitest';
import { typeChecker } from 'smaltkit/typechecker';
import { runExample, toTable } from '../docs/.vitepress/theme/smaltml.js';
import type { SetupStep } from '../docs/.vitepress/theme/playground.js';

/**
 * Examples are tests (BRAND.md, Documentation, rule 5). Every ```ts example in docs/ runs as its playground runs it
 * (theme/smaltml.ts): after the page's examples above it, in one scope. The table printed under it is its tested
 * result, row for row; a long one prints its first rows and `… N more rows`. An example marked
 * `<!-- docs:error CODE -->` must fail with CODE. A fence marked `noplay`, or after `<!-- docs:skip -->`, is not run.
 *
 * `UPDATE_DOCS=1 npx vitest run test/docs-examples.test.ts` writes the printed results into the pages.
 */
const UPDATE = !!process.env.UPDATE_DOCS;
const LONG = 12;
const SHORT = 6;
const DOCS = new URL('../docs/', import.meta.url).pathname;

function markdown(dir: string): string[] {
  return readdirSync(dir)
    .sort()
    .flatMap((f) => {
      const p = join(dir, f);
      if (f.startsWith('.') || f === 'node_modules' || f === 'public') return [];
      return statSync(p).isDirectory() ? markdown(p) : p.endsWith('.md') ? [p] : [];
    });
}

interface Example {
  line: number;
  end: number;
  table: { from: number; to: number } | null;
  code: string;
  fails?: string;
  setup: SetupStep[];
}

function examples(file: string): Example[] {
  const lines = readFileSync(file, 'utf8').split('\n');
  const out: Example[] = [];
  const setup: SetupStep[] = [];
  let markers: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const marker = /^<!--\s*(docs:[\s\S]*?)\s*-->\s*$/.exec(line.trim());
    if (marker) {
      markers.push(marker[1]!);
      continue;
    }
    const open = /^```(\S*)\s*(.*)$/.exec(line);
    if (!open) {
      if (line.trim()) markers = [];
      continue;
    }
    const start = i;
    for (i++; i < lines.length && lines[i] !== '```'; i++);
    const [lang, info] = [open[1]!, open[2]!];
    const kinds = markers.map((m) => m.split(/\s+/)[0]);
    const fails = markers.find((m) => m.startsWith('docs:error'))?.split(/\s+/)[1];
    markers = [];
    if (lang !== 'ts' || kinds.includes('docs:skip')) continue;
    // As the kit's playground does: a noplay fence doesn't run, but is setup for the examples below it.
    if (info.split(/\s+/).includes('noplay')) {
      setup.push(lines.slice(start + 1, i).join('\n').trim());
      continue;
    }
    const code = lines.slice(start + 1, i).join('\n');
    let j = i + 1;
    while (j < lines.length && !lines[j]!.trim()) j++;
    let table: Example['table'] = null;
    if (lines[j]?.startsWith('|')) {
      const from = j;
      while (j < lines.length && lines[j]!.startsWith('|')) j++;
      table = { from, to: j };
    }
    out.push({ line: start + 1, end: i, table, code, fails, setup: setup.slice() });
    setup.push(fails !== undefined ? { sql: code.trim(), fails: true } : code.trim());
  }
  return out;
}

/** A result as the Markdown table under its example. */
function render(t: { columns: string[]; rows: string[][]; note?: string }): string[] {
  if (t.note !== undefined) return [];
  const rows = t.rows.length > LONG ? t.rows.slice(0, SHORT) : t.rows;
  // Escaped for Markdown: | ends a cell, and VitePress reads {…} at a cell's end as attributes.
  const esc = (c: string) => c.replace(/[|{}]/g, '\\$&');
  const out = [`| ${t.columns.map(esc).join(' | ')} |`, `| ${t.columns.map(() => '---').join(' | ')} |`, ...rows.map((r) => `| ${r.map(esc).join(' | ')} |`)];
  if (t.rows.length > LONG) out.push(`| … ${t.rows.length - SHORT} more rows |${' |'.repeat(Math.max(0, t.columns.length - 1))}`);
  return out;
}

/**
 * The examples type-checked by smaltty, the smalt apps' own checker (through smaltkit/typechecker), against smaltml's
 * source by its package names, as a reader's editor would check them: a misspelt method or option, or an option of the
 * wrong type, fails here with tsc's message. Each example is checked with the examples above it on its page (they share
 * one scope); only errors in its own lines are its own.
 */
const SRC = new URL('../src/', import.meta.url).pathname;
function sources(dir: string): [string, string][] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? sources(p) : p.endsWith('.ts') ? [[`/src/${p.slice(SRC.length)}`, readFileSync(p, 'utf8')] as [string, string]] : [];
  });
}
const AREAS = ['core', 'preprocessing', 'linear', 'metrics', 'selection', 'pipeline', 'io', 'gpu', 'parallel', 'registry', 'worker'];
const checkTypes = typeChecker({
  files: Object.fromEntries(sources(SRC)),
  packages: { '@smaltworks/smaltml': '/src/index.ts', ...Object.fromEntries(AREAS.map((a) => [`@smaltworks/smaltml/${a}`, `/src/${a}/index.ts`])) },
  // smaltty 0.15.0's further checks (stenhoor/smaltty#38): a model awaited from fit() has its members checked, an
  // overloaded function its argument count, and an annotated variable its value.
  checks: ['this', 'overloads', 'variables'],
});

/** The code smaltty checks for an example: a module, the setup above it, then the example; repeated imports once. */
function checkedCode(ex: Example): { code: string; first: number } {
  const seen = new Set<string>();
  const once = (part: string) =>
    part
      .split('\n')
      .filter((l) => !/^\s*import\s/.test(l) || (!seen.has(l.trim()) && !!seen.add(l.trim())))
      .join('\n');
  const before = ['export {};', ...ex.setup.map((s) => (typeof s === 'string' ? s : s.sql)).map(once)].join('\n;\n') + '\n;\n';
  return { code: before + once(ex.code), first: before.split('\n').length };
}

const files = markdown(DOCS);
const updates = new Map<string, { at: Example; table: string[] }[]>();

for (const file of files) {
  const list = examples(file);
  if (list.length === 0) continue;
  describe(file.slice(DOCS.length), () => {
    for (const ex of list) {
      test(`line ${ex.line} type-checks (smaltty)`, () => {
        const { code, first } = checkedCode(ex);
        const own = checkTypes(code).filter((w) => w.line >= first);
        expect(own.map((w) => `line ${ex.line + w.line - first + 1}: TS${w.code} ${w.message}`), `${file}:${ex.line}`).toEqual([]);
      });
      test(`line ${ex.line}`, async () => {
        if (ex.fails !== undefined) {
          await expect(runExample(ex.code, ex.setup)).rejects.toMatchObject({ code: ex.fails });
          return;
        }
        const table = render(toTable(await runExample(ex.code, ex.setup)));
        const lines = readFileSync(file, 'utf8').split('\n');
        const printed = ex.table ? lines.slice(ex.table.from, ex.table.to) : [];
        if (UPDATE) {
          const list = updates.get(file) ?? [];
          list.push({ at: ex, table });
          updates.set(file, list);
          return;
        }
        expect(printed.map((l) => l.trim()), `${file}:${ex.line}: run UPDATE_DOCS=1 to rewrite`).toEqual(table);
      });
    }
  });
}

if (UPDATE) {
  test('write the printed results', () => {
    for (const [file, list] of updates) {
      const lines = readFileSync(file, 'utf8').split('\n');
      // From the bottom up, so earlier line numbers stay right.
      for (const { at, table } of [...list].sort((a, b) => b.at.end - a.at.end)) {
        if (at.table) lines.splice(at.table.from, at.table.to - at.table.from, ...table);
        else if (table.length) lines.splice(at.end + 1, 0, '', ...table);
      }
      writeFileSync(file, lines.join('\n'));
    }
  });
}
