// smaltml's runner for the site kit's playgrounds (playground.ts, BRAND.md, Documentation, rule 7). Not a kit file.
//
// Every ```ts example runs where it stands. A page's examples share one scope, as a notebook's cells do: an example
// runs after the examples above it (its setup), so a model fitted above is there below. Imports of
// `@smaltworks/smaltml` (or any of its entry points) take smaltml's exports; the types are removed with smaltkit's
// TypeScript (smaltty). The example's last line, if it is an expression, is its result, shown as a table.
//
// test/docs-examples.test.ts runs every example the same way (runExample) and checks its printed result.
import { stripTypes } from 'smaltkit/typescript'
import * as smaltml from '../../../src/index'
import type { PlayResult, SetupStep } from './playground'

/** The code of a setup step. */
const codeOf = (s: SetupStep): string => (typeof s === 'string' ? s : s.sql)

/**
 * `import { a, b as c } from '@smaltworks/smaltml/…'` as `var { a, b: c } = __smaltml;` (`var`, so two examples on a
 * page may import the same name); types-only imports go.
 */
function rewriteImports(code: string): string {
  return code.replace(/^\s*import\s+(type\s+)?\{([^}]*)\}\s+from\s+['"]@smaltworks\/smaltml(?:\/\w+)?['"];?[ \t]*$/gm, (_m, typeOnly: string | undefined, names: string) => {
    if (typeOnly) return ''
    const values = names
      .split(',')
      .map((n) => n.trim())
      .filter((n) => n && !n.startsWith('type '))
      .map((n) => n.replace(/\s+as\s+/, ': '))
    return values.length ? `var { ${values.join(', ')} } = __smaltml;` : ''
  })
}

/** Whether a line starts a statement rather than an expression. */
const STATEMENT = /^(const|let|var|import|export|function|async\s+function|class|if|for|while|do|switch|try|throw|return|type|interface|\/\/|\/\*|\}|\{)\b|^[}\])]/

/**
 * The example's code with its result returned: its last top-level line, when that line is a whole expression and
 * not a statement. Lines that continue an expression (indented, or ending in an operator) don't count.
 */
function withResult(code: string): string {
  const lines = code.replace(/\s+$/, '').split('\n')
  const last = lines.length - 1
  const line = lines[last] ?? ''
  if (!line.trim() || /^\s/.test(line) || STATEMENT.test(line.trim())) return code
  // An expression that started on an earlier line (that line ends open) isn't the last line's alone.
  if (last > 0 && /[(,[{=+\-*/&|?:.]\s*$/.test(lines[last - 1]!)) return code
  lines[last] = `return (${line.trim().replace(/;$/, '')});`
  return lines.join('\n')
}

const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor as new (...args: string[]) => (...a: unknown[]) => Promise<unknown>

/** Runs an example after its setup, in one scope, and returns its result (its last line's value, or undefined). */
export async function runExample(code: string, setup: readonly SetupStep[]): Promise<unknown> {
  const before = setup.filter((s) => !(typeof s === 'object' && s.fails)).map(codeOf)
  const source = [...before, withResult(code)].map((c) => rewriteImports(c)).join('\n;\n')
  const js = stripTypes(source, { filename: 'example.ts' })
  const fn = new AsyncFunction('__smaltml', js)
  return fn(smaltml)
}

/** A number as the examples print it: whole numbers as they are, others to 6 significant digits. */
export function printedNumber(v: number): string {
  if (Number.isNaN(v)) return 'NaN'
  if (Number.isInteger(v)) return String(v)
  const s = Number(v.toPrecision(6))
  return Object.is(s, -0) ? '0' : String(s)
}

/** A value as a result cell shows it. The examples test escapes it for a Markdown table. */
export function printed(v: unknown): string {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'number') return printedNumber(v)
  if (typeof v === 'string') return v
  if (ArrayBuffer.isView(v)) return `[${Array.from(v as unknown as ArrayLike<number>, printedNumber).join(', ')}]`
  if (Array.isArray(v)) return `[${v.map(printed).join(', ')}]`
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/** A result as a table: columns and rows of printed cells, or a note. */
export function toTable(value: unknown): { columns: string[]; rows: string[][]; note?: string } {
  const one = (name: string, v: unknown) => ({ columns: [name], rows: [[printed(v)]] })
  if (value === undefined) return { columns: [], rows: [], note: 'Done' }
  if (value instanceof smaltml.Matrix) {
    const columns = Array.from({ length: value.cols }, (_, j) => value.columnName(j))
    return { columns, rows: value.toRows().map((r) => r.map(printedNumber)) }
  }
  if (isModel(value)) {
    return { columns: [], rows: [], note: `${value.kind}, ${value.fitted ? 'fitted' : 'not fitted'}` }
  }
  if (ArrayBuffer.isView(value) || (Array.isArray(value) && value.every((v) => !isObject(v)))) {
    return { columns: ['value'], rows: Array.from(value as ArrayLike<unknown>, (v) => [printed(v)]) }
  }
  if (Array.isArray(value) && value.every((v) => isObject(v) && !Array.isArray(v))) {
    const columns = [...new Set(value.flatMap((r) => Object.keys(r as object)))]
    return { columns, rows: value.map((r) => columns.map((c) => printed((r as Record<string, unknown>)[c]))) }
  }
  if (isObject(value) && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>)
    // An object of columns (every value a list of the same length) is a table; otherwise one row.
    const lists = entries.every(([, v]) => Array.isArray(v) || ArrayBuffer.isView(v))
    if (lists && entries.length) {
      const n = (entries[0]![1] as ArrayLike<unknown>).length
      if (entries.every(([, v]) => (v as ArrayLike<unknown>).length === n)) {
        return {
          columns: entries.map(([k]) => k),
          rows: Array.from({ length: n }, (_, i) => entries.map(([, v]) => printed((v as ArrayLike<unknown>)[i]))),
        }
      }
    }
    return { columns: entries.map(([k]) => k), rows: [entries.map(([, v]) => printed(v))] }
  }
  return one('value', value)
}

function isObject(v: unknown): v is object {
  return v !== null && typeof v === 'object' && !ArrayBuffer.isView(v)
}

function isModel(v: unknown): v is smaltml.Model {
  return isObject(v) && typeof (v as smaltml.Model).kind === 'string' && typeof (v as smaltml.Model).toJSON === 'function'
}

/** The runner for ```ts playgrounds. */
export function smaltmlRunner() {
  return async () => ({
    async run(code: string, setup: readonly SetupStep[]): Promise<PlayResult> {
      const t = toTable(await runExample(code, setup))
      return t.note !== undefined ? { columns: [], rows: [], note: t.note } : { columns: t.columns, rows: t.rows }
    },
  })
}
