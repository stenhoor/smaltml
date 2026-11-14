import { fail } from '../core/errors.js';
import { Matrix } from '../core/matrix.js';
import { checkDocument, document, isTransformer, type Labels, type Model, type ModelDocument, type Transformer } from '../core/model.js';
import { isText, toTable, type Table, type TableInput } from '../core/table.js';

/** Columns by name (`['age', 'spend']`) or by position (`[0, 1]`). */
export type ColumnSelection = readonly string[] | readonly number[];

/** One part of a ColumnTransformer: its name, what it does, and the columns it takes. */
export type ColumnPart = readonly [name: string, transformer: Transformer | 'passthrough' | 'drop', columns: ColumnSelection];

export interface ColumnTransformerParams {
  /** What happens to the columns no part names: dropped (`'drop'`) or kept as they are (`'passthrough'`). Default `'drop'`. */
  remainder?: 'drop' | 'passthrough';
}

interface State {
  nFeaturesIn: number;
  featureNamesIn: readonly string[] | undefined;
  /** Each part's column positions, then the remainder's. */
  positions: number[][];
  remainder: number[];
  featureNamesOut: string[];
}

/**
 * Runs a different transformer on different columns, side by side, and puts their outputs together: scale the
 * numbers, one-hot encode the text. scikit-learn's `ColumnTransformer`, with dense output. Output columns are named
 * `part__column`.
 */
export class ColumnTransformer implements Transformer {
  readonly kind = 'ColumnTransformer';
  readonly parts: readonly ColumnPart[];
  readonly remainder: 'drop' | 'passthrough';
  #state: State | null = null;

  constructor(parts: readonly ColumnPart[], params: ColumnTransformerParams = {}) {
    if (parts.length === 0) fail('ML_PARAM', 'a ColumnTransformer needs at least one part');
    this.remainder = params.remainder ?? 'drop';
    if (this.remainder !== 'drop' && this.remainder !== 'passthrough') fail('ML_PARAM', `remainder is 'drop' or 'passthrough', not ${String(this.remainder)}`);
    const seen = new Set<string>();
    for (const [name, t, cols] of parts) {
      if (seen.has(name)) fail('ML_PARAM', `two parts are named ${name}`);
      seen.add(name);
      if (t !== 'drop' && t !== 'passthrough' && !isTransformer(t)) fail('ML_PARAM', `part ${name} must be a transformer, 'passthrough' or 'drop'`);
      if (!Array.isArray(cols)) fail('ML_PARAM', `part ${name}'s columns must be a list of names or positions`);
    }
    this.parts = parts;
  }

  get fitted(): boolean {
    return this.#state !== null;
  }
  get nFeaturesIn(): number {
    return this.#fitted().nFeaturesIn;
  }
  get featureNamesIn(): readonly string[] | undefined {
    return this.#fitted().featureNamesIn;
  }

  /** The names of the output columns: `part__column`. */
  getFeatureNamesOut(): string[] {
    return [...this.#fitted().featureNamesOut];
  }

  /** The part with this name's transformer. */
  part(name: string): Transformer | 'passthrough' | 'drop' {
    const p = this.parts.find(([n]) => n === name);
    return p ? p[1] : fail('ML_PARAM', `no part is named ${name}`);
  }

  getParams(): Record<string, unknown> {
    return {
      remainder: this.remainder,
      parts: Object.fromEntries(this.parts.map(([n, t]) => [n, typeof t === 'string' ? t : t.getParams()])),
    };
  }

  async fit(X: TableInput, y?: Labels): Promise<this> {
    return this.fitSync(X, y);
  }

  fitSync(X: TableInput, y?: Labels): this {
    const t = toTable(X);
    if (t.rows === 0) fail('ML_SHAPE', 'X has no rows');
    const positions = this.parts.map(([name, , cols]) => resolve(t, cols, name));
    const used = new Set(positions.flat());
    const remainder = this.remainder === 'passthrough' ? t.names.map((_, j) => j).filter((j) => !used.has(j)) : [];
    const names: string[] = [];
    this.parts.forEach(([name, tr], k) => {
      if (tr === 'drop') return;
      const sub = select(t, positions[k]!);
      if (tr === 'passthrough') {
        names.push(...Object.keys(sub).map((c) => `${name}__${c}`));
        return;
      }
      const out = tr.fitSync(sub as never, y).transformSync(sub as never);
      names.push(...outputNames(out, tr).map((c) => `${name}__${c}`));
    });
    names.push(...Object.keys(select(t, remainder)).map((c) => `remainder__${c}`));
    this.#state = { nFeaturesIn: t.columns.length, featureNamesIn: t.named ? t.names : undefined, positions, remainder, featureNamesOut: names };
    return this;
  }

  async transform(X: TableInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: TableInput): Matrix {
    const s = this.#fitted();
    const t = toTable(X);
    if (t.columns.length !== s.nFeaturesIn) fail('ML_SHAPE', `X has ${t.columns.length} columns; this ColumnTransformer was fitted on ${s.nFeaturesIn}`);
    const blocks: Matrix[] = [];
    this.parts.forEach(([, tr], k) => {
      if (tr === 'drop') return;
      const sub = select(t, s.positions[k]!);
      blocks.push(tr === 'passthrough' ? numeric(sub, t.rows) : Matrix.from(tr.transformSync(sub as never)));
    });
    if (s.remainder.length) blocks.push(numeric(select(t, s.remainder), t.rows));
    const cols = blocks.reduce((c, b) => c + b.cols, 0);
    const data = new Float64Array(t.rows * cols);
    let offset = 0;
    for (const b of blocks) {
      data.set(b.data, offset * t.rows);
      offset += b.cols;
    }
    return new Matrix(t.rows, cols, data, s.featureNamesOut);
  }

  async fitTransform(X: TableInput, y?: Labels): Promise<Matrix> {
    return this.fitSync(X, y).transformSync(X);
  }

  toJSON(): ModelDocument {
    const s = this.#state;
    return document(
      this.kind,
      { remainder: this.remainder, parts: this.parts.map(([n, t, c]) => ({ name: n, columns: [...c], transformer: typeof t === 'string' ? t : t.kind })) },
      {
        parts: this.parts.map(([, t]) => (typeof t === 'string' ? t : t.toJSON())),
        ...(s ? { fitted: { nFeaturesIn: s.nFeaturesIn, featureNamesIn: s.featureNamesIn ?? null, positions: s.positions, remainder: s.remainder, featureNamesOut: s.featureNamesOut } } : {}),
      },
    );
  }

  /** A ColumnTransformer from its document; `load` turns each part's document into its transformer. */
  static fromJSON(doc: ModelDocument, load: (doc: ModelDocument) => Model): ColumnTransformer {
    const d = checkDocument(doc, 'ColumnTransformer');
    const parts = d.params.parts as { name: string; columns: ColumnSelection }[] | undefined;
    const models = d.state?.parts as (ModelDocument | 'drop' | 'passthrough')[] | undefined;
    if (!Array.isArray(parts) || !Array.isArray(models) || parts.length !== models.length) fail('ML_FORMAT', 'a ColumnTransformer document needs its parts');
    const ct = new ColumnTransformer(
      parts.map((p, k) => {
        const m = models[k]!;
        return [p.name, typeof m === 'string' ? m : (load(m) as Transformer), p.columns] as const;
      }),
      { remainder: d.params.remainder as 'drop' | 'passthrough' },
    );
    const f = d.state?.fitted as (Omit<State, 'featureNamesIn'> & { featureNamesIn: string[] | null }) | undefined;
    if (f) ct.#state = { ...f, featureNamesIn: f.featureNamesIn ?? undefined };
    return ct;
  }

  #fitted(): State {
    return this.#state ?? fail('ML_NOT_FITTED', "this ColumnTransformer isn't fitted yet: call fit() first");
  }
}

/** Column positions for a selection: names need named columns. */
function resolve(t: Table, cols: ColumnSelection, part: string): number[] {
  return (cols as readonly (string | number)[]).map((c) => {
    if (typeof c === 'number') {
      if (!Number.isInteger(c) || c < 0 || c >= t.columns.length) fail('ML_PARAM', `part ${part}: there is no column ${c}`);
      return c;
    }
    const j = t.named ? t.names.indexOf(c) : -1;
    if (j < 0) fail('ML_PARAM', `part ${part}: there is no column named ${c}${t.named ? '' : ' (X has no column names: select by position)'}`);
    return j;
  });
}

/** The selected columns as named columns: by name, or `x0`, `x1` … by position where X had no names. */
function select(t: Table, positions: number[]): Record<string, Float64Array | (string | null)[]> {
  return Object.fromEntries(positions.map((j) => [t.named ? t.names[j]! : `x${j}`, t.columns[j]!]));
}

function numeric(cols: Record<string, Float64Array | (string | null)[]>, rows: number): Matrix {
  for (const [name, c] of Object.entries(cols)) if (isText(c)) fail('ML_VALUE', `column ${name} holds text, so it can't pass through: encode it`);
  if (Object.keys(cols).length === 0) return new Matrix(rows, 0);
  return Matrix.fromColumns(cols as Record<string, Float64Array>);
}

/** A transformer's output column names: its getFeatureNamesOut() where it has one, else the output's names. */
function outputNames(out: Matrix, tr: Transformer): string[] {
  const own = (tr as Partial<{ getFeatureNamesOut(): string[] }>).getFeatureNamesOut;
  if (own) return own.call(tr);
  return Array.from({ length: out.cols }, (_, j) => out.names?.[j] ?? `${j}`);
}
