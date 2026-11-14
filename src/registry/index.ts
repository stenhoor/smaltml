/**
 * The model registry: fitted models kept by name and version, and predict() by name (ADR 0024, sections 4 and 6).
 * Notebooks and pipelines save models; apps and notebooks predict with them. A store keeps the saved documents: in
 * memory, or as rows of a table in a SQL database such as smaltdb.
 * @module
 */
import { fail } from '../core/errors.js';
import type { Matrix, MatrixInput } from '../core/matrix.js';
import { checkDocument, isClassifier, isPredictor, type Label, type Model, type ModelDocument } from '../core/model.js';
import type { Device } from '../gpu/index.js';
import { loadModel } from '../io/index.js';

/** One saved version of a model. */
export interface ModelEntry {
  name: string;
  version: number;
  /** The model's class, such as `LogisticRegression`. */
  model: string;
  /** When it was saved, as an ISO 8601 time. */
  savedAt: string;
}

/** Where a registry keeps its models' documents. */
export interface ModelStore {
  /** Saves a document as the next version of `name`, and returns that version (1 for the first). */
  put(name: string, doc: ModelDocument): Promise<number>;
  /** The document of `name` at `version`, or its latest version; undefined when there is none. */
  get(name: string, version?: number): Promise<ModelDocument | undefined>;
  /** Every saved version, by name, then version. */
  list(): Promise<ModelEntry[]>;
  /** Removes `name` at `version`, or every version of it; returns how many it removed. */
  delete(name: string, version?: number): Promise<number>;
}

/** Keeps models in memory: for tests, docs, and apps that save models somewhere of their own. */
export class MemoryStore implements ModelStore {
  #versions = new Map<string, { doc: ModelDocument; entry: ModelEntry }[]>();

  async put(name: string, doc: ModelDocument): Promise<number> {
    checkName(name);
    const list = this.#versions.get(name) ?? [];
    const version = (list.at(-1)?.entry.version ?? 0) + 1;
    list.push({ doc: structuredClone(doc), entry: { name, version, model: doc.model, savedAt: new Date().toISOString() } });
    this.#versions.set(name, list);
    return version;
  }

  async get(name: string, version?: number): Promise<ModelDocument | undefined> {
    const list = this.#versions.get(name);
    const found = version === undefined ? list?.at(-1) : list?.find((v) => v.entry.version === version);
    return found && structuredClone(found.doc);
  }

  async list(): Promise<ModelEntry[]> {
    return [...this.#versions.keys()].sort().flatMap((n) => this.#versions.get(n)!.map((v) => ({ ...v.entry })));
  }

  async delete(name: string, version?: number): Promise<number> {
    const list = this.#versions.get(name) ?? [];
    const keep = version === undefined ? [] : list.filter((v) => v.entry.version !== version);
    if (keep.length) this.#versions.set(name, keep);
    else this.#versions.delete(name);
    return list.length - keep.length;
  }
}

/**
 * The part of a SQL database a TableStore uses: smaltdb's `Database` (with smaltql) has it, and so do others that take
 * `?` parameters. `query` returns rows as objects.
 */
export interface SqlDatabase {
  query(sql: string, params?: unknown[]): Promise<readonly Record<string, unknown>[]>;
  exec(sql: string, params?: unknown[]): Promise<unknown>;
}

/**
 * Keeps models as rows of a table, `smaltml_models` unless you name another: one row per name and version, with the
 * model's class, the smaltml version that saved it, when, and the document as JSON text. The table is created when
 * it isn't there.
 */
export class TableStore implements ModelStore {
  readonly db: SqlDatabase;
  readonly table: string;
  #ready: Promise<unknown> | undefined;

  constructor(db: SqlDatabase, options: { table?: string } = {}) {
    this.db = db;
    this.table = options.table ?? 'smaltml_models';
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(this.table)) fail('ML_PARAM', `${this.table} isn't a plain table name: letters, digits and _`);
  }

  #init(): Promise<unknown> {
    return (this.#ready ??= this.db.exec(
      `CREATE TABLE IF NOT EXISTS ${this.table} (name TEXT NOT NULL, version INTEGER NOT NULL, model TEXT NOT NULL, smaltml TEXT NOT NULL, saved_at TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY (name, version))`,
    ));
  }

  async put(name: string, doc: ModelDocument): Promise<number> {
    checkName(name);
    await this.#init();
    const rows = await this.db.query(`SELECT MAX(version) AS v FROM ${this.table} WHERE name = ?`, [name]);
    const version = Number(rows[0]?.v ?? 0) + 1;
    await this.db.exec(`INSERT INTO ${this.table} (name, version, model, smaltml, saved_at, document) VALUES (?, ?, ?, ?, ?, ?)`, [
      name,
      version,
      doc.model,
      doc.smaltml,
      new Date().toISOString(),
      JSON.stringify(doc),
    ]);
    return version;
  }

  async get(name: string, version?: number): Promise<ModelDocument | undefined> {
    await this.#init();
    const rows =
      version === undefined
        ? await this.db.query(`SELECT document FROM ${this.table} WHERE name = ? ORDER BY version DESC LIMIT 1`, [name])
        : await this.db.query(`SELECT document FROM ${this.table} WHERE name = ? AND version = ?`, [name, version]);
    const text = rows[0]?.document;
    if (text === undefined) return undefined;
    try {
      return JSON.parse(String(text)) as ModelDocument;
    } catch {
      return fail('ML_FORMAT', `model ${name} in ${this.table} isn't JSON`);
    }
  }

  async list(): Promise<ModelEntry[]> {
    await this.#init();
    const rows = await this.db.query(`SELECT name, version, model, saved_at FROM ${this.table} ORDER BY name, version`);
    return rows.map((r) => ({ name: String(r.name), version: Number(r.version), model: String(r.model), savedAt: String(r.saved_at) }));
  }

  async delete(name: string, version?: number): Promise<number> {
    await this.#init();
    const before = (await this.list()).filter((e) => e.name === name && (version === undefined || e.version === version)).length;
    if (version === undefined) await this.db.exec(`DELETE FROM ${this.table} WHERE name = ?`, [name]);
    else await this.db.exec(`DELETE FROM ${this.table} WHERE name = ? AND version = ?`, [name, version]);
    return before;
  }
}

/** Fitted models by name: save() a model as a new version, load() or predict() with one by name. */
export class ModelRegistry {
  readonly store: ModelStore;
  /** Loaded models, by name and version, so predict() decodes a document once. */
  #cache = new Map<string, Model>();

  constructor(store: ModelStore = new MemoryStore()) {
    this.store = store;
  }

  /** Saves a fitted model as the next version of `name`, and returns the version. */
  async save(name: string, model: Model): Promise<number> {
    if (!model.fitted) fail('ML_NOT_FITTED', `this ${model.kind} isn't fitted: fit it before saving it as ${name}`);
    const version = await this.store.put(name, model.toJSON());
    this.#cache.delete(`${name}\u0000latest`);
    return version;
  }

  /** The model saved as `name`, at `version` or its latest. Fails with ML_FORMAT when there is none. */
  async load(name: string, version?: number): Promise<Model> {
    const key = `${name}\u0000${version ?? 'latest'}`;
    const cached = this.#cache.get(key);
    if (cached) return cached;
    const doc = await this.store.get(name, version);
    if (!doc) fail('ML_FORMAT', `no model is saved as ${name}${version === undefined ? '' : ` version ${version}`}`);
    const model = loadModel(checkDocument(doc));
    this.#cache.set(key, model);
    return model;
  }

  /** Every saved version. */
  list(): Promise<ModelEntry[]> {
    return this.store.list();
  }

  /** Removes `name` at `version`, or every version of it. */
  async delete(name: string, version?: number): Promise<number> {
    for (const key of [...this.#cache.keys()]) if (key.startsWith(`${name}\u0000`)) this.#cache.delete(key);
    return this.store.delete(name, version);
  }

  /** The predictions of the model saved as `name`. */
  async predict(name: string, X: MatrixInput, options: PredictOptions = {}): Promise<Float64Array | Label[]> {
    return predict(await this.load(name, options.version), X, options);
  }

  /** The class probabilities of the classifier saved as `name`. */
  async predictProba(name: string, X: MatrixInput, options: PredictOptions = {}): Promise<Matrix> {
    return predictProba(await this.load(name, options.version), X, options);
  }
}

export interface PredictOptions {
  /** Where the work runs (The CPU, WebGPU and workers). */
  device?: Device;
  /** The registry a model name is looked up in. Default: the one useRegistry() set, else an in-memory one. */
  registry?: ModelRegistry;
  /** The version of a named model. Default: its latest. */
  version?: number;
}

let current: ModelRegistry | undefined;

/** The registry predict() looks names up in, unless a call names another. */
export function useRegistry(registry: ModelRegistry): void {
  current = registry;
}

/** The registry predict() looks names up in: the one useRegistry() set, else one in memory. */
export function defaultRegistry(): ModelRegistry {
  return (current ??= new ModelRegistry());
}

/**
 * The predictions of a model, or of the model saved under a name: `await predict(model, X)` or
 * `await predict('churn', X)`. A model name is looked up in `options.registry`, else the default registry.
 */
export async function predict(model: Model | string, X: MatrixInput, options: PredictOptions = {}): Promise<Float64Array | Label[]> {
  const m = typeof model === 'string' ? await (options.registry ?? defaultRegistry()).load(model, options.version) : model;
  if (!isPredictor(m)) fail('ML_PARAM', `${m.kind} doesn't predict`);
  return m.predict(X, options.device ? { device: options.device } : {});
}

/** Each class's probability from a classifier, or from the classifier saved under a name. */
export async function predictProba(model: Model | string, X: MatrixInput, options: PredictOptions = {}): Promise<Matrix> {
  const m = typeof model === 'string' ? await (options.registry ?? defaultRegistry()).load(model, options.version) : model;
  const proba = (m as Partial<{ predictProba(X: MatrixInput, o: object): Promise<Matrix> }>).predictProba;
  if (!isClassifier(m) && !(m.kind === 'Pipeline' && proba)) fail('ML_PARAM', `${m.kind} doesn't give probabilities`);
  return (m as unknown as { predictProba(X: MatrixInput, o: object): Promise<Matrix> }).predictProba(X, options.device ? { device: options.device } : {});
}

function checkName(name: string): void {
  if (typeof name !== 'string' || name.length === 0 || name.includes('\u0000')) fail('ML_PARAM', 'a model name is a non-empty string');
}
