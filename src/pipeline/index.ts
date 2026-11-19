/**
 * Pipelines: steps that run in order as one model. scikit-learn's `Pipeline` and `make_pipeline`.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import type { TableInput } from '../core/table.js';
import {
  checkDocument,
  document,
  isClassifier,
  isPredictor,
  isTransformer,
  type Label,
  type Labels,
  type Model,
  type ModelDocument,
  type Transformer,
} from '../core/model.js';

/** The models whose fit takes `sampleWeight`; a pipeline given weights fails for any other last step. */
const WEIGHTED = new Set(['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression', 'DecisionTreeRegressor', 'DecisionTreeClassifier', 'RandomForestRegressor', 'RandomForestClassifier', 'ExtraTreesRegressor', 'ExtraTreesClassifier']);

function checkWeighted(last: Model, options: { sampleWeight?: unknown }): void {
  if (options.sampleWeight !== undefined && !WEIGHTED.has(last.kind)) {
    fail('ML_PARAM', `sampleWeight isn't supported by ${last.kind} yet: leave it out, or end the pipeline with a model that takes it`);
  }
}

/** A step: a model, named after its class unless you name it (`['scale', new StandardScaler()]`). */
export type Step = Model | readonly [string, Model];

/**
 * Steps that run in order as one model: every step but the last transforms X for the next; the last one predicts or
 * transforms. fit() fits each step on what the steps before it give. The pipeline does what its last step does:
 * predict(), predictProba() and transform() are there when the last step has them.
 */
export class Pipeline implements Model {
  readonly kind = 'Pipeline';
  /** The steps as [name, model] pairs. */
  readonly steps: readonly (readonly [string, Model])[];

  constructor(steps: readonly Step[]) {
    if (steps.length === 0) fail('ML_PARAM', 'a pipeline needs at least one step');
    const named = steps.map((s) => (Array.isArray(s) ? (s as readonly [string, Model]) : ([(s as Model).kind.toLowerCase(), s as Model] as const)));
    const seen = new Set<string>();
    for (const [name] of named) {
      if (seen.has(name)) fail('ML_PARAM', `two steps are named ${name}: name them, as ['${name}2', model]`);
      seen.add(name);
    }
    named.slice(0, -1).forEach(([name, model]) => {
      if (!isTransformer(model)) fail('ML_PARAM', `step ${name} (${model.kind}) comes before the last step, so it must transform X`);
    });
    // Its predict() gives a Matrix, which a pipeline's doesn't yet.
    if (named[named.length - 1]![1].kind === 'MultiOutputRegressor') fail('ML_PARAM', "a pipeline can't end in a MultiOutputRegressor yet: transform X first, then fit the MultiOutputRegressor on it");
    this.steps = named;
  }

  get fitted(): boolean {
    return this.steps.every(([, m]) => m.fitted);
  }

  /** The last step. */
  get last(): Model {
    return this.steps[this.steps.length - 1]![1];
  }

  /** The step with this name. */
  step(name: string): Model {
    const found = this.steps.find(([n]) => n === name);
    return found ? found[1] : fail('ML_PARAM', `no step is named ${name}`);
  }

  /** The last step's classes, when it is a classifier. */
  get classes(): readonly Label[] {
    const last = this.last;
    return isClassifier(last) ? last.classes : fail('ML_PARAM', `the last step, ${last.kind}, isn't a classifier`);
  }

  getParams(): Record<string, unknown> {
    return Object.fromEntries(this.steps.map(([n, m]) => [n, m.getParams()]));
  }

  /**
   * Fits each step. The last step's own fit() gets the options: `threads`, so a heavy model at the end runs on several
   * cores, and `sampleWeight`, the rows' weights for the model. The steps before it fit without weights.
   */
  async fit(X: TableInput, y?: Labels, options: { threads?: number | 'auto'; sampleWeight?: ArrayLike<number> } = {}): Promise<this> {
    checkWeighted(this.last, options);
    let data = X as MatrixInput;
    const steps = this.steps.map(([, m]) => m);
    for (const model of steps.slice(0, -1)) data = (model as Transformer).fitSync(data, y).transformSync(data);
    await (this.last as unknown as { fit(X: MatrixInput, y?: Labels, o?: object): Promise<unknown> }).fit(data, y, options);
    return this;
  }

  /** fit() on this thread; `sampleWeight` reaches the last step, as in fit(). */
  fitSync(X: TableInput, y?: Labels, options: { sampleWeight?: ArrayLike<number> } = {}): this {
    checkWeighted(this.last, options);
    // The first step gets X as it came, so it may hold text for an encoder or a ColumnTransformer.
    let data = X as MatrixInput;
    const last = this.steps.length - 1;
    this.steps.forEach(([, model], k) => {
      if (k < last) data = (model as Transformer).fitSync(data, y).transformSync(data);
      else (model as unknown as { fitSync(X: MatrixInput, y?: Labels, o?: object): unknown }).fitSync(data, y, options);
    });
    return this;
  }

  /** X through the steps, then the last step's predict(); `device` reaches the last step. */
  async predict(X: TableInput, options: { device?: 'auto' | 'cpu' | 'gpu' } = {}): Promise<Float64Array | Label[]> {
    const last = this.last;
    if (!isPredictor(last)) fail('ML_PARAM', `the last step, ${last.kind}, doesn't predict`);
    return last.predict(this.#through(X), options);
  }

  predictSync(X: TableInput): Float64Array | Label[] {
    const last = this.last;
    if (!isPredictor(last)) fail('ML_PARAM', `the last step, ${last.kind}, doesn't predict`);
    return last.predictSync(this.#through(X));
  }

  async predictProba(X: TableInput, options: { device?: 'auto' | 'cpu' | 'gpu' } = {}): Promise<Matrix> {
    const last = this.last;
    if (!isClassifier(last)) fail('ML_PARAM', `the last step, ${last.kind}, doesn't give probabilities`);
    return last.predictProba(this.#through(X), options);
  }

  predictProbaSync(X: TableInput): Matrix {
    const last = this.last;
    if (!isClassifier(last)) fail('ML_PARAM', `the last step, ${last.kind}, doesn't give probabilities`);
    return last.predictProbaSync(this.#through(X));
  }

  async transform(X: TableInput): Promise<Matrix> {
    return this.transformSync(X);
  }

  transformSync(X: TableInput): Matrix {
    const last = this.last;
    if (!isTransformer(last)) fail('ML_PARAM', `the last step, ${last.kind}, doesn't transform`);
    return last.transformSync(this.#through(X));
  }

  async fitTransform(X: TableInput, y?: Labels): Promise<Matrix> {
    return this.fitSync(X, y).transformSync(X);
  }

  toJSON(): ModelDocument {
    // Each step's own document says whether it is fitted.
    return document(this.kind, { steps: this.steps.map(([name]) => name) }, { steps: this.steps.map(([, m]) => m.toJSON()) });
  }

  /** A pipeline from its document; `load` turns each step's document into its model (loadModel() does). */
  static fromJSON(doc: ModelDocument, load: (doc: ModelDocument) => Model): Pipeline {
    const d = checkDocument(doc, 'Pipeline');
    const names = d.params.steps;
    const steps = d.state?.steps;
    if (!Array.isArray(names) || !Array.isArray(steps) || names.length !== steps.length) {
      fail('ML_FORMAT', 'a Pipeline document needs a name and a model for each step');
    }
    return new Pipeline(names.map((n, k) => [String(n), load(steps[k] as ModelDocument)] as const));
  }

  /** X through every step but the last. */
  #through(X: TableInput): MatrixInput {
    let data = X as MatrixInput;
    for (const [, model] of this.steps.slice(0, -1)) data = (model as Transformer).transformSync(data);
    return data;
  }
}

/** A pipeline of these steps, each named after its class in lower case (`standardscaler`). scikit-learn's `make_pipeline`. */
export function pipeline(...steps: Step[]): Pipeline {
  return new Pipeline(steps);
}

export { ColumnTransformer, type ColumnPart, type ColumnSelection, type ColumnTransformerParams } from './column-transformer.js';
