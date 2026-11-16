/**
 * Saving and loading models as JSON (ADR 0024, section 6).
 * @module
 */
import { fail } from '../core/errors.js';
import { checkDocument, type Model, type ModelDocument } from '../core/model.js';
import { LinearRegression } from '../linear/linear-regression.js';
import { LogisticRegression } from '../linear/logistic-regression.js';
import { ElasticNet, Lasso, Ridge } from '../linear/penalized.js';
import { ColumnTransformer, Pipeline } from '../pipeline/index.js';
import { MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, SimpleImputer, StandardScaler } from '../preprocessing/index.js';

export { MODEL_FORMAT, MODEL_FORMAT_VERSION, type ModelDocument } from '../core/model.js';

const LOADERS: Record<string, (doc: ModelDocument) => Model> = {
  StandardScaler: (d) => StandardScaler.fromJSON(d),
  MinMaxScaler: (d) => MinMaxScaler.fromJSON(d),
  RobustScaler: (d) => RobustScaler.fromJSON(d),
  SimpleImputer: (d) => SimpleImputer.fromJSON(d),
  OneHotEncoder: (d) => OneHotEncoder.fromJSON(d),
  OrdinalEncoder: (d) => OrdinalEncoder.fromJSON(d),
  LinearRegression: (d) => LinearRegression.fromJSON(d),
  LogisticRegression: (d) => LogisticRegression.fromJSON(d),
  Ridge: (d) => Ridge.fromJSON(d),
  Lasso: (d) => Lasso.fromJSON(d),
  ElasticNet: (d) => ElasticNet.fromJSON(d),
  Pipeline: (d) => Pipeline.fromJSON(d, loadDocument),
  ColumnTransformer: (d) => ColumnTransformer.fromJSON(d, loadDocument),
};

/**
 * Teaches loadModel() a model class from outside this module (GridSearchCV registers itself when
 * `@smaltworks/smaltml/selection` loads, and MultiOutputRegressor when `@smaltworks/smaltml/multioutput` does, as io
 * can't import either without a cycle).
 */
export function registerLoader(kind: string, load: (doc: ModelDocument) => Model): void {
  LOADERS[kind] = load;
}

/** The model as JSON text: its parameters and, once fitted, what it learned. */
export function saveModel(model: Model): string {
  return JSON.stringify(model.toJSON());
}

/** A model from the JSON text saveModel() wrote, or from its parsed document. */
export function loadModel(json: string | ModelDocument): Model {
  let doc: unknown = json;
  if (typeof json === 'string') {
    try {
      doc = JSON.parse(json);
    } catch (e) {
      fail('ML_FORMAT', `not JSON: ${(e as Error).message}`);
    }
  }
  return loadDocument(doc as ModelDocument);
}

function loadDocument(doc: ModelDocument): Model {
  const d = checkDocument(doc);
  const load = LOADERS[d.model];
  if (!load) fail('ML_FORMAT', `this smaltml can't load a ${d.model} model${d.model === 'GridSearchCV' ? ': import @smaltworks/smaltml/selection first' : d.model === 'MultiOutputRegressor' ? ': import @smaltworks/smaltml/multioutput first' : ''}`);
  return load(d);
}
