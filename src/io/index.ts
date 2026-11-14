/**
 * Saving and loading models as JSON (ADR 0024, section 6).
 * @module
 */
import { fail } from '../core/errors.js';
import { checkDocument, type Model, type ModelDocument } from '../core/model.js';
import { LinearRegression } from '../linear/linear-regression.js';
import { LogisticRegression } from '../linear/logistic-regression.js';
import { Pipeline } from '../pipeline/index.js';
import { StandardScaler } from '../preprocessing/standard-scaler.js';

export { MODEL_FORMAT, MODEL_FORMAT_VERSION, type ModelDocument } from '../core/model.js';

const LOADERS: Record<string, (doc: ModelDocument) => Model> = {
  StandardScaler: (d) => StandardScaler.fromJSON(d),
  LinearRegression: (d) => LinearRegression.fromJSON(d),
  LogisticRegression: (d) => LogisticRegression.fromJSON(d),
  Pipeline: (d) => Pipeline.fromJSON(d, loadDocument),
};

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
  if (!load) fail('ML_FORMAT', `this smaltml can't load a ${d.model} model`);
  return load(d);
}
