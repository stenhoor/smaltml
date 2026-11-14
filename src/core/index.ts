/**
 * The core: smaltml's matrix, errors, model interfaces and seeded random numbers.
 * @module
 */
export { Matrix, type MatrixInput, type VectorInput } from './matrix.js';
export { type Category, type TableInput } from './table.js';
export { SmaltmlError, type ErrorCode } from './errors.js';
export { Random } from './random.js';
export {
  MODEL_FORMAT,
  MODEL_FORMAT_VERSION,
  type Classifier,
  type Label,
  type Labels,
  type Model,
  type ModelDocument,
  type Predictor,
  type Transformer,
} from './model.js';
export { VERSION } from '../version.js';
