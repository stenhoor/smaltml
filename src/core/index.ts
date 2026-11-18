/**
 * The core: smaltml's matrix, errors, model interfaces, seeded random numbers, and the kernels another set can replace.
 * @module
 */
export { Matrix, type MatrixInput, type VectorInput } from './matrix.js';
export { type Category, type TableInput } from './table.js';
export { SmaltmlError, type ErrorCode, type WarningCode } from './errors.js';
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
export {
  KERNELS_VERSION,
  kernelsInUse,
  kernelsReady,
  useKernels,
  type KernelName,
  type Kernels,
  type KernelSet,
  type KernelsInUse,
} from './kernel-set.js';
export { VERSION } from '../version.js';
