/**
 * smaltml: machine learning for smaltdb, on the hardware you have. Every area is also an entry point of its own
 * (`@smaltworks/smaltml/linear` and so on), so an app loads only what it imports.
 * @module
 */
export * from './core/index.js';
export * from './preprocessing/index.js';
export * from './linear/index.js';
export * from './tree/index.js';
export * from './ensemble/index.js';
export * from './metrics/index.js';
export * from './selection/index.js';
export * from './pipeline/index.js';
export * from './multioutput/index.js';
export * from './io/index.js';
export * from './gpu/index.js';
export * from './registry/index.js';
export {
  configureThreads,
  sharedMemoryAvailable,
  stopThreads,
  PARALLEL_MIN_VALUES,
  type ThreadOptions,
  type Threads,
} from './parallel/index.js';
