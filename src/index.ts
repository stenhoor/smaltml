/**
 * smaltml: machine learning for smaltdb, on the hardware you have. Every area is also an entry point of its own
 * (`@smaltworks/smaltml/linear` and so on), so an app loads only what it imports.
 * @module
 */
export * from './core/index.js';
export * from './preprocessing/index.js';
export * from './linear/index.js';
export * from './metrics/index.js';
export * from './selection/index.js';
export * from './pipeline/index.js';
export * from './io/index.js';
export * from './gpu/index.js';
export * from './registry/index.js';
