/** What the main thread and the thread pool's workers say to each other. */
import type { Label, ModelDocument } from '../core/model.js';

/** A matrix in shared memory: its numbers in a SharedArrayBuffer, column by column. */
export interface SharedMatrix {
  buffer: SharedArrayBuffer;
  rows: number;
  cols: number;
}

/** The work for one worker: an operation over a run of whole chunks of rows. */
export type Task =
  | { op: 'gram'; X: SharedMatrix; mean: Float64Array | null; weights: SharedArrayBuffer | null; ones: boolean; chunks: [number, number][] }
  | { op: 'xtv'; X: SharedMatrix; v: SharedArrayBuffer; mean: Float64Array | null; chunks: [number, number][] }
  | { op: 'xw'; X: SharedMatrix; w: Float64Array; b: number; mean: Float64Array | null; out: SharedArrayBuffer; chunks: [number, number][] }
  /** Before any other task, where useKernels() replaced kernels: import `module` and use the set its default export gives. */
  | { op: 'kernels'; module: string; options: unknown }
  | { op: 'logistic'; X: SharedMatrix; y: SharedArrayBuffer; K: number; intercept: boolean; x: Float64Array; weights: SharedArrayBuffer | null; chunks: [number, number][] }
  /** A forest's trees, one per stream (src/ensemble/grow.ts): the trees' parameters, never their seeds. */
  | {
      op: 'trees';
      X: SharedMatrix;
      names: string[] | null;
      classifier: boolean;
      y: Float64Array | Label[];
      weights: Float64Array | null;
      params: Record<string, unknown>;
      bootstrap: boolean;
      draws: number;
      oob: boolean;
      streams: number[];
    };

export interface Request {
  id: number;
  task: Task;
}

/** One result per chunk, in the order of the task's chunks: a vector, or a loss and its gradient. */
export type Response =
  | {
      id: number;
      ok: true;
      parts: { loss?: number; values: Float64Array }[];
      kernels?: string[];
      /** The trees of a 'trees' task, in stream order, each with its out-of-bag rows and predictions. */
      trees?: { doc: ModelDocument; outOfBag: Int32Array | null; oob: Float64Array | null }[];
    }
  | { id: number; ok: false; message: string };
